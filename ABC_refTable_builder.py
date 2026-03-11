#!/usr/bin/env python3
'''
ABC reference-table generator for two 1D stepping-stone models in msprime.


- Model 1: ancestral -> N demes at T1; 1D stepping-stone migration.
- Model 2 (SC): ancestral -> left/right at T1; left splits at T2L into nL demes,
          right splits at T2R into nR demes; 1D stepping-stone within lineages;
          secondary contact L(last) <-> R0 from T3. ( supports ASYMMETRIC bridge.)
- Model 3 (NO-SC): same as Model 2 but NO secondary contact/bridge at any time.
- Priors on times, Ne (ancestral, lineages, demes), migration either in m (probabilities)
  or in Nem (expected number of migrants per generation), as configured.
  * Model 1 within-chain: symmetric; draw from m or Nem.
  * Model 2 within-chain (Left/Right): symmetric within each side; draw from m or Nem.
  * Model 2 bridge (L->R and R->L): asymmetric; draw from m or from Nem per direction.
- Choose up to 5 demes to sample per model (5 diploids per deme).
- Summaries:
    per pop: pi, thetaW, Tajima's D, folded 1D SFS bins per bp (MAC 1..half; can drop MAC1)
    per pair: dXY, dA, FST (Hudson).
- Two compute paths:
    from_sfs = False ? tskit stats (fast), optional across-locus variances
    from_sfs = True  ? derive ALL stats from SFS (no variances)
- Parallelism: process pool across prior draws (`jobs:` in config).
- Outputs:
    outdir/
      +- abc_summary.txt
      +- effective_config.json
      +- ref_table_model1.csv
      +- ref_table_model2.csv
      +- ref_table_model3.csv
'''

import os, sys, csv, math, json, random
from datetime import datetime
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import msprime

# ------------------------- load config -------------------------

def load_config(path):
    if not os.path.exists(path):
        raise SystemExit(f"Config file not found: {path}")
    if path.lower().endswith((".yml", ".yaml")):
        try:
            import yaml
        except ImportError:
            raise SystemExit("YAML config requested but PyYAML not installed. `pip install pyyaml`.")
        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        with open(path, "r") as f:
            cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise SystemExit("Config must be a key:value mapping.")
    return cfg

DEFAULTS = dict(
    # core sim design
    mu=None,             # required (per-bp per-generation)
    recomb_rate=0.0,     # NEW: per-bp per-generation recombination (rho). Set >0 to enable recombination within loci.
    reps=10_000,         # number of independent loci
    length=150.0,        # bp per locus
    seed=None,           # base seed (int) or None
    from_sfs=False,      # derive all stats from SFS (no variances)
    variance=False,      # across-locus variances (only if from_sfs=False)
    no_singleton=False,  # drop folded MAC1 bin

    # parallel rows
    jobs=1,              # processes across prior draws

    # deme counts
    m1_n=20,
    m2_nl=10,
    m2_nr=10,

    # sampled pops (=5 per model, comma-separated string is accepted)
    m1_pops="P0,P9,P19",
    m2_pops="L5,R5",

    # time priors (T1 absolute)
    prior_t1_min=1e2,  prior_t1_max=1e6,

    # ---- fraction priors for T2 and T3 (Beta on [0,1] mapped to [min,max]) ----
    t2_frac_min=0.1,
    t2_frac_max=0.9,
    t3_frac_min=0.1,
    t3_frac_max=0.9,
    t2_beta_alpha=2.0,
    t2_beta_beta=1.0,
    t3_beta_alpha=3.0,
    t3_beta_beta=1.2,

    # ---- migration priors directly in m-space (probabilities) ----
    # Within-block priors for Left and Right (Model 2) and Model 1
    m_neighbor1_min=1.0e-5,  # Model 1 within-chain m
    m_neighbor1_max=0.2,
    m_neighborL_min=1.0e-5,  # Model 2 within Left chain m
    m_neighborL_max=0.2,
    m_neighborR_min=1.0e-5,  # Model 2 within Right chain m
    m_neighborR_max=0.2,

    # Bridge migration priors (Model 2)  ASYMMETRIC m
    m_bridge_L2R_min=1.0e-6,
    m_bridge_L2R_max=1.0e-2,
    m_bridge_R2L_min=1.0e-6,
    m_bridge_R2L_max=1.0e-2,

    # ---- Nem priors (migrants per generation) ----
    # Toggles for drawing within-chain from Nem for M1 and M2
    draw_within_from_Nem=False,  # if True, use Nem to derive m for Model 1 and Model 2 within-chains
    # Model 1 within-chain Nem
    Nem_neighbor1_min=1e-4,
    Nem_neighbor1_max=25.0,
    # Model 2 within-chains Nem
    Nem_neighborL_min=1e-4,
    Nem_neighborL_max=25.0,
    Nem_neighborR_min=1e-4,
    Nem_neighborR_max=25.0,

    # Bridge (Model 2)  draw from Nem per direction if enabled
    draw_bridge_from_Nem=False,
    Nem_bridge_L2R_min=1e-5,
    Nem_bridge_L2R_max=5e-1,
    Nem_bridge_R2L_min=1e-5,
    Nem_bridge_R2L_max=5e-1,

    # Choice of distribution for Nem and m draws
    use_loguniform_Nem=False,  # if True, draw Nem from log-uniform between the given min/max
    use_loguniform_m=False,    # if True, draw m from log-uniform between the given min/max

    # Safety cap on migration probabilities
    mig_cap=0.5,

    # Ne priors
    # Model 1
    prior_NeA_min=1e2,  prior_NeA_max=2e5,
    prior_NeD1_min=1e2, prior_NeD1_max=2e5,
    # Model 2 / 3
    prior_NeA2_min=1e2,   prior_NeA2_max=2e5,
    prior_NeLlin_min=1e2, prior_NeLlin_max=2e5,
    prior_NeRlin_min=1e2, prior_NeRlin_max=2e5,
    prior_NeDL_min=1e2,   prior_NeDL_max=2e5,
    prior_NeDR_min=1e2,   prior_NeDR_max=2e5,

    # number of rows per model
    n_sims1=100,
    n_sims2=100,   # Model 2 (SC)
    n_sims3=100,   # Model 3 (NO-SC)

    # output
    outdir=None,
)

def resolve_config(user_cfg):
    cfg = DEFAULTS.copy()
    cfg.update({k: v for k, v in user_cfg.items() if v is not None})
    if cfg["mu"] is None:
        raise SystemExit("Config must set `mu` (mutation rate per bp per generation).")

    # normalize population lists: accept a single comma-separated string
    for key in ("m1_pops", "m2_pops"):
        s = cfg[key]
        cfg[key] = ",".join([x.strip() for x in str(s).split(",") if x.strip()])

    if cfg["from_sfs"] and cfg["variance"]:
        raise SystemExit("`from_sfs: true` is incompatible with `variance: true`.")
    cfg["jobs"] = max(1, int(cfg["jobs"] or 1))

    # ---- force numeric types to avoid 'str' values from YAML ----
    _as_int = [
        "jobs","reps","n_sims1","n_sims2","n_sims3",
        "m1_n","m2_nl","m2_nr","n_diploid_per_pop","seed"
    ]
    _as_float = [
        "mu","recomb_rate","length",
        "prior_t1_min","prior_t1_max",
        "t2_frac_min","t2_frac_max","t3_frac_min","t3_frac_max",
        "t2_beta_alpha","t2_beta_beta","t3_beta_alpha","t3_beta_beta",
        "prior_NeA_min","prior_NeA_max","prior_NeD1_min","prior_NeD1_max",
        "prior_NeA2_min","prior_NeA2_max","prior_NeLlin_min","prior_NeLlin_max",
        "prior_NeRlin_min","prior_NeRlin_max","prior_NeDL_min","prior_NeDL_max",
        "prior_NeDR_min","prior_NeDR_max",
        "m_neighbor1_min","m_neighbor1_max",
        "m_neighborL_min","m_neighborL_max","m_neighborR_min","m_neighborR_max",
        "m_bridge_L2R_min","m_bridge_L2R_max","m_bridge_R2L_min","m_bridge_R2L_max",
        "Nem_neighbor1_min","Nem_neighbor1_max",
        "Nem_neighborL_min","Nem_neighborL_max","Nem_neighborR_min","Nem_neighborR_max",
        "Nem_bridge_L2R_min","Nem_bridge_L2R_max","Nem_bridge_R2L_min","Nem_bridge_R2L_max",
        "mig_cap",
    ]
    for k in _as_int:
        if k in cfg and cfg[k] is not None:
            cfg[k] = int(cfg[k])
    for k in _as_float:
        if k in cfg and cfg[k] is not None:
            cfg[k] = float(cfg[k])

    # booleans (handle YAML strings)
    for bkey in (
        "from_sfs","variance","no_singleton",
        "draw_within_from_Nem","draw_bridge_from_Nem",
        "use_loguniform_Nem","use_loguniform_m"
    ):
        if isinstance(cfg.get(bkey), str):
            cfg[bkey] = cfg[bkey].strip().lower() in ("1","true","yes","y","on")
        else:
            cfg[bkey] = bool(cfg.get(bkey, False))

    return cfg

# ------------------------- utils -------------------------

DIPLOIDS_PER_DEME = 5
HAP_PER_DEME = 2 * DIPLOIDS_PER_DEME

def parse_pop_list(s, prefixes, nL=None, nR=None, nP=None):
    if not s:
        return []
    pops = [x.strip() for x in s.split(",") if x.strip()]
    if len(pops) > 5:
        raise SystemExit("Please list at most 5 populations per model.")
    for p in pops:
        ok = any(p.startswith(pref) for pref in prefixes)
        if not ok:
            raise SystemExit(f"Bad population name '{p}' (expected prefixes {prefixes}).")
        if p.startswith("P") and nP is not None:
            idx = int(p[1:])
            if not (0 <= idx < nP):
                raise SystemExit(f"{p} invalid for Model 1 with m1_n={nP}.")
        if (p.startswith("L") or p.startswith("R")) and (nL is not None and nR is not None):
            lab = p[0]; idx = int(p[1:])
            if lab == "L" and not (0 <= idx < nL): raise SystemExit(f"{p} invalid for m2_nl={nL}.")
            if lab == "R" and not (0 <= idx < nR): raise SystemExit(f"{p} invalid for m2_nr={nR}.")
    return pops

def harmonic_numbers(n):
    a1 = sum(1.0/i for i in range(1, n))
    a2 = sum(1.0/(i*i) for i in range(1, n))
    return a1, a2

def tajimas_D_from_totals(n_hap, S_total, pi_total):
    if S_total <= 0:
        return 0.0
    a1, a2 = harmonic_numbers(n_hap)
    b1 = (n_hap + 1) / (3 * (n_hap - 1))
    b2 = 2 * (n_hap**2 + n_hap + 3) / (9 * (n_hap * (n_hap - 1)))
    c1 = b1 - (1 / a1)
    c2 = b2 - (n_hap + 2) / (a1 * n_hap) + (a2 / (a1 * a1))
    e1 = c1 / a1
    e2 = c2 / (a1*a1 + a2)
    denom = math.sqrt(e1 * S_total + e2 * S_total * (S_total - 1))
    if denom == 0:
        return 0.0
    return (pi_total - (S_total / a1)) / denom

def fold_unfolded_1d_sfs(sfs_unfolded):
    n = len(sfs_unfolded) - 1
    half = n // 2
    folded = np.zeros(half + 1, dtype=float)
    for k in range(0, n + 1):
        folded[min(k, n - k)] += sfs_unfolded[k]
    return folded

def weight_pi_1d(n, i):
    return (2.0 * i * (n - i)) / (n * (n - 1)) if n > 1 else 0.0

def weight_dxy_2d(n1, n2, i, j):
    return (i / n1) + (j / n2) - (2.0 * i * j) / (n1 * n2)

def pair_key(a, b):
    return tuple(sorted((a, b)))

# ---- helpers for Beta-fraction priors ----

def _beta_draw(alpha, beta, rng):
    x = rng.gammavariate(alpha, 1.0)
    y = rng.gammavariate(beta,  1.0)
    if x <= 0.0 and y <= 0.0:
        return 0.5
    return x / (x + y)

def _map01(x, lo, hi):
    return lo + (hi - lo) * x

# ---- log-uniform helper for Nem/m draws ----

def _logU(rng, a, b):
    """Draw from log-uniform on (a,b). Bounds must be > 0."""
    a = float(a); b = float(b)
    if a <= 0.0 or b <= 0.0:
        raise ValueError("log-uniform bounds must be > 0")
    lo = math.log(min(a, b))
    hi = math.log(max(a, b))
    return math.exp(rng.uniform(lo, hi))

# ------------------------- demographies (unchanged topology; args are m) -------------------------

def model1_demography(T1, NeA, NeDeme, m_within, n_demes=20):
    """
    m_within is a migration probability (symmetric), used directly.
    """
    dem = msprime.Demography()
    dem.add_population(name="ancestral", initial_size=NeA)

    pops = [f"P{i}" for i in range(n_demes)]
    for p in pops:
        dem.add_population(name=p, initial_size=NeDeme)

    # Single multi-derived split (ancestor -> all demes) at time T1
    dem.add_population_split(time=T1, derived=pops, ancestral="ancestral")

    # Neighbor (stepping-stone) symmetric migration
    m = m_within
    for i in range(n_demes - 1):
        dem.set_migration_rate(pops[i], pops[i+1], m)
        dem.set_migration_rate(pops[i+1], pops[i], m)

    dem.sort_events()
    return dem


def model2_demography(T1, T2L, T2R, T3,
                      NeA, NeL_line, NeR_line, NeDL, NeDR,
                      m_within_left, m_within_right,
                      m_bridge_L2R, m_bridge_R2L,
                      nL=10, nR=10):
    """
    m_within_left / m_within_right are migration probabilities (symmetric)
    applied along the left/right chains; bridge is ASYMMETRIC with distinct
    m for L->R and R->L switched on at time T3 (Model 2 only).
    """
    if nL > 100 or nR > 100:
        raise SystemExit("msprime allows =100 derived pops per split; choose m2_nl/m2_nr = 100.")
    dem = msprime.Demography()
    dem.add_population(name="ancestral", initial_size=NeA)
    dem.add_population(name="left",  initial_size=NeL_line)
    dem.add_population(name="right", initial_size=NeR_line)
    dem.add_population_split(time=T1, derived=["left", "right"], ancestral="ancestral")

    L = [f"L{i}" for i in range(nL)]
    R = [f"R{i}" for i in range(nR)]
    for p in L: dem.add_population(name=p, initial_size=NeDL)
    for p in R: dem.add_population(name=p, initial_size=NeDR)

    dem.add_population_split(time=T2L, derived=L, ancestral="left")
    dem.add_population_split(time=T2R, derived=R, ancestral="right")

    mL = m_within_left
    mR = m_within_right
    for i in range(nL - 1):
        dem.set_migration_rate(L[i], L[i+1], mL)
        dem.set_migration_rate(L[i+1], L[i], mL)
    for i in range(nR - 1):
        dem.set_migration_rate(R[i], R[i+1], mR)
        dem.set_migration_rate(R[i+1], R[i], mR)

    if nL >= 1 and nR >= 1:
        dem.add_migration_rate_change(time=T3, source=f"L{nL-1}", dest="R0", rate=m_bridge_L2R)
        dem.add_migration_rate_change(time=T3, source="R0",     dest=f"L{nL-1}", rate=m_bridge_R2L)

    dem.sort_events()
    return dem

def model3_demography(T1, T2L, T2R,
                      NeA, NeL_line, NeR_line, NeDL, NeDR,
                      m_within_left, m_within_right,
                      nL=10, nR=10):
    """
    Model 3: same as Model 2 but NO secondary contact (no bridge).
    Left/right chains can have distinct symmetric migration rates.
    """
    if nL > 100 or nR > 100:
        raise SystemExit("msprime allows =100 derived pops per split; choose m2_nl/m2_nr = 100.")
    dem = msprime.Demography()
    dem.add_population(name="ancestral", initial_size=NeA)
    dem.add_population(name="left",  initial_size=NeL_line)
    dem.add_population(name="right", initial_size=NeR_line)
    dem.add_population_split(time=T1, derived=["left", "right"], ancestral="ancestral")

    L = [f"L{i}" for i in range(nL)]
    R = [f"R{i}" for i in range(nR)]
    for p in L: dem.add_population(name=p, initial_size=NeDL)
    for p in R: dem.add_population(name=p, initial_size=NeDR)

    dem.add_population_split(time=T2L, derived=L, ancestral="left")
    dem.add_population_split(time=T2R, derived=R, ancestral="right")

    mL = m_within_left
    mR = m_within_right
    for i in range(nL - 1):
        dem.set_migration_rate(L[i], L[i+1], mL)
        dem.set_migration_rate(L[i+1], L[i], mL)
    for i in range(nR - 1):
        dem.set_migration_rate(R[i], R[i+1], mR)
        dem.set_migration_rate(R[i+1], R[i], mR)

    dem.sort_events()
    return dem

# ------------------------- ancestry iterator -------------------------

def reps_iterator(demography, pop_names, L_bp, reps, workers, rand_seed=None, recomb_rate=0.0):
    samples = {name: DIPLOIDS_PER_DEME for name in pop_names}
    return msprime.sim_ancestry(
        samples=samples,
        sequence_length=L_bp,
        recombination_rate=recomb_rate,  # NEW: user-configurable recombination
        demography=demography,
        num_replicates=reps,
        ploidy=2,
        random_seed=rand_seed,
    )

# ------------------------- compute paths -------------------------

def compute_summary_tskit(demography, pop_names, reps, L_bp, mu,
                          with_variance=False, drop_singleton=False, workers=None, rand_seed=None,
                          out_labels=None, recomb_rate=0.0):
    """
    pop_names: original demography population names (e.g., P0,L5,...)
    out_labels: labels to use in output (e.g., ['P1','P2',...]) in the SAME order.
    """
    labels = out_labels if out_labels is not None else pop_names
    name_to_id = {p.name: i for i, p in enumerate(demography.populations)}
    L_total = reps * L_bp

    pi_totals   = {lab: 0.0 for lab in labels}
    S_totals    = {lab: 0   for lab in labels}
    sfs_unf_tot = {lab: np.zeros(HAP_PER_DEME + 1) for lab in labels}
    dxy_totals  = {pair_key(*x): 0.0 for x in combinations(labels, 2)}

    if with_variance:
        def mk(): return dict(n=0, s=0.0, ss=0.0)
        pi_acc    = {lab: mk() for lab in labels}
        thW_acc   = {lab: mk() for lab in labels}
        tajD_acc  = {lab: mk() for lab in labels}
        dxy_acc   = {pair_key(*x): mk() for x in combinations(labels, 2)}
        da_acc    = {pair_key(*x): mk() for x in combinations(labels, 2)}
        fst_acc   = {pair_key(*x): mk() for x in combinations(labels, 2)}

    reps_iter = reps_iterator(demography, pop_names, L_bp, reps, workers, rand_seed=rand_seed, recomb_rate=recomb_rate)

    for ts in reps_iter:
        mts = msprime.sim_mutations(ts, rate=mu, model="binary")
        node_sets = [mts.samples(population=name_to_id[p]) for p in pop_names]

        pi_site_vec = np.atleast_1d(
            mts.diversity(sample_sets=node_sets, mode="site", span_normalise=True)
        )

        perpop_sfs = []
        for idx, (p, lab) in enumerate(zip(pop_names, labels)):
            sfs_u = mts.allele_frequency_spectrum(
                sample_sets=[node_sets[idx]], polarised=True, mode="site", span_normalise=False
            )
            perpop_sfs.append(sfs_u)
            sfs_unf_tot[lab] += sfs_u
            S_i = int(np.sum(sfs_u[1:-1]))
            S_totals[lab] += S_i
            pi_totals[lab] += float(pi_site_vec[idx]) * L_bp

        for (i, j) in combinations(range(len(pop_names)), 2):
            lab1, lab2 = labels[i], labels[j]
            key = pair_key(lab1, lab2)
            dxy_site = mts.divergence(
                sample_sets=node_sets, indexes=[(i, j)],
                mode="site", span_normalise=True
            )[0]
            dxy_totals[key] += float(dxy_site) * L_bp

        if with_variance:
            a1, _ = harmonic_numbers(HAP_PER_DEME)
            for idx, lab in enumerate(labels):
                S_i = int(np.sum(perpop_sfs[idx][1:-1]))
                pi_site = float(pi_site_vec[idx])
                theta_site = (S_i / a1) / L_bp
                tajD_i = tajimas_D_from_totals(HAP_PER_DEME, S_i, pi_site * L_bp)
                for acc, val in ((pi_acc[lab], pi_site), (thW_acc[lab], theta_site), (tajD_acc[lab], tajD_i)):
                    acc["n"] += 1; acc["s"] += val; acc["ss"] += val * val

            pi_map = {lab: float(v) for lab, v in zip(labels, pi_site_vec)}
            for (i, j) in combinations(range(len(pop_names)), 2):
                lab1, lab2 = labels[i], labels[j]
                key = pair_key(lab1, lab2)
                dxy_site = mts.divergence(
                    sample_sets=node_sets, indexes=[(i, j)],
                    mode="site", span_normalise=True
                )[0]
                da_site = dxy_site - (pi_map[lab1] + pi_map[lab2]) / 2.0
                fst_site = 0.0 if dxy_site <= 0 else max(0.0, min(1.0, da_site / dxy_site))
                for acc, val in ((dxy_acc[key], dxy_site), (da_acc[key], da_site), (fst_acc[key], fst_site)):
                    acc["n"] += 1; acc["s"] += val; acc["ss"] += val * val

    out = {}
    for lab in labels:
        a1, _ = harmonic_numbers(HAP_PER_DEME)
        out[f"pi_{lab}"]      = pi_totals[lab] / L_total
        out[f"thetaW_{lab}"]  = (S_totals[lab] / a1) / L_total
        out[f"TajD_{lab}"]    = tajimas_D_from_totals(HAP_PER_DEME, S_totals[lab], pi_totals[lab])

        sfs_fold = fold_unfolded_1d_sfs(sfs_unf_tot[lab])
        half = HAP_PER_DEME // 2
        for k in range(1, half + 1):
            if drop_singleton and k == 1: continue
            out[f"sfs1d_fold_{lab}_mac{k}_perbp"] = sfs_fold[k] / L_total

        if with_variance:
            def finish(acc):
                n = acc["n"]
                if n <= 1: return 0.0
                return max(0.0, (acc["ss"] - acc["s"]*acc["s"]/n)) / (n - 1)
            out[f"var_pi_{lab}"]     = finish(pi_acc[lab])
            out[f"var_thetaW_{lab}"] = finish(thW_acc[lab])
            out[f"var_TajD_{lab}"]   = finish(tajD_acc[lab])

    if with_variance:
        def finish(acc):
            n = acc["n"]
            if n <= 1: return 0.0
            return max(0.0, (acc["ss"] - acc["s"]*acc["s"]/n)) / (n - 1)

    for (lab1, lab2) in combinations(labels, 2):
        key = pair_key(lab1, lab2)
        dxy_per_site = dxy_totals[key] / L_total
        da_per_site  = dxy_per_site - (out[f"pi_{lab1}"] + out[f"pi_{lab2}"]) / 2.0
        fst = 0.0 if dxy_per_site <= 0 else max(0.0, min(1.0, da_per_site / dxy_per_site))
        out[f"dxy_{lab1}_{lab2}"] = dxy_per_site
        out[f"da_{lab1}_{lab2}"]  = da_per_site
        out[f"Fst_{lab1}_{lab2}"] = fst
        if with_variance:
            out[f"var_dxy_{lab1}_{lab2}"] = finish(dxy_acc[key])
            out[f"var_da_{lab1}_{lab2}"]  = finish(da_acc[key])
            out[f"var_Fst_{lab1}_{lab2}"] = finish(fst_acc[key])
    return out

def compute_summary_from_sfs(demography, pop_names, reps, L_bp, mu, drop_singleton=False, workers=None, rand_seed=None, out_labels=None, recomb_rate=0.0):
    labels = out_labels if out_labels is not None else pop_names
    name_to_id = {p.name: i for i, p in enumerate(demography.populations)}
    L_total = reps * L_bp
    sfs_unf_tot = {lab: np.zeros(HAP_PER_DEME + 1) for lab in labels}
    jsfs_tot = {pair_key(*x): None for x in combinations(labels, 2)}

    reps_iter = reps_iterator(demography, pop_names, L_bp, reps, workers, rand_seed=rand_seed, recomb_rate=recomb_rate)
    for ts in reps_iter:
        mts = msprime.sim_mutations(ts, rate=mu, model="binary")
        node_sets = [mts.samples(population=name_to_id[p]) for p in pop_names]

        for idx, lab in enumerate(labels):
            sfs_u = mts.allele_frequency_spectrum(
                sample_sets=[node_sets[idx]], polarised=True, mode="site", span_normalise=False
            )
            sfs_unf_tot[lab] += sfs_u

        for (i, j) in combinations(range(len(labels)), 2):
            lab1, lab2 = labels[i], labels[j]
            key = pair_key(lab1, lab2)
            js = mts.allele_frequency_spectrum(
                sample_sets=[node_sets[i], node_sets[j]],
                polarised=True, mode="site", span_normalise=False
            )
            if jsfs_tot[key] is None:
                jsfs_tot[key] = js
            else:
                jsfs_tot[key] += js

    out = {}
    for lab in labels:
        sfs_u = sfs_unf_tot[lab]
        n = len(sfs_u) - 1
        S_total = float(np.sum(sfs_u[1:-1]))
        weights = np.array([weight_pi_1d(n, i) for i in range(n + 1)], dtype=float)
        pi_total = float(np.dot(weights, sfs_u))
        a1, _ = harmonic_numbers(n)
        out[f"pi_{lab}"] = pi_total / L_total
        out[f"thetaW_{lab}"] = (S_total / a1) / L_total
        out[f"TajD_{lab}"] = tajimas_D_from_totals(n, S_total, pi_total)

        sfs_fold = fold_unfolded_1d_sfs(sfs_u)
        half = n // 2
        for k in range(1, half + 1):
            if drop_singleton and k == 1: continue
            out[f"sfs1d_fold_{lab}_mac{k}_perbp"] = sfs_fold[k] / L_total

    for (lab1, lab2) in combinations(labels, 2):
        key = pair_key(lab1, lab2)
        mat = jsfs_tot[key]
        if mat is None:
            out[f"dxy_{lab1}_{lab2}"] = 0.0
            out[f"da_{lab1}_{lab2}"]  = 0.0
            out[f"Fst_{lab1}_{lab2}"] = 0.0
            continue
        n1 = mat.shape[0] - 1
        n2 = mat.shape[1] - 1
        w = np.fromfunction(lambda i, j: weight_dxy_2d(n1, n2, i, j), mat.shape, dtype=float)
        dxy_total = float(np.sum(w * mat))
        dxy_per_site = dxy_total / L_total
        da_per_site  = dxy_per_site - (out[f"pi_{lab1}"] + out[f"pi_{lab2}"]) / 2.0
        fst = 0.0 if dxy_per_site <= 0 else max(0.0, min(1.0, da_per_site / dxy_per_site))
        out[f"dxy_{lab1}_{lab2}"] = dxy_per_site
        out[f"da_{lab1}_{lab2}"]  = da_per_site
        out[f"Fst_{lab1}_{lab2}"] = fst
    return out

# ------------------------- per-draw simulation helpers -------------------------

def _rng_from(base_seed, draw_idx, model_offset=0):
    base = 0 if base_seed is None else int(base_seed)
    # Ensure 1 <= seed <= 2**32 - 1; avoid 0 which msprime rejects
    seed = (base + (model_offset << 28) + int(draw_idx)) % (2**32 - 1)
    if seed == 0:
        seed = 1
    return seed

def simulate_row_model1(draw_idx, cfg, pop_names, out_labels):
    seed = _rng_from(cfg.get("seed"), draw_idx, model_offset=0)
    rng = random.Random(seed)
    U = lambda a, b: rng.uniform(a, b)
    draw_Nem = (lambda a, b: _logU(rng, a, b)) if cfg.get("use_loguniform_Nem") else U
    draw_m   = (lambda a, b: _logU(rng, a, b)) if cfg.get("use_loguniform_m")   else U

    T1     = U(cfg["prior_t1_min"], cfg["prior_t1_max"])
    NeA    = U(cfg["prior_NeA_min"],  cfg["prior_NeA_max"])
    NeDeme = U(cfg["prior_NeD1_min"], cfg["prior_NeD1_max"])

    # draw within-chain either from Nem or m (symmetric)
    m_cap = float(cfg.get("mig_cap") or 1.0)
    Nem_neighbor = None
    if cfg.get("draw_within_from_Nem", False):
        Nem_neighbor = draw_Nem(cfg["Nem_neighbor1_min"], cfg["Nem_neighbor1_max"])
        m_within = min(Nem_neighbor / NeDeme, m_cap)
    else:
        m_within = draw_m(cfg["m_neighbor1_min"], cfg["m_neighbor1_max"])

    dem = model1_demography(T1, NeA, NeDeme, m_within, n_demes=int(cfg["m1_n"]))

    if cfg["from_sfs"]:
        stats = compute_summary_from_sfs(dem, pop_names, int(cfg["reps"]), float(cfg["length"]),
                                         float(cfg["mu"]), drop_singleton=cfg["no_singleton"],
                                         workers=cfg["workers"], rand_seed=seed, out_labels=out_labels,
                                         recomb_rate=float(cfg["recomb_rate"]))
    else:
        stats = compute_summary_tskit(dem, pop_names, int(cfg["reps"]), float(cfg["length"]),
                                      float(cfg["mu"]), with_variance=cfg["variance"],
                                      drop_singleton=cfg["no_singleton"], workers=cfg["workers"],
                                      rand_seed=seed, out_labels=out_labels,
                                      recomb_rate=float(cfg["recomb_rate"]))
    row = {
        "model": 1,
        "draw": draw_idx,
        "T1": T1,
        "Ne_anc": NeA,
        "Ne_deme": NeDeme,
        "m_neighbor": m_within,             # m used in demography
        "mu": cfg["mu"],
        "recomb_rate": cfg["recomb_rate"],  # NEW: report recombination
        "L_bp": cfg["length"],
        "reps": cfg["reps"],
        "pops": ";".join(out_labels),
        "m1_n": cfg["m1_n"],
    }
    # also report Nem if drawn from Nem; else compute implied Nem for convenience
    if cfg.get("draw_within_from_Nem", False):
        row["Nem_neighbor"] = Nem_neighbor
    else:
        row["Nem_neighbor"] = m_within * NeDeme
    row.update(stats)
    return row

def simulate_row_model2(draw_idx, cfg, pop_names, out_labels):
    seed = _rng_from(cfg.get("seed"), draw_idx, model_offset=1)
    rng = random.Random(seed)
    U = lambda a, b: rng.uniform(a, b)
    draw_Nem = (lambda a, b: _logU(rng, a, b)) if cfg.get("use_loguniform_Nem") else U
    draw_m   = (lambda a, b: _logU(rng, a, b)) if cfg.get("use_loguniform_m")   else U

    # times as Beta-fractions of T1
    T1 = U(cfg["prior_t1_min"], cfg["prior_t1_max"])
    fL = _map01(_beta_draw(cfg["t2_beta_alpha"], cfg["t2_beta_beta"], rng),
                cfg["t2_frac_min"], cfg["t2_frac_max"])
    fR = _map01(_beta_draw(cfg["t2_beta_alpha"], cfg["t2_beta_beta"], rng),
                cfg["t2_frac_min"], cfg["t2_frac_max"])
    T2L = fL * T1
    T2R = fR * T1
    g   = _map01(_beta_draw(cfg["t3_beta_alpha"], cfg["t3_beta_beta"], rng),
                 cfg["t3_frac_min"], cfg["t3_frac_max"])
    T3  = g * min(T2L, T2R)
    if not (T1 > max(T2L, T2R) and min(T2L, T2R) > T3 and T3 >= 0):
        raise RuntimeError("Invalid time ordering from fraction priors for Model 2.")

    NeA      = U(cfg["prior_NeA2_min"],   cfg["prior_NeA2_max"])
    NeL_line = U(cfg["prior_NeLlin_min"], cfg["prior_NeLlin_max"])
    NeR_line = U(cfg["prior_NeRlin_min"], cfg["prior_NeRlin_max"])
    NeDL     = U(cfg["prior_NeDL_min"],   cfg["prior_NeDL_max"])
    NeDR     = U(cfg["prior_NeDR_min"],   cfg["prior_NeDR_max"])

    # within-chains: symmetric per side; draw from Nem if requested
    m_cap = float(cfg.get("mig_cap") or 1.0)
    NemL = NemR = None
    if cfg.get("draw_within_from_Nem", False):
        NemL = draw_Nem(cfg["Nem_neighborL_min"], cfg["Nem_neighborL_max"])
        NemR = draw_Nem(cfg["Nem_neighborR_min"], cfg["Nem_neighborR_max"])
        m_within_L = min(NemL / NeDL, m_cap)
        m_within_R = min(NemR / NeDR, m_cap)
    else:
        m_within_L = draw_m(cfg["m_neighborL_min"], cfg["m_neighborL_max"])
        m_within_R = draw_m(cfg["m_neighborR_min"], cfg["m_neighborR_max"])

    # bridge: ASYMMETRIC; draw per-direction either from Nem or from m
    if cfg.get("draw_bridge_from_Nem", False):
        Nem_L2R = draw_Nem(cfg["Nem_bridge_L2R_min"], cfg["Nem_bridge_L2R_max"])
        Nem_R2L = draw_Nem(cfg["Nem_bridge_R2L_min"], cfg["Nem_bridge_R2L_max"])
        m_bridge_L2R = min(Nem_L2R / NeDR, m_cap)  # into R0
        m_bridge_R2L = min(Nem_R2L / NeDL, m_cap)  # into L(last)
    else:
        # per-direction m priors
        m_bridge_L2R = min(draw_m(cfg["m_bridge_L2R_min"], cfg["m_bridge_L2R_max"]), m_cap)
        m_bridge_R2L = min(draw_m(cfg["m_bridge_R2L_min"], cfg["m_bridge_R2L_max"]), m_cap)
        # implied Nem for reporting
        Nem_L2R = m_bridge_L2R * NeDR
        Nem_R2L = m_bridge_R2L * NeDL

    dem = model2_demography(T1, T2L, T2R, T3,
                            NeA, NeL_line, NeR_line, NeDL, NeDR,
                            m_within_L, m_within_R,
                            m_bridge_L2R, m_bridge_R2L,
                            nL=int(cfg["m2_nl"]), nR=int(cfg["m2_nr"]))

    if cfg["from_sfs"]:
        stats = compute_summary_from_sfs(dem, pop_names, int(cfg["reps"]), float(cfg["length"]),
                                         float(cfg["mu"]), drop_singleton=cfg["no_singleton"],
                                         workers=cfg["workers"], rand_seed=seed, out_labels=out_labels,
                                         recomb_rate=float(cfg["recomb_rate"]))
    else:
        stats = compute_summary_tskit(dem, pop_names, int(cfg["reps"]), float(cfg["length"]),
                                      float(cfg["mu"]), with_variance=cfg["variance"],
                                      drop_singleton=cfg["no_singleton"], workers=cfg["workers"],
                                      rand_seed=seed, out_labels=out_labels,
                                      recomb_rate=float(cfg["recomb_rate"]))

    row = {
        "model": 2,
        "draw": draw_idx,
        "T1": T1, "T2_left": T2L, "T2_right": T2R, "T3": T3,
        "Ne_anc": NeA,
        "Ne_left_lineage": NeL_line,
        "Ne_right_lineage": NeR_line,
        "Ne_deme_left": NeDL,
        "Ne_deme_right": NeDR,
        "m_neighbor_left":  m_within_L,
        "m_neighbor_right": m_within_R,
        "m_bridge_LtoR": m_bridge_L2R,
        "m_bridge_RtoL": m_bridge_R2L,
        "mu": cfg["mu"],
        "recomb_rate": cfg["recomb_rate"],
        "L_bp": cfg["length"],
        "reps": cfg["reps"],
        "pops": ";".join(out_labels),
        "m2_nl": cfg["m2_nl"],
        "m2_nr": cfg["m2_nr"],
    }
    # within-chain Nem reporting
    if cfg.get("draw_within_from_Nem", False):
        row["Nem_neighbor_left"]  = NemL
        row["Nem_neighbor_right"] = NemR
    else:
        row["Nem_neighbor_left"]  = m_within_L * NeDL
        row["Nem_neighbor_right"] = m_within_R * NeDR
    # bridge Nem reporting
    row["Nem_bridge_LtoR"] = Nem_L2R
    row["Nem_bridge_RtoL"] = Nem_R2L

    row.update(stats)
    return row

def simulate_row_model3(draw_idx, cfg, pop_names, out_labels):
    """
    Model 3 (NO-SC): same priors as Model 2 for times, but no T3 and NO bridge.
    Left/Right have distinct within-block migration rates; within-chains can be drawn from Nem or m.
    """
    seed = _rng_from(cfg.get("seed"), draw_idx, model_offset=2)
    rng = random.Random(seed)
    U = lambda a, b: rng.uniform(a, b)
    draw_Nem = (lambda a, b: _logU(rng, a, b)) if cfg.get("use_loguniform_Nem") else U
    draw_m   = (lambda a, b: _logU(rng, a, b)) if cfg.get("use_loguniform_m")   else U

    # times as Beta-fractions of T1 (no T3)
    T1 = U(cfg["prior_t1_min"], cfg["prior_t1_max"])
    fL = _map01(_beta_draw(cfg["t2_beta_alpha"], cfg["t2_beta_beta"], rng),
                cfg["t2_frac_min"], cfg["t2_frac_max"])
    fR = _map01(_beta_draw(cfg["t2_beta_alpha"], cfg["t2_beta_beta"], rng),
                cfg["t2_frac_min"], cfg["t2_frac_max"])
    T2L = fL * T1
    T2R = fR * T1
    if not (T1 > max(T2L, T2R)):
        raise RuntimeError("Invalid time ordering from fraction priors for Model 3.")

    NeA      = U(cfg["prior_NeA2_min"],   cfg["prior_NeA2_max"])
    NeL_line = U(cfg["prior_NeLlin_min"], cfg["prior_NeLlin_max"])
    NeR_line = U(cfg["prior_NeRlin_min"], cfg["prior_NeRlin_max"])
    NeDL     = U(cfg["prior_NeDL_min"],   cfg["prior_NeDL_max"])
    NeDR     = U(cfg["prior_NeDR_min"],   cfg["prior_NeDR_max"])

    # distinct within-block m's, possibly derived from Nem
    m_cap = float(cfg.get("mig_cap") or 1.0)
    NemL = NemR = None
    if cfg.get("draw_within_from_Nem", False):
        NemL = draw_Nem(cfg["Nem_neighborL_min"], cfg["Nem_neighborL_max"])
        NemR = draw_Nem(cfg["Nem_neighborR_min"], cfg["Nem_neighborR_max"])
        m_within_L = min(NemL / NeDL, m_cap)
        m_within_R = min(NemR / NeDR, m_cap)
    else:
        m_within_L = draw_m(cfg["m_neighborL_min"], cfg["m_neighborL_max"])
        m_within_R = draw_m(cfg["m_neighborR_min"], cfg["m_neighborR_max"])

    dem = model3_demography(T1, T2L, T2R,
                            NeA, NeL_line, NeR_line, NeDL, NeDR,
                            m_within_L, m_within_R,
                            nL=int(cfg["m2_nl"]), nR=int(cfg["m2_nr"]))

    if cfg["from_sfs"]:
        stats = compute_summary_from_sfs(dem, pop_names, int(cfg["reps"]), float(cfg["length"]),
                                         float(cfg["mu"]), drop_singleton=cfg["no_singleton"],
                                         workers=cfg["workers"], rand_seed=seed, out_labels=out_labels,
                                         recomb_rate=float(cfg["recomb_rate"]))
    else:
        stats = compute_summary_tskit(dem, pop_names, int(cfg["reps"]), float(cfg["length"]),
                                      float(cfg["mu"]), with_variance=cfg["variance"],
                                      drop_singleton=cfg["no_singleton"], workers=cfg["workers"],
                                      rand_seed=seed, out_labels=out_labels,
                                      recomb_rate=float(cfg["recomb_rate"]))

    row = {
        "model": 3,
        "draw": draw_idx,
        "T1": T1, "T2_left": T2L, "T2_right": T2R,
        "Ne_anc": NeA,
        "Ne_left_lineage": NeL_line,
        "Ne_right_lineage": NeR_line,
        "Ne_deme_left": NeDL,
        "Ne_deme_right": NeDR,
        "m_neighbor_left":  m_within_L,
        "m_neighbor_right": m_within_R,
        "mu": cfg["mu"],
        "recomb_rate": cfg["recomb_rate"],
        "L_bp": cfg["length"],
        "reps": cfg["reps"],
        "pops": ";".join(out_labels),
        "m2_nl": cfg["m2_nl"],
        "m2_nr": cfg["m2_nr"],
    }
    if cfg.get("draw_within_from_Nem", False):
        row["Nem_neighbor_left"]  = NemL
        row["Nem_neighbor_right"] = NemR
    else:
        row["Nem_neighbor_left"]  = m_within_L * NeDL
        row["Nem_neighbor_right"] = m_within_R * NeDR

    row.update(stats)
    return row

# ------------------------- main -------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python Build_ABC_refTable.py <config.json|yaml>")
        return 2
    raw_cfg = load_config(sys.argv[1])
    cfg = resolve_config(raw_cfg)

    outdir = cfg["outdir"] or (
        f"abc_mu{cfg['mu']}_rho{cfg['recomb_rate']}_L{cfg['length']}_reps{cfg['reps']}"
        f"_T1[{cfg['prior_t1_min']}-{cfg['prior_t1_max']}]"
        f"_T2frac[{cfg['t2_frac_min']}-{cfg['t2_frac_max']}]"
        f"_T3frac[{cfg['t3_frac_min']}-{cfg['t3_frac_max']}]"
        f"_M1n{cfg['m1_n']}_M2nL{cfg['m2_nl']}_M2nR{cfg['m2_nr']}"
        + ("_VAR" if cfg["variance"] else "")
        + ("_SFS" if cfg["from_sfs"] else "")
        + ("_NOS1" if cfg["no_singleton"] else "")
        + (f"_J{cfg['jobs']}" if cfg["jobs"] and cfg["jobs"] > 1 else "")
    )
    os.makedirs(outdir, exist_ok=True)

    # logs
    with open(os.path.join(outdir, "abc_summary.txt"), "w") as f:
        f.write("# ABC setup (Ne priors; T2/T3 as fractions; fixed mu; recombination; m- or Nem-based migration; distinct L/R within-block m; ASYMMETRIC bridge)\n")
        f.write(f"time: {datetime.now().isoformat()}\n")
        f.write(f"script: {os.path.basename(sys.argv[0])}\n")
        f.write(f"config: {os.path.abspath(sys.argv[1])}\n")
        for k in sorted(cfg.keys()):
            f.write(f"{k} = {cfg[k]}\n")
    with open(os.path.join(outdir, "effective_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # parse pop selections (bounds-checked)
    m1_pops = parse_pop_list(cfg["m1_pops"], ("P",), nP=int(cfg["m1_n"]))
    m2_pops = parse_pop_list(cfg["m2_pops"], ("L", "R"), nL=int(cfg["m2_nl"]), nR=int(cfg["m2_nr"]))
    # standardized labels for outputs (P1, P2, ...)
    m1_labels = [f"P{i+1}" for i in range(len(m1_pops))]
    m2_labels = [f"P{i+1}" for i in range(len(m2_pops))]
    m3_pops   = m2_pops      # SAME sampled demes for Model 2 and 3
    m3_labels = m2_labels    # SAME standardized labels

    jobs = int(cfg["jobs"])

    # ----- concurrent, interleaved build of all reference tables -----
    path1 = os.path.join(outdir, "ref_table_model1.csv")
    path2 = os.path.join(outdir, "ref_table_model2.csv")
    path3 = os.path.join(outdir, "ref_table_model3.csv")
    # open files only if we will actually write to them
    n1 = int(cfg["n_sims1"]) if m1_pops else 0
    n2 = int(cfg["n_sims2"]) if m2_pops else 0
    n3 = int(cfg["n_sims3"]) if m3_pops else 0
    fh1 = open(path1, "w", newline="") if (m1_pops and n1 > 0) else None
    fh2 = open(path2, "w", newline="") if (m2_pops and n2 > 0) else None
    fh3 = open(path3, "w", newline="") if (m3_pops and n3 > 0) else None
    writer1 = writer2 = writer3 = None

    total = n1 + n2 + n3

    def write_row(row):
        nonlocal writer1, writer2, writer3
        if row["model"] == 1:
            if writer1 is None:
                writer1 = csv.DictWriter(fh1, fieldnames=list(row.keys()))
                writer1.writeheader()
            writer1.writerow(row)
        elif row["model"] == 2:
            if writer2 is None:
                writer2 = csv.DictWriter(fh2, fieldnames=list(row.keys()))
                writer2.writeheader()
            writer2.writerow(row)
        else:
            if writer3 is None:
                writer3 = csv.DictWriter(fh3, fieldnames=list(row.keys()))
                writer3.writeheader()
            writer3.writerow(row)

    if jobs > 1 and total > 0:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = []
            i = j = k = 0
            # round-robin submission so all requested models start immediately
            while i < n1 or j < n2 or k < n3:
                if i < n1:
                    futs.append(ex.submit(simulate_row_model1, i, cfg, m1_pops, m1_labels))
                    i += 1
                if j < n2:
                    futs.append(ex.submit(simulate_row_model2, j, cfg, m2_pops, m2_labels))
                    j += 1
                if k < n3:
                    futs.append(ex.submit(simulate_row_model3, k, cfg, m3_pops, m3_labels))
                    k += 1

            done = done1 = done2 = done3 = 0
            for fut in as_completed(futs):
                row = fut.result()
                write_row(row)
                done += 1
                if row["model"] == 1:
                    done1 += 1
                elif row["model"] == 2:
                    done2 += 1
                else:
                    done3 += 1
                if done % 50 == 0:
                    print(f"[progress] total {done}/{total} | M1 {done1}/{n1} | M2 {done2}/{n2} | M3 {done3}/{n3}")
    else:
        # sequential fallback (deterministic order)
        if n1:
            for i in range(n1):
                write_row(simulate_row_model1(i, cfg, m1_pops, m1_labels))
        if n2:
            for j in range(n2):
                write_row(simulate_row_model2(j, cfg, m2_pops, m2_labels))
        if n3:
            for k in range(n3):
                write_row(simulate_row_model3(k, cfg, m3_pops, m3_labels))

    if fh1:
        fh1.close(); print(f"[OK] {path1}")
    else:
        print("Note: Model 1 skipped or n_sims1=0.")
    if fh2:
        fh2.close(); print(f"[OK] {path2}")
    else:
        print("Note: Model 2 skipped or n_sims2=0.")
    if fh3:
        fh3.close(); print(f"[OK] {path3}")
    else:
        print("Note: Model 3 skipped or n_sims3=0.")

    print("Done.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

