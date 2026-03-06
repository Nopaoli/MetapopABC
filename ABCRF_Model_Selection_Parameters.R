setwd("C:/Users/paolo/Dropbox/LocalDocuments/UniTus/Ricerca/SHARKS/ABC/NEW_PIPELINE/JAN26/")
# load required libraries

library(abc)    # for approximate Bayesian computation functions
require(hexbin) # for plots of PCA
require(grid)   # for plots of PCA
require(abcrf)           # abc random forest: model choice
require(quantregForest)  # abc random forest: parameter estimation
library(plyr)
library(faraway)
library(scales)
library(dplyr)
library(tidyr)
library(ggplot2)
num_of_threads=6



# Load data and prepare reference tables and target summary stats ---------------------------------------------------------------


#South gbr het6 unfolded projected 
targetStats <-read.csv ("../../OBS/observed_summary.csv ", header = T)
df_cleaned <- targetStats[, colSums(targetStats != 0) > 0]
SS_names <- colnames(df_cleaned)


# Load reference table for every simulation model 

Model1<-read.csv("All_models_70kLoci/ref_table_model1.csv", header = T)
Model1b<-read.csv("All_models_70k6cLoci/ref_table_model1.csv", header = T)
Model2<-read.csv("All_models_70kLoci/ref_table_model2.csv", header = T)
Model2b<-read.csv("All_models_70k6cLoci/ref_table_model2.csv", header = T)
Model3<-read.csv("All_models_70kLoci/ref_table_model3.csv", header = T)
Model3b<-read.csv("All_models_70k6cLoci/ref_table_model3.csv", header = T)


colnames(Model1[c(1,14:46)])
colnames(Model2[c(1,27:59)])
colnames(Model3[c(1,22:54)])


Model1$col<-"blue"
Model1b$col<-"blue"

Model2$col<-"red"
Model2b$col<-"red"

Model3$col<-"purple"
Model3b$col<-"purple"

# Merge reference tables from single models into one ref table. Since i ran 50k simulations from Model2 (the winner), subsample the same number (10k) from it. Order is irrelevant

ref_tableALL <- rbind.fill(
  Model1[c(1, 14:46)],
  Model1b[c(1, 14:46)],
  Model2[c(1, 27:59)],
  Model2b[c(1, 27:59)],
  Model3[c(1, 22:54)],
  Model3b[c(1, 22:54)]
)

ref_table

library(dplyr)

set.seed(123)  # for reproducibility

ref_table <- ref_tableALL %>%
  group_by(model) %>%      # my_var takes values 1, 2, 3
  slice_sample(n = 10000) %>%
  ungroup()


modindex <- as.factor(ref_table$model)
sumstats <- ref_table[, 2:34]
nscenarios <- 3

# Define scenarios from modindex
scenarios <- levels(modindex)
post_cols <- paste0("Votes_", scenarios)
out_cols  <- c("nsims", "post_best", "Best Model",post_cols, "prior_err")



# ABCRF -------------------------------------------------------------------

# from entire data

output <- setNames(data.frame(matrix(ncol = length(out_cols), nrow = 0)), out_cols)

for (i in 1:10) {
  
  mc.rf <- abcrf(
    modindex ~ .,
    data  = data.frame(modindex, sumstats),
    ntree = 1000,
    paral = TRUE, ncores = num_of_threads
  )
  
  pred.rf <- predict(
    object         = mc.rf,
    obs            = targetStats,
    training       = data.frame(modindex, sumstats),
    ntree          = 1000,
    paral          = TRUE,  ncores          = num_of_threads,
    paral.predict  = TRUE,  ncores.predict  = num_of_threads
  )
  
  ## build the row (keep numeric best-model ID)
  mid <- unlist(pred.rf[2:(length(pred.rf) - 1)], use.names = FALSE)  # c(best_id, votes1, votes2, votes3)
  
  row_vec <- c(
    nrow(sumstats),                                 # nsims
    round(as.numeric(pred.rf[length(pred.rf)]), 4), # post_best
    as.numeric(mid),                                # Best Model + Votes_*
    round(as.numeric(mc.rf$prior.err), 4)           # prior_err
  )
  
  ## make a 1xk data.frame with EXACT names (avoid X... headers)
  new_row <- as.data.frame(t(row_vec), check.names = FALSE)
  colnames(new_row) <- out_cols
  
  output <- rbind(output, new_row)
}



write.table(output, "./ABC_full.txt", sep = "\t", quote = FALSE, row.names = FALSE)





# abcrf 2 models (1 and 3) , to compare SST and two separate metapops-------------------------------------------------



reftable13<- ref_table[which(ref_table$model != 2),]

modindex13 <- as.factor(reftable13$model)
nscenarios13 <- 2
sumstats13 <- reftable13[, 2:34]

scenarios13 <- levels(modindex13)
post_cols13 <- paste0("Votes_", scenarios13)
out_cols13  <- c("nsims", "post_best", "Best Model",post_cols13, "prior_err")


output13 <- setNames(data.frame(matrix(ncol = length(out_cols13), nrow = 0)), out_cols13)

# note: varibale 32 is redunant (can be scribed by linear combinations of other varibales9 and needs to be removed. )
for (i in 1:10) {
  
  mc.rf13 <- abcrf(
    modindex13 ~ .,
    data  = data.frame(modindex13, sumstats13),
    ntree = 1000,
    paral = TRUE, ncores = num_of_threads
  )
  
  pred.rf13 <- predict(
    object         = mc.rf13,
    obs            = targetStats,
    training       = data.frame(modindex13, sumstats13),
    ntree          = 1000,
    paral          = TRUE,  ncores          = num_of_threads,
    paral.predict  = TRUE,  ncores.predict  = num_of_threads
  )
  
  ## build the row (keep numeric best-model ID)
  mid <- unlist(pred.rf13[2:(length(pred.rf13) - 1)], use.names = FALSE)  # c(best_id, votes1, votes2, votes3)
  
  row_vec <- c(
    nrow(sumstats13),                                 # nsims
    round(as.numeric(pred.rf13[length(pred.rf13)]), 4), # post_best
    as.numeric(mid),                                # Best Model + Votes_*
    round(as.numeric(mc.rf13$prior.err), 4)           # prior_err
  )
  
  ## make a 1xk data.frame with EXACT names (avoid X... headers)
  new_row <- as.data.frame(t(row_vec), check.names = FALSE)
  colnames(new_row) <- out_cols13
  
  output13 <- rbind(output13, new_row)
}



write.table(output13, "./ABC_full13.txt", sep = "\t", quote = FALSE, row.names = FALSE)



# Now test model 2 vs model 3 ---------------------------------------------


reftable23<- ref_table[which(ref_table$model != 1),]

modindex23 <- as.factor(reftable23$model)
nscenarios23 <- 2
sumstats23 <- reftable23[, 2:34]

scenarios23 <- levels(modindex23)
post_cols23 <- paste0("Votes_", scenarios23)
out_cols23  <- c("nsims", "post_best", "Best Model",post_cols23, "prior_err")


output23 <- setNames(data.frame(matrix(ncol = length(out_cols23), nrow = 0)), out_cols23)

# note: varibale 32 is redunant (can be scribed by linear combinations of other varibales9 and needs to be removed. )
for (i in 1:10) {
  
  mc.rf23 <- abcrf(
    modindex23 ~ .,
    data  = data.frame(modindex23, sumstats23),
    ntree = 1000,
    paral = TRUE, ncores = num_of_threads
  )
  
  pred.rf23 <- predict(
    object         = mc.rf23,
    obs            = targetStats,
    training       = data.frame(modindex23, sumstats23),
    ntree          = 1000,
    paral          = TRUE,  ncores          = num_of_threads,
    paral.predict  = TRUE,  ncores.predict  = num_of_threads
  )
  
  ## build the row (keep numeric best-model ID)
  mid <- unlist(pred.rf23[2:(length(pred.rf23) - 1)], use.names = FALSE)  # c(best_id, votes1, votes2, votes3)
  
  row_vec <- c(
    nrow(sumstats23),                                 # nsims
    round(as.numeric(pred.rf23[length(pred.rf23)]), 4), # post_best
    as.numeric(mid),                                # Best Model + Votes_*
    round(as.numeric(mc.rf23$prior.err), 4)           # prior_err
  )
  
  ## make a 1xk data.frame with EXACT names (avoid X... headers)
  new_row <- as.data.frame(t(row_vec), check.names = FALSE)
  colnames(new_row) <- out_cols23
  
  output23 <- rbind(output23, new_row)
}



write.table(output23, "./ABC_full23.txt", sep = "\t", quote = FALSE, row.names = FALSE)




# From two sampled populatiosn only ---------------------------------------

targetStats_noP3 <- targetStats[, !grepl("P3", colnames(targetStats))]
sumstats_noP3<- sumstats[, !grepl("P3", colnames(sumstats13))]


output_noP3 <- setNames(data.frame(matrix(ncol = length(out_cols13), nrow = 0)), out_cols13)

for (i in 1:10) {
  
  mc.rf <- abcrf(
    modindex ~ .,
    data  = data.frame(modindex, sumstats_noP3),
    ntree = 1000,
    paral = TRUE, ncores = num_of_threads
  )
  
  pred.rf <- predict(
    object         = mc.rf,
    obs            = targetStats_noP3,
    training       = data.frame(modindex, sumstats_noP3),
    ntree          = 1000,
    paral          = TRUE,  ncores          = num_of_threads,
    paral.predict  = TRUE,  ncores.predict  = num_of_threads
  )
  
  ## build the row (keep numeric best-model ID)
  mid <- unlist(pred.rf[2:(length(pred.rf) - 1)], use.names = FALSE)  # c(best_id, votes1, votes2, votes3)
  
  row_vec <- c(
    nrow(sumstats_noP3),                                 # nsims
    round(as.numeric(pred.rf[length(pred.rf)]), 4), # post_best
    as.numeric(mid),                                # Best Model + Votes_*
    round(as.numeric(mc.rf$prior.err), 4)           # prior_err
  )
  
  ## make a 1xk data.frame with EXACT names (avoid X... headers)
  new_row <- as.data.frame(t(row_vec), check.names = FALSE)
  colnames(new_row) <- out_cols
  
  output_noP3 <- rbind(output_noP3, new_row)
}


write.table(output_noP3, "./ABC_2POPS.txt", sep = "\t", quote = FALSE, row.names = FALSE)





# Two sampled pops, SST vs HIER -------------------------------------------
targetStats_noP3 <- targetStats[, !grepl("P3", colnames(targetStats))]
sumstats13_noP3<- sumstats13[, !grepl("P3", colnames(sumstats13))]


output13_noP3 <- setNames(data.frame(matrix(ncol = length(out_cols13), nrow = 0)), out_cols13)

for (i in 1:10) {
  
  mc.rf <- abcrf(
    modindex13 ~ .,
    data  = data.frame(modindex13, sumstats13_noP3),
    ntree = 1000,
    paral = TRUE, ncores = num_of_threads
  )
  
  pred.rf <- predict(
    object         = mc.rf,
    obs            = targetStats_noP3,
    training       = data.frame(modindex13, sumstats13_noP3),
    ntree          = 1000,
    paral          = TRUE,  ncores          = num_of_threads,
    paral.predict  = TRUE,  ncores.predict  = num_of_threads
  )
  
  ## build the row (keep numeric best-model ID)
  mid <- unlist(pred.rf[2:(length(pred.rf) - 1)], use.names = FALSE)  # c(best_id, votes1, votes2, votes3)
  
  row_vec <- c(
    nrow(sumstats13_noP3),                                 # nsims
    round(as.numeric(pred.rf[length(pred.rf)]), 4), # post_best
    as.numeric(mid),                                # Best Model + Votes_*
    round(as.numeric(mc.rf$prior.err), 4)           # prior_err
  )
  
  ## make a 1xk data.frame with EXACT names (avoid X... headers)
  new_row <- as.data.frame(t(row_vec), check.names = FALSE)
  colnames(new_row) <- out_cols13
  
  output13_noP3 <- rbind(output13_noP3, new_row)
}


write.table(output13_noP3, "./ABC_2POPS13.txt", sep = "\t", quote = FALSE, row.names = FALSE)



# Two sampled pops, HIER vs HIERSC ----------------------------------------

sumstats23_noP3<- sumstats23[, !grepl("P3", colnames(sumstats23))]


output23_noP3 <- setNames(data.frame(matrix(ncol = length(out_cols23), nrow = 0)), out_cols23)

for (i in 1:10) {
  
  mc.rf <- abcrf(
    modindex23 ~ .,
    data  = data.frame(modindex23, sumstats23_noP3),
    ntree = 1000,
    paral = TRUE, ncores = num_of_threads
  )
  
  pred.rf <- predict(
    object         = mc.rf,
    obs            = targetStats_noP3,
    training       = data.frame(modindex23, sumstats23_noP3),
    ntree          = 1000,
    paral          = TRUE,  ncores          = num_of_threads,
    paral.predict  = TRUE,  ncores.predict  = num_of_threads
  )
  
  ## build the row (keep numeric best-model ID)
  mid <- unlist(pred.rf[2:(length(pred.rf) - 1)], use.names = FALSE)  # c(best_id, votes1, votes2, votes3)
  
  row_vec <- c(
    nrow(sumstats23_noP3),                                 # nsims
    round(as.numeric(pred.rf[length(pred.rf)]), 4), # post_best
    as.numeric(mid),                                # Best Model + Votes_*
    round(as.numeric(mc.rf$prior.err), 4)           # prior_err
  )
  
  ## make a 1xk data.frame with EXACT names (avoid X... headers)
  new_row <- as.data.frame(t(row_vec), check.names = FALSE)
  colnames(new_row) <- out_cols23
  
  output23_noP3 <- rbind(output23_noP3, new_row)
}


write.table(output23_noP3, "./Results_1k_5K_UnformNem/ABC_2POPS23.txt", sep = "\t", quote = FALSE, row.names = FALSE)




# PlOT LDA and err --------------------------------------------------------




# Make a figure showing a) variables' relative 
# importance as well as b) LDA showing simulations and observed data
pdf("./Results/LDA.pdf", width= 6, height = 5)
plot(mc.rf,
     training=data.frame(modindex, sumstats),
     obs=targetStats)

dev.off()

pdf("./Results/Prior_ErrRate.pdf", width= 6, height = 4)
err.abcrf(mc.rf,
          training=data.frame(modindex, sumstats),
          paral=T,
          ncores=num_of_threads)
dev.off()




model_RF <- abcrf(modindex~.,
                  data=data.frame(modindex, sumstats),
                  ntree=1000,
                  paral=T,
                  ncores=num_of_threads)

# Print the prior error rate
model_RF$prior.err






res <- pca_lda_plot_consistent(refTable = ref_table[,1:34],
                               obs = targetStats,          # one-row data.frame
                               post_summ = NULL,
                               var_thresh = 0.99,
                               model_col = "model",
                               keep_all_pcs = FALSE,
                               pca_center = TRUE,
                               pca_scale  = TRUE,
                               point_alpha = 0.03,
                               legend_labels = c("1"="SST", "2"="HIERSC","3"="HIER"), # NEW: named vector c("old_level"="New Label", ...),
                                               )
pdf(".//LDA.pdf", width= 6, height = 4)
res
dev.off()
# pretty solid results. Very clearly there are 2 metapops, good support for secondary contact


# Estimate parameters from Model 2 ----------------------------------------


sumstats2 <-  Model2[,27:59]




## --- Force shapes and names ---
#parameters  <- Model2[, c(3:15,21,22,23,24,25,26)]        # ensure data.frame

## inputs assumed to exist:
## Model2, sumstats2, targetStats, SS_names, num_of_threads

parameters  <- Model2[, c(3:15,21,22,23,24,25,26)]        # ensure data.frame
targetStats <- as.data.frame(targetStats[, SS_names, drop = FALSE])  # 1-row df

ntree_fit <- 100

stopifnot(nrow(parameters) == nrow(sumstats2))
stopifnot(identical(colnames(sumstats2), colnames(targetStats)))

RFmodels   <- list()
errors     <- list()
metrics    <- list()
post_rows  <- list()

for (param in colnames(parameters)) {
  
  y_raw <- parameters[[param]]
  keep  <- complete.cases(y_raw, sumstats2)
  y     <- y_raw[keep]
  X     <- sumstats2[keep, , drop = FALSE]
  
  if (sum(is.finite(y)) < 5L || length(unique(y[is.finite(y)])) < 2L) {
    metrics[[param]] <- data.frame(parameter = param, MSE = NA_real_, NMSE = NA_real_, Q2 = NA_real_)
    next
  }
  
  train_df <- data.frame(y = y, X, check.names = FALSE)
  colnames(train_df)[1] <- param
  
  form <- stats::reformulate(termlabels = ".", response = param)
  
  reg.rf <- regAbcrf(form,
                     data  = train_df,
                     ntree = ntree_fit,
                     paral = TRUE, ncores = num_of_threads)
  
  pred <- predict(reg.rf,
                  obs      = targetStats,
                  training = train_df,
                  ntree    = ntree_fit,
                  paral    = TRUE, ncores = num_of_threads)
  
  er <- err.regAbcrf(object   = reg.rf,
                     training = train_df,
                     paral    = TRUE, ncores = num_of_threads)
  
  ## --- metrics ---------------------------------------------------------------
  mse <- nmse <- q2 <- NA_real_
  
  if (is.matrix(er) || inherits(er, "array") || is.data.frame(er)) {
    er_df <- as.data.frame(er)
    last  <- nrow(er_df)
    
    if ("oob_mse" %in% names(er_df)) {
      mse <- as.numeric(er_df$oob_mse[last])
    } else if ("MSE" %in% names(er_df)) {
      mse <- as.numeric(er_df$MSE[last])
    } else {
      mse <- as.numeric(er_df[last, 2])
    }
    
    vY <- stats::var(train_df[[param]], na.rm = TRUE)
    if (is.finite(vY) && vY > 0 && is.finite(mse)) {
      nmse <- mse / vY
      q2   <- 1 - nmse
    }
    
  } else if (is.list(er)) {
    if (!is.null(er$MSE))  mse  <- as.numeric(er$MSE)
    if (!is.null(er$NMSE)) nmse <- as.numeric(er$NMSE)
    if (!is.null(er$Q2))   q2   <- as.numeric(er$Q2)
    
    if (is.na(nmse) && is.finite(mse)) {
      vY <- stats::var(train_df[[param]], na.rm = TRUE)
      if (is.finite(vY) && vY > 0) {
        nmse <- mse / vY
        if (is.na(q2)) q2 <- 1 - nmse
      }
    }
    
  } else if (is.numeric(er) && length(er) == 1L) {
    nmse <- as.numeric(er)
    q2   <- 1 - nmse
  }
  
  ## --- store ---------------------------------------------------------------
  metrics[[param]]  <- data.frame(parameter = param, MSE = mse, NMSE = nmse, Q2 = q2)
  RFmodels[[param]] <- reg.rf
  errors[[param]]   <- er
  
  pred_df <- as.data.frame(pred, check.names = FALSE)
  pred_df$parameter <- param
  post_rows[[param]] <- pred_df
  
  ## --- density plots (posterior vs prior) -----------------------------------
  try({
    densityPlot(object   = reg.rf,
                obs      = targetStats,
                training = train_df,
                main     = paste("Posterior vs prior for", param),
                paral    = TRUE, ncores = num_of_threads)
    
    # overlay prior density of simulated parameter (using *the same filtered y*)
    lines(stats::density(y), col = "forestgreen", lwd = 2)
  }, silent = TRUE)
}

## bind at the end
param_metrics <- do.call(rbind, metrics)
row.names(param_metrics) <- NULL
param_metrics <- param_metrics[
  order(-ifelse(is.finite(param_metrics$Q2), param_metrics$Q2, -Inf)),
  , drop = FALSE
]

posterior_table <- plyr::rbind.fill(post_rows)
posterior_table <- posterior_table[, c("parameter", setdiff(names(posterior_table), "parameter")), drop = FALSE]
row.names(posterior_table) <- NULL
# Plots Stats -------------------------------------------------------------


stats_to_plot <- c("pi_P1","thetaW_P1","TajD_P1","pi_P2","thetaW_P2","TajD_P2", "pi_P3","thetaW_P3","TajD_P3", "dxy_P1_P2",  "da_P1_P2",
                   "Fst_P1_P2",  "dxy_P1_P3",  "da_P1_P3",   "Fst_P1_P3",  "dxy_P2_P3",  "da_P2_P3",   "Fst_P2_P3")
# filtra le simulazioni e i valori osservati

df_sim_sub <- ref_table %>%
  select(model, all_of(SS_names))

df_obs_sub <- targetStats  %>%
  select(all_of(SS_names))

sim_long <- df_sim_sub %>%
  pivot_longer(-model, names_to = "stat", values_to = "value")

obs_long <- df_obs_sub %>%
  pivot_longer(everything(), names_to = "stat", values_to = "obs_value")

# Replico i punti osservati su ogni modello (così compaiono in ogni violino)
obs_points <- tidyr::crossing(
  model = unique(sim_long$model),
  obs_long
)


# --- Plot: violini per modello + punto osservato ---
p <- ggplot(sim_long, aes(x = model, y = value, fill = model)) +
  geom_violin(trim = FALSE, scale = "width", alpha = 0.8) +
  # (opzionale) jitter dei campioni simulati per dare idea di densità
  # geom_jitter(width = 0.12, alpha = 0.15, size = 0.6) +
  # punto osservato
  geom_point(data = obs_points, aes(y = obs_value),
             inherit.aes = TRUE,
             shape = 21, fill = "red", stroke = 1, size = 2.6) +
  # (opzionale) mediana delle simulazioni
  stat_summary(fun = median, geom = "point", color = "black",
               shape = 95, size = 6, position = position_dodge(width = 0.9)) +
  facet_wrap(~ stat, scales = "free_y", ncol = 3) +
  labs(title = "Distribuzione delle simulazioni per modello (violini)\ncon valore osservato sovrapposto",
       x = NULL, y = NULL, fill = "model") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top",
        strip.text = element_text(face = "bold"),
        axis.text.x = element_text(angle = 30, hjust = 1))

print(p)
# Helper: estrai colonne per gruppo tramite regex sui nomi
get_cols_by_regex <- function(df, pattern) {
  setdiff(grep(pattern, names(df), value = TRUE), "model")
}

# Gruppi (regex robuste ai tuoi nomi)
stats_pi_theta <- get_cols_by_regex(df_sim_sub, "^(pi|thetaW)_")
stats_Fst      <- get_cols_by_regex(df_sim_sub, "^Fst_")
stats_dxy_da   <- get_cols_by_regex(df_sim_sub, "^(dxy|da)_")
stats_mac      <- get_cols_by_regex(df_sim_sub, "^sfs1d_fold_.*_mac[0-9]+_perbp$")
stats_TD      <- get_cols_by_regex(df_sim_sub, "^TajD")


#------------------------------------------------------------
# Funzione: violini per un set di statistiche + punto osservato
#------------------------------------------------------------
make_violin <- function(df_sim, stats_vec, df_obs = NULL, title = NULL, ncol = 3) {
  stopifnot("model" %in% names(df_sim))
  stopifnot(length(stats_vec) > 0)
  stopifnot(all(stats_vec %in% names(df_sim)))
  
  df_sim2 <- df_sim %>% select(model, all_of(stats_vec))
  
  sim_long <- df_sim2 %>%
    pivot_longer(-model, names_to = "stat", values_to = "value") %>%
    mutate(stat = factor(stat, levels = stats_vec))
  
  # punti osservati replicati per ogni modello (se forniti)
  if (!is.null(df_obs)) {
    stopifnot(nrow(df_obs) == 1)
    stopifnot(all(stats_vec %in% names(df_obs)))
    obs_long <- df_obs %>%
      select(all_of(stats_vec)) %>%
      pivot_longer(everything(), names_to = "stat", values_to = "obs_value") %>%
      mutate(stat = factor(stat, levels = stats_vec))
    
    obs_points <- tidyr::crossing(
      model = unique(sim_long$model),
      obs_long
    )
  } else {
    obs_points <- NULL
  }
  
  # ordine modelli (mediana della prima stat)
  model_order <- sim_long %>%
    filter(stat == stats_vec[1]) %>%
    group_by(model) %>% summarize(med = median(value, na.rm = TRUE), .groups = "drop") %>%
    arrange(med) %>% pull(model)
  sim_long  <- sim_long  %>% mutate(model = factor(model, levels = model_order))
  if (!is.null(obs_points)) {
    obs_points <- obs_points %>% mutate(model = factor(model, levels = model_order))
  }
  
  p <- ggplot(sim_long, aes(x = model, y = value, fill = model)) +
    geom_violin(trim = FALSE, scale = "width", alpha = 0.8) +
    # punto osservato (se presente)
    { if (!is.null(obs_points))
      geom_point(data = obs_points, aes(y = obs_value),
                 inherit.aes = TRUE, shape = 21, fill = "white",
                 stroke = 1, size = 2.6) } +
    # mediana delle simulazioni
    stat_summary(fun = median, geom = "point", color = "black",
                 shape = 95, size = 6, position = position_dodge(width = 0.9)) +
    facet_wrap(~ stat, scales = "free_y", ncol = ncol) +
    labs(title = title, x = NULL, y = NULL, fill = "model") +
    theme_minimal(base_size = 12) +
    theme(legend.position = "top",
          strip.text = element_text(face = "bold"),
          axis.text.x = element_text(angle = 30, hjust = 1))
  p
}

#------------------------------------------------------------
# Costruzione dei 4 plot
#------------------------------------------------------------
p_pi_theta <- make_violin(df_sim_sub, stats_pi_theta, df_obs = df_obs_sub,
                          title = "pi & thetaW", ncol = 4)

p_Fst      <- make_violin(df_sim_sub, stats_Fst, df_obs = df_obs_sub,
                          title = "Fst", ncol = 3)

p_dxy_da   <- make_violin(df_sim_sub, stats_dxy_da, df_obs = df_obs_sub,
                          title = "dxy & da", ncol = 3)

p_mac      <- make_violin(df_sim_sub, stats_mac, df_obs = df_obs_sub,
                          title = "MAC (folded SFS per bp)", ncol = 5)
TajD       <- make_violin(df_sim_sub, stats_TD, df_obs = df_obs_sub,
                          title = "MAC (folded SFS per bp)", ncol = 3)
# Stampa
pdf(file="Pi_theta_plot.pdf", width = 12, height = 6 )
print(p_pi_theta)
dev.off()

pdf(file="FST_plot.pdf", width = 12, height = 4 )
print(p_Fst)
dev.off()

pdf(file="Dxy_plot.pdf", width=12, height = 4 )
print(p_dxy_da)
dev.off()

pdf(file="Mac.pdf", width =12, height = 8 )
print(p_mac)
dev.off()

pdf(file="TajD.pdf", width =12, height =4 )
print(TajD)
dev.off()


# Estimate parameters -----------------------------------------------------
sumstats2 <-  Model2[,27:59]

## --- Prep --------------------------------------------------------------------
parameters <- plyr::rbind.fill(Model2[, c(3:11,23,24,25,26)])
to_num <- function(x){ if (is.factor(x)) x <- as.character(x); suppressWarnings(as.numeric(x)) }
parameters <- as.data.frame(lapply(parameters, to_num), check.names = FALSE)

targetStats <- as.data.frame(targetStats[, SS_names, drop = FALSE])
stopifnot(nrow(parameters) == nrow(sumstats2))
stopifnot(identical(colnames(sumstats2), colnames(targetStats)))

## --- Loop --------------------------------------------------------------------
RFmodels <- list()
errors   <- list()
metrics  <- list()
post_rows <- list()

ntree_fit <- 1000

for (param in colnames(parameters)) {
  y_raw <- parameters[[param]]
  keep  <- complete.cases(y_raw, sumstats2)
  y     <- y_raw[keep]
  X     <- sumstats2[keep, , drop = FALSE]
  
  if (sum(is.finite(y)) < 5L || length(unique(y[is.finite(y)])) < 2L) {
    metrics[[param]] <- data.frame(parameter = param, MSE = NA_real_, NMSE = NA_real_, Q2 = NA_real_)
    next
  }
  
  train_df <- data.frame(y = y, X, check.names = FALSE)
  colnames(train_df)[1] <- param
  form <- as.formula(sprintf("`%s` ~ .", param))
  
  reg.rf <- regAbcrf(form, data = train_df, ntree = ntree_fit, paral = TRUE, ncores = num_of_threads)
  pred   <- predict(reg.rf, obs = targetStats, training = train_df, ntree = ntree_fit,
                    paral = TRUE, ncores = num_of_threads)
  er     <- err.regAbcrf(object = reg.rf, training = train_df, paral = TRUE, ncores = num_of_threads)
  
  # --- metrics (from OOB MSE vs ntree matrix) ---
  mse <- nmse <- q2 <- NA_real_
  if (is.matrix(er)) {
    col_oob <- if (!is.null(colnames(er)) && "oob_mse" %in% colnames(er)) "oob_mse" else 2
    mse     <- as.numeric(er[nrow(er), col_oob])
    vY      <- stats::var(train_df[[param]], na.rm = TRUE)
    if (is.finite(vY) && vY > 0) { nmse <- mse / vY; q2 <- 1 - nmse }
  } else if (is.list(er)) {
    if (!is.null(er$MSE))  mse  <- er$MSE
    if (!is.null(er$NMSE)) nmse <- er$NMSE
    if (!is.null(er$Q2))   q2   <- er$Q2
    if (is.na(nmse) && is.finite(mse)) {
      vY <- stats::var(train_df[[param]], na.rm = TRUE)
      if (is.finite(vY) && vY > 0) { nmse <- mse / vY; if (is.na(q2)) q2 <- 1 - nmse }
    }
  } else if (is.numeric(er) && length(er) == 1L) {
    nmse <- as.numeric(er); q2 <- 1 - nmse
  }
  
  metrics[[param]] <- data.frame(parameter = param, MSE = mse, NMSE = nmse, Q2 = q2)
  RFmodels[[param]] <- reg.rf
  errors[[param]]   <- er
  
  # --- simple posterior row bind: keep original column names exactly ----------
  pred_df <- as.data.frame(pred, check.names = FALSE)
  pred_df$parameter <- param
  post_rows[[param]] <- pred_df
}


param_metrics <- do.call(rbind, metrics)
row.names(param_metrics) <- NULL
param_metrics <- param_metrics[order(-ifelse(is.finite(param_metrics$Q2), param_metrics$Q2, -Inf)), ]

# bind all 1-row posteriors; tolerate any missing columns across params
posterior_table <- plyr::rbind.fill(post_rows)
# put 'parameter' first
posterior_table <- posterior_table[, c("parameter", setdiff(names(posterior_table), "parameter")), drop = FALSE]
row.names(posterior_table) <- NULL

# View
param_metrics
posterior_table

# Optional: save
write.csv(param_metrics,  "./Results70k/abcrf_param_metrics_1500sim.csv",  row.names = FALSE)
write.csv(posterior_table, "./Results70k//abcrf_posteriors__1500sim.csv",    row.names = FALSE)
        
# Functions PCA -----------------------------------------------------------

 n_pcs_for_var <- function(pr, thresh = 0.95) {
   v <- pr$sdev^2
   which(cumsum(v) / sum(v) >= thresh)[1]
 }
 
        pca_lda_plot_consistent <- function(refTable,
                                            obs,
                                            post_summ = NULL,
                                            model_col = "model",
                                            keep_all_pcs = TRUE,
                                            var_thresh = 0.95,   # ignored if keep_all_pcs=TRUE
                                            max_pcs = 1000,
                                            pca_center = TRUE,
                                            pca_scale  = FALSE,  # set TRUE only if your plain LDA used standardized stats
                                            prior_alpha = 0.25,  # alpha for priors density
                                            point_alpha = 0.2,   # alpha for scatter/rug
                                            align_sign_two_models = TRUE) {
          
          stopifnot(model_col %in% names(refTable))
          y <- as.factor(refTable[[model_col]])
          X <- refTable |> dplyr::select(-dplyr::all_of(model_col))
          
          if (is.vector(obs)) obs <- as.data.frame(as.list(obs))
          obs <- obs |> dplyr::select(dplyr::all_of(names(X)))
          if (!is.null(post_summ)) post_summ <- post_summ |> dplyr::select(dplyr::all_of(names(X)))
          
          # --- PCA (rotation) on priors only ---
          pca <- prcomp(X, center = pca_center, scale. = pca_scale)
          
          if (keep_all_pcs) {
            k <- min(ncol(X), max_pcs)
          } else {
            v <- pca$sdev^2; k <- which(cumsum(v)/sum(v) >= var_thresh)[1]
            k <- min(k, max_pcs, ncol(X))
          }
          
          X_pca    <- as.data.frame(pca$x[, 1:k, drop = FALSE])
          obs_pca  <- as.data.frame(predict(pca, newdata = obs)[, 1:k, drop = FALSE])
          post_pca <- if (is.null(post_summ)) NULL else as.data.frame(predict(pca, newdata = post_summ)[, 1:k, drop = FALSE])
          
          # --- LDA on PCs ---
          lda_fit <- MASS::lda(y ~ ., data = cbind(y = y, X_pca))
          
          pri_mat  <- as.matrix(predict(lda_fit)$x)
          obs_mat  <- as.matrix(predict(lda_fit, newdata = obs_pca)$x)
          post_mat <- if (is.null(post_pca)) NULL else as.matrix(predict(lda_fit, newdata = post_pca)$x)
          
          nLD <- ncol(pri_mat)
          colnames(pri_mat)  <- paste0("LD", seq_len(nLD))
          colnames(obs_mat)  <- paste0("LD", seq_len(nLD))
          if (!is.null(post_mat)) colnames(post_mat) <- paste0("LD", seq_len(nLD))
          
          pri_lda  <- as.data.frame(pri_mat); pri_lda[[model_col]] <- y
          obs_lda  <- as.data.frame(obs_mat)
          post_lda <- if (is.null(post_mat)) NULL else as.data.frame(post_mat)
          
          # --- Optional: sign alignment for 2-model case ---
          if (align_sign_two_models && nlevels(y) == 2 && nLD >= 1) {
            levs <- levels(y)
            m1 <- mean(pri_lda$LD1[pri_lda[[model_col]] == levs[1]])
            m2 <- mean(pri_lda$LD1[pri_lda[[model_col]] == levs[2]])
            if (m2 < m1) {
              pri_lda[grep("^LD", names(pri_lda))]  <- lapply(pri_lda[grep("^LD", names(pri_lda))], `*`, -1)
              obs_lda[grep("^LD", names(obs_lda))]  <- lapply(obs_lda[grep("^LD", names(obs_lda))], `*`, -1)
              if (!is.null(post_lda))
                post_lda[grep("^LD", names(post_lda))] <- lapply(post_lda[grep("^LD", names(post_lda))], `*`, -1)
            }
          }
          
          # --- Plot ---
          if (nLD < 2) {
            df <- pri_lda |> dplyr::rename(model = !!model_col)
            p <- ggplot2::ggplot(df, ggplot2::aes(x = LD1, fill = model, color = model)) +
              ggplot2::geom_density(alpha = prior_alpha) +
              ggplot2::geom_vline(xintercept = obs_lda$LD1[1], linetype = 2, linewidth = 0.7) +
              { if (!is.null(post_lda)) ggplot2::geom_rug(data = post_lda, ggplot2::aes(x = LD1),
                                                          inherit.aes = FALSE, alpha = point_alpha) } +
              ggplot2::theme_minimal(base_size = 12) +
              ggplot2::theme(
                panel.grid = element_blank(),
                panel.border = element_rect(color = "black", fill = NA, linewidth = 0.7),
                plot.background = element_rect(color = "black", fill = "white", linewidth = 0.7)
              ) +
              ggplot2::labs(
                title = "Priors vs Observed in LDA space",
                x = "LD1", y = "Density", fill = "Model", color = "Model"
              ) +
              ggplot2::guides(
                color = ggplot2::guide_legend(override.aes = list(alpha = 1)),
                fill  = ggplot2::guide_legend(override.aes = list(alpha = 1))
              )
          } else {
            df <- pri_lda |> dplyr::rename(model = !!model_col)
            p <- ggplot2::ggplot() +
              ggplot2::geom_point(data = df, ggplot2::aes(LD1, LD2, color = model), alpha = point_alpha, size = 2) +
              { if (!is.null(post_lda)) ggplot2::geom_point(data = post_lda, ggplot2::aes(LD1, LD2),
                                                            shape = 1, alpha = point_alpha, size = 2) } +
              ggplot2::geom_point(data = obs_lda, ggplot2::aes(LD1, LD2),
                                  color = "black", shape = 4, size = 3, stroke = 1.1) +
              ggplot2::theme_minimal(base_size = 12) +
              ggplot2::theme(
                panel.grid = element_blank(),
                panel.border = element_rect(color = "black", fill = NA, linewidth = 0.7),
                plot.background = element_rect(color = "black", fill = "white", linewidth = 0.7)
              ) +
              ggplot2::labs(
                title = "Priors vs Observed in LDA space",
                subtitle = paste0("PCA (", k, " PCs; center=", pca_center, ", scale=", pca_scale, ") LDA"),
                x = "LD1", y = "LD2", color = "Model"
              ) +
              ggplot2::guides(
                color = ggplot2::guide_legend(override.aes = list(alpha = 1)),
                fill  = ggplot2::guide_legend(override.aes = list(alpha = 1))
              )
          }
          
          list(plot = p, pca = pca, k_pcs = k, lda = lda_fit,
               priors_lda = pri_lda, observed_lda = obs_lda,
               posterior_lda = post_lda, nLD = nLD)
        }
        