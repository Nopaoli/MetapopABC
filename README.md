The script Build_ABC_refTable.py generates the reference tables required for ABC inference. Each row of the reference table corresponds to a simulated dataset generated under a set of demographic parameters drawn from prior distributions. For each simulation, the script records:
1.	The demographic parameters used for the simulation
2.	A set of population genetic summary statistics computed from the simulated data
These reference tables are then used as training datasets for ABC-Random Forest (ABC-RF) analyses or other ABC approaches.
The simulations are performed using msprime, a fast coalescent simulator that efficiently generates genealogical trees and mutations. The script supports three demographic mteapopulation models representing different spatial population histories and patterns of gene flow.

An indepth guide to use Build_ABC_refTable.py to generate simulations under the three main demographic scenarios is provided. 
