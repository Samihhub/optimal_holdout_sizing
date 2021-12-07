ninteractions = c(0, 2)
families = c("log_reg", "rand_forest")

for (family in families) {
  for (ninters in ninteractions) {
    source("sim_system.R")
    data_to_save <- as.data.frame(cbind(frac_ho, t(deaths_boot_tot[1,,])))
    write.csv(x=data_to_save, file=paste("data", ninters, family, ".csv", sep=''))
  }
}