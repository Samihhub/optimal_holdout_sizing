nsamp <- 2000    # How many bootstrap samples?

pi_booted <- c()
deaths_booted <- c()
frac_booted <- c()

for (i in 1:nsamp) {
  mask <- sample(1:boots, boots, replace = T)
  
  deaths_booted[i] <- min(colMeans(deaths_boot_tot[1, mask, ]))
  pi_booted[i] <- frac_ho[which.min(colMeans(deaths_boot_tot[1, mask, ]))]
}


#pi_test <- frac_ho[apply(deaths_boot_tot[1,, ], 1, which.min)]


# pi* obtained from minimising L
frac_ho[which.min(colMeans(deaths_boot_tot[dr_vars, , ]))]
# pi* obtained from bootstrap. NOTE do not use this one
mean(pi_booted)
# pi*'s sd
sd(pi_booted)

# L* obtained from samples
min(colMeans(deaths_boot_tot[dr_vars, , ]))
# L* obtained from bootstrap DO NOT USE
mean(deaths_booted)
# L*'s sd
sd(deaths_booted)
