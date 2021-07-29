#### Simulation ####
## Simulation for the estimation of the optimal holdout set size ##
## for intervention on a population, guided by a risk score      ##

# Ctrl+Alt+T to run section by section

# ---------------------------------------------------------------------------- #
#### Package loading ####
library("dplyr")
library("progress")



# ---------------------------------------------------------------------------- #
#### Function Definitions ####

## Model and Mathematical function Definitions
logistic <- function(x) 1/(1 + exp(-x))
logit <- function(x) -log(1/x - 1)


gen_dr_coefs <- function(coefs, noise = TRUE, num_vars = 3) {
  if (!noise) {
    coefs <- coefs[1:num_vars]
  } 
  else {
    coefs <- coefs + rnorm(length(coefs), sd = 2 ** (num_vars - 1))
  }
  
  return(coefs)
}


# Estimate Dr's predictions
oracle_pred <- function(X, coefs, num_vars = 3, noise = TRUE) {
  # We model Dr behaviour as a logistic regression model that has access to the
  # coefficients, with added noise to them, generating imperfect predictions.
  # There is a double layer of indeterminism: firstly, the coefficients are
  # recalculated with new noise every time the function is called, making each
  # prediction different than previous ones; secondly, the returned classes are
  # sampled from a binomial distribution. This is done this way to simulate
  # the fact that human behaviour is not deterministic.
  
  # Flag to limit power by adding noise to coefficients or limit access to just a number of them
  
  if ("Y" %in% colnames(X)) {
    X <- X %>% select(-Y)
  }
  
  if (!noise) {
    X <- X %>% select(all_of(1:num_vars))
  }
  
  nobs <- dim(X)[1]
  
  lin_comb <- rowSums(t(t(X) * coefs))
  probs <- logistic(lin_comb)
  
  return(rbinom(nobs, 1, probs))
}


# Generate the predictors matrix
gen_preds <- function(nobs, npreds) {
  # Generate gaussian covariates matrix
  X <- as.data.frame(matrix( rnorm(nobs * npreds), nobs, npreds))
  return(X)
}


# Generate Y
gen_resp <- function(X, coefs = NA, coefs_sd = 1, retprobs = FALSE) {
  # For now, combines predictors linearly and applies binomial to find class
  nobs <- dim(X)[1]
  npreds <- dim(X)[2]
  
  # Generate coefficients for each predictor for the linear combination
  if (any(is.na(coefs))) {
    denom <- npreds ** 0.48
    if (!denom) denom <- 1
    coefs <- rnorm(npreds, sd = coefs_sd/denom)
  }
  
  # First term: linear combination of X's. Second term: Matrix of noise
  #lin_comb <- rowSums(t(t(X) * coefs) + matrix(rnorm(nobs * npreds), nobs, npreds))
  lin_comb <- rowSums(t(t(X) * coefs))
  probs <- logistic(lin_comb)
  
  if (retprobs) return(probs)
  # Round to get binary classes, assuming cuttoff at 0.5
  #classes <- as.data.frame(round(logistic(lin_comb)))
  classes <- rbinom(nobs, 1, probs)
  return(list("classes" = classes, "coefs" = coefs))
}


# Split dataset into holdout and intervention set
split_data <- function(X, frac) {
  # Returns a mask for the observations belonging in the training (holdout) set
  nobs <- dim(X)[1]
  mask <- rep(FALSE, nobs) #sample(FALSE, nobs, replace = TRUE)
  train_lim <- floor(nobs * frac)
  mask[1:train_lim] <- TRUE
  mask <- sample(mask)  # Shuffle to avoid correlation between sizes
  return(mask)
}



# -----------------------------------------------------------------------------#
#### Dynamics of the system ####

## Set seed for reproducibility
set.seed(1234)

# Initialisation of patient data
boots <- 200      # Number of bootstrap resamplings
nobs <- 5000                      # Number of observations, i.e patients
nobs_boot <- 5000         # Number of observations to resample in each bootstrap iter
npreds <- 7                    # Number of predictors


# Flag and vars to generate multiple dr predictive powers.
max_dr_vars <- 1 # When testing different Dr predictive powers, maximum power to be tested
run_many_powers <- FALSE # Flag for analysing several Dr powers at once
if(max_dr_vars > npreds) stop("max_dr_vars cannot be larger than npreds")
if(max_dr_vars > 1 && !run_many_powers) stop("Flag set to run only 1 power, but max_dr_vars > 1")
if(max_dr_vars == 1 && run_many_powers) stop("Flag set to run many powers, but max_dr_vars = 1")


# Definition of holdout set sizes to test
min_frac <- 0.02
max_frac <- 0.15
num_sizes <- 50
# Fraction of patients assigned to the holdout set
frac_ho <- seq(min_frac, max_frac, length.out = num_sizes)


# Variables for the threshold estimation
c_tn <- 0 # Cost of true neg
c_tp <- 0.5 # Cost of true pos
c_fp <- 0.5 # Cost of false pos
c_fn <- 1 # Cost of false neg
num_thresh <- 10 # Number of probabilities to scan
prob_vals <- seq(0, 1, length.out = num_thresh) # Vector with probability thresh to scan
thresh_vals <- seq()
k <- 5 # Number of CV folds
set_thresh <- TRUE # Flag to estimate classification threshold


# Cost estimation
costs_boot <- matrix(nrow = boots, ncol = num_sizes)
costs <- seq()
cost_type <- "gen_cost" # gen_cost = generalised cost, deaths = deaths (i.e. fn)
cost_mat <- rbind(c(c_tn, c_fp), c(c_fn, c_tp))


# Matrix with the number of deaths per epoch and holdout size
deaths_boot_inter <- deaths_boot_ho <- matrix(nrow = boots, ncol = num_sizes)
deaths_inter <- deaths_ho <- seq()


# Initialise Progress Bar
#if (exists("prog_bar")) rm(prog_bar)
#prog_bar <- progress_bar$new()
old_progress <- 0


# Initialise Dataset
set.seed(107)  
X <- gen_preds(nobs, npreds)
#newdata <- gen_resp(X, coefs_sd = 6)
#Y <- newdata$classes
coefs_general <- gen_resp(X)$coefs
coefs_dr <- gen_dr_coefs(coefs_general)

# Arrays to store data from sweeping through Dr's different powers
deaths_per_frac <- array(0, dim = c(max_dr_vars, num_sizes))
deaths_boot_tot <- array(0, dim = c(max_dr_vars, boots, num_sizes))
deaths_sd <- array(0, dim = c(max_dr_vars, num_sizes))


for (i in 1:num_sizes) {  # sweep through h.o. set sizes of interest
  for (b in 0:boots) {  # b=0 is the point estimate, b>0 are bootstrap samples
    
    progress <- 100 * (((i - 1) * boots) + b) / (num_sizes * (boots + 1))
    
    if (abs(floor(old_progress) - floor(progress)) > 0) {
      cat(floor(progress), "%\n")
      #for(aux in 1:abs(floor(old_progress) - floor(progress))) prog_bar$tick()
    }
    
    set.seed(b + i*boots)
    thresh <- 0.5 # Decision boundary
    
    X <- gen_preds(nobs, npreds)
    newdata <- gen_resp(X, coefs = coefs_general)
    Y <- newdata$classes
    coefs <- newdata$coefs
    
    ## Resample Patient data
    #if (b) {
    #  samp_mask <- sample(1:nobs, nobs_boot, replace = TRUE) # mask to resample data
    #  pat_data <- cbind(X, Y)[samp_mask,]
    #} else pat_data <- cbind(X, Y)
    
    
    
    pat_data <- cbind(X, Y)
    
    
    pat_data["Y"] <- lapply(pat_data["Y"], factor)
    
    # For each holdout size, split data into intervention and holdout set
    mask <- split_data(pat_data, frac_ho[i])
    data_interv <- pat_data[!mask,]
    data_hold <- pat_data[mask,]
    
    # Calculate optimal threshold
    indices <- sample(1:nrow(data_hold))
    folds <- cut(1:length(indices), breaks = k, labels = FALSE) # Mask to partition data in CV analysis
    cost_tot <- numeric(num_thresh)
    
    for (f in 1:k) {
      # Train model for each fold of CV, then scan all probs to find loss
      val_indices <- folds == f
      val_data <- data_hold[val_indices,]
      partial_train_data <- data_hold[!val_indices,]
      
      thresh_model <- glm(Y ~ ., data = partial_train_data, 
                          family = binomial(link = "logit"))
      
      thresh_pred <- predict(thresh_model, newdata = val_data, 
                             type = "response")
      
      for(p in 1:num_thresh) {
        num_tn <- sum(as.numeric(val_data["Y"] == 0) & as.numeric(thresh_pred < prob_vals[p]))
        num_fn <- sum(as.numeric(val_data["Y"] == 1) & as.numeric(thresh_pred < prob_vals[p]))
        num_fp <- sum(as.numeric(val_data["Y"] == 0) & as.numeric(thresh_pred >= prob_vals[p]))
        num_tp <- sum(as.numeric(val_data["Y"] == 1) & as.numeric(thresh_pred >= prob_vals[p]))
        
        cost_tot[p] <- cost_tot[p] + c_tn * num_tn + 
          c_tp * num_tp + 
          c_fn * num_fn + 
          c_fp * num_fp
      }
    }
    
    # Rescale loss
    cost_tot <- cost_tot / k
    
    
    # THIS NEEDS REWRITTING!!!!!!!!!
    # CURRENT COSTS ARE CALCULATED WITH THRESHOLD ESTIMATION, EVEN WHEN FLAG IS OFF
    # Train model
    glm_model <- glm(Y ~ ., data = data_hold, 
                     family = binomial(link = "logit"))
    thresh <- ifelse(set_thresh, prob_vals[which.min(cost_tot)], 0.5)
    if (b) costs_boot[b, i] <- min(cost_tot) else costs <- min(cost_tot)
    
    # Predict
    glm_pred <- predict(glm_model, newdata = data_interv, type = "response")
    class_pred <- ifelse(glm_pred > thresh, '1', '0')
    
    
    for (dr_vars in 1:max_dr_vars) { # sweep through different dr predictive powers
      if (run_many_powers) dr_pred <- oracle_pred(data_hold, 
                                                  coefs_dr, 
                                                  num_vars = dr_vars)
      else dr_pred <- oracle_pred(data_hold, coefs_dr)
      
      
      
      ##### CAREFUL HERE!!! CONSIDER USING THE THRESHOLD FOR H.O SET AS WELL, INSTEAD
      ##### OF THE BERNOULLI DISTRIBUTION!!!!!!!!!!!
      
      
      
      # Those with disease, predicted not to die, will die
      # This "if clause" calculates the cost for each h.o. set size 
      # as the number of deaths for each of them
      if (cost_type == "deaths") {
        if (b) {
          deaths_boot_inter[b, i] <- sum(data_interv$Y == 1 & class_pred != 1)
          deaths_boot_ho[b, i] <- sum(data_hold$Y == 1 & dr_pred != 1)
        } else {
          deaths_inter[i] <- sum(data_interv$Y == 1 & class_pred != 1)
          deaths_ho[i] <- sum(data_hold$Y == 1 & dr_pred != 1)
        }
      }
      
      
      # Alternatively, use a generalised cost function with a specific cost for
      # each of fn, fp, tp and tn
      if (cost_type == "gen_cost"){
        # Generate confusion matrices
        confus_inter <- table(factor(data_interv$Y, levels=0:1), 
                               factor(class_pred, levels=0:1))
        
        confus_hold <- table(factor(data_hold$Y, levels=0:1), 
                             factor(dr_pred, levels=0:1))
      
      # CAN SAVE MEMORY BY SAVING STRAIGHT AWAY INTO deaths_per_frac and _boot_tot
      # BUT THEN I DON'T HAVE THEM SEPARATELY. CONSIDER FLAG?
        if (b) {
          deaths_boot_inter[b, i] <- sum(confus_inter * cost_mat)
          deaths_boot_ho[b, i] <- sum(confus_hold * cost_mat)
          deaths_boot_tot[dr_vars, b, i] <- deaths_boot_ho[b, i] + deaths_boot_inter[b, i]
        } 
        else {
          deaths_inter[i] <- sum(confus_inter * cost_mat)
          deaths_ho[i] <- sum(confus_hold * cost_mat)
          deaths_per_frac[dr_vars, i] <- deaths_ho[i] + deaths_inter[i]
        }
      }
      
    }
    
    old_progress <- progress
  }
}

for (dr_vars in 1:max_dr_vars) 
  deaths_sd[dr_vars, ] <- apply(deaths_boot_tot[dr_vars, , ], 2, sd)

par(mfrow = c(1, 1))

# Test for multiple powers
plot(frac_ho, deaths_per_frac[1, ], type = "n", 
     ylab = "L",
     xlab = expression(pi),
     ylim = c(min(deaths_per_frac - deaths_sd), max(deaths_per_frac + deaths_sd))
     #ylim = range(colMeans(deaths_boot_tot[3,,]), na.rm=T)
     )

#colours = c("#fcba03", "#59b7ff","", "#b51d27")

for (dr_vars in 1:max_dr_vars){#1:max_dr_vars) {
  #points(frac_ho, 
  lines(frac_ho, 
         colMeans(deaths_boot_tot[dr_vars, , ], na.rm = T), 
         pch = 16, 
         lwd = 2,
         col = dr_vars)
  points(frac_ho[which.min(colMeans(deaths_boot_tot[dr_vars, , ]))],
         min(colMeans(deaths_boot_tot[dr_vars, , ])), 
         na.rm = T,
         pch = 4,
         col = 1)
}

if (run_many_powers) {
  legend("topleft", legend = (2 ** (c(0, 1, 3))), 
         fill = c("#fcba03", "#59b7ff", "#b51d27"), title = expression(sigma))
}




# -----------------------------------------------------------------------------#
#### Analysis Plots ####

#### Plot for 1/n behaviour of deaths in intervention set
# THIS SHOWS WHAT I WANT BUT THE NORMALISATION IS WRONG. IT'S NOT THE ACTUAL
# MISCLASSIFICATION ERROR IF I'M USING DEATHS, HAVE TO USE GEN COST WITH FN = FP = 1,
# TP = TN = 0
inv_x_behaviour <- (colMeans(deaths_boot_inter) / frac_ho) /
                        max(colMeans(deaths_boot_inter) / frac_ho)
plot(frac_ho, inv_x_behaviour, 
     xlab = "holdout set size", 
     ylab = "Normalised cost in intervention set")
lines(frac_ho, (8 / frac_ho) / max(8 / frac_ho) - 0.022)
lines(frac_ho, (8 / (frac_ho * nobs)))


lin_x_behaviour <- (colMeans(deaths_boot_ho) / frac_ho)
plot(frac_ho, lin_x_behaviour, col = 2)



###### Snippet for hist of probs
set.seed(107)  
npreds = 7
X <- gen_preds(nobs, npreds)
probs_hist <- gen_resp(X, retprobs = TRUE, coefs_sd = 1)

par(mfrow = c(1, 3))

hist(probs_hist, 
     xlab = "f(X)",
     main = "Hist of all samples")

hist(probs_hist[which(as.logical(rbinom(nobs, 1, probs_hist)))], 
     xlab = "f(X)",
     main = "Hist of Y=1")

hist(probs_hist[which(!as.logical(rbinom(nobs, 1, probs_hist)))], 
     xlab = "f(X)",
     main = "Hist of Y=0")



##### Snippet to test the threshold decision
c_tn <- 0 # Cost of true neg
c_tp <- 0.5 # Cost of true pos
c_fp <- 0.5 # Cost of false pos
c_fn <- 1 # Cost of false neg

num_probs <- 10 # Number of porbabilities to scan
prob_vals <- seq(0, 1, length.out = num_probs)
cost_tot <- numeric(length(prob_vals))


# Plot cost as a function of threshold
for(i in 1:num_probs){
  model_pred$set_threshold(prob_vals[i])
  num_tp <- model_pred$confusion[1]
  num_fn <- model_pred$confusion[2]
  num_fp <- model_pred$confusion[3]
  num_tn <- model_pred$confusion[4]
  cost_tot[i] <- c_tn * num_tn + 
    c_tp * num_tp + 
    c_fn * num_fn + 
    c_fp * num_fp
}
plot(prob_vals, cost_tot)
cost_tot


# Plot cost vs threshold case where FP are penalised beacuse too expensive
for(i in 1:num_probs){
  model_pred$set_threshold(prob_vals[i])
  num_tp <- model_pred$confusion[1]
  num_fn <- model_pred$confusion[2]
  num_fp <- model_pred$confusion[3]
  num_tn <- model_pred$confusion[4]
  cost_tot[i] <- c_tn * num_tn + 
    c_tp * num_tp + 
    c_fn * num_fn + 
    c_fp * num_fp * 4 
}
plot(prob_vals, cost_tot)
cost_to


plot(frac_ho, deaths_per_frac[1, ], type = "n", ylim = c(1500, 2100))

lines(frac_ho, 
               colMeans(deaths_boot_inter[ , ], na.rm = T), 
               pch = 16, 
               lwd = 2,
               col = dr_vars)

lines(frac_ho, 
      colMeans(deaths_boot_ho[ , ] + 1500, na.rm = T), 
      pch = 16, 
      lwd = 2,
      col = dr_vars)
