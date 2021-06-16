################################################################################
## Set seed for reproducibility
set.seed(1234)

################################################################################
## Call libraries
library("mlr3pipelines")
library("mlr3tuning")
library("mlr3filters")
library("mlr3verse")
library("dplyr")
library("progress")

################################################################################
## Model and Mathematical function Definitions
logistic <- function(x) 1/(1 + exp(-x))
logit <- function(x) -log(1/x - 1)


# Risk score before intervention, given rho and covariates before intervention
f <- function(xs, xa) {
  logistic(xs + xa)
}


# Risk score after intervention, given rho and covariates before intervention
h <- function(xs, xa, r) f(xs, g(r, xa))


# Intervention function, taking X(0) -> X(1)
g <- function(rho, xa) {
  (xa + 0.5 * (xa + sqrt(1 + xa^2)))*(1 - rho) + 
    (xa - 0.5 * (xa + sqrt(1 + xa^2))) * rho
}


# Risk score at e calculated from the value at e-1
rho_update <- function(xs, xa, e, old_rho){
  if (e == 0) {
    return(f(xs, xa))
  }
  
  return(h(xs, xa, old_rho))
}


drift <- function(X, coefs) {
  # For now, implemented drift for the model, changing the coefficients of the
  # logreg model that calculates the true classes.
  nobs <- dim(X)[1]
  npreds <- dim(X)[2]
  #X <- X + matrix(rnorm(nobs * npreds) / 10, nobs, npreds)
  coefs <- coefs + rnorm(npreds) / 10
  newdata <- gen_resp(X, coefs)
  
  return(list("X" = X, "Y" = newdata$classes, "coefs" = newdata$coefs))
}


oracle_pred <- function(X, coefs){
  #Dr is oracle to the first predictor and blind to the other ones.
  X <- X %>% select(1, 2)
  nobs <- dim(X)[1]
  coefs <- coefs[1:2]
  
  # Trying adding noise to make it harder for dr
  lin_comb <- rowSums(t(t(X) * coefs))
  probs <- round(logistic(lin_comb))# + rnorm(nobs, sd = 0.2))
  # Now check that the noise hasn't spoiled probabilities
  probs[which(probs > 1)] = 1
  probs[which(probs < 0)] = 0
  
  return(rbinom(nobs, 1, probs))
}




################################################################################
## Function definitions

# Generate the predictors matrix
gen_preds <- function(nobs, npreds) {
  # Generate gaussian covariates matrix
  X_vals <- as.data.frame(matrix( rnorm(nobs * npreds), nobs, npreds))
  return(X_vals)
}


# Generate samples of Y
gen_resp <- function(X, coefs = NA, coefs_sd = 1, retprobs = FALSE) {
  # For now, combines predictors linearly and applies binomial to find class
  nobs <- dim(X)[1]
  npreds <- dim(X)[2]
  
  # Generate coefficients for each predictor for the linear combination
  if (is.na(coefs)) {
    #denom <- log(npreds)
    denom <- npreds ** 1
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
  mask <- sample(FALSE, nobs, replace = TRUE)
  train_lim <- floor(nobs * frac)
  mask[1:train_lim] <- TRUE
  return(mask)
}



################################################################################
## Dynamics of the system
# Initialisation of patient data
boots <- 100      # Number of bootstrap resamplings
nobs <- 50000                      # Number of observations, i.e patients
nobs_boot <- 50000         # Number of observations to resample in each bootstrap iter
npreds <- 7                    # Number of predictors

# Definition of holdout set sizes to test
min_frac <- 0.002
max_frac <- 0.02
num_sizes <- 150


# Variables for the threshold estimation
c_tn <- 0 # Cost of true neg
c_tp <- 0.5 # Cost of true pos
c_fp <- 0.5 # Cost of false pos
c_fn <- 1 # Cost of false neg
num_thresh <- 10 # Number of probabilities to scan
prob_vals <- seq(0, 1, length.out = num_thresh) # Vector with probability thresh to scan
thresh_vals <- seq()
k <- 5 # Number of CV folds


# Fraction of patients assigned to the holdout set
frac_ho <- seq(min_frac, max_frac, length.out = num_sizes)
dyn_cols = colorRampPalette(c("red", "blue"))(num_sizes) # Colour palette

costs_boot <- matrix(nrow = boots, ncol = num_sizes)
costs <- seq()


# Matrix with the number of deaths per epoch and holdout size
deaths_boot_inter <- deaths_boot_ho <- matrix(nrow = boots, ncol = num_sizes)
deaths_inter <- deaths_ho <- seq()


# Initialise Progress Bar
#if (exists(prog_bar)) rm(prog_bar)
#prog_bar <- progress_bar$new()
old_progress <- 0


# Initialise Dataset
set.seed(100)  
X <- gen_preds(nobs, npreds)
newdata <- gen_resp(X, coefs_sd = 6)
Y <- newdata$classes
coefs <- newdata$coefs

for (i in 1:num_sizes) {
#  thresh <- 0.5
#  pat_data <- cbind(X, Y)
#  pat_data["Y"] <- lapply(pat_data["Y"], factor)
#  
#  mask <- split_data(pat_data, frac_ho[i])
#  data_interv <- pat_data[!mask,]
#  data_hold <- pat_data[mask,]
#  
#  # Calculate optimal threshold
#  indices <- sample(1:nrow(data_hold))
#  folds <- cut(1:length(indices), breaks = k, labels = FALSE) # Mask to partition data in CV analysis
#  cost_tot <- numeric(num_thresh)
#  
#  for (f in 1:k) {
#    # Train model for each fold of CV, then scan all probs to find loss
#    val_indices <- folds == f
#    val_data <- data_hold[val_indices,]
#    partial_train_data <- data_hold[!val_indices,]
#    
#    thresh_model <- glm(Y ~ ., data = partial_train_data, 
#                     family = binomial(link = "logit"))
#    
#    thresh_pred <- predict(thresh_model, newdata = val_data, 
#                           type = "response")
#    
#    for(p in 1:num_thresh){
#      num_tn <- sum(as.numeric(val_data["Y"] == 0) & as.numeric(thresh_pred < prob_vals[p]))
#      num_fn <- sum(as.numeric(val_data["Y"] == 1) & as.numeric(thresh_pred < prob_vals[p]))
#      num_fp <- sum(as.numeric(val_data["Y"] == 0) & as.numeric(thresh_pred >= prob_vals[p]))
#      num_tp <- sum(as.numeric(val_data["Y"] == 1) & as.numeric(thresh_pred >= prob_vals[p]))
#      
#      cost_tot[p] <- cost_tot[p] + c_tn * num_tn + 
#        c_tp * num_tp + 
#        c_fn * num_fn + 
#        c_fp * num_fp
#    }
#  }
#  
#  # Rescale loss
#  cost_tot <- cost_tot / k
#  
#  # Minimise loss to find classif threshold
#  thresh <- prob_vals[which.min(cost_tot)]
#  
#  # Predict
#  glm_model <- glm(Y ~ ., data = data_hold, 
#                   family = binomial(link = "logit"))
#  glm_pred <- predict(glm_model, newdata = data_interv, type = "response")
#  class_pred <- ifelse(glm_pred > thresh, '1', '0')
#  
#  # Dr is an oracle for the first predictor and blind for the rest of them
#  dr_pred <- oracle_pred(data_hold, coefs)
#  
#  
#  # Those with disease, predicted not to die, will die
#  deaths_inter[i] <- sum(data_interv$Y == 1 & class_pred != 1)
#  deaths_ho[i] <- sum(data_hold$Y == 1 & dr_pred != 1)
  
  # Bootstrapping loop, the iteration 0 uses the whole dataset to calculate the
  # estimates of the predictions.
  
  for (b in 0:boots) {
    progress <- 100 * (((i - 1) * boots) + b) / (num_sizes * (boots + 1))
    
    if (abs(floor(old_progress) - floor(progress)) > 0) {
      cat(floor(progress), "%\n")
      #for(aux in 1:abs(floor(old_progress) - floor(progress))) prog_bar$tick()
    }
    
    set.seed(b + i*boots)
    thresh <- 0.5 # Decision boundary
    
    # Resample Patient data
    if (b) {
      samp_mask <- sample(1:nobs, nobs_boot, replace = TRUE) # mask to resample data
      pat_data <- cbind(X, Y)[samp_mask,]
    } else pat_data <- cbind(X, Y)
    
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
      
      for(p in 1:num_thresh){
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
    
    # Train model
    glm_model <- glm(Y ~ ., data = data_hold, 
                     family = binomial(link = "logit"))
    thresh <- prob_vals[which.min(cost_tot)]
    if (b) costs_boot[b, i] <- min(cost_tot) else costs <- min(cost_tot)
    
    # Predict
    glm_pred <- predict(glm_model, newdata = data_interv, type = "response")
    class_pred <- ifelse(glm_pred > thresh, '1', '0')
    
    # Dr is an oracle for the first predictor and blind for the rest of them
    dr_pred <- oracle_pred(data_hold, coefs)
    
    # Those with disease, predicted not to die, will die
    if (b){
      deaths_boot_inter[b, i] <- sum(data_interv$Y == 1 & class_pred != 1)
      deaths_boot_ho[b, i] <- sum(data_hold$Y == 1 & dr_pred != 1)
    } else {
      deaths_inter[i] <- sum(data_interv$Y == 1 & class_pred != 1)
      deaths_ho[i] <- sum(data_hold$Y == 1 & dr_pred != 1)
    }
    
    old_progress <- progress
  }
}

deaths_per_frac <- deaths_ho + deaths_inter
deaths_boot_tot <- deaths_boot_ho + deaths_boot_inter
deaths_sd <- apply(deaths_boot_tot, 2, sd)

plot(frac_ho, deaths_per_frac,
#plot(costs, deaths_inter,
     pch = 16,
     ylab = "Deaths",
     xlab = "Holdout set size",
     #ylim = c(min(deaths_per_frac - deaths_sd), max(deaths_per_frac + deaths_sd)))
    ylim = c(0, max(deaths_per_frac + deaths_sd)))

points(frac_ho, deaths_inter, pch = 16, col = 2)

arrows(frac_ho, 
       deaths_per_frac - deaths_sd,
       frac_ho,
       deaths_per_frac + deaths_sd, 
       length=0.05, angle=90, code=3)


##### Snippet for hist of probs
set.seed(100)  
npreds = 7
X <- gen_preds(nobs, npreds)
probs_hist <- gen_resp(X, retprobs = TRUE, coefs_sd = 6)
hist(probs_hist)



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
cost_tot