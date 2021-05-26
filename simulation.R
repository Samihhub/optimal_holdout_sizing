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

# Risk score at epoch e, calculated recursively from all previous values.
# Consider implementing by saving previous risk scores to avoid unnecessary computations.
# This algorithm, for 100 epochs, calculates a total of 5000 risk scores per point. If
# the simulations are going to need lots of calculations of risk scores, it will be
# worth it to optimise this.
#oldre <- function(xs, xa, e) {
#  # No intervention at e==0, so call f directly. Otherwise, model intervention through g
#  if (e == 0) {
#    return(f(xs, xa))
#  }
#  
#  return(h(xs, xa, oldre(xs, xa, e-1)))
#}

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
nobs <- 10000                      # Number of observations, i.e patients
npreds <- 10                     # Number of predictors
#num_action <- 8                   # Number of actionable predictors

# Definition of holdout set sizes to test
min_frac <- 0.001
max_frac <- 0.02
num_sizes <- 20

# Threshold optimisation variables
cost_fp <- 1 # Cost of further testing a non-ill patient
cost_fn <- 1 # Cost of not treating an ill patient, down the line
cost_tp <- 1 # Cost of threating an ill patient
cost_tot <- 0
res_tot <- 1000 # Avilable resources
thresh <- 0.5
num_thresh <- 100

# Fraction of patients assigned to the holdout set
frac_ho <- seq(min_frac, max_frac, length.out = num_sizes)
dyn_cols = colorRampPalette(c("red", "blue"))(num_sizes) # Colour palette

costs <- matrix(0, num_sizes, 1)

# Variables for the threshold estimation
c_tn <- 0 # Cost of true neg
c_tp <- 0.5 # Cost of true pos
c_fp <- 0.5 # Cost of false pos
c_fn <- 1 # Cost of false neg
num_probs <- 10 # Number of porbabilities to scan
prob_vals <- seq(0, 1, length.out = num_probs)
cost_tot <- numeric(num_probs)    


# Number of epochs to study
e_max <- 1
# Matrix with the number of deaths per epoch and holdout size
deaths_inter <- deaths_ho <- matrix(nrow = e_max, ncol = num_sizes)
old_progress <- 0
avg_disease <- seq()

if (exists(prog_bar)) rm(prog_bar)
prog_bar <- progress_bar$new()
ticks = 0

for (i in 1:num_sizes) {
  # Set seed at each frac to ensure same predictor evolution for all of them
  set.seed(100)  
  X <- gen_preds(nobs, npreds)
  newdata <- gen_resp(X, coefs_sd = 6)
  Y <- newdata$classes
  coefs <- newdata$coefs
  avg_disease[i] <- 0
  thresh <- 0.5 # Decision boundary
  
  for (e in 1:e_max) {
    avg_disease[i] <- avg_disease[i] + sum(Y)
    
    progress <- 100 * ((i - 1) * e_max + e) / (num_sizes * e_max)
    if (abs(floor(old_progress) - floor(progress)) >= 0) {
      #cat(floor(progress), "%\n")
      for(aux in 1:abs(floor(old_progress) - floor(progress))) prog_bar$tick()
    }
    # Patient data
    pat_data <- cbind(X, Y)
    pat_data["Y"] <- lapply(pat_data["Y"], factor)
    
    # For each holdout size, split data into intervention and holdout set
    mask <- split_data(X, frac_ho[i])
    data_interv <- pat_data[!mask,]
    data_hold <- pat_data[mask,]
    
    # In holdout set, split for threshold calc
    
    # Holdout set mlr3 task
    hold_task <- TaskClassif$new(id = "Holdout",
                                 backend = data_hold,
                                 target = "Y",
                                 positive = '1')
    
    # Intervention set mlr3 task
    interv_task <- TaskClassif$new(id = "Intervention",
                                   backend = data_interv,
                                   target = "Y",
                                   positive = '1')
    
    # Initialise logreg Learner
    lrn_patient  <- lrn("classif.log_reg", predict_type = "prob")
    
    # Calculate optimal threshold
    k <- 5 # Number of CV folds
    indices <- sample(1:nrow(data_hold))
    folds <- cut(1:length(indices), breaks = k, labels = FALSE) # Mask to partition data in CV analysis
    
    for(p in 1:num_probs){
      for (f in 1:k) {
        val_indices <- folds == f
        val_data <- data_hold[val_indices,]
        partial_train_data <- data_hold[!val_indices,]
        
        partial_task <- TaskClassif$new(id = "CV_ho_train",
                                        backend = partial_train_data,
                                        target = "Y",
                                        positive = '1')
        
        partial_val_task <- TaskClassif$new(id = "CV_ho_val",
                                            backend = val_data,
                                            target = "Y",
                                            positive = '1')
        
        lrn_patient$train(partial_task)
        model_thresh <- lrn_patient$predict(partial_val_task)
        
        model_thresh$set_threshold(prob_vals[p])
        num_tp <- model_thresh$confusion[1]
        num_fn <- model_thresh$confusion[2]
        num_fp <- model_thresh$confusion[3]
        num_tn <- model_thresh$confusion[4]
        cost_tot[p] <- cost_tot[p] + c_tn * num_tn + 
          c_tp * num_tp + 
          c_fn * num_fn + 
          c_fp * num_fp
      }
      cost_tot[p] <- cost_tot[p] / k
    }
    
    # Train model
    lrn_patient$train(hold_task)
    thresh <- prob_vals[which.min(cost_tot)]
    model_thresh$set_threshold(thresh)
    costs[i] <- min(cost_tot)
    
    # Predict
    model_pred <- lrn_patient$predict(interv_task)
    model_resp <- model_pred$response
    
    
    # Dr is an oracle for the first predictor and blind for the rest of them
    dr_pred <- oracle_pred(data_hold, coefs)
    
    
    # Those with disease, predicted not to die, will die
    deaths_inter[e, i] <- sum(data_interv$Y == 1 & model_resp != 1)
    deaths_ho[e, i] <- sum(data_hold$Y == 1 & dr_pred != 1)
    
    old_progress <- progress
    
    set.seed(e)
    X <- gen_preds(nobs, npreds)
    newdata <- gen_resp(X, coefs_sd = 6)          # I use the name newdata for diff things. CHANGE
    Y <- newdata$classes
    coefs <- newdata$coefs
  }
  
  avg_disease[i] <- avg_disease[i] / e_max
}

tot_deaths <- deaths_ho + deaths_inter
tot_sd <- apply(tot_deaths, 2, sd)

#plot(0, type="n", xlim = c(0, e_max), ylim = c(min(tot_deaths), max(tot_deaths)),
#     xaxs = "i", yaxs = "i",
#     xlab = "Epoch",
#     ylab = "Total Number of deaths")
#
#for (i in 1:num_sizes) lines(1:e_max, tot_deaths[, i], col = dyn_cols[i])
#legend("topright", legend = frac_ho, pch = 16, col = dyn_cols, bty = "n",
#       title = "Holdout set size")

deaths_per_frac <- colSums(tot_deaths) / e_max
min_deaths <- deaths_per_frac - tot_sd
max_deaths <- deaths_per_frac + tot_sd


#plot(frac_ho, deaths_inter,
     plot(costs, deaths_inter,
     pch = 16,
     ylab = "Deaths",
     xlab = "Holdout set size", 
     ylim = c(min(tot_deaths), max(tot_deaths)))
#arrows(frac_ho, min_deaths, frac_ho, max_deaths, 
#       length=0.05, angle=90, code=3)
points(frac_ho, deaths_per_frac, col = 2, pch = 16)
points(frac_ho, deaths_ho, col = 3, pch = 16)
#points(frac_ho, avg_disease, col = 4, pch = 16)



#### Debug plots
plot(0, type="n", xlim = c(0, e_max), ylim = c(0, max(tot_deaths)))
lines(1:e_max, deaths_inter[, 1], col = 2)
lines(1:e_max, deaths_ho[, 1], col = 2)
lines(1:e_max, deaths_inter[, 15], col = 3)
lines(1:e_max, deaths_ho[, 15], col = 3)

plot(0, type="n", xlim = c(0, e_max), ylim = c(0, max(tot_deaths)))
lines(1:e_max, tot_deaths[, 1], col = 2)
lines(1:e_max, tot_deaths[, 5], col = 1)
lines(1:e_max, tot_deaths[, 10], col = 4)
lines(1:e_max, tot_deaths[, 15], col = 3)



##### Snippet for hist of probs
set.seed(100)  
npreds = 10
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

# Plot Normal
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

# Plot case where FP are penalised beacuse too expensive
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
