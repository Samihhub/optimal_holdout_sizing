#### Simulation ####
## Simulation for the estimation of the optimal holdout set size ##
## on a population where intervention is guided by a risk score  ##

# Ctrl+Alt+T to run section by section

# ---------------------------------------------------------------------------- #
#### Package loading ####
library("dplyr")
library("ranger")



# ---------------------------------------------------------------------------- #
#### Function Definitions ####

## Model and Mathematical function Definitions
logistic <- function(x) 1/(1 + exp(-x))

gen_dr_coefs <- function(coefs, noise = TRUE, num_vars = 2, max_dr_powers = 1) {
  if (!noise) {
    return(coefs[1:num_vars])
  } 
  
  coefs_dr <- matrix(nrow = max_dr_powers, ncol = length(coefs))
  
  if (max_dr_powers - 1){
    for (dr_vars in 1:max_dr_powers)
      # If we are estimating the cost for more than one dr power, we use the ad-hoc
      # formula below for the standard deviation
      coefs_dr[dr_vars, ] <- coefs + rnorm(length(coefs), sd = 2 ** (dr_vars - 2))#1))
  } else {
    coefs_dr[1, ] <- coefs + rnorm(length(coefs), sd = 1)
  }
  
  return(coefs_dr)
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
    denom <- npreds ** 0.5
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


# Wrapper function to train model independently of chosen model to study
model_train <- function(train_data, model_family = "log_reg") {
  # Takes training data and the model family, returns a model trained on that data
  # Options:
  # log_reg for logistic regression
  # rand_forest for random forest
  
  if (model_family == "log_reg"){
    model <- glm(Y ~ ., data = train_data, family = binomial(link = "logit"))
  } else if (model_family == "rand_forest") {
    model <- ranger(Y ~ ., data = train_data, probability = TRUE)
  }
  
  return(model)
}


# Wrapper function to predict outcome for new data. Necessary as different models have 
# different calls to predict()
model_predict <- function(data_test, trained_model, return_type, threshold = NULL, model_family = NULL) {
  if (model_family == "log_reg") {
    predictions <- predict(trained_model, newdata = data_test, type = "response")
  } else if (model_family == "rand_forest") {
    predictions <- predict(trained_model, data = data_test, type = 'response')$predictions[ ,2]
  } else if (is.null(model_family)) {
    stop("model_predict: Please provide a correct model family")
  }
  
  if (return_type == "class") {
    return(ifelse(predictions > threshold, '1', '0'))
  } else if(return_type == "probs"){
    return(predictions)
  } else stop("model_predict: Wrong return type. Specify in return_type")
}


# -----------------------------------------------------------------------------#
#### Dynamics of the system ####

## Set seed for reproducibility
set.seed(1234)

# Initialisation of patient data
boots <- 10      # Number of point estimates to be calculated
nobs <- 5000                      # Number of observations, i.e patients
npreds <- 5                    # Number of predictors
family <- "rand_forest" # Model family


# Flag and vars to generate multiple dr predictive powers.
max_dr_powers <- 1 # When testing different Dr predictive powers, maximum power to be tested
run_many_powers <- !(max_dr_powers == 1) # Flag for analysing several Dr powers at once
if(max_dr_powers > npreds) stop("max_dr_powers cannot be larger than npreds")
if(max_dr_powers > 1 && !run_many_powers) stop("Flag set to run only 1 power, but max_dr_powers > 1")
if(max_dr_powers == 1 && run_many_powers) stop("Flag set to run many powers, but max_dr_powers = 1")


# Definition of holdout set sizes to test
min_frac <- 0.02
max_frac <- 0.15
num_sizes <- 6
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


old_progress <- 0


# Generate coefficients for underlying model and for predictions in h.o. set
set.seed(107)  
X <- gen_preds(nobs, npreds)
coefs_general <- gen_resp(X)$coefs
coefs_dr <- gen_dr_coefs(coefs_general, max_dr_powers = max_dr_powers)


# Arrays to store data from sweeping through Dr's different powers
deaths_per_frac <- array(0, dim = c(max_dr_powers, num_sizes))
deaths_boot_tot <- array(0, dim = c(max_dr_powers, boots, num_sizes))
deaths_sd <- array(0, dim = c(max_dr_powers, num_sizes))


for (i in 1:num_sizes) {  # sweep through h.o. set sizes of interest
  for (b in 1:boots) {
    
    progress <- 100 * (((i - 1) * boots) + b) / (num_sizes * (boots + 1))
    
    if (abs(floor(old_progress) - floor(progress)) > 0) {
      cat(floor(progress), "%\n")
    }
    
    set.seed(b + i*boots)
    thresh <- 0.5 # Decision boundary
    
    X <- gen_preds(nobs, npreds)
    newdata <- gen_resp(X, coefs = coefs_general)
    Y <- newdata$classes
    coefs <- newdata$coefs
    
    
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
      
      thresh_model <- model_train(partial_train_data, model_family = family)
      
      thresh_pred <- model_predict(val_data, thresh_model, return_type = "probs",
                                   model_family = family)
      
      
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
    
    # Train model
    trained_model <- model_train(data_hold, model_family = family)
    #glm(Y ~ ., data = data_hold, family = binomial(link = "logit"))
    thresh <- ifelse(set_thresh, prob_vals[which.min(cost_tot)], 0.5)
    if (b) costs_boot[b, i] <- min(cost_tot) else costs <- min(cost_tot)
    
    # Predict
    #model_probs <- predict(trained_model, newdata = data_interv, type = "response")
    #class_pred <- ifelse(model_probs > thresh, '1', '0')
    class_pred <- model_predict(data_interv, trained_model, return_type = "class", threshold = thresh, model_family = family)
    
    
    
    for (dr_vars in 1:max_dr_powers) { # sweep through different dr predictive powers
      if (run_many_powers) dr_pred <- oracle_pred(data_hold, 
                                                  coefs_dr[dr_vars, ], num_vars = dr_vars)
      else dr_pred <- oracle_pred(data_hold, coefs_dr[dr_vars, ])
      
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

# Calculate Standard Deviations and Errors
for (dr_vars in 1:max_dr_powers) {
  deaths_sd[dr_vars, ] <- 
    apply(deaths_boot_tot[dr_vars, , ], 2, sd)
}
deaths_se <- deaths_sd / sqrt(boots)




# ---------------------------------------------------------------------------- #
#### PLOTTING SECTION ####
# Set plot to one subfigure and define colours
par(mfrow = c(1, 1))

# Colours for plots
reds <- c(0.07, 0.95, 0.22, 0.65)
greens <- c(0.6, 0.7, 0.72, 0.1)
blues <- c(0.85, 0, 0.25, 0.12)
colours_line <- c()
colours_fill <- c()

for (i in 1:max_dr_powers) {
  colours_line[i] <- as.character(rgb(red = reds[i], green = greens[i], blue = blues[i]))
  colours_fill[i] <- as.character(rgb(red = reds[i], green = greens[i], blue = blues[i], alpha = 0.2))
}


# Create Figure
plot(frac_ho, deaths_per_frac[1, ], type = "n", 
     ylab = "L",
     xlab = expression(pi),
     ylim = c(min(colMeans(deaths_boot_tot[dr_vars, , ]) - deaths_sd[1, ]), 
              max(colMeans(deaths_boot_tot[dr_vars, , ]) + deaths_sd[1, ]))
     #ylim = c(2025, 2125)
)


for (dr_vars in 1:max_dr_powers) {
  # Plot Cost line
  lines(frac_ho, 
        colMeans(deaths_boot_tot[dr_vars, , ]), 
        pch = 16, 
        lwd = 1,
        col = colours_line[dr_vars])
  
  # Mark minima in plot
  points(frac_ho[which.min(colMeans(deaths_boot_tot[dr_vars, , ]))],
         min(colMeans(deaths_boot_tot[dr_vars, , ])),
         pch = 4,
         col = 1)
  
  # Plot SE bands
  polygon(c(frac_ho, rev(frac_ho)), 
          c(colMeans(deaths_boot_tot[dr_vars, , ]) - deaths_se[dr_vars, ], 
            rev(colMeans(deaths_boot_tot[dr_vars, , ]) + deaths_se[dr_vars, ])),
          col = colours_fill[dr_vars],
          #rgb(red = reds[dr_vars], green = greens[dr_vars], blue = blues[dr_vars], alpha = 0.2),
          border = NA)
}


if (run_many_powers) {
  # Legend
  legend("topleft", legend = (2 ** (1:max_dr_powers - 1)), 
         fill = colours_line[1:max_dr_powers], title = expression(lambda))
} else {
  # Standard Deviation Bands, if only plotting 1 line 
  polygon(c(frac_ho, rev(frac_ho)), 
          c(colMeans(deaths_boot_tot[1,,]) - deaths_sd[1, ], 
            rev(colMeans(deaths_boot_tot[1,,]) + deaths_sd[1, ])),
          col = colours_fill[dr_vars],
          #rgb(red = reds[dr_vars], green = greens[dr_vars], blue = blues[dr_vars], alpha = 0.2),
          border = NA)
  
  # Legend
  legend("topleft", legend = c("SD", "SEM"), 
         fill = c(rgb(red = reds[1], green = greens[1], blue = blues[1], alpha = 0.2),
                  rgb(red = reds[1], green = greens[1], blue = blues[1], alpha = 0.4)), 
         #title = "Uncertainty"
  )
}






# -----------------------------------------------------------------------------#
#### Analysis Plots ####

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





##### Fitting linear curve according to parametrisations from literature ####
max_index <- 3
test_inter <- colMeans(deaths_boot_inter) / (nobs * (1-frac_ho))
df <- as.data.frame(cbind(frac_ho[1:max_index], test_inter[1:max_index]))
colnames(df) <- c("V1", "V2")
model_inter <- coef(nls(V2 ~ a / (V1 *nobs) + b, df, start = list(a = 8, b = 0.4)))

test_ho <- colMeans(deaths_boot_ho) / (frac_ho * nobs)
df2 <- as.data.frame(cbind(frac_ho, test_ho))
colnames(df2) <- c("V1", "V2")
fit_vals <- (model_inter[1] / (frac_ho * nobs) + model_inter[2]) * (nobs * (1-frac_ho)) + mean(test_ho) * nobs * frac_ho
lines(frac_ho, fit_vals, col="#636363")

min_fit <- which.min(fit_vals)
# Mark minimum in plot
points(frac_ho[min_fit],
       fit_vals[min_fit],
       pch = 4,
       col = 2)
frac_ho[min_fit]



plot(frac_ho, test_inter, 
     pch = 16,
     col = colours_line[dr_vars],
     ylab = "Relative L",
     xlab = expression(pi))
lines(frac_ho, 1.92 / (frac_ho * nobs) + 0.400,
      lwd = 2,
      col = 2)

plot(frac_ho, test_ho, 
     pch = 16,
     col = colours_line[dr_vars],
     ylab = "Relative L",
     xlab = expression(pi))
