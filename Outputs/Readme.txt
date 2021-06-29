Plot Key:

*PlotBootstrapAverages.svg: 
Example of deaths as a function of ho_frac, with uncertainty estimated by bootstrapping - resampling size smaller than dataset size.

*Plot7preds1-7drpreds.svg:
Example trialing a sweep of the Dr's predictive power, i.e. simulation ran for which the oracle has access to an increasing num of preds.
5000 samples, 50 ho sizes, resampling 5000 per bootstrap, with 100 resamples. Errors saved in its workspace, Data7preds1-7drpreds.RData.

*Data7preds-drpreds_w_noise.svg:
Same as above, but now rather than being an oracle, I introduced an increasing noise in the coefficients to simulate the predictions from Drs.
For this, I used "coefs <- coefs + rnorm(npreds, sd = (num_vars - 1))" in the function oracle_pred.
Data saved in Data7preds-drpreds_w_noise.RData