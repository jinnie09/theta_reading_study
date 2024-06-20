import pandas as pd
import pingouin as pg

df = pd.read_csv('theta_1018.csv')

# Research Question 1 - Logistic Regression: Compare models with and without the variable of interest (theta power) 

import statsmodels.api as sm
import statsmodels.stats.power as smp

# Add a constant for the intercept
df['Intercept'] = 1

# Fit the first logistic regression model (covariates only)
model_covariates = sm.Logit(df['read.impair'], df[['Intercept', 'Age', 'WISC_MR_raw', 'epoch']])
result_covariates = model_covariates.fit()

# Print the summary of the first model
print(result_covariates.summary())

# AIC for the first model
aic_covariates = result_covariates.aic
print(f'AIC (Covariates Only): {aic_covariates}')

# BIC for the first model
bic_covariates = result_covariates.bic
print(f'BIC (Covariates Only): {bic_covariates}')

# Log-Likelihood for the first model
llf_covariates = result_covariates.llf
print(f'Log-Likelihood (Covariates Only): {llf_covariates}')

# Null Log-Likelihood for the first model
llnull_covariates = result_covariates.llnull
print(f'Null Log-Likelihood (Covariates Only): {llnull_covariates}')

# McFadden's Pseudo R-squared for the first model
pseudo_r_squared_covariates = 1 - (llf_covariates / llnull_covariates)
print(f'McFadden\'s Pseudo R-squared (Covariates Only): {pseudo_r_squared_covariates}')

# Fit the second logistic regression model (covariates + X)
model_full = sm.Logit(df['read.impair'], df[['Intercept', 'avg.front.log', 'Age', 'WISC_MR_raw', 'epoch']])
result_full = model_full.fit()

# Print the summary of the second model
print(result_full.summary())

# AIC for the second model
aic_full = result_full.aic
print(f'AIC (Covariates + X): {aic_full}')

# BIC for the second model
bic_full = result_full.bic
print(f'BIC (Covariates + X): {bic_full}')

# Log-Likelihood for the second model
llf_full = result_full.llf
print(f'Log-Likelihood (Covariates + X): {llf_full}')

# Null Log-Likelihood for the second model
llnull_full = result_full.llnull
print(f'Null Log-Likelihood (Covariates + X): {llnull_full}')

# McFadden's Pseudo R-squared for the second model
pseudo_r_squared_full = 1 - (llf_full / llnull_full)
print(f'McFadden\'s Pseudo R-squared (Covariates + X): {pseudo_r_squared_full}')

# Compare AIC
print(f'Difference in AIC: {aic_covariates - aic_full}')

# Compare BIC
print(f'Difference in BIC: {bic_covariates - bic_full}')

# Compare McFadden's Pseudo R-squared
print(f'Difference in McFadden\'s Pseudo R-squared: {pseudo_r_squared_full - pseudo_r_squared_covariates}')


# Research Question 2 - Partial Correlation between theta power and PA

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pingouin import partial_corr, power_corr

# Calculate Spearman partial correlation
partial_corr_result = partial_corr(data=df, x='avg.front.log', y='CTOPP.com.raw', covar=['Age', 'WISC_MR_raw', 'epoch'], method='spearman')
print(partial_corr_result)

# Extract the Spearman partial correlation coefficient
r = partial_corr_result['r'].values[0]
n = df.shape[0]

# Calculate post hoc power
power = power_corr(r=r, n=n)
print(f'Post hoc power: {power}')
