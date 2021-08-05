import os
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
from pandas.plotting import parallel_coordinates
from scipy import stats
from theano import shared, tensor


data = stats.norm(4,0.5).rvs(size=57)
print(data)

with pm.Model() as model:
    mu = pm.Normal("mu", 0, 10)
    print("1111111111111111111")
    sd = pm.HalfNormal("sd", 25)
    print("22222222222222222222222222")
    y = pm.Normal("y,", mu, sd, observed=data)
    print("333333333333333333333333333333")
    # Compute both prior, and prior predictive
    prior_predictive = pm.sample_prior_predictive()
    print("444444444444444444444444444444444444")
    # Compute posterior
    trace = pm.sample(500,cores =1) #隨意設定 posterior (?
    print("55555555555555555555555555555")
    # Compute posterior predictive
    posterior_predictive = pm.sample_posterior_predictive(trace)
    print("66666666666666666666666666666")

dataset = az.from_pymc3(trace=trace, posterior_predictive=posterior_predictive, prior=prior_predictive)
print(dataset)
#
#prior distributions
az.plot_posterior(dataset.prior, var_names=["mu", "sd"]) #plot_posterior method 可用來畫 priors
plt.show()
# print("prior:" , dataset.prior)

# Compare above plot to posterior distribution below, as well as to original parameters in distribution
az.plot_posterior(dataset)
plt.show()

# prior predictivep
print(dataset.prior_predictive["y,"].values.shape)
print(dataset.prior)
prior_predictive = dataset.prior_predictive["y,"].values.flatten()
az.plot_kde(prior_predictive)
plt.show()
#
#posterior predictive
az.plot_ppc(dataset)
plt.show()