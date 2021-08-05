import os
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
from pandas.plotting import parallel_coordinates
from scipy import stats
from theano import shared, tensor

data_df = pd.read_csv('BattingAverage.csv')
data_df.head(5)

# pymc3 can't handle this dataset, so reduce number of samples
data_df = data_df.head(5)

# convert categorical positions to numerical values
positions, pos_encoding = np.unique(data_df['PriPos'], return_inverse=True)
data_df['PriPos'] = pos_encoding

with pm.Model() as model:
    # define priors on the top level parameters
    omegaO = pm.Beta('omegaO', 1, 1, transform=None)
    kappaO = pm.Gamma('kappaO', 0.01, 0.01, transform=None)

    # define categories' variables
    alphaO = omegaO * kappaO + 1
    betaO = (1 - omegaO) * kappaO + 1

    n_cat = len(data_df['PriPos'].unique())
    omega = pm.Beta('omega', alphaO, betaO, shape=n_cat, transform=None)
    kappa = pm.Gamma('kappa', 0.01, 0.01, shape=n_cat, transform=None)

    # define subjects' variables
    for i, row in data_df[['PriPos', 'Hits', 'AtBats']].iterrows():
        s = row['PriPos']
        N_observ = row['AtBats']
        z_observ = row['Hits']

        alpha = omega[s] * kappa[s] + 1
        beta = (1 - omega[s]) * kappa[s] + 1

        theta_name = 'theta_%d' % (i + 1)
        theta = pm.Beta(theta_name, alpha, beta, transform=None)

        z_name = 'z_%d' % (i + 1)
        z = pm.Binomial(z_name, n=N_observ, p=theta, observed=z_observ)

    # use Metropolis sampling as it's the fastest among the other 2
    step = pm.Metropolis(tune_interval=1000)
    # step = pm.Slice()
    # step = None

    trace = pm.sample(500, step=step, cores=1)  # burn in
    # trace = pm.sample(50000, step=step, start=trace[-1], core=1)

import pystan as ps
import seaborn as sns
from dbda2e_utils import plotPost
# reload data
data_df = pd.read_csv('BattingAverage.csv')

# convert categorical positions to numerical values
positions, pos_encoding = np.unique(data_df['PriPos'], return_inverse=True)
data_df['PriPos'] = pos_encoding

model_code = """
data {
    int<lower=0> n_cat;    
    int<lower=0> n_subj;
    int<lower=0> categories[n_subj];
    int<lower=0> observations[n_subj];
    int<lower=0> successes[n_subj];
}
parameters {
    real<lower=0,upper=1> omegaO;
    real<lower=0.01> kappaO;

    real<lower=0,upper=1> omegas[n_cat];
    real<lower=0.01> kappas[n_cat];

    real<lower=0,upper=1> thetas[n_subj];
}
transformed parameters {
    real alphaO;
    real betaO;

    real alphas[n_cat];
    real betas[n_cat];    

    alphaO = omegaO*kappaO + 1;
    betaO = (1-omegaO)*kappaO + 1;

    for (c in 1:n_cat) {
        alphas[c] = omegas[c]*kappas[c] + 1;
        betas[c] = (1-omegas[c])*kappas[c] + 1;
    }
}
model {
    int cat;

    omegaO ~ beta(1, 1);
    kappaO ~ gamma(0.01 , 0.01);

    for (c in 1:n_cat) {
        omegas[c] ~ beta(alphaO, betaO);
        kappas[c] ~ gamma(0.01 , 0.01);
    }

    for (s in 1:n_subj) {
        cat = categories[s] + 1; // indexing starts with 1
        thetas[s] ~ beta(alphas[cat], betas[cat]);
        successes[s] ~ binomial(observations[s], thetas[s]);
    }
}
"""

data = {
    'n_cat': len(data_df['PriPos'].unique()),
    'n_subj': data_df.shape[0],
    'categories': data_df['PriPos'].values,
    'observations': data_df['AtBats'].values,
    'successes': data_df['Hits'].values
}

fit = ps.stan(model_code=model_code, data=data, iter=11000, warmup=1000, chains=4)
samples = fit.extract(permuted=False, inc_warmup=False)
# samples.shape

# Check MCMC conversion
f, ax = plt.subplots(1,2,figsize=(10,5))
for chain_id in range(4):
    smpl = samples[:, chain_id, fit.flatnames.index('omegaO')]
    label = 'chain %d' % chain_id
    sns.kdeplot(smpl, label=label, ax=ax[0])
ax[0].set_title('omegaO chains')

for chain_id in range(4):
    smpl = samples[:, chain_id, fit.flatnames.index('kappaO')]
    label = 'chain %d' % chain_id
    sns.kdeplot(smpl, label=label, ax=ax[1])
ax[1].set_title('kappaO chains')
plt.legend()
plt.show()

param_samples = samples[:, :, :-1] # exclude log prob (the last on the params dimension)
n_samples, n_chains, n_params = param_samples.shape
param_samples = param_samples.reshape((n_samples*n_chains, n_params)) # join all chains

samples_df = pd.DataFrame(param_samples.reshape((n_samples*n_chains, n_params)), columns=fit.flatnames)


def plot_mcmc_posterior(data, titles, xlabels):
    points1, points2 = data
    title1, title2 = titles
    xlabel1, xlabel2 = xlabels

    f, axs = plt.subplots(2, 2, figsize=(10, 10))

    # marginal posterior for param1
    plotPost(points1, axs[0, 0], title=title1, xlabel=xlabel1)

    # marginal posterios for param1 - param2
    ax = axs[0, 1]
    title = '%s - %s' % (title1, title2)
    xlabel = '%s - %s' % (xlabel1, xlabel2)
    plotPost(points1 - points2, ax, title=title, xlabel=xlabel1)
    ax.vlines(0, ax.get_ylim()[0], ax.get_ylim()[1], color='r', linestyle='--')

    # marginal posterior for 2
    plotPost(points2, axs[1, 1], title=title2, xlabel=xlabel2)

    ax = axs[1, 0]
    ax.scatter(points1, points2)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot([0, 1], [0, 1], color='red', linestyle='--')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(title1)
    ax.set_ylabel(title2)
    return


def analyze_pairwise(param1, param2, param_type):
    if param_type == 'positions':
        param_name1 = '%s[%d]' % ('omegas', positions.tolist().index(param1))
        param_name2 = '%s[%d]' % ('omegas', positions.tolist().index(param2))
    else:
        # players
        param_name1 = '%s[%d]' % ('thetas', data_df['Player'].tolist().index(param1))
        param_name2 = '%s[%d]' % ('thetas', data_df['Player'].tolist().index(param2))

    plot_mcmc_posterior(data=[samples_df[param_name1], samples_df[param_name2]],
                        titles=[param1, param2],
                        xlabels=[param_name1, param_name2])
analyze_pairwise('Pitcher', 'Catcher', 'positions')