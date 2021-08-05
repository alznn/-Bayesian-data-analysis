import os
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
from pandas.plotting import parallel_coordinates
from scipy import stats
from theano import shared, tensor

#preprocess
color = '#87ceeb'
df2 = pd.read_csv('BattingAverage.csv', usecols=[0,1,2,3], dtype={'PriPos':'category'})
df2.info()

df2['BatAv'] = df2.Hits.divide(df2.AtBats)
df2.head(10)
df2.groupby('PriPos')['Hits','AtBats'].sum().pipe(lambda x: x.Hits/x.AtBats)

pripos_idx = df2.PriPos.cat.codes.values
pripos_codes = df2.PriPos.cat.categories
n_pripos = pripos_codes.size

# df2 contains one entry per player
n_players = df2.index.size

with pm.Model() as model:
    omega = pm.Beta('omega', 1, 1)
    kappa_minus2 = pm.Gamma('kappa_minus2', 0.01, 0.01)
    kappa = pm.Deterministic('kappa', kappa_minus2 + 2)

    # Parameters for categories (Primary field positions)
    omega_c = pm.Beta('omega_c',
                      omega * (kappa - 2) + 1, (1 - omega) * (kappa - 2) + 1,
                      shape=n_pripos)

    kappa_c_minus2 = pm.Gamma('kappa_c_minus2',
                              0.01, 0.01,
                              shape=n_pripos)
    kappa_c = pm.Deterministic('kappa_c', kappa_c_minus2 + 2)

    # Parameter for individual players
    theta = pm.Beta('theta',
                    omega_c[pripos_idx] * (kappa_c[pripos_idx] - 2) + 1,
                    (1 - omega_c[pripos_idx]) * (kappa_c[pripos_idx] - 2) + 1,
                    shape=n_players)

    y2 = pm.Binomial('y2', n=df2.AtBats.values, p=theta, observed=df2.Hits)
print("model done")
pm.model_to_graphviz(model)

with model:
    trace2 = pm.sample(500, cores=1)
print("trace2:",trace2)

# fig, axes = plt.subplots(3,3, figsize=(14,8))
#
# for i, ax in enumerate(axes.T.flatten()):
#     pm.plot_posterior(trace2['omega_c'][:,i], ax=ax, point_estimate='mode', color=color)
#     ax.set_title(pripos_codes[i], fontdict={'fontsize':16, 'fontweight':'bold'})
#     ax.set_xlabel('omega_c__{}'.format(i), fontdict={'fontsize':14})
#     ax.set_xlim(0.10,0.30)
#
# plt.tight_layout(h_pad=3)