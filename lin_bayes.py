import pandas as pd
import numpy as np
import theano.tensor as tt
import pymc3 as pm
import seaborn as sns
import matplotlib.pyplot as plt
from Clean import Clean


df = pd.read_csv('FM_2000-2019.csv')
print(df.shape)
df_all = df[df['gp_all_0_a'] >= 30]
df = df_all[0:-100]
df_test = df_all[-100:]
print(df.shape)

games = 30
q = 1

clean = Clean(df,games)
features = clean.get_features(['e-def-rating','e-off-rating','e-pace'],q)
y = clean.get_target(q).values

cols = features.columns
x = features.values
x1 = x
print(type(x))
print(x.shape)

print(type(y))


basic_model = pm.Model()

with basic_model:

    # Priors for unknown model parameters
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=1, shape=x.shape[1])
    sigma = pm.HalfNormal('sigma', sigma=1)

    # Expected value of outcome
    mu = alpha + pm.math.dot(x,beta)

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y)

    # instantiate sampler
    # draw 500 posterior samples

    trace = pm.sample(cores=1,target_accept = 0.95)

    #pm.summary(trace).round(2)
    #pm.plot_posterior(trace)


'''
map_est = pm.find_MAP(model=basic_model)
a = map_est['alpha']
b = map_est['beta']

clean_test = Clean(df_test,games)
features_test = clean_test.get_features(['e-def-rating','e-off-rating','e-pace'],q)
for i,r in features_test.iterrows():
    val = a + np.dot(r.values,b)
    print(val)
'''
