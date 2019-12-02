import pandas as pd
import numpy as np
import theano.tensor as tt
import pymc3 as pm
import seaborn as sns
import matplotlib.pyplot as plt
from Clean import Clean


df = pd.read_csv('./data/FM_2000-2019.csv')
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


def run_basic_model():

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

    return basic_model


def run_prior_model(prior):

    prior_model = pm.Model()

    for i in range(len(cols)):
        print(i,cols[i])

    with prior_model:

        # Priors for unknown model parameters
        alpha = pm.Normal('alpha', mu=0, sigma=100)

        if prior == 'normal':
            beta_def = pm.Normal('beta_def', mu=-1, sigma=1, shape=8)
            beta_off = pm.Normal('beta_off', mu=1, sigma=1, shape=8)
            beta_pace =pm.Normal('beta_pace', mu=1, sigma=1, shape=8)

        if prior == 'uniform':
            beta_def = pm.Uniform('beta_def', upper = 0, lower = -2, shape=8)
            beta_off = pm.Uniform('beta_off', upper = 2, lower = 0, shape=8)
            beta_pace =pm.Uniform('beta_pace', upper = 2, lower = 0, shape=8)

        sigma = pm.HalfNormal('sigma', sigma=1)

        # Expected value of outcome
        mu = alpha
        for i in ['off','def','pace']:

            if i == 'off':

                col_list = [j*3 for j in range(8)]
                x = features.iloc[:,col_list].values
                mu += pm.math.dot(x,beta_off)

            elif i == 'def':

                col_list = [j*3 + 1 for j in range(8)]
                x = features.iloc[:,col_list].values
                mu += pm.math.dot(x,beta_def)

            elif i == 'pace':

                col_list = [j*3 + 2 for j in range(8)]
                x = features.iloc[:,col_list].values
                mu += pm.math.dot(x,beta_pace)

        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y)

    return prior_model

model = run_prior_model('normal')
map_est = pm.find_MAP(model=model)

'''
a = map_est['alpha']
b = map_est['beta']

clean_test = Clean(df_test,games)
features_test = clean_test.get_features(['e-def-rating','e-off-rating','e-pace'],q)
for i,r in features_test.iterrows():
    val = a + np.dot(r.values,b)
    print(val)
'''
