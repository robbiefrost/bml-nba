import pandas as pd
import numpy as np
import theano.tensor as tt
import pymc3 as pm
import seaborn as sns
import matplotlib.pyplot as plt
from Clean import Clean
import time
import copy



def prep_data():

    print("prepping data...")

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
    x = features.values

    return features,x,y,df_test


def build_basic_model(features,x,y):

    print("building basic model")
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


def build_prior_model(prior,features,x,y):

    cols = features.columns

    print('building model with prior:', prior)
    prior_model = pm.Model()

    for i in range(len(cols)):
        print(i,cols[i])

    with prior_model:

        # Priors for unknown model parameters
        alpha = pm.Normal('alpha', mu=0, sigma=10)

        if prior == 'normal':
            beta_def = pm.Normal('beta_def', mu=-.25, sigma=.25, shape=8)
            beta_off = pm.Normal('beta_off', mu=.25, sigma=.25, shape=8)
            beta_pace = pm.Normal('beta_pace', mu=.25, sigma=.25, shape=8)

        if prior == 'uniform':
            beta_def = pm.Uniform('beta_def', upper = 0, lower = -.5, shape=8)
            beta_off = pm.Uniform('beta_off', upper = .5, lower = 0, shape=8)
            beta_pace =pm.Uniform('beta_pace', upper = .5, lower = 0, shape=8)

        if prior == 'truncnormal':
            beta_def = pm.TruncatedNormal('beta_def', mu = -.25 , sigma=.25, upper = 0 , shape=8)
            beta_off = pm.TruncatedNormal('beta_off', mu = .25 , sigma=.25, lower = 0 , shape=8)
            beta_pace = pm.TruncatedNormal('beta_pace', mu = .25 , sigma=.25, lower = 0 , shape=8)



        sigma = pm.HalfNormal('sigma', sigma=1)

        # Expected value of outcome
        mu = alpha
        for i in ['off','def','pace']:

            if i == 'off':

                off_col_list = [j*3 for j in range(8)]
                x = features.iloc[:,off_col_list].values
                mu += pm.math.dot(x,beta_off)

            elif i == 'def':

                def_col_list = [j*3 + 1 for j in range(8)]
                x = features.iloc[:,def_col_list].values
                mu += pm.math.dot(x,beta_def)

            elif i == 'pace':

                pace_col_list = [j*3 + 2 for j in range(8)]
                x = features.iloc[:,pace_col_list].values
                mu += pm.math.dot(x,beta_pace)

        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y)

    return prior_model,off_col_list,def_col_list,pace_col_list

def fit_model(model):
    start = time.time()
    with model:
        step = pm.NUTS()
        trace = pm.sample(2000, step=step, chains=1, cores=1, tune=1000)
        print(time.time() - start)
        pm.plot_posterior(trace)
        pm.traceplot(trace)

features,x,y,df_test = prep_data()
#model,off_col_list,def_col_list,pace_col_list = build_prior_model('normal',features,x,y)
model = build_basic_model(features,x,y)
fit_model(model)
'''
map_est = pm.find_MAP(model=model)

a = map_est['alpha']
b = map_est['beta']

clean_test = Clean(df_test,games)
features_test = clean_test.get_features(['e-def-rating','e-off-rating','e-pace'],q)
for i,r in features_test.iterrows():
    val = a + np.dot(r.values,b)
    print(val)


map_est = pm.find_MAP(model=model)

alpha = map_est['alpha']
b_def = map_est['beta_def']
b_off = map_est['beta_off']
b_pace = map_est['beta_pace']


clean_test = Clean(df_test,30)
features_test = clean_test.get_features(['e-def-rating','e-off-rating','e-pace'],1)
for i,r in features_test.iterrows():
    val = copy.copy(alpha)
    val += np.dot(r[off_col_list].values,b_off)
    val += np.dot(r[def_col_list].values,b_def)
    val += np.dot(r[pace_col_list].values,b_pace)
    print(val)
'''
