import pandas as pd
import pickle
import argparse
from functions import *
from sklearn.preprocessing import *
from sklearn import linear_model
import pymc3 as pm
import seaborn as sns
import theano
import copy
import torch
#torch.set_default_tensor_type(torch.cuda.FloatTensor)
import gpytorch
import numpy as np
import os

def calc_disc(ppc,true_y,func):

    print("ppc shape", ppc.shape)
    sum = 0
    samp_size = ppc.shape[0]
    for i in range(samp_size):

        if func == 'mean':
            sum += abs(ppc[i,:].mean() - true_y.mean())
        elif func == 'std':
            sum += abs(np.std(ppc[i,:]) - np.std(true_y))

    return sum/samp_size

class GPModelSKI(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(GPModelSKI, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module.initialize(constant=50.)
        self.covar_module = gpytorch.kernels.AdditiveStructureKernel(
            gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.ScaleKernel(kernel)
                , grid_size=1000, num_dims=1)
            , num_dims=train_x.shape[1])

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, 1000))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(1000, 500))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(500, 50))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(50, 2))


class GPModelDKL(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(GPModelDKL, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module.initialize(constant=50.)
        self.covar_module = gpytorch.kernels.AdditiveStructureKernel(
            gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.ScaleKernel(kernel)
                , grid_size=1000, num_dims=1)
            , num_dims=train_x.shape[1])
        self.feature_extractor = LargeFeatureExtractor(train_x.shape[1]).cuda()

    def forward(self, x):
        projected_x = self.feature_extractor(x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class Modelling(object):

    def __init__(self,period,model_type,feature_classes,remove_features,restrict_features,\
    hp_dict,normalize,trace_samp,burn_in,post_samp,chains,cores):

        self.period = period
        self.model_type = model_type
        self.feature_classes = feature_classes
        self.remove_features = remove_features
        self.restrict_features = restrict_features
        self.hp_dict = hp_dict
        self.normalize = normalize
        self.trace_samp = trace_samp
        self.burn_in = burn_in
        self.post_samp = post_samp
        self.chains = chains
        self.cores = cores

        #selct model type
        if model_type == 'Lasso':
            print()
            self.reg = linear_model.Lasso(alpha = hp_dict['alpha'])
        elif model_type == 'Ridge':
            print()
            self.reg = linear_model.Ridge(alpha = hp_dict['alpha'])
        elif model_type == 'MAP':
            self.basic_model = pm.Model()


    #train method
    def train(self,df_train,first_feature, file_name):

        self.first_feature = first_feature
        #get rid of identifiers and team record info
        index = list(df_train.columns).index(first_feature)
        features = df_train.iloc[:,index:]
        features = remove_stats(features,['w','l','gp','min'])

        #get rid of irrelevant quarter features
        features = get_quarter_features(features,self.period)
        for f in self.remove_features:
            features = remove_stats(features,[f])

        for f in self.restrict_features:
            features = restrict_stats(features,[f])


        print("feature classes", self.feature_classes)
        #choose features here
        if self.feature_classes != 'all':
            features = restrict_stats(features,self.feature_classes)

        self.feature_cols = list(features.columns)

        if self.normalize:
            self.features_prenorm = features
            mm_scaler = MinMaxScaler()
            features = mm_scaler.fit_transform(features)
            features = pd.DataFrame(data=features,columns = self.feature_cols)


        if self.period == 0:
            Y = df_train['pts_a'] + df_train['pts_h']
        elif self.period == 5:
            Y = df_train['pts_qtr1_a'] + df_train['pts_qtr2_a'] + df_train['pts_qtr1_h'] + df_train['pts_qtr2_h']
        elif self.period == 6:
            Y = df_train['pts_qtr3_a'] + df_train['pts_qtr4_a'] + df_train['pts_qtr3_h'] + df_train['pts_qtr4_h']
        else:
            Y = df_train[f'pts_qtr{self.period}_a'] + df_train[f'pts_qtr{self.period}_h']


        if self.model_type == 'Lasso' or self.model_type == 'Ridge':

            self.reg.fit(features,Y)


        else:
            x = features.values
            model_split = self.model_type.split('-')
            cat = model_split[0]
            prior = model_split[1]
            self.prior = prior
            self.cat = cat

            if prior == 'basic':

                if cat == 'bayes':

                    print("x shape" , x.shape)
                    self.x_shared = theano.shared(x)
                    self.y_shared = theano.shared(Y.values)
                    print("Y shape", Y.values.shape)

                self.basic_model = pm.Model()

                with self.basic_model:
                    # Priors for unknown model parameters
                    alpha = pm.Normal('alpha', mu=0, sigma=10)
                    beta = pm.Normal('beta', mu=0, sigma=1, shape=x.shape[1])
                    sigma = pm.HalfNormal('sigma', sigma=1)

                    if cat == 'MAP':

                        mu = alpha + pm.math.dot(x,beta)
                        Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=Y)

                    elif cat == 'bayes':
                        mu = alpha + pm.math.dot(self.x_shared,beta)
                        Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=self.y_shared)
                        trace = pm.load_trace('./trace/' + file_name)
                        if trace.nchains > 0:
                            self.trace = trace
                        else:
                            self.trace = pm.sample(self.trace_samp, step=pm.NUTS(), chains=self.chains,
                                                   cores=self.cores, tune=self.burn_in)
                            pm.save_trace(self.trace, './trace/' + file_name)

            elif cat == 'bayes' and prior == 'normal':

                print("x shape" , x.shape)
                self.x_shared = theano.shared(x)
                self.y_shared = theano.shared(Y.values)
                print("Y shape", Y.values.shape)

                self.basic_model = pm.Model()

                with self.basic_model:

                    alpha = pm.Normal('alpha', mu=0, sigma=10)

                    beta_def = pm.Normal('beta_def', mu=-.25, sigma=.25, shape=8)
                    beta_off = pm.Normal('beta_off', mu=.25, sigma=.25, shape=8)
                    beta_pace =pm.Normal('beta_pace', mu=.25, sigma=.25, shape=8)

                    sigma = pm.HalfNormal('sigma', sigma=1)

                    mu = alpha
                    for i in ['off','def','pace']:

                        if i == 'off':

                            off_col_list = [j*3 for j in range(8)]
                            x_off = self.x_shared[:,off_col_list]
                            mu += theano.tensor.dot(x_off,beta_off)

                        elif i == 'def':

                            def_col_list = [j*3 + 1 for j in range(8)]
                            x_def = self.x_shared[:,def_col_list]
                            mu += theano.tensor.dot(x_def,beta_def)

                        elif i == 'pace':

                            pace_col_list = [j*3 + 2 for j in range(8)]
                            x_pace = self.x_shared[:,pace_col_list]
                            mu += theano.tensor.dot(x_pace,beta_pace)

                    # Likelihood (sampling distribution) of observations
                    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=self.y_shared)
                    self.trace = pm.sample(self.trace_samp, step=pm.Slice(), chains=self.chains, cores=self.cores, tune=self.burn_in)


            elif cat == 'MAP':
                cols = features.columns

                print('building model with prior:', prior)
                self.prior_model = pm.Model()

                for i in range(len(cols)):
                    print(i,cols[i])

                with self.prior_model:

                    # Priors for unknown model parameters
                    alpha = pm.Normal('alpha', mu=0, sigma=10)

                    if prior == 'normal':
                        beta_def = pm.Normal('beta_def', mu=-.25, sigma=.25, shape=8)
                        beta_off = pm.Normal('beta_off', mu=.25, sigma=.25, shape=8)
                        beta_pace =pm.Normal('beta_pace', mu=.25, sigma=.25, shape=8)

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

                            self.off_col_list = [j*3 for j in range(8)]
                            x = features.iloc[:,self.off_col_list].values
                            mu += pm.math.dot(x,beta_off)

                        elif i == 'def':

                            self.def_col_list = [j*3 + 1 for j in range(8)]
                            x = features.iloc[:,self.def_col_list].values
                            print("X SHAPE", x.shape)
                            exit()
                            mu += pm.math.dot(x,beta_def)

                        elif i == 'pace':

                            self.pace_col_list = [j*3 + 2 for j in range(8)]
                            x = features.iloc[:,self.pace_col_list].values
                            mu += pm.math.dot(x,beta_pace)

                    # Likelihood (sampling distribution) of observations
                    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=Y)

            elif cat == 'GP':

                train_x = torch.from_numpy(x).double().cuda()
                train_y = torch.from_numpy(Y.values).double().cuda()
                # train_x = train_x.cuda()
                # train_y = train_y.cuda()
                self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

                if prior == 'RBF':
                    kernel = gpytorch.kernels.RBFKernel()
                elif prior == 'Exponential':
                    kernel = gpytorch.kernels.MaternKernel(nu=0.5)

                # self.gp_model = GPModelSKI(train_x,train_y,self.likelihood,kernel).double().cuda()
                self.gp_model = GPModelDKL(train_x, train_y, self.likelihood, kernel).double().cuda()

                # Find optimal model hyperparameters
                # self.gp_model.double()
                self.gp_model.train()
                self.likelihood.train()

                # optimizer = torch.optim.Adam([{'params': self.gp_model.parameters()}], lr=0.1)
                optimizer = torch.optim.Adam([
                    {'params': self.gp_model.mean_module.parameters()},
                    {'params': self.gp_model.covar_module.parameters()},
                    {'params': self.gp_model.likelihood.parameters()},
                    {'params': self.gp_model.feature_extractor.parameters(), 'weight_decay': 1e-3}
                ], lr=0.01)

                mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)


                def gp_train(training_iter):
                    print('training gp model ' + file_name)
                    for i in range(training_iter):
                        optimizer.zero_grad()
                        output = self.gp_model(train_x)
                        loss = -mll(output, train_y)
                        loss.backward()
                        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()), end='\r')
                        optimizer.step()
                    print('Iter %d/%d - Loss: %.3f' % (training_iter, training_iter, loss.item()))

                with gpytorch.settings.use_toeplitz(False):
                    if os.path.isfile('./gp-models/'+file_name):
                        print('loading gp model ' + file_name)
                        self.gp_model.load_state_dict(torch.load('./gp-models/'+file_name))
                    else:
                        gp_train(500)
                        torch.save(self.gp_model.state_dict(), './gp-models/'+file_name)

    #predict method
    def predict(self,df_test):

        index = list(df_test.columns).index(self.first_feature)
        features = df_test.iloc[:,index:]
        features = features[self.feature_cols]
        rows = features.shape[0]

        if self.normalize:
            data = []
            mm_scaler = MinMaxScaler()
            count = 0
            for i,row in features.iterrows():
                count += 1
                if count % 100 == 0:
                    print(f'{count}/{rows}')
                X_train_temp = copy.deepcopy(self.features_prenorm)
                both = X_train_temp.append(row)
                both = mm_scaler.fit_transform(both)
                data.append(both[-1])
                features = pd.DataFrame(data = data)

        if self.model_type == 'Lasso' or self.model_type == 'Ridge':
            pred = self.reg.predict(features)

        elif self.cat == 'bayes':

            pred = []
            c = 0
            with self.basic_model:
                for i,r in features.iterrows():
                    c += 1
                    print(f'generating post pred for game {c}/{features.shape[0]}')
                    new_x = r.values
                    self.x_shared.set_value([new_x])
                    self.y_shared.set_value([0])
                    post_pred = pm.sample_posterior_predictive(self.trace, samples=self.post_samp)
                    vals = post_pred['Y_obs']
                    print(f'predictive mean: {vals.mean()} , variance: {vals.var()}')
                    pred.append(vals)

        elif self.cat == 'MAP' and self.prior=='basic':

            map_est = pm.find_MAP(model=self.basic_model)
            a = map_est['alpha']
            b = map_est['beta']
            pred = []
            for i,r in features.iterrows():
                val = a + np.dot(r.values,b)
                pred.append(val)

            pred = np.array(pred)

        elif self.cat == 'MAP':
            map_est = pm.find_MAP(model=self.prior_model)
            alpha = map_est['alpha']
            b_def = map_est['beta_def']
            b_off = map_est['beta_off']
            b_pace = map_est['beta_pace']
            pred = []
            for i,r in features.iterrows():
                val = copy.copy(alpha)
                val += np.dot(r[self.off_col_list].values,b_off)
                val += np.dot(r[self.def_col_list].values,b_def)
                val += np.dot(r[self.pace_col_list].values,b_pace)
                pred.append(val)

            pred = np.array(pred)

        elif self.cat == 'GP':

            test_x = torch.from_numpy(features.values).double()
            test_x = test_x.cuda()

            self.gp_model.eval()
            self.likelihood.eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                ppc = self.likelihood(self.gp_model(test_x))
                pred = ppc.sample(sample_shape=torch.Size([self.post_samp,])).cpu().numpy()

        return pred


    def Pop_PC(self,df_test):

        index = list(df_test.columns).index(self.first_feature)
        features = df_test.iloc[:,index:]
        features = features[self.feature_cols]
        new_x = features.values

        Y = df_test[f'pts_qtr{self.period}_a'] + df_test[f'pts_qtr{self.period}_h']

        with self.basic_model:
            self.x_shared.set_value(new_x)
            y_dummy = np.zeros(new_x.shape[0],dtype=int)
            self.y_shared.set_value(y_dummy)
            post_pred = pm.sample_posterior_predictive(self.trace, samples=self.post_samp)

        vals = post_pred['Y_obs']
        print(vals.shape)
        mean = calc_disc(vals,Y.values,'mean')
        std = calc_disc(vals,Y.values,'std')
        print(mean,std)
        return vals,mean,std

def test():

    with open(f'./data/fm_2000-2019.pkl', 'rb') as handle:
        df = pickle.load(handle)

    df = df[df['gp_all_0_a'] >= 30][df['gp_all_0_h'] >= 30]
    df = df[df['gp_all_0_a'] <= 82][df['gp_all_0_h'] <= 82]
    df.dropna()

    #set the model specificiations here
    period = 1
    model_type = 'GP-RBF'
    hp_dict = {'alpha':.08}
    feature_classes = ['e-off-rating','e-def-rating','e-pace']
    trace_samp = 5000
    burn_in = 200
    post_samp = 500
    chains = 4
    cores = 1

    my_model = Modelling(period=period,model_type=model_type,feature_classes=feature_classes,remove_features=[]\
    ,restrict_features=[],hp_dict=hp_dict,normalize=False,trace_samp=trace_samp,burn_in=burn_in,post_samp=post_samp,
    chains=chains,cores=cores)

    print(df.shape)
    my_model.train(df,'gp_all_0_a')
    pred = my_model.predict(df[-10:])
    print(pred.mean())


def main():
    test()

if __name__ == '__main__':
    main()
