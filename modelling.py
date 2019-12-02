import pandas as pd
import pickle
import argparse
from functions import *
from sklearn.preprocessing import *
from sklearn import linear_model
import pymc3 as pm
import copy


class Modelling(object):

    def __init__(self,period,model_type,feature_classes,remove_features,restrict_features,hp_dict,normalize):

        self.period = period
        self.model_type = model_type
        self.feature_classes = feature_classes
        self.remove_features = remove_features
        self.restrict_features = restrict_features
        self.hp_dict = hp_dict
        self.normalize = normalize


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
    def train(self,df_train,first_feature):

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

        elif self.model_type == 'MAP':

            x = features.values
            
            self.basic_model = pm.Model()

            with self.basic_model:
                # Priors for unknown model parameters
                alpha = pm.Normal('alpha', mu=0, sigma=10)
                beta = pm.Normal('beta', mu=0, sigma=1, shape=x.shape[1])
                sigma = pm.HalfNormal('sigma', sigma=1)

                # Expected value of outcome
                mu = alpha + pm.math.dot(x,beta)

                # Likelihood (sampling distribution) of observations
                Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=Y)


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
        else:
            map_est = pm.find_MAP(model=self.basic_model)
            a = map_est['alpha']
            b = map_est['beta']
            pred = []
            for i,r in features.iterrows():
                val = a + np.dot(r.values,b)
                pred.append(val)

            pred = np.array(pred)

        return pred


def test():

    with open(f'./data/fm_2000-2019.pkl', 'rb') as handle:
        df = pickle.load(handle)

    df = df[df['gp_all_0_a'] >= 30][df['gp_all_0_h'] >= 30]
    df = df[df['gp_all_0_a'] <= 82][df['gp_all_0_h'] <= 82]
    df.dropna()

    #set the model specificiations here
    period = 1
    model_type = 'MAP'
    hp_dict = {'alpha':.08}
    feature_classes = ['e-off-rating','e-def-rating','e-pace']

    my_model = Modelling(period=period,model_type=model_type,feature_classes=feature_classes,remove_features=[]\
    ,restrict_features=[],hp_dict=hp_dict,normalize=False)

    my_model.train(df,'gp_all_0_a')
    pred = my_model.predict(df[-100:])
    print(pred)

def main():
    test()

if __name__ == '__main__':
    main()
