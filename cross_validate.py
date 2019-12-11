import pandas as pd
import numpy as np
import math
import pickle
from modelling import Modelling
from functions import *
from scipy.stats import norm
import pymc3 as pm

def cross_validate(df,my_model,start,end,thresh,vegas_years,first_feature,normalize):

    period = my_model.period

    print(df.shape)
    df = df[df['gp_all_0_a'] >= start]
    print(df.shape)
    df = df[df['gp_all_0_h'] >= start]
    print(df.shape)
    df = df[df['gp_all_0_a'] <= end]
    print(df.shape)
    df = df[df['gp_all_0_h'] <= end]
    print(df.shape)
    df = df.dropna()
    print(df.shape)


    df_vegas = pd.read_csv('./data/vegas_totals.csv')
    df_vegas = df_vegas[df_vegas['court'] == 'Away']
    df_vegas = df_vegas[df_vegas['period'] == period]
    vegas_dict = {}
    for i,r in df_vegas.iterrows():
        vegas_dict[str(r['game_id'])] = r['totals']


    season_list = sorted(list(set(list(df['season']))))
    if vegas_years == 'all':
        vegas_years = season_list

    over_wins_total = 0
    over_losses_total = 0
    under_wins_total = 0
    under_losses_total = 0
    num_seasons = 0
    error_total = 0
    ve_total = 0

    for season in season_list:

        if season not in vegas_years:
            continue

        print(f'running season {season}')
        num_seasons += 1

        df_train = df[df['season'] != season]
        df_test = df[df['season'] == season][0:10]

        my_model.train(df_train,first_feature)
        pred = my_model.predict(df_test)

        count = 0
        over_wins = 0
        over_losses = 0
        under_wins = 0
        under_losses = 0
        error = 0
        vegas_error = 0
        v_count = 0
        e_count = 0

        for i,r in df_test.iterrows():


            '''
            #print(df_test.columns)
            if r['gp_all_0_a'] > 30 or r['gp_all_0_h'] > 30:
                continue
            '''

            if str(r['game_id']) not in vegas_dict:
                vegas = 0
                #print('skipped')
            else:
                vegas = vegas_dict[str(r['game_id'])]
                #print(r['date'] , r['team_name_a'], r['team_name_h'],vegas)

            result_dict = {}
            #get actual total value
            if period == 0:
                result_dict['total'] = r['pts_a'] + r['pts_h']
                result_dict['spread'] = r['pts_h'] - r['pts_a']
            elif period == 5:
                result_dict['total'] = r['pts_qtr1_a'] + r['pts_qtr2_a'] + r['pts_qtr1_h'] + r['pts_qtr2_h']
                result_dict['spread'] = r['pts_qtr1_h'] + r['pts_qtr2_h'] - (r['pts_qtr1_a'] + r['pts_qtr2_a'])
            elif period == 6:
                result_dict['total'] = r['pts_qtr3_a'] + r['pts_qtr4_a'] + r['pts_qtr3_h'] + r['pts_qtr4_h']
                result_dict['spread'] = r['pts_qtr3_h'] + r['pts_qtr4_h'] - (r['pts_qtr3_a'] + r['pts_qtr4_a'])
            else:
                result_dict['total'] = r[f'pts_qtr{period}_a'] + r[f'pts_qtr{period}_h']
                result_dict['spread'] = r[f'pts_qtr{period}_h'] - r[f'pts_qtr{period}_a']

            #calc error
            result  = result_dict['total']

            if my_model.model_type.split('-')[0] == 'bayes':

                samp = pred[count]
                over = np.sum(samp >= vegas)
                under = len(samp) - over
                over_perc = over/len(samp)
                under_perc = under/len(samp)

                print(f'vegas: {vegas}, predictive mean {samp.mean()}, under percentage: {under_perc}, over percentage: {over_perc}')
                error += abs(pred[count].mean() - result)

                if vegas != 0 and season in vegas_years:
                    v_count += 1
                    vegas_error += abs(vegas - result)
                    #calc vegas winnings
                    if over_perc > thresh and result > vegas:
                        over_wins += 1
                        #print("home win")
                    elif over_perc > thresh and result < vegas:
                        over_losses += 1
                        #print("home loss")
                    elif under_perc > thresh and result < vegas:
                        #print("away win")
                        under_wins += 1
                    elif under_perc > thresh and result > vegas:
                        #print("away loss")
                        under_losses += 1


            else:

                error += abs(pred[count] - result)

                if vegas != 0 and season in vegas_years:
                    v_count += 1
                    vegas_error += abs(vegas - result)
                    #calc vegas winnings
                    if pred[count] > vegas + thresh and result > vegas:
                        over_wins += 1
                        #print("home win")
                    elif pred[count] > vegas + thresh and result < vegas:
                        over_losses += 1
                        #print("home loss")
                    elif pred[count] < vegas - thresh and result < vegas:
                        #print("away win")
                        under_wins += 1
                    elif pred[count] < vegas - thresh and result > vegas:
                        #print("away loss")
                        under_losses += 1


            count += 1

        print(r['season'])
        if over_wins + over_losses > 0:
            print(f'Over: -- wins: {over_wins} , losses: {over_losses}, win%: {over_wins/(over_wins+over_losses)}')
            print(f'Profit: {1000*over_wins - 1110*over_losses}')

        print()
        if under_wins + under_losses > 0:
            print(f'Under: -- wins: {under_wins} , losses: {under_losses}, win%: {under_wins/(under_wins+under_losses)}')
            print(f'Profit: {1000*under_wins - 1100*under_losses}')

        print()

        over_wins_total += over_wins
        over_losses_total += over_losses
        under_wins_total += under_wins
        under_losses_total += under_losses
        error_total += error/count
        ve_total += vegas_error/v_count


    print('------------------------------------------------------------------------------------------')
    print()


    ave_error = error_total/num_seasons
    ave_profit_under = (1000*under_wins_total - 1100*under_losses_total)/len(vegas_years)
    ave_profit_over = (1000*over_wins_total - 1100*over_losses_total)/len(vegas_years)
    winper_over = over_wins_total/(over_wins_total+over_losses_total)
    winper_under = under_wins_total/(under_wins_total+under_losses_total)
    print(f'Number of Seasons: {num_seasons}')
    print(f'Period: {period}')
    print(f'my_model: Type - {my_model.model_type}')
    print(f'HyperParams: {my_model.hp_dict}')
    print(f'Thresh: {thresh}')
    print(f'Normalize: {normalize}')
    print(f'Feature Classes: {my_model.feature_classes}')
    print(f'Remove Features: {my_model.remove_features}')
    print(f'Restrict Features: {my_model.restrict_features}')
    print(f'Over Total: -- wins: {over_wins_total} , losses: {over_losses_total}, win%: {winper_over}')
    print(f'Average Profit: {ave_profit_over}')
    print()
    print(f'Under Total: -- wins: {under_wins_total} , losses: {under_losses_total}, win%: {winper_under}')
    print(f'Average Profit: {ave_profit_under}')
    print(f'Average Absolute Prediciton Error: {ave_error}')

    print("WINS")
    print(under_wins_total + over_wins_total)
    print("LOSSES")
    print(under_losses_total + over_losses_total)
    print("N")
    print(under_wins_total + over_wins_total + under_losses_total + over_losses_total)
    print("WIN%")
    print((under_wins_total + over_wins_total)/(under_wins_total + over_wins_total + under_losses_total + over_losses_total))
    print("PROFIT")
    print((ave_profit_under + ave_profit_over)/1000)
    print("AVERAGE VEGAS ERROR:" , ve_total/num_seasons)

    #return (ave_error,ave_profit_over,winper_over,ave_profit_under,winper_under)


def main():

    with open('./data/fm_2000-2019.pkl','rb') as handle:
        df = pickle.load(handle)

    start_game = 30
    end_game = 82
    vegas_years = ['2013-14','2014-15','2015-16','2016-17','2017-18','2018-19']
    #vegas_years = ['2018-19']
    first_feature = 'gp_all_0_a'
    model_type = 'bayes-basic'
    hp_dict = {'alpha':.05}
    feature_classes = ['e-off-rating','e-def-rating','e-pace']
    thresh = .57
    period = 1
    trace_samp = 2000
    burn_in = 1000
    post_samp = 500
    chains = 4
    cores = 1

    my_model = Modelling(period=period,model_type=model_type,feature_classes=feature_classes,remove_features=[]\
    ,restrict_features=[],hp_dict=hp_dict,normalize=False,trace_samp=trace_samp,burn_in=burn_in,post_samp=post_samp,
    chains=chains,cores=cores)

    cross_validate(df,my_model,start_game,end_game,thresh,vegas_years,first_feature,normalize=False)


if __name__ == '__main__':
    main()
