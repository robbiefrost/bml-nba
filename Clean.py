import pandas as pd

class Clean():

    def __init__(self,df,games):

        df = df[df['gp_all_0_a'] >= 30]
        df = df[df['gp_all_0_h'] >= 30]
        self.df = df

    def get_features(self,stat_list,q):

        def restrict_stats(df,stat_list):
            cols = df.columns
            index_list = []
            for c in range(len(cols)):
                for stat in stat_list:
                    if stat in cols[c].split('_'):
                        index_list.append(c)
            df = df.iloc[:, [j for j, c in enumerate(df.columns) if j in index_list]]
            return df

        def remove_stats(df,stat_list):
            cols = df.columns
            index_list = []
            for c in range(len(cols)):
                for stat in stat_list:
                    if stat in cols[c].split('_'):
                        index_list.append(c)
            df = df.iloc[:, [j for j, c in enumerate(df.columns) if j not in index_list]]
            return df

        def get_quarter_features(df,q):

            l = list(range(4))
            del l[q-1]
            fq = remove_stats(df,[f'{k+1}' for k in l])
            return fq

        df_quarter = get_quarter_features(self.df,q)
        features = restrict_stats(df_quarter,stat_list)

        return features.reset_index(drop=True)

    def get_target(self,q):

        y = self.df[f'pts_qtr{q}_a'] + self.df[f'pts_qtr{q}_h']
        return y.reset_index(drop=True)


if __name__ == '__main__':
    games = 30
    q = 1
    df = pd.read_csv('FM_2000-2019.csv')
    clean = Clean(df,30)
    x = clean.get_features(['e-net-rating'],q)
    y = clean.get_target(q)
