#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reated on Fri Oct 27 06:09:51 2017

@author: dfox
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
import math
import copy
import os
import datetime as dt

location = '.'

team_list = [
      'ARI',
      'ATL',
      'BAL',
      'BUF',
      'CAR',
      'DAL',
      'CHI',
      'CIN',
      'CLE',
      'DEN',
      'DET',
      'GBP',
      'HOU',
      'IND',
      'JAC',
      'KCC',
      'LAR',
      'MIA',
      'MIN',
      'NEP',
      'NOS',
      'NYG',
      'NYJ',
      'OAK',
      'PHI',
      'PIT',
      'SDC',
      'SEA',
      'SFF',
      'TBB',
      'TEN',
      'WAS'
    ]

tdhr = {
      'ARI': -0.192401,
      'ATL': 0.035722,
      'BAL': 0.265690,
      'BUF': -0.313357,
      'CAR': 0.000447,
      'DAL': 0.108233,
      'CHI': -0.072653,
      'CIN': -0.080662,
      'CLE': -1.069229,
      'DEN': 0.581034,
      'DET': -0.638797,
      'GBP': 0.676169,
      'HOU': -0.684752,
      'IND': 0.642366,
      'JAC': -0.524207,
      'KCC': -0.016966,
      'LAR': -0.509101,
      'MIA': -0.147602,
      'MIN': 0.036765,
      'NEP': 1.088545,
      'NOS': 0.115790,
      'NYG': 0.055463,
      'NYJ': -0.065944,
      'OAK': -0.648259,
      'PHI': 0.278435,
      'PIT': 0.625936,
      'SDC': 0.072101,
      'SEA': 0.280138,
      'SFF': 0.062144,
      'TBB': -0.387991,
      'TEN': -0.174533,
      'WAS': -0.409294
      }

class NFLP:
    nfl_games_fn = os.path.join(location, 'NFLGamesDB.csv')
    team_names_map_fn = os.path.join(location, 'TeamNamesMap.csv')
    neutral_sites_fn = os.path.join(location, 'UpTo2012NeutralSites.xlsx')
    planned_neutral_sites_fn = os.path.join(location, 'PlannedNeutralSites.xlsx')

    def __init__(self):
        starttime = dt.datetime.now()
        print("start")
        
        self.td = {}
        self.rstd = {}
        self.tds = {}
        self.rstds = {}
        
        gdf = pd.read_csv(NFLP.nfl_games_fn)
        tdf = pd.read_csv(NFLP.team_names_map_fn)
#        xl = pd.ExcelFile(NFLP.team_names_map_fn)
#        tdf = xl.parse()
        xl = pd.ExcelFile(NFLP.neutral_sites_fn)
        nudf = xl.parse()
        xl = pd.ExcelFile(NFLP.planned_neutral_sites_fn)
        pnudf = xl.parse()

        count = 0
        count = count + 1
        print("ms{} {}".format(count, dt.datetime.now() - starttime))
        gdf['VPF'] = gdf['Q1VS'] + gdf['Q2VS'] + gdf['Q3VS'] + gdf['Q4VS'] + gdf['OTVS']
        gdf['VRPF'] = gdf['VPF'] - gdf['OTVS']
        gdf['VPA'] = gdf['Q1HS'] + gdf['Q2HS'] + gdf['Q3HS'] + gdf['Q4HS'] + gdf['OTHS']
        gdf['VRPA'] = gdf['VPA'] - gdf['OTHS']
        gdf['VRNP'] = gdf['VRPF'] - gdf['VRPA']
        gdf['VNP'] = gdf['VPF'] - gdf['VPA']
        gdf['OT'] = (gdf['VRPF'] == gdf['VRPA'])
        gdf['VWIN'] = (gdf['VPF'] > gdf['VPA'])
        gdf['TIE'] = (gdf['VPF'] == gdf['VPA'])
        gdf['VQ1NP'] = gdf['Q1VS'] - gdf['Q1HS']
        gdf['VQ2NP'] = gdf['Q2VS'] - gdf['Q2HS']
        gdf['VQ3NP'] = gdf['Q3VS'] - gdf['Q3HS']
        gdf['VQ4NP'] = gdf['Q4VS'] - gdf['Q4HS']
        gdf['VOTNP'] = gdf['OTVS'] - gdf['OTHS']

        """
        gsdf = gdf[gdf['Year'] > 1977].copy()
        for index, row in gsdf.iterrows():
            gsdf.at[index, 'RegOddsDiff'] = abs(gsdf.at[index, 'VRNP'] + gsdf.at[index, 'VSpread'])
            gsdf.at[index, 'OddsDiff'] = abs(gsdf.at[index, 'VNP'] + gsdf.at[index, 'VSpread'])
            gsdf.at[index, 'VRegOddsDiff'] = gsdf.at[index, 'VRNP'] + gsdf.at[index, 'VSpread']
            gsdf.at[index, 'VOddsDiff'] = gsdf.at[index, 'VNP'] + gsdf.at[index, 'VSpread']
            gsdf.at[index, 'FRegOddsDiff'] = gsdf.at[index, 'VRNP'] + gsdf.at[index, 'VSpread']
            gsdf.at[index, 'FOddsDiff'] = gsdf.at[index, 'VNP'] + gsdf.at[index, 'VSpread']
            if (gsdf.at[index, 'VOddsDiff'] > 0):
                gsdf.at[index, 'VBeatSpread'] = 1.0
            elif (gsdf.at[index, 'VOddsDiff'] < 0):
                gsdf.at[index, 'VBeatSpread'] = 0.0
            else:
                gsdf.at[index, 'VBeatSpread'] = 0.5
            if (gsdf.at[index, 'VSpread'] > 0):
                if (gsdf.at[index, 'VOddsDiff'] < 0):
                    gsdf.at[index, 'FBeatSpread'] = 1.0
                elif (gsdf.at[index, 'VOddsDiff'] > 0):
                    gsdf.at[index, 'FBeatSpread'] = 0.0
                else:
                    gsdf.at[index, 'FBeatSpread'] = 0.5
                if (gsdf.at[index, 'VOddsDiff'] > 0):
                    gsdf.at[index, 'FBeatSpread'] = 0.0
                elif (gsdf.at[index, 'VOddsDiff'] < 0):
                    gsdf.at[index, 'FBeatSpread'] = 1.0
                else:
                    gsdf.at[index, 'FBeatSpread'] = 0.5
        """
        
        for index, row in gdf.iterrows():
            v = tdf[tdf['Name'] == row.Visitor]
            v = v[row['Year'] >= v['Start']]
            v = v[row['Year'] <= v['End']]
            v.reset_index(inplace=True, drop=True)
        #    gdf.set_value(index, 'Visitor Symbol', v.get_value(0, 'Symbol'))
        #    gdf.loc[index, 'Visitor Symbol'] = v.loc[0, 'Symbol']
            gdf.at[index, 'Visitor Symbol'] = v.at[0, 'Symbol']
            h = tdf[tdf['Name'] == row.Home].copy()
            h = h[row['Year'] >= h['Start']]
            h = h[row['Year'] <= h['End']]
            h.reset_index(inplace=True, drop=True)
        #    gdf.set_value(index, 'Home Symbol', h.get_value(0, 'Symbol'))
        #    gdf.loc[index, 'Home Symbol'] = h.loc[0, 'Symbol']
            gdf.at[index, 'Home Symbol'] = h.at[0, 'Symbol']
        #    print(h.loc[0, 'Symbol'])
        #    gdf.set_value(index, 'LNVRNP', math.log(abs(gdf.get_value(index, 'VRNP'))+1))
        #    if (gdf.get_value(index, 'VRNP') < 0):
        #        gdf.set_value(index, 'LNVRNP', -gdf.get_value(index, 'LNVRNP'))
        
        count = count + 1
        print("ms{} {}".format(count, dt.datetime.now() - starttime))
        gdf['Visitor'] = gdf['Visitor Symbol']
        gdf['Home'] = gdf['Home Symbol']
        gdf.drop(['Visitor Symbol', 'Home Symbol'], axis=1, inplace=True)
        gdf.sort_values(['Year', 'Week'], axis=0, inplace=True)
        #gdf.sort_values(['Year', 'Week', 'Visitor'], axis=0, inplace=True)
        gdf.reset_index(drop=True, inplace=True)
        gdf['Neutral'] = False
        
        nudf.dropna(inplace=True)
        nudf['Season'] = nudf['Season'].astype(int)
        nudf = nudf[nudf.Season >= 1970]
        nudf = nudf[nudf['Week'] != 'SB']
        nudf = nudf[(nudf['Designated'] != 'Oakland Raiders') | (nudf["State/Int'l"] != 'CA')]
        nudf = nudf[(nudf['Designated'] != 'San Francisco 49ers') | (nudf["State/Int'l"] != 'CA')]
        nudf = nudf[(nudf['Designated'] != 'New York Jets') | (nudf["State/Int'l"] != 'NJ')]
        nudf = nudf[(nudf['Designated'] != 'Chicago Bears') | (nudf["State/Int'l"] != 'IL')]
        nudf = nudf[(nudf['Designated'] != 'Minnesota Vikings') | (nudf["State/Int'l"] != 'MN')]
        nudf = nudf[(nudf['Designated'] != 'Buffalo Bills') | (nudf["City"] != 'Toronto')]
        nudf = nudf[nudf["State/Int'l"] != 'England']
        nudf = nudf[nudf["State/Int'l"] != 'Mexico']
        nudf['Year'] = nudf['Season']
        nudf['Home'] = nudf['Designated']
        nudf['Visitor'] = nudf['Away Team']
        nudf['Week'] = nudf['Week'].map(lambda x: x.lstrip('Wk '))
        nudf['Neutral'] = True
        nudf = nudf[['Year', 'Week', 'Visitor', 'Home', 'Neutral']]
        
        for index, row in nudf.iterrows():
            v = tdf[tdf['Name'] == row.Visitor].copy()
            v = v[row['Year'] >= v['Start']]
            v = v[row['Year'] <= v['End']]
            v.reset_index(inplace=True, drop=True)
        #    nudf.set_value(index, 'Visitor Symbol', v.get_value(0, 'Symbol'))
            nudf.at[index, 'Visitor Symbol'] = v.at[0, 'Symbol']
            h = tdf[tdf['Name'] == row.Home].copy()
            h = h[row['Year'] >= h['Start']]
            h = h[row['Year'] <= h['End']]
            h.reset_index(inplace=True, drop=True)
        #    nudf.set_value(index, 'Home Symbol', h.get_value(0, 'Symbol'))
            nudf.at[index, 'Home Symbol'] = h.at[0, 'Symbol']
        #    gdf.set_value(index, 'LNVRNP', math.log(abs(gdf.get_value(index, 'VRNP'))+1))
        #    if (gdf.get_value(index, 'VRNP') < 0):
        #        gdf.set_value(index, 'LNVRNP', -gdf.get_value(index, 'LNVRNP'))
        
        count = count + 1
        print("ms{} {}".format(count, dt.datetime.now() - starttime))
        nudf['Visitor'] = nudf['Visitor Symbol']
        nudf['Home'] = nudf['Home Symbol']
        nudf.drop(['Visitor Symbol', 'Home Symbol'], axis=1, inplace=True)
        nudf.sort_values(['Year', 'Week', 'Visitor'], axis=0, inplace=True)
        nudf['Week'] = nudf['Week'].astype(int)
        
        pnudf.drop(pnudf.head(1).index, inplace=True)
        prev = 1926
        for index, row in pnudf.iterrows():
        #    print(type(row.Season))
            if (pd.isnull(row.Season)):
                pnudf.at[index, 'Season'] = prev
        #        print("set season =", prev)
            else:
                prev = row.Season
        #        print("prev =", prev)
        pnudf = pnudf[pnudf.Type == 'REG']
        pnudf['Season'] = pnudf['Season'].astype(int)
        pnudf = pnudf[(pnudf.Season >= 1970) & (pnudf.Type == 'REG')]
        pnudf = pnudf[(pnudf.Season != 2018)]
        pnudf = pnudf[(pnudf['Winning/Tied Team'] != 'Buffalo Bills') | (pnudf["City"] != 'Toronto')]
        pnudf = pnudf[(pnudf['Losing/Tied Team'] != 'Buffalo Bills') | (pnudf["City"] != 'Toronto')]
        for index, row in pnudf.iterrows():
            w = tdf[tdf['Name'] == row['Winning/Tied Team']].copy()
            w = w[row['Season'] >= w['Start']]
            w = w[row['Season'] <= w['End']]
            w.reset_index(inplace=True, drop=True)
        #    pnudf.set_value(index, 'Win Symbol', v.get_value(0, 'Symbol'))
            pnudf.at[index, 'Win Symbol'] = w.at[0, 'Symbol']
            l = tdf[tdf['Name'] == row['Losing/Tied Team']].copy()
            l = l[row['Season'] >= l['Start']]
            l = l[row['Season'] <= l['End']]
            l.reset_index(inplace=True, drop=True)
        #    pnudf.set_value(index, 'Lose Symbol', h.get_value(0, 'Symbol'))
            pnudf.at[index, 'Lose Symbol'] = l.at[0, 'Symbol']
            score = pnudf.at[index, 'Score']
            if (score[1] == "-"):
                wscore = score[:1]
                lscore = score[2:4]
            else:
                wscore = score[:2]
                lscore = score[3:5]
        #    pnudf.set_value(index, 'WNP', int(int(wscore) - int(lscore)))
            pnudf.at[index, 'WNP'] = int(int(wscore) - int(lscore))
        
        pnudf['WNP'] = pnudf['WNP'].astype(int)
        
        count = count + 1
        print("ms{} {}".format(count, dt.datetime.now() - starttime))
        for index, row in nudf.iterrows():
        #    print(row.Year, row.Week, row.Visitor)
            gndf = gdf[(gdf.Year == row.Year) & (gdf.Week == row.Week) & (gdf.Visitor == row.Visitor)]
            for gindex, grow in gndf.iterrows():
                gdf.at[gindex, 'Neutral'] = True
#            for gindex, grow in gdf.iterrows():
#                if(grow.Year == row.Year and grow.Week == row.Week and grow.Visitor == row.Visitor):
        #            print("Setting neutral to True")
        #            gdf.set_value(gindex, 'Neutral', True)
#                    gdf.at[gindex, 'Neutral'] = True
        
        for index, row in pnudf.iterrows():
            gpvdf = gdf[(gdf.Year == row.Season) & (gdf.Visitor == row['Win Symbol']) & (gdf.Home == row['Lose Symbol']) & (gdf.VNP == row.WNP)]
            for gindex, grow in gpvdf.iterrows():
                gdf.at[gindex, 'Neutral'] = True
            gphdf = gdf[(gdf.Year == row.Season) & (gdf.Visitor == row['Lose Symbol']) & (gdf.Home == row['Win Symbol']) & (gdf.VNP == -row.WNP)]
            for gindex, grow in gphdf.iterrows():
                gdf.at[gindex, 'Neutral'] = True
            """
            for gindex, grow in gdf.iterrows():
                if(grow.Year == row.Season and
                   (((grow.Visitor == row['Win Symbol']) and
                     (grow.Home == row['Lose Symbol']) and
                     (grow.VNP == row.WNP)) or
                   (((grow.Home == row['Win Symbol']) and
                     (grow.Visitor == row['Lose Symbol']) and
                     (grow.VNP == -row.WNP))))):
        #            print("Setting neutral to True Year", grow.Year, grow.Visitor, grow.Home, grow.VNP)
        #            gdf.set_value(gindex, 'Neutral', True)
                    gdf.at[gindex, 'Neutral'] = True
            """
        
        y = 1970
        while y <= 2017:
            ygdf = gdf[gdf.Year == y]
            week_max = ygdf['Week'].max()
            gmdf = ygdf[ygdf.Week == week_max]
            for index, row in gmdf.iterrows():
                gdf.at[index, 'Neutral'] = True
            """
            for index, row in gdf.iterrows():
                if (row.Year == y and row.Week == week_max):
        #            gdf.set_value(index, 'Neutral', True)
                    gdf.at[index, 'Neutral'] = True
            """
            y = y + 1
            
        count = count + 1
        print("ms{} {}".format(count, dt.datetime.now() - starttime))
        gdf['Playoffs'] = False
        for index, row in gdf.iterrows():
            if (row.Week > 9 and row.Year == 1982):
                gdf.at[index, 'Playoffs'] = True
            if (row.Week > 14 and row.Year < 1978):
                gdf.at[index, 'Playoffs'] = True
            if (row.Week > 15 and row.Year == 1987):
                gdf.at[index, 'Playoffs'] = True
            if (row.Week > 16 and row.Year < 1990):    
                gdf.at[index, 'Playoffs'] = True
            if (row.Week > 17 and row.Year != 1993):
                gdf.at[index, 'Playoffs'] = True
            if (row.Week > 18 and row.Year == 1993):
                gdf.at[index, 'Playoffs'] = True
        
        
        gdf['OT'] = gdf['OT'].astype(int)
        gdf['VWIN'] = gdf['VWIN'].astype(int)
        gdf['TIE'] = gdf['TIE'].astype(int)
        gdf['Neutral'] = gdf['Neutral'].astype(int)
        gdf['Playoffs'] = gdf['Playoffs'].astype(int)
        
        count = count + 1
        print("ms{} {}".format(count, dt.datetime.now() - starttime))
        gsdf = gdf[gdf['Year'] > 1977].copy()
        for index, row in gsdf.iterrows():
            gsdf.at[index, 'RegOddsDiff'] = abs(gsdf.at[index, 'VRNP'] + gsdf.at[index, 'VSpread'])
            gsdf.at[index, 'OddsDiff'] = abs(gsdf.at[index, 'VNP'] + gsdf.at[index, 'VSpread'])
            gsdf.at[index, 'VRegOddsDiff'] = gsdf.at[index, 'VRNP'] + gsdf.at[index, 'VSpread']
            gsdf.at[index, 'VOddsDiff'] = gsdf.at[index, 'VNP'] + gsdf.at[index, 'VSpread']
            gsdf.at[index, 'FRegOddsDiff'] = gsdf.at[index, 'VRNP'] + gsdf.at[index, 'VSpread']
            gsdf.at[index, 'FOddsDiff'] = gsdf.at[index, 'VNP'] + gsdf.at[index, 'VSpread']
            if (gsdf.at[index, 'VOddsDiff'] > 0):
                gsdf.at[index, 'VBeatSpread'] = 1.0
            elif (gsdf.at[index, 'VOddsDiff'] < 0):
                gsdf.at[index, 'VBeatSpread'] = 0.0
            else:
                gsdf.at[index, 'VBeatSpread'] = 0.5
            if (gsdf.at[index, 'VSpread'] > 0):
                if (gsdf.at[index, 'VOddsDiff'] < 0):
                    gsdf.at[index, 'FBeatSpread'] = 1.0
                elif (gsdf.at[index, 'VOddsDiff'] > 0):
                    gsdf.at[index, 'FBeatSpread'] = 0.0
                else:
                    gsdf.at[index, 'FBeatSpread'] = 0.5
                if (gsdf.at[index, 'VOddsDiff'] > 0):
                    gsdf.at[index, 'FBeatSpread'] = 0.0
                elif (gsdf.at[index, 'VOddsDiff'] < 0):
                    gsdf.at[index, 'FBeatSpread'] = 1.0
                else:
                    gsdf.at[index, 'FBeatSpread'] = 0.5
        
#        gdf.to_csv('{}NFLGamesDBDrived.csv'.format(location))
        
        rsdf = gdf[gdf['Playoffs'] == False].copy()
        rssdf = gsdf[gsdf['Playoffs'] == False].copy()
        
        self.gdf = gdf
        
        count = count + 1
        print("ms{} {}".format(count, dt.datetime.now() - starttime))
        for symbol in team_list:
            self.td[symbol] = gdf[(gdf.Visitor == symbol) | (gdf.Home == symbol)].copy()
            self.td[symbol].reset_index(drop=True, inplace=True)
            self.tds[symbol] = gsdf[(gsdf.Visitor == symbol) | (gsdf.Home == symbol)].copy()
            self.tds[symbol].reset_index(drop=True, inplace=True)
            self.rstd[symbol] = rsdf[(rsdf.Visitor == symbol) | (rsdf.Home == symbol)].copy()
            self.rstd[symbol].reset_index(drop=True, inplace=True)
            self.rstds[symbol] = rssdf[(rssdf.Visitor == symbol) | (rssdf.Home == symbol)].copy()
            self.rstds[symbol].reset_index(drop=True, inplace=True)
            
    def get_teams_all(self):
        return self.td
    
    def get_teams_spread(self):
        return self.tds
    
    def get_teams_reg(self):
        return self.rstd
    
    def get_teams_reg_spread(self):
        return self.rstds
    
    def get_dataframe_overall(self):
        return self.gdf
        
def split_Xy(X, col):
    y = X[col]
    X.drop([col], axis=1, inplace=True)
    return y

def logrec_score(X, y, runs, numvar):

    train_score = []
    test_score = []

    i = 0
    c = []
    while (i < numvar):
        ci = []
        c.append(ci)
        i = i + 1

    j = 0
    for j in range(runs):

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        X_train.head()

#mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10])
#mlp.fit(X_train, y_train)

#train_score = mlp.score(X_train, y_train)
#test_score = mlp.score(X_test, y_test)

        logreg = LogisticRegression(C=1)
        logreg.fit(X_train, y_train)
        
        train_score.append(logreg.score(X_train, y_train))
        test_score.append(logreg.score(X_test, y_test))

        i = 0
        while (i < numvar):
            c[i].append(logreg.coef_[0][i])
            i = i + 1

    i = 0
    cols = []
    cavg = []
    while (i < numvar):
        cols.append(i)
        cavg.append(np.mean(c[i]))
        i = i + 1
    
    xdf = pd.DataFrame(cavg, cols)

    return xdf, np.mean(train_score), np.mean(test_score), ((np.mean(train_score)*3 + np.mean(test_score))/4), cavg

def logrec_score2(X, y, numvar):

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    grid_values = {'penalty': ['l1', 'l2'], 'C': [.001, .01, .1, 1, 10, 100]}
    logreg = LogisticRegression(random_state=0)
    
    grid_logreg = GridSearchCV(logreg, param_grid=grid_values, scoring='accuracy')
    grid_logreg.fit(X_train, y_train)
    y_dec_scores_acc = grid_logreg.decision_function(X_test)
    
    print(grid_logreg.best_params_)
    print(grid_logreg.best_score_)
    print(grid_logreg.cv_results_)
    
    logreg = LogisticRegression(penalty=grid_logreg.best_params_['penalty'],
                                C=grid_logreg.best_params_['C'],
                                random_state=0)

    logreg.fit(X_train, y_train)
    i = 0
    cols = []
#    cavg = []
    while (i < numvar):
        cols.append(i)
#        cavg.append(np.mean(c[i]))
        i = i + 1
        
    xdf = pd.DataFrame(logreg.coef_[0], cols)
    trs = logreg.score(X_train, y_train)
    tes = logreg.score(X_test, y_test)
    ovs = (trs*3 + tes) / 4

    return xdf, trs, tes, ovs, logreg.coef_[0]

def ridgereg_score(X, y):

    ridgecv = RidgeCV(alphas=(.001, .01, .1, 1.0, 5.0, 10.0, 20.0, 50.0, 75.0, 100.0, 1000.0), store_cv_values=True)
    ridgecv.fit(X, y)
    score = ridgecv.score(X, y)
    
    print("intercept =", ridgecv.intercept_)
    print("coef =", ridgecv.coef_)
    print("alpha =", ridgecv.alpha_)
    print("r2 =", score)
    

    return ridgecv, score, ridgecv.coef_, ridgecv.intercept_