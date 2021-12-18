# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 03:36:03 2018

@author: Dan
"""

import pandas as pd
from sklearn.linear_model import RidgeCV
import random
import NFLP
import datetime as dt
import os
import shutil

RATINGS_ALG_ID = 1
"""
above as defined in the get_X_game_NP() code, if X and Y columns are
changed, increment the TRAIN_XY_ID
if how they are calculated changes in the NFLPCalcRatings.py code,
increment the RATINGS_ALG_ID

Yes, better code would handle this much better.
"""

def get_X_game_NP(nflp, rwdf, prev_games, begin_year, end_year, week, week_end):
    # year/week the year and week to PREDICT
    # X is a row for each team that played 32 previous games
    # each of 32 games has two columns: Net Points; Opponent Rating For That Week
    
    TRAIN_XY_ID = 1
    
    gdf = nflp.get_games_overall()
    td = nflp.get_teams_all()

    Xcols = []
    for game in range(32):
        Xcols.append('VORG{}'.format(game))
        Xcols.append('VNPG{}'.format(game))
    for game in range(32):
        Xcols.append('HORG{}'.format(game))
        Xcols.append('HNPG{}'.format(game))

    # rownum = 0
    gamedf = gdf[(gdf.Year >= begin_year) & (gdf.Year <= end_year) & (gdf.Week >= week) & (gdf.Week <= week_end)].copy()
    gamedf.reset_index(drop=True, inplace=True)
    for index, row in gamedf.iterrows():
        # predict each game that we have 32 prev games played by each team
        vtd = td[row.Visitor]
        vtd = vtd[(vtd.Year < row.Year) | ((vtd.Year == row.Year) & (vtd.Week < row.Week))].copy()
        vtd.reset_index(drop=True, inplace=True)
        if (len(vtd.index) < prev_games):
            continue;

        htd = td[row.Home]
        htd = htd[(htd.Year < row.Year) | ((htd.Year == row.Year) & (htd.Week < row.Week))].copy()
        htd.reset_index(drop=True, inplace=True)
        if (len(htd.index) < prev_games):
            continue;
        
        # now put in visitor block of 32 previous games
        vtd32 = vtd.loc[len(vtd.index)-prev_games:len(vtd.index)].copy()
        vtd32.reset_index(drop=True, inplace=True)
        for tindex, trow in vtd32.iterrows():
#            ordf = rwdf[(rwdf.Team == gamedf.Visitor) & (rwdf.Year == trow.Year) & (rwdf.Week == trow.Week)]
#            ordf.reset_index(drop=True, inplace=True)
#            gamedf.at[index, 'VORG{}'.format(tindex)] = ordf.at[0, 'Rating'] 
            idx = (rwdf.index[(rwdf.Team == trow.Visitor) & (rwdf.Year == trow.Year) & (rwdf.Week == trow.Week)]).tolist()
            if (len(idx) > 0):
                gamedf.at[index, 'VORG{}'.format(tindex)] = rwdf.at[idx[0], 'Rating'] 
            else:
                gamedf.at[index, 'VORG{}'.format(tindex)] = 0.0
            gamedf.at[index, 'VNPG{}'.format(tindex)] = trow.VNP                

        # now put in home block of 32 previous games
        htd32 = htd.loc[len(htd.index)-32:len(htd.index)].copy()
        htd32.reset_index(drop=True, inplace=True)
        for tindex, trow in htd32.iterrows():
#            ordf = rwdf[(rwdf.Team == gamedf.Home) & (rwdf.Year == trow.Year) & (rwdf.Week == trow.Week)]
#            ordf.reset_index(drop=True, inplace=True)
#            gamedf.at[index, 'HORG{}'.format(tindex)] = ordf.at[0, 'Rating'] 
            idx = (rwdf.index[(rwdf.Team == trow.Home) & (rwdf.Year == trow.Year) & (rwdf.Week == trow.Week)]).tolist()
            if (len(idx) > 0):
                gamedf.at[index, 'HORG{}'.format(tindex)] = rwdf.at[idx[0], 'Rating'] 
            else:
                gamedf.at[index, 'HORG{}'.format(tindex)] = 0.0
            gamedf.at[index, 'HNPG{}'.format(tindex)] = trow.VNP

    X = gamedf[Xcols]
    y = gamedf['VNPA']

    return TRAIN_XY_ID, gamedf, X, y

start_time = dt.datetime.now()
nfl = NFLP.NFLP()
print("nfl initialization complete, {}".format(dt.datetime.now() - start_time))

ratings_weekly_df = pd.read_csv("./RatingsWeekly_{}.csv".format(RATINGS_ALG_ID))
print("ratings load complete, {}".format(dt.datetime.now() - start_time))
# def get_X_game_NP(nflp, rwdf, begin_year, end_year, week):

rescols = ['Train_Algorithm', 'Train_XY_ID', 'Ratings_Algorithm_ID',
           'Parameters_And_Results',
           'Year_Begin_Predict', 'Year_End_Predict', 
           'Week_Begin_Predict', 'Week_End_Predict', 'Num_Games',
           'Win_Correct', 'Win_Correct_Fraction', 'Num_Games_No_Push',
           'Spread_Correct', 'Spread_Correct_Fraction']

if os.path.exists("./MLPredictResults.csv"):
    shutil.copyfile("./MLPredictResults.csv", "./MLPredictResultsOld_{}_{}_{}_{}_{}_{}.csv".format(start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second))
    results_prior_df = pd.read_csv("./MLPredictResults.csv")
else:
    results_prior_df = pd.DataFrame(columns=rescols)
result_df = pd.DataFrame(columns=rescols)
# result_rownum = 0

for gnum in range(1, 17+1):
    gnum_end = gnum
    if gnum == 17:
        gnum_end = 21
    
    xy_id, gamedf, X, y = get_X_game_NP(nfl, ratings_weekly_df, 32, 2004, 2012, gnum, gnum_end)        

    random.seed(17)        
    ridgecv = RidgeCV(alphas=(.001, .01, .1, 1.0, 5.0, 10.0, 20.0, 50.0, 75.0, 100.0, 1000.0), store_cv_values=True)
    ridgecv.fit(X, y)
    score = ridgecv.score(X, y)
    
    param_res = ("Xtrain_prevgames=32"
                 + ":Xtrain_years=2004-2012"
                 + ":Xtrain_weeks={}-{}".format(gnum, gnum_end)
                 + ":alpha={}".format(ridgecv.alpha_)
                 + ":r2={}".format(round(score, 3))
                 + ":seed=17")
        
    # print("X[{},{}] {}".format(X.shape[0], X.shape[1], [col for col in X]))
    # print("y[{},] {}".format(y.shape[0], [value for value in y]))
    # print("intercept =", ridgecv.intercept_)
    # print("coef[{}] = {}".format(len(ridgecv.coef_), ridgecv.coef_))
    # print("alpha =", ridgecv.alpha_)
    # print("r2 =", score)

    # calculate new ratings
    
    overall_correct = 0
    overall_correct_spread = 0
    overall_num_games = 0
    overall_num_games_spread = 0
    for year in range(2013, 2017+1):
    
        _, pred_gamedf, pred_X, actual_pred_y = get_X_game_NP(nfl, ratings_weekly_df, 32, year, year, gnum, gnum_end)
        # print([col for col in pred_gamedf])
        # print(pred_gamedf['VSpread'])
        
        pred_y = ridgecv.predict(pred_X)
        # since Y is 'VNPA', it is visitor net points adjusted for home field
        # re-adjust back to the actual net points by subtracting 2.7 for
        # home field disadvantage
        pred_y_adj = pred_y - 2.7
        pred_y_list = [value for value in pred_y_adj]
        # actual_pred_y_adj = actual_pred_y - 2.7
        # actual_pred_y_list = [value for value in actual_pred_y_adj]
        # print("pred_y {}".format(pred_y_list))
        # print("actual_pred_y {}".format(actual_pred_y_list))
        correct = 0
        correct_spread = 0
        push_spread = 1
        pred_gamedf.reset_index(drop=True,inplace=True)
        for index, pred_val in enumerate(pred_y_list):
            # act_val = round(actual_pred_y_list[index]
            act_val = round(float(pred_gamedf.at[index, 'VNP']), 0)
            if pred_val * act_val > 0.0:
                correct += 1
            spread = round(float(pred_gamedf.at[index, 'VSpread']), 1)
            if pred_val > spread:
                visitor_pred = True
            else:
                visitor_pred = False
            if act_val > spread:
                visitor_actual = True
                push_actual = False
            elif act_val == spread:
                visitor_actual = False
                push_actual = True
            else:
                visitor_actual = False
                push_actual = False
            if push_actual:
                push_spread += 1
            elif visitor_pred == visitor_actual:
                correct_spread += 1
        num_games = len(pred_y_list)
        num_games_spread = len(pred_y_list) - push_spread
        correct_fraction = round(float(correct) / float(num_games), 3)
        correct_spread_fraction = round(float(correct_spread) / float(num_games_spread), 3)
        
        print("{} correct for year {}, week {}-{}, out of {}, accuracy {},  {}".format(correct, year, gnum, gnum_end, num_games, correct_fraction, dt.datetime.now() - start_time))
        print("{} correct spread for year {}, week {}-{}, out of {}, push {}, overtime {}, accuracy {}, {}".format(correct_spread, year, gnum, gnum_end, num_games_spread, push_spread, gamedf['OT'].sum(), correct_spread_fraction, dt.datetime.now() - start_time))
        # overall_num_games += num_games
        # overall_num_games_spread += num_games_spread
        # overall_correct += correct
        # overall_correct_spread += correct_spread
    
        # rescols = ['Train_Algorithm', 'Train_XY_ID', 'Ratings_Algorithm_ID',
        #            'Parameters_And_Results',
        #            'Year_Begin_Predict', 'Year_End_Predict', 
        #            'Week_Begin_Predict', 'Week_End_Predict', 'Num_Games',
        #            'Win_Correct', 'Win_Correct_Fraction', 'Num_Games_No_Push',
        #            'Spread_Correct', 'Spread_Correct_Fraction']

        rec = ['Ridge', xy_id, RATINGS_ALG_ID,
               param_res,
               year, year, gnum, gnum_end, num_games,
               correct, correct_fraction, 
               num_games_spread,
               correct_spread, 
               correct_spread_fraction]

        recseries = pd.Series(rec, rescols)
        result_df = result_df.append([recseries], ignore_index=True)
        # result_df.loc[result_rownum] = rec
        # result_rownum += 1

    

    # overall_correct_fraction = round(float(overall_correct)
    #                                  / float(overall_num_games), 3)
    # overall_correct_spread_fraction = round(float(overall_correct_spread)
    #                                  / float(overall_num_games_spread), 3)
    # result_df.loc[result_rownum] = ['Ridge', xy_id, RATINGS_ALG_ID, param_res,
    #                                 2013, 2017, gnum, gnum_end, 
    #                                 overall_num_games,
    #                                 overall_correct, correct_fraction, 
    #                                 overall_num_games_spread,
    #                                 overall_correct_spread, 
    #                                 overall_correct_spread_fraction]
    # result_rownum += 1

year_result_df = pd.DataFrame(columns=rescols)
week_result_df = pd.DataFrame(columns=rescols)
summary_result_df = pd.DataFrame(columns=rescols)
for year in range(2013, 2017+1):
    rdf = result_df[result_df.Year_Begin_Predict == year]
    week_begin = rdf['Week_Begin_Predict'].min()
    week_end = rdf['Week_End_Predict'].max()
    param_res = ("Xtrain_prevgames=32"
                 + ":Xtrain_years=2004-2012"
                 + ":Xtrain_weeks={}-{}".format(week_begin, week_end))
    num_games = rdf['Num_Games'].sum()
    print(type(num_games))
    print(num_games)
    if num_games <= 0:
        num_games = 1
    correct = rdf['Win_Correct'].sum()
    correct_fraction = round(float(correct) / float(num_games), 3)
    num_games_spread = rdf['Num_Games_No_Push'].sum()
    if num_games_spread <= 0:
        num_games_spread = 1
    correct_spread = rdf['Spread_Correct'].sum()
    correct_spread_fraction = round(float(correct_spread) / float(num_games_spread), 3)
    rec = ['Ridge', xy_id, RATINGS_ALG_ID,
           param_res,
           year, year, week_begin,
           week_end, num_games,
           correct, correct_fraction, 
           num_games_spread,
           correct_spread, 
           correct_spread_fraction]
    recseries = pd.Series(rec, rescols)
    year_result_df = year_result_df.append([recseries], ignore_index=True)
    # result_df.loc[result_rownum] = rec
    # result_rownum += 1
    print("Year {}: {} correct out of {}, accuracy {}".format(year, correct, num_games, correct_fraction))
    print("Year {}: {} correct spread out of {}, accuracy {}".format(year, correct_spread, num_games_spread, correct_spread_fraction))
for week in range(1, 17+1):
    rdf = result_df[result_df.Week_Begin_Predict == week]
    year_begin = rdf['Year_Begin_Predict'].min()
    year_end = rdf['Year_End_Predict'].max()
    param_res = ("Xtrain_prevgames=32"
                 + ":Xtrain_years=2004-2012"
                 + ":Xtrain_weeks={}-{}".format(week, week))
    num_games = rdf['Num_Games'].sum()
    if num_games <= 0:
        num_games = 1
    correct = rdf['Win_Correct'].sum()
    correct_fraction = round(float(correct) / float(num_games), 3)
    num_games_spread = rdf['Num_Games_No_Push'].sum()
    if num_games_spread <= 0:
        num_games_spread = 1
    correct_spread = rdf['Spread_Correct'].sum()
    correct_spread_fraction = round(float(correct_spread) / float(num_games_spread), 3)
    rec = ['Ridge', xy_id, RATINGS_ALG_ID,
           param_res,
           year_begin, year_end, week, week, num_games,
           correct, correct_fraction, 
           num_games_spread,
           correct_spread, 
           correct_spread_fraction]
    recseries = pd.Series(rec, rescols)
    week_result_df = week_result_df.append([recseries], ignore_index=True)
    # result_df.loc[result_rownum] = rec
    # result_rownum += 1
    print("Years {}-{}, Week {}: {} correct out of {}, accuracy {}".format(year_begin, year_end, week, correct, num_games, correct_fraction))
    print("Years {}-{}, Week {}: {} correct spread out of {}, accuracy {}".format(year_begin, year_end, week, correct_spread, num_games_spread, correct_spread_fraction))

param_res = ("Xtrain_prevgames=32"
             + ":Xtrain_years=2004-2012"
             + ":Xtrain_weeks={}-{}".format(1, 21))
num_games = result_df['Num_Games'].sum()
if num_games <= 0:
    num_games = 1
correct = result_df['Win_Correct'].sum()
correct_fraction = round(float(correct) / float(num_games), 3)
num_games_spread = result_df['Num_Games_No_Push'].sum()
if num_games_spread <= 0:
    num_games_spread = 1
correct_spread = result_df['Spread_Correct'].sum()
correct_spread_fraction = round(float(correct_spread) / float(num_games_spread), 3)
rec = ['Ridge', xy_id, RATINGS_ALG_ID,
       param_res,
       2013, 2017, 1, 21, num_games,
       correct, correct_fraction, 
       num_games_spread,
       correct_spread, 
       correct_spread_fraction]
recseries = pd.Series(rec, rescols)
summary_result_df = week_result_df.append([recseries], ignore_index=True)
print("Years {}-{}, Weeks {}-{}: {} correct out of {}, accuracy {}".format(2013,2017, 1, 21, correct, num_games, correct_fraction))
print("Years {}-{}, Weeks {}-{}: {} correct spread out of {}, accuracy {}".format(2013,2017, 1, 21, correct_spread, num_games_spread, correct_spread_fraction))

## overall_correct_fraction = round(float(overall_correct) / float(overall_num_games), 3)
# overall_correct_spread_fraction = round(float(overall_correct_fraction) / float(overall_num_games), 3)
# print("{} correct for week {}-{} out of {}, {}".format(overall_correct, 1, 21, overall_num_games, dt.datetime.now() - start_time))
# print("{} correct spread for week {}-{} out of {}, {}".format(overall_correct_spread, 1, 21, overall_num_games, dt.datetime.now() - start_time))
results_prior_df = results_prior_df.append(result_df)
results_prior_df = results_prior_df.append(year_result_df)
results_prior_df = results_prior_df.append(week_result_df)
results_prior_df = results_prior_df.append(summary_result_df)
results_prior_df.to_csv("./MLPredictResults.csv", index=False)

