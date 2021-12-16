# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 03:36:03 2018

@author: Dan
"""
"""
Ratings Algorithm 1
"""
import pandas as pd
import NFLP
import datetime as dt

start_time = dt.datetime.now()

RATINGS_ALG_ID = 1

nfl = NFLP.NFLP()
print("NFLP() done, {}", dt.datetime.now() - start_time)

yrcols = ['Team', 'Year', 'Rating', 'Score']
wrcols = ['Team', 'Year', 'Week', 'Rating', 'Prev_Rating', 'Opponent', 'Score']

def calc_initial_ratings(nflp):
    ratings_yearly_df = pd.DataFrame(columns=yrcols)
    ratings_weekly_df = pd.DataFrame(columns=wrcols)
    rownum = 0
    yearrownum = 0
    td = nflp.get_teams_all()
    rstd = nflp.get_teams_reg()
    for team in td:
        print("calculating initial for team {}, {}".format(team, dt.datetime.now() - start_time))
        # print(team)
        rstdt = rstd[team]
        tdt = td[team]
        year = 1970
        while (year < 2019):
            week = 1
            while (week < 23 and (year != 2018 or week == 1)):
                rstdtyw = rstdt[(rstdt['Year'] == year) & (rstdt['Week'] < week)].copy()
                rstdtyw.reset_index(drop=True, inplace=True)
                if (len(rstdtyw.index) > 0):
                    rstdtywv = rstdtyw[rstdtyw['Visitor'] == team]
                    rstdtywh = rstdtyw[rstdtyw['Home'] == team]
                    num_games = len(rstdtywv.index) + len(rstdtywh.index)
                    net_points = 0.0
                    if (len(rstdtywv.index) > 0):
                        net_points = rstdtywv['VNPA'].sum()
                    if (len(rstdtywh.index) > 0):
                        net_points = net_points - rstdtywh['VNPA'].sum()
                else:
                    num_games = 0
                    net_points = 0.0
                # if team == 'NEP' or team == 'ARI':
                #     print("week {}, net points {}, num games {}".format(week, net_points, num_games))
                # get yearly rating
                temp = ratings_yearly_df[(ratings_yearly_df['Team'] == team) & (ratings_yearly_df['Year'] == year - 1)].copy()
                if (len(temp.index) == 0):
                    # not assigned yet, so assign
                    temp2 = ratings_weekly_df[(ratings_weekly_df['Team'] == team) & (ratings_weekly_df['Year'] == year - 1) & (ratings_weekly_df['Week'] == 22)].copy()
    #            temp = ratings_yearly_df[(ratings_yearly_df['Team'] == team) & (ratings_yearly_df['Year'] == year - 1)]
                    if (len(temp2.index) > 0):
                        temp2.reset_index(drop=True, inplace=True)
                        prev_year_rating = temp2.at[0, 'Rating']
                        prev_year_score = temp2.at[0, 'Score']
                    else:
                        prev_year_rating = 0.0
                        prev_year_score = 0.0
                    if (week == 1):
                        ratings_yearly_df.loc[yearrownum] = [team, year-1, prev_year_rating, prev_year_score]
                        yearrownum = yearrownum + 1
                else:
                    temp.reset_index(drop=True, inplace=True)
                    prev_year_rating = temp.at[0, 'Rating']
                    prev_year_score = temp.at[0, 'Score']
                # if team == 'NEP' or team == 'ARI':
                #     print("year {}, rating {}, score {}".format(year, prev_year_rating, prev_year_score))
                rating = ((prev_year_rating * 3 + (net_points / 2))/(num_games + 3))
                # We divide net points by 2 to get the week's rating.
                # For example, if you are the visitor and win by 8, your
                # net points here will be 10.7 (adjusted for home-field)
                # Now you are 10.7 points better than your opponent, which
                # translates to your rating of 5.35 and your opponent of -5.35
                #
                # Later, when we update through iteration, we calculate a
                # fulcrum from the midpoint of the rating of the two teams
                # that played, and apply the adjustment to the fulcrum.
                # See in calc_new_ratings()
                #
                # tdtyw = tdt[((tdt['Year'] == year) & (tdt['Week'] < week) | (tdt['Year'] < year))]
                # tdtyw.reset_index(drop=True, inplace=True)
                # if (len(tdtyw.index) > 0):
                #     if (tdtyw.at[len(tdtyw.index)-1, 'Home'] == team):
                #         prev_opp = tdtyw.at[len(tdtyw.index)-1, 'Visitor']
                #     else:
                #         prev_opp = tdtyw.at[len(tdtyw.index)-1, 'Home']
                # else:
                #     prev_opp = None
                tdtyw = tdt[(tdt['Year'] == year) & (tdt['Week'] == week)]
                tdtyw.reset_index(drop=True, inplace=True)
                if (len(tdtyw.index) > 0):
                    if (tdtyw.at[0, 'Home'] == team):
                        opp = tdtyw.at[0, 'Visitor']
                    else:
                        opp = tdtyw.at[0, 'Home']
                else:
                    opp = None
                score_df = tdt[(tdt.Year == year) & (tdt.Week == week)].copy()
                score_df.reset_index(drop=True, inplace=True)
                if len(score_df.index) > 0:
                    if score_df.at[0, 'Home'] == team:
                        score = -score_df.at[0, 'VNPA']
                    else:
                        # print([col for col in score_df])
                        score = score_df.at[0, 'VNPA']
                    # if team == 'NEP' or team == 'ARI':
                    #     print("week {}, score {}".format(week, score))
                else:
                    score = None
                ratings_weekly_df.loc[rownum] = [team, year, week, rating, rating, opp, score]
                # if team == 'NEP' or team == 'ARI':
                #     print(ratings_weekly_df.loc[rownum].tolist())
                # ratings_weekly_df.loc[rownum] = [team, year, week, rating, rating, prev_opp]
                rownum = rownum + 1
                week = week + 1
            year = year + 1
    
    return ratings_yearly_df, ratings_weekly_df

def get_prev_rating(df, team, year, week):
    idx = (df.index[(df.Team == team) & (df.Year == year) & (df.Week == week)]).tolist()
#    df.reset_index(drop=True, inplace=True)
    if (len(idx) > 0):
        return df.at[idx[0], 'Prev_Rating']
    return 0.0

def set_rating(df, team, year, week, rating):
    idx = (df.index[(df.Team == team) & (df.Year == year) & (df.Week == week)]).tolist()
    if (len(idx) > 0):
        df.at[idx[0], 'Rating'] = rating
    # if team == 'ARI' or team == 'NEP':
    #     print(df.loc[idx[0]].tolist())
        
def set_score(df, team, year, week, score):
    idx = (df.index[(df.Team == team) & (df.Year == year) & (df.Week == week)]).tolist()
    if (len(idx) > 0):
        df.at[idx[0], 'Score'] = score
    # if team == 'ARI' or team == 'NEP':
    #     print(df.loc[idx[0]].tolist())
        
def calc_new_ratings(nflp, ratings_yearly_df, ratings_weekly_df):
    ratings_weekly_df['Prev_Rating'] = ratings_weekly_df['Rating']
    # td = nflp.get_teams_all()
    gdf = nflp.get_games_overall()
    prev_team = None
    for index, row in ratings_weekly_df.iterrows():
        # print(row.Team)
        if row.Team != prev_team:
            print("calculating new for team {}, {}".format(row.Team, dt.datetime.now() - start_time))
            # print(row.Team)
            prev_team = row.Team
        # tdt = td[row.Team]
        tgdf = gdf[((gdf.Visitor == row.Team) | (gdf.Home == row.Team))
                    & (gdf.Year == row.Year) & (gdf.Week == row.Week)].copy()
        tgdf.reset_index(drop=True, inplace=True)
        # print(tgdf)
        if row.Week == 1:
            prev_year = row.Year - 1
            prev_week = 22
        else:
            prev_year = row.Year
            prev_week = row.Week - 1
        team_rating = get_prev_rating(ratings_weekly_df, row.Team, prev_year, prev_week)
        if len(tgdf.index) > 0:
            if row.Team == tgdf.at[0, 'Visitor']:
                opp = tgdf.at[0, 'Home']
                np = tgdf.at[0, 'VNPA']
            else:
                opp = tgdf.at[0, 'Visitor']
                np = - tgdf.at[0, 'VNPA']
            opp_rating = get_prev_rating(ratings_weekly_df, opp, prev_year, prev_week)
            new_score = (np + team_rating + opp_rating)/2
            # We take the relative strength of the two teams and use that
            # as a fulcrum/baseline. Then we apply the net points adjustment
            # Thus, say a team with a rating of 8.0 plays a team with a 6.0
            # The average is 7.0, and if np = 10.7 (visitor won by 8, then
            # adjusted for home field), then the visitor's strength score
            # for the game is 7 + 10.7/2 = 12.35, and the home's strength
            # score is 7 - 10.7/2 = 1.65. So the 12.35 is 10.7 greater than
            # the 1.65
        else:
            opp = None
            np = 0.0
            new_score = None
        # if row.Team == 'ARI' or row.Team == 'NEP':
        #     print("team rating {}, opponent rating {}, net points {}, new score {}".format(team_rating, opp_rating, np, new_score))
        set_score(ratings_weekly_df, row.Team, row.Year, row.Week, new_score)
        # if row.Week == 22:
        #     rdf = ratings_yearly_df
        #     tyrdf = rdf[(rdf.Team == row.Team) & (rdf.Year == row.Year)]
        #     rdfindex = tyrdf.index[0]
        #     ratings_yearly_df.at[rdfindex, 'Score'] = new_score
        rwdf = ratings_weekly_df
        rydf = ratings_yearly_df
        trwdf = rwdf[(rwdf.Team == row.Team) & (rwdf.Year == row.Year) & (rwdf.Week < row.Week)].copy()
        trydf = rydf[(rydf.Team == row.Team) & (rydf.Year == (row.Year - 1))].copy()
        nr = 0.0
        ng = 0
        if len(trwdf.index) > 0:
            nr += trwdf['Score'].sum()
            ng += trwdf['Score'].count()
        if len(trydf.index) > 0:
            trydf.reset_index(drop=True, inplace=True)
            nr += trydf.at[0, 'Rating'] * 3.0
            ng += 3
        nr = nr / ng
        set_rating(ratings_weekly_df, row.Team, row.Year, row.Week, nr)
        if row.Week == 22:
            rdf = ratings_yearly_df
            tyrdf = rdf[(rdf.Team == row.Team) & (rdf.Year == row.Year)]
            rdfindex = tyrdf.index[0]
            ratings_yearly_df.at[rdfindex, 'Rating'] = nr
    diffseries = (ratings_weekly_df['Rating'] - ratings_weekly_df['Prev_Rating']).apply(abs)
    return diffseries.sum()
    
ratings_yearly_df, ratings_weekly_df = calc_initial_ratings(nfl)
for i in range(20):
    print("Begin Iteration {}, {}".format(i+1, dt.datetime.now() - start_time))
    diff = calc_new_ratings(nfl, ratings_yearly_df, ratings_weekly_df)
    print("Iteration {}, Ratings diff is {}, {}".format(i+1, diff, dt.datetime.now() - start_time))
    if diff < 0.1:
        break;

print("Write to csv, {}".format(dt.datetime.now() - start_time))
ratings_weekly_df.to_csv("RatingsWeekly_{}.csv".format(RATINGS_ALG_ID), index=False)
print("Done, {}".format(dt.datetime.now() - start_time))

