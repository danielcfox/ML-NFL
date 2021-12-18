#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reated on Fri Oct 27 06:09:51 2017

@author: dfox
"""

import pandas as pd
import os
# import datetime as dt

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


class NFLP:
    """
    NFL Prediction class.

    It must be instantiated to use the NFL Predictor.
    This takes a long time to instantiate because it's loading in all games
    from 1970-2017. TBD - Add games from 2018-2021 in NFLGamesDB.csv
    """

    nfl_games_fn = os.path.join(location, 'NFLGamesDB.csv')
    team_names_map_fn = os.path.join(location, 'TeamNamesMap.csv')
    neutral_sites_fn = os.path.join(location, 'UpTo2012NeutralSites.xlsx')
    planned_neutral_sites_fn = os.path.join(location,
                                            'PlannedNeutralSites.xlsx')

    def __init__(self):
        # starttime = dt.datetime.now()
        # print("start")

        self.td = {}
        self.rstd = {}
        self.tds = {}
        self.rstds = {}

        gdf = pd.read_csv(NFLP.nfl_games_fn)
        tdf = pd.read_csv(NFLP.team_names_map_fn)
        xl = pd.ExcelFile(NFLP.neutral_sites_fn)
        nudf = xl.parse()
        xl = pd.ExcelFile(NFLP.planned_neutral_sites_fn)
        pnudf = xl.parse()

        # count = 0
        # count = count + 1
        # print("ms{} {}".format(count, dt.datetime.now() - starttime))
        gdf['VPF'] = (gdf['Q1VS'] + gdf['Q2VS'] + gdf['Q3VS'] + gdf['Q4VS']
                      + gdf['OTVS'])
        gdf['VRPF'] = gdf['VPF'] - gdf['OTVS']
        gdf['VPA'] = (gdf['Q1HS'] + gdf['Q2HS'] + gdf['Q3HS'] + gdf['Q4HS']
                      + gdf['OTHS'])
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

        for index, row in gdf.iterrows():
            v = tdf[tdf['Name'] == row.Visitor]
            v = v[row['Year'] >= v['Start']]
            v = v[row['Year'] <= v['End']]
            v.reset_index(inplace=True, drop=True)
            gdf.at[index, 'Visitor Symbol'] = v.at[0, 'Symbol']
            h = tdf[tdf['Name'] == row.Home].copy()
            h = h[row['Year'] >= h['Start']]
            h = h[row['Year'] <= h['End']]
            h.reset_index(inplace=True, drop=True)
            gdf.at[index, 'Home Symbol'] = h.at[0, 'Symbol']

        # count = count + 1
        # print("ms{} {}".format(count, dt.datetime.now() - starttime))
        gdf['Visitor'] = gdf['Visitor Symbol']
        gdf['Home'] = gdf['Home Symbol']
        gdf.drop(['Visitor Symbol', 'Home Symbol'], axis=1, inplace=True)
        gdf.sort_values(['Year', 'Week'], axis=0, inplace=True)
        gdf.reset_index(drop=True, inplace=True)
        gdf['Neutral'] = False

        nudf.dropna(inplace=True)
        nudf['Season'] = nudf['Season'].astype(int)
        nudf = nudf[nudf.Season >= 1970]
        nudf = nudf[nudf['Week'] != 'SB']
        nudf = nudf[(nudf['Designated'] != 'Oakland Raiders')
                    | (nudf["State/Int'l"] != 'CA')]
        nudf = nudf[(nudf['Designated'] != 'San Francisco 49ers')
                    | (nudf["State/Int'l"] != 'CA')]
        nudf = nudf[(nudf['Designated'] != 'New York Jets')
                    | (nudf["State/Int'l"] != 'NJ')]
        nudf = nudf[(nudf['Designated'] != 'Chicago Bears')
                    | (nudf["State/Int'l"] != 'IL')]
        nudf = nudf[(nudf['Designated'] != 'Minnesota Vikings')
                    | (nudf["State/Int'l"] != 'MN')]
        nudf = nudf[(nudf['Designated'] != 'Buffalo Bills')
                    | (nudf["City"] != 'Toronto')]
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
            nudf.at[index, 'Visitor Symbol'] = v.at[0, 'Symbol']
            h = tdf[tdf['Name'] == row.Home].copy()
            h = h[row['Year'] >= h['Start']]
            h = h[row['Year'] <= h['End']]
            h.reset_index(inplace=True, drop=True)
            nudf.at[index, 'Home Symbol'] = h.at[0, 'Symbol']

        # count = count + 1
        # print("ms{} {}".format(count, dt.datetime.now() - starttime))
        nudf['Visitor'] = nudf['Visitor Symbol']
        nudf['Home'] = nudf['Home Symbol']
        nudf.drop(['Visitor Symbol', 'Home Symbol'], axis=1, inplace=True)
        nudf.sort_values(['Year', 'Week', 'Visitor'], axis=0, inplace=True)
        nudf['Week'] = nudf['Week'].astype(int)

        pnudf.drop(pnudf.head(1).index, inplace=True)
        prev = 1926
        for index, row in pnudf.iterrows():
            if (pd.isnull(row.Season)):
                pnudf.at[index, 'Season'] = prev
            else:
                prev = row.Season
        pnudf = pnudf[pnudf.Type == 'REG']
        pnudf['Season'] = pnudf['Season'].astype(int)
        pnudf = pnudf[(pnudf.Season >= 1970) & (pnudf.Type == 'REG')]
        pnudf = pnudf[(pnudf.Season != 2018)]
        pnudf = pnudf[(pnudf['Winning/Tied Team'] != 'Buffalo Bills')
                      | (pnudf["City"] != 'Toronto')]
        pnudf = pnudf[(pnudf['Losing/Tied Team'] != 'Buffalo Bills')
                      | (pnudf["City"] != 'Toronto')]
        for index, row in pnudf.iterrows():
            windf = tdf[(tdf['Name'] == row['Winning/Tied Team'])
                        | (row['Season'] >= tdf['Start'])
                        | (row['Season'] <= tdf['End'])].copy()
            windf.reset_index(inplace=True, drop=True)
            pnudf.at[index, 'Win Symbol'] = windf.at[0, 'Symbol']
            lossdf = tdf[(tdf['Name'] == row['Losing/Tied Team'])
                         | (row['Season'] >= tdf['Start'])
                         | (row['Season'] <= tdf['End'])].copy()
            lossdf.reset_index(inplace=True, drop=True)
            pnudf.at[index, 'Lose Symbol'] = lossdf.at[0, 'Symbol']
            score = pnudf.at[index, 'Score']
            if (score[1] == "-"):
                wscore = score[:1]
                lscore = score[2:4]
            else:
                wscore = score[:2]
                lscore = score[3:5]
            pnudf.at[index, 'WNP'] = int(int(wscore) - int(lscore))

        pnudf['WNP'] = pnudf['WNP'].astype(int)

        # count = count + 1
        # print("ms{} {}".format(count, dt.datetime.now() - starttime))
        for index, row in nudf.iterrows():
            gndf = gdf[(gdf.Year == row.Year) & (gdf.Week == row.Week)
                       & (gdf.Visitor == row.Visitor)]
            for gindex, grow in gndf.iterrows():
                gdf.at[gindex, 'Neutral'] = True

        for index, row in pnudf.iterrows():
            gpvdf = gdf[(gdf.Year == row.Season)
                        & (gdf.Visitor == row['Win Symbol'])
                        & (gdf.Home == row['Lose Symbol'])
                        & (gdf.VNP == row.WNP)]
            for gindex, grow in gpvdf.iterrows():
                gdf.at[gindex, 'Neutral'] = True
            gphdf = gdf[(gdf.Year == row.Season)
                        & (gdf.Visitor == row['Lose Symbol'])
                        & (gdf.Home == row['Win Symbol'])
                        & (gdf.VNP == -row.WNP)]
            for gindex, grow in gphdf.iterrows():
                gdf.at[gindex, 'Neutral'] = True

        y = 1970
        while y <= 2017:
            ygdf = gdf[gdf.Year == y]
            week_max = ygdf['Week'].max()
            gmdf = ygdf[ygdf.Week == week_max]
            for index, row in gmdf.iterrows():
                gdf.at[index, 'Neutral'] = True
            y = y + 1

        # count = count + 1
        # print("ms{} {}".format(count, dt.datetime.now() - starttime))
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
            if (row.Week > 17 and row.Year < 2021 and row.Year != 1993):
                gdf.at[index, 'Playoffs'] = True
            if (row.Week > 18 and (row.Year == 1993 or row.Year >= 2021)):
                gdf.at[index, 'Playoffs'] = True

        gdf['OT'] = gdf['OT'].astype(int)
        gdf['VWIN'] = gdf['VWIN'].astype(int)
        gdf['TIE'] = gdf['TIE'].astype(int)
        gdf['Neutral'] = gdf['Neutral'].astype(bool)
        gdf['Playoffs'] = gdf['Playoffs'].astype(int)

        # count = count + 1
        # print("ms{} {}".format(count, dt.datetime.now() - starttime))
        for index, row in gdf.iterrows():
            if row['OT']:
                vnp = 0.0
            else:
                vnp = row['VNP']
            if row['Neutral']:
                gdf.at[index, 'VNPA'] = round(float(vnp), 0)
            else:
                gdf.at[index, 'VNPA'] = round(vnp + 2.7, 1)
            if row.Year > 1977:
                gdf.at[index, 'RegOddsDiff'] = abs(gdf.at[index, 'VRNP']
                                                   + gdf.at[index, 'VSpread'])
                gdf.at[index, 'OddsDiff'] = abs(gdf.at[index, 'VNP']
                                                + gdf.at[index, 'VSpread'])
                gdf.at[index, 'VRegOddsDiff'] = (gdf.at[index, 'VRNP']
                                                 + gdf.at[index, 'VSpread'])
                gdf.at[index, 'VOddsDiff'] = (gdf.at[index, 'VNP']
                                              + gdf.at[index, 'VSpread'])
                gdf.at[index, 'FRegOddsDiff'] = (gdf.at[index, 'VRNP']
                                                 + gdf.at[index, 'VSpread'])
                gdf.at[index, 'FOddsDiff'] = (gdf.at[index, 'VNP']
                                              + gdf.at[index, 'VSpread'])
                if (gdf.at[index, 'VOddsDiff'] > 0):
                    gdf.at[index, 'VBeatSpread'] = 1.0
                elif (gdf.at[index, 'VOddsDiff'] < 0):
                    gdf.at[index, 'VBeatSpread'] = 0.0
                else:
                    gdf.at[index, 'VBeatSpread'] = 0.5
                if (gdf.at[index, 'VSpread'] > 0):
                    if (gdf.at[index, 'VOddsDiff'] < 0):
                        gdf.at[index, 'FBeatSpread'] = 1.0
                    elif (gdf.at[index, 'VOddsDiff'] > 0):
                        gdf.at[index, 'FBeatSpread'] = 0.0
                    else:
                        gdf.at[index, 'FBeatSpread'] = 0.5
                    if (gdf.at[index, 'VOddsDiff'] > 0):
                        gdf.at[index, 'FBeatSpread'] = 0.0
                    elif (gdf.at[index, 'VOddsDiff'] < 0):
                        gdf.at[index, 'FBeatSpread'] = 1.0
                    else:
                        gdf.at[index, 'FBeatSpread'] = 0.5

        gsdf = gdf[gdf.Year > 1977]
        rsdf = gdf[gdf.Playoffs == False]
        rssdf = gsdf[gsdf.Playoffs == False]

        self.gdf = gdf

        # count = count + 1
        # print("ms{} {}".format(count, dt.datetime.now() - starttime))
        for symbol in team_list:
            self.td[symbol] = gdf[(gdf.Visitor == symbol)
                                  | (gdf.Home == symbol)].copy()
            self.td[symbol].reset_index(drop=True, inplace=True)
            self.tds[symbol] = gsdf[(gsdf.Visitor == symbol)
                                    | (gsdf.Home == symbol)].copy()
            self.tds[symbol].reset_index(drop=True, inplace=True)
            self.rstd[symbol] = rsdf[(rsdf.Visitor == symbol)
                                     | (rsdf.Home == symbol)].copy()
            self.rstd[symbol].reset_index(drop=True, inplace=True)
            self.rstds[symbol] = rssdf[(rssdf.Visitor == symbol)
                                       | (rssdf.Home == symbol)].copy()
            self.rstds[symbol].reset_index(drop=True, inplace=True)

    def get_teams_all(self):
        """
        Get the dictionary of each teams's games.

        Returns
        -------
        dict of pd.DataFrame
            A dictionary of dataframes for each team's games.
        """
        return self.td

    def get_teams_spread(self):
        """
        Get the dictionary of each team's games where a poinspread is present.

        Returns
        -------
        dict of pd.DataFrame
            A dictionary of dataframes for each team's games where a
            pointspread is present.
        """
        return self.tds

    def get_teams_reg(self):
        """
        Get the dictionary of each teams's regular season games.

        Returns
        -------
        dict of pd.DataFrame
            A dictionary of dataframes for each team's regular season games.
        """
        return self.rstd

    def get_teams_reg_spread(self):
        """
        Return each team's regular season games where a poinspread is present.

        Returns
        -------
        dict of pd.DataFrame
            A dictionary of dataframes for each team's games where a
            pointspread is present.
        """
        return self.rstds

    def get_games_overall(self):
        """
        Get records of all games.

        Returns
        -------
        pd.DataFrame
            The dataframe of all games in the database (currently 1970-2017).
        """
        return self.gdf


nfl = NFLP()
