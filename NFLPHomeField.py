# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 09:01:40 2017

@author: Dan
"""

# Calculate home-field advantage

import NFLP
import numpy as np
import matplotlib.pyplot as plt

nfl = NFLP.NFLP()
gdf = nfl.get_games_overall()

nngdf = gdf[gdf.Neutral == False]

print("Overall home field advantage is", -nngdf['VNP'].mean())

fyadv = []

year = 2016
while year > 1984:
    ynngdf = nngdf[(nngdf.Year <= year) & (nngdf.Year > (year-11))]
    fyadv.append(-ynngdf['VNP'].mean())
    print(year-10, "to", year, "home field advantage is", -ynngdf['VNP'].mean())
    year = year - 1

yadv = []
xadv = []

year = 1970
i = 0
while year < 2018:
    ynngdf = nngdf[nngdf.Year == year]
    yadv.append(-ynngdf['VNP'].mean())
    print(year, "home field advantage is", -ynngdf['VNP'].mean())
    xadv.append(i)
    i = i + 1
    year = year + 1
#                (nngdf.Year == (year-1)) |
#                (nngdf.Year == (year-2)) |
#                (nngdf.Year == (year-3)) |
#                (nngdf.Year == (year-4))] 
#    print(year-4, "to", year, "home field advantage is", -ynngdf['VNP'].mean())
#    year = year -5

plt.scatter(xadv, yadv)

c = np.polyfit(xadv, yadv, 4)
p = np.poly1d(c)
plt.plot(xadv, p(xadv))

#print("home field advantage for", 2017, "is", c[0]*47 + c[1])

tadv= []
    
for team, tdf in nfl.get_teams_all().items():
    tadv.append(-tdf['VNP'].mean())
    print(team, "home field advantage is", -tdf['VNP'].mean())
    
print("stddev for years is", np.std(yadv))
print("stddev for teams is", np.std(tadv))

#Conclusion
#
# home field advantage from line fit is 2.73 for 2017
# there is a slight but clear trend downward over time, from 2.97 to 2.74
#
# varies from team to team, but stddev of teams is 
# significantly less than stddev of years
# therefore, no significant team bias