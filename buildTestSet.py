#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 08:34:00 2018

@author: bking
"""

import pandas as pd
import gc #garbage collector
import matplotlib.pyplot as plt
import dask.dataframe as dd
from pandas import pivot_table

from sklearn import cross_validation as cv
from sklearn.metrics.pairwise import pairwise_distances, cosine_distances,cosine_similarity
from sklearn.metrics import mean_squared_error
from math import sqrt
import time
import numpy as np

#df_playlists_test = pd.read_hdf('data/df_playlists_test.hdf')
#df_playlists_test_info = pd.read_hdf('data/df_playlists_test_info.hdf')
    
# Unique (run on local)
df_unique_tracks = pd.read_hdf('data/df_tracks.hdf')
df_tracks = pd.read_hdf('data/df_playlists.hdf')
df_playlists = pd.read_hdf('data/df_playlists_info.hdf')    

# Run on local with different directory
#df_unique_tracks = pd.read_hdf('data/df_data/df_tracks.hdf')
#df_tracks = pd.read_hdf('data/df_data/df_playlists.hdf')
#df_playlists = pd.read_hdf('data/df_data/df_playlists_info.hdf')   

#- All tracks in the challenge set appear in the MPD
#- All holdout tracks appear in the MPD
#
#The test set contains X difference challenges:
#- Predict tracks for a playlist given its title and the first 5 tracks
#- Predict tracks for a playlist given its title and the first 10 tracks
#- Predict tracks for a playlist given its title and the first 25 tracks
#- Predict tracks for a playlist given its title and 25 random tracks
#- Predict tracks for a playlist given its title and the first 100 tracks
#- Predict tracks for a playlist given its title and 100 random tracks


# Pandas table sorted by 
df_temp = df_tracks.copy()
df_temp['count'] = 1
df_track_distr = df_temp.groupby(['tid'])['count'].sum().sort_values(ascending=False)
del df_temp

#criteria_list = [200,100,50,25,10,5]
criteria_list = [200,100]
pid_list = []
for criteria in criteria_list:
    #criteria = 200
    df_temp = df_playlists.loc[:,['num_tracks','pid']]
    # filter playlists that have more than criteria = 100 tracks 
    df_filter = df_temp[df_temp.num_tracks > criteria]
    
    # Create an empty list to contain the pid
    list_size = 10
    list_1000 = []
    while (True):
        # randomly pick 1 pid in df_filter
        ran = df_filter.sample(n=1)
        # get the value of pid
        pid = ran.pid.values[0]
        # get list of tid (this code uses so many memory)
        tid_arr=df_tracks[df_tracks.pid == pid].tid
        # decrease frequency by 1
        temp = df_track_distr[tid_arr] - 1
        
        if (temp.any() != 0):
            df_track_distr[tid_arr] = temp
            list_1000.append(pid)
            if (len(list_1000) == list_size):
                break

    pid_list.append(list_1000)


#- Predict tracks for a playlist given its 200 random tracks
df_playlists_challenge = pd.DataFrame()
for i in range(len(pid_list)):
    df_temp =  df_playlists[df_playlists.pid.isin(pid_list[i])]
    df_playlists_challenge = pd.concat([df_playlists_challenge,df_temp])

df_playlists_challenge.to_hdf('data/df_playlists_challenge.hdf', key='abc')

print("finish")
# Write a small test
#assert(df_playlists_challenge.pid[:10].isin(pid_list[0]).all())
    
    