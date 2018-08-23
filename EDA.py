# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 10:39:17 2018

@author: mrthl

Basic EDA

Usage: python EDA.py
"""

import json
import pandas as pd
import os
import glob
import time
import gc #garbage collector

gc.enable()

# Unique
df_tracks = pd.read_hdf('data/df_data/df_tracks.hdf')

df_playlists = pd.read_hdf('data/df_data/df_playlists.hdf')
df_playlists_info = pd.read_hdf('data/df_data/df_playlists_info.hdf')
df_playlists_test = pd.read_hdf('data/df_data/df_playlists_test.hdf')
df_playlists_test_info = pd.read_hdf('data/df_data/df_playlists_test_info.hdf')



# Number of ...
print("Number of Playlist: ",df_playlists_info.shape[0])
print("Number of Tracks: ",df_playlists.shape[0])
print("Number of unique Tracks: ",df_tracks.shape[0])
print("Number of unique Albums: ",len(pd.unique(df_tracks['album_uri'])))
print("Number of unique Artists: ",len(pd.unique(df_tracks['artist_uri'])))

# Distribution of
df_playlists_info[['duration_ms','num_albums','num_artists','num_edits','num_followers','num_tracks']].hist()

# Median of 
print("Median of playlist length: ",df_playlists_info['duration_ms'].median())
print("Median of number of albums in each playlist: ",df_playlists_info['num_albums'].median())
print("Median of number of artists in each playlist: ",df_playlists_info['num_artists'].median())
print("Median of number of edits in each playlist: ",df_playlists_info['num_artists'].median())
print("Median of number of followers in each playlist: ",df_playlists_info['num_followers'].median())
print("Median of number of tracks in each playlist: ",df_playlists_info['num_tracks'].median())

# Top of
# Track Name
# Artist Name
df_top = df_playlists.merge(df_tracks,on='tid',how='outer')

b = df_top.iloc[:10000,:]
a = b.groupby(['track_uri']).count().a

df_top['artist_name'].value_counts()[:20]

top_20_track = df_top['track_uri'].value_counts()[:20]

def concat_str(x):
    value = df_tracks.loc[df_tracks['track_uri'] == x,['track_name','artist_name']].values
    return value
#return value[0] + ' by ' + value[1]
top_20_track['track_uri'] = top_20_track.index
a = top_20_track.apply(concat_str)

