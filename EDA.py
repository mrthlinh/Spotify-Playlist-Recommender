# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 10:39:17 2018

@author: mrthl

Basic EDA

Usage: python EDA.py
"""

#import json
import pandas as pd
#import os
#import glob
#import time
import gc #garbage collector
import matplotlib.pyplot as plt
import dask.dataframe as dd



# df_playlists: number of tracks in playlists
# df_playlists_info: number of playlists
# df_tracks: number of unique tracks

# Unique
df_unique_tracks = pd.read_hdf('data/df_data/df_tracks.hdf')
df_tracks = pd.read_hdf('data/df_data/df_playlists.hdf')
df_playlists = pd.read_hdf('data/df_data/df_playlists_info.hdf')

#df_tracks_test = pd.read_hdf('data/df_data/df_playlists_test.hdf')
#df_playlists_test = pd.read_hdf('data/df_data/df_playlists_test_info.hdf')


# Number of ...
print("Number of Playlist: ",df_playlists.shape[0])
print("Number of Tracks: ",df_tracks.shape[0])
print("Number of unique Tracks: ",df_unique_tracks.shape[0])
print("Number of unique Albums: ",len(pd.unique(df_unique_tracks['album_uri'])))
print("Number of unique Artists: ",len(pd.unique(df_unique_tracks['artist_uri'])))

# Distribution of
plt.figure(figsize=(10,8))
df_playlists[['duration_ms','num_albums','num_artists','num_edits','num_followers','num_tracks']].hist()
plt.tight_layout()
plt.show()

# Median of 
print("Median of playlist length: ",df_playlists['duration_ms'].median())
print("Median of number of albums in each playlist: ",df_playlists['num_albums'].median())
print("Median of number of artists in each playlist: ",df_playlists['num_artists'].median())
print("Median of number of edits in each playlist: ",df_playlists['num_artists'].median())
print("Median of number of followers in each playlist: ",df_playlists['num_followers'].median())
print("Median of number of tracks in each playlist: ",df_playlists['num_tracks'].median())

# Top of
# Track Name
# Artist Name
df_top = df_tracks.merge(df_unique_tracks,on='tid',how='outer')

#df_top = df1.merge(df2,on='tid',how='outer')

top_20_track = df_top.groupby(['track_uri']).agg({'track_uri':'count', 'artist_name':'first','track_name':'first'})
top_20_track['full_title'] = top_20_track.apply(lambda x: x[2] + " by " + x[1],axis=1)
top_20_track = top_20_track.sort_values(by=['track_uri'],ascending = False)
top_20_track = top_20_track.head(20)

top_20_track = top_20_track.iloc[:,[0,3]]
top_20_track.columns = ['Top 20','Appearance']
top_20_track = top_20_track.set_index('Tittle')
top_20_track.iloc[::-1].plot(kind = 'barh', title='Top 20 Songs in Spotify Playlist')


top_20_artist= df_top.groupby(['artist_uri']).agg({'artist_uri':'count', 'artist_name':'first'})
top_20_artist = top_20_artist.sort_values(by=['artist_uri'],ascending = False)
top_20_artist = top_20_artist.head(20)

top_20_artist.columns = ['Number of songs','Artist Name']
top_20_artist = top_20_artist.set_index('Artist Name')
top_20_artist.iloc[::-1].plot(kind = 'barh', title='Top 20 Artists in Spotify Playlist')









