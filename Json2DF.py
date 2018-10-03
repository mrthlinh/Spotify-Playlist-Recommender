# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 16:06:47 2018
@author: mrthl

Create dataframe from JSON

Usage: python Json2DF.py
"""

import os
import json
import pandas as pd
import time
from tqdm import tqdm
import gc
def create_df_data(path):
    
    playlist_col = ['collaborative', 'duration_ms', 'modified_at', 
                'name', 'num_albums', 'num_artists', 'num_edits',
                'num_followers', 'num_tracks', 'pid']
    tracks_col = ['album_name', 'album_uri', 'artist_name', 'artist_uri', 
                  'duration_ms', 'track_name', 'track_uri'] 
    playlist_test_col = ['name', 'num_holdouts', 'num_samples', 'num_tracks', 'pid']
    
    filenames = os.listdir(path+'/mpd/data')
    
    data_playlists = []
    data_tracks = []
    playlists = []

    tracks = set()
    
    total_time = 0
    
    print("Reading the dataset")
    for filename in tqdm(filenames):
        start_time = time.time()  
        fullpath = os.sep.join((path+'/mpd/data/', filename))
        f = open(fullpath)
        js = f.read()
        f.close()

        mpd_slice = json.loads(js)

        for playlist in mpd_slice['playlists']:
            data_playlists.append([playlist[col] for col in playlist_col])
            for track in playlist['tracks']:
                playlists.append([playlist['pid'], track['track_uri'], track['pos']])
                if track['track_uri'] not in tracks:
                    data_tracks.append([track[col] for col in tracks_col])
                    tracks.add(track['track_uri'])
        duration = time.time() - start_time
        total_time += duration
#        print("Time elapsed: ",duration)
    print("Total time elapsed: ",total_time)
    gc.collect()
    print("Reading the challenge dataset")
    f = open(path+'/challenge/challenge_set.json')
    js = f.read()
    f.close()
    mpd_slice = json.loads(js)

    data_playlists_test = []
    playlists_test = []

    for playlist in tqdm(mpd_slice['playlists']):
        data_playlists_test.append([playlist.get(col, '') for col in playlist_test_col])
        for track in playlist['tracks']:
            playlists_test.append([playlist['pid'], track['track_uri'], track['pos']])
            if track['track_uri'] not in tracks:
                data_tracks.append([track[col] for col in tracks_col])
                tracks.add(track['track_uri'])
                
    df_playlists_info = pd.DataFrame(data_playlists, columns=playlist_col)
    df_playlists_info['collaborative'] = df_playlists_info['collaborative'].map({'false': False, 'true': True})

    df_tracks = pd.DataFrame(data_tracks, columns=tracks_col)
    df_tracks['tid'] = df_tracks.index

    track_uri2tid = df_tracks.set_index('track_uri').tid

#    df_playlists = pd.DataFrame.from_records(playlists, columns=['pid', 'tid', 'pos'])
    df_playlists = pd.DataFrame.from_records(playlists, columns=['pid', 'tid', 'pos'])
    df_playlists.tid = df_playlists.tid.map(track_uri2tid)

    df_playlists_test_info = pd.DataFrame.from_records(data_playlists_test, columns=playlist_test_col)

    df_playlists_test = pd.DataFrame.from_records(playlists_test, columns=['pid', 'tid', 'pos'])
    df_playlists_test.tid = df_playlists_test.tid.map(track_uri2tid)

    df_tracks.to_hdf(path+'/df_data/df_tracks_info.hdf', key='abc')
    df_playlists.to_hdf(path+'/df_data/df_tracks.hdf', key='abc')
    df_playlists_info.to_hdf(path+'/df_data/df_playlists_info.hdf', key='abc')
    df_playlists_test.to_hdf(path+'/df_data/challenge_set/df_tracks_test.hdf', key='abc')
    df_playlists_test_info.to_hdf(path+'/df_data/challenge_set/df_playlists_test_info.hdf', key='abc')
    
if __name__ == "__main__":
    print(__doc__)
    path ="data"
    # Check existence of folder df_data
    list_dir = os.listdir(path)
    if ("df_data" not in list_dir):
        print("Create df_data folder")
        os.makedirs(path+"/df_data")
        os.makedirs(path+"/df_data/challenge_set")

    # call main function      
    create_df_data(path)