#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 17:22:00 2018

@author: bking
"""
import pandas as pd
df_playlists_challenge = pd.read_hdf('data/df_data/my_challenge_set/df_playlists_challenge.hdf', key='abc')

df_tracks_challenge = pd.read_hdf('data/df_data/my_challenge_set/df_tracks_challenge.hdf', key='abc')
df_tracks_challenge_incomplete = pd.read_hdf('data/df_data/my_challenge_set/df_tracks_challenge_incomplete.hdf', key='abc')
    
df_tracks = pd.read_hdf('data/df_data/df_tracks.hdf')

drop_index = list(df_tracks_challenge_incomplete.index)
df_filter  = df_tracks.drop(drop_index)

assert df_filter.shape[0] + len(drop_index) == df_tracks.shape[0]

# train file = df_filter + df_tracks_incomplete
df_train = pd.concat([df_filter,df_tracks_challenge_incomplete])

print("Check the size of pid and tid")
num_tid = len(df_tracks.tid.unique())
num_pid = len(df_tracks.pid.unique())

assert df_train.tid.unique().shape[0] == df_tracks.tid.unique().shape[0] 
assert df_train.pid.unique().shape[0] == df_tracks.pid.unique().shape[0] 

# test file truth
df_test = df_tracks_challenge_incomplete.copy()

# test file truth
df_test_truth = df_tracks_challenge.copy()

df_train.to_hdf('data/df_data/df_train_new.hdf', key='abc')
df_test.to_hdf('data/df_data/df_test_new.hdf', key='abc')
df_test_truth.to_hdf('data/df_data/df_test_truth_new.hdf', key='abc')
    