#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 08:34:00 2018

@author: bking
"""

import pandas as pd
from tqdm import tqdm
from helper import alertError,alertFinishJob

#df_playlists_test = pd.read_hdf('data/df_data/challenge_set/df_playlists_test.hdf')
#df_playlists_test_info = pd.read_hdf('data/df_playlists_test_info.hdf')
def main():
    
    # Read data
    df_unique_tracks = pd.read_hdf('data/df_data/df_tracks_info.hdf')
    df_tracks = pd.read_hdf('data/df_data/df_tracks.hdf')
    df_playlists = pd.read_hdf('data/df_data/df_playlists_info.hdf')    
    
    
    # Pandas table sorted by 
    df_temp = df_tracks.copy()
    df_temp['count'] = 1
    
    # Ensure any track must appear in training
    df_track_distr = df_temp.groupby(['tid'])['count'].sum().sort_values(ascending=False)
    del df_temp
    
    #list_size = 5
    #criteria_list = [200,100]
    
#    list_size = 1000
#    criteria_list = [200,200,100,100,50,50,25,25,10,5]

    list_size = 100
    criteria_list = [200,200,100,100,50,50,25,25,10,5]
    
    pid_list = []
    #pid_unfold_list=[]
    df_temp = df_playlists.loc[:,['num_tracks','pid']]
    df_temp = df_temp.set_index('pid')
    for criteria in tqdm(criteria_list):
        #criteria = 200
        
        # filter playlists that have more than criteria = 100 tracks 
        df_filter = df_temp[df_temp.num_tracks > criteria]
        
        # Create an empty list to contain the pid
        
        list_1000 = []
        while (True):
            # randomly pick 1 pid in df_filter
            ran = df_filter.sample(n=1)
            # get the value of pid
            pid = ran.index[0]
    #        pid = ran.pid.values[0]
                    
            # get list of tid (this code uses so many memory)
            tid_arr = df_tracks[df_tracks.pid == pid].tid
            # decrease frequency by 1
            temp = df_track_distr[tid_arr] - 1
            
            if (temp.any() != 0):
                df_track_distr[tid_arr] = temp
                list_1000.append(pid)
                
                df_filter = df_filter.drop(pid)
                
                if (len(list_1000) == list_size):
                    break
                
        pid_list.append(list_1000)
        df_temp = df_temp.drop(list_1000)
    
    # Assemble challenge set
    df_playlists_challenge = pd.DataFrame()
    df_tracks_challenge = pd.DataFrame()
    
    for i in tqdm(range(len(pid_list))):
        df_temp =  df_playlists[df_playlists.pid.isin(pid_list[i])]
        df_playlists_challenge = pd.concat([df_playlists_challenge,df_temp])
        
        df_temp_ =  df_tracks[df_tracks.pid.isin(pid_list[i])]
        df_tracks_challenge = pd.concat([df_tracks_challenge,df_temp_])
    
    # Assemble training set
    df_playlists_training = df_playlists[~df_playlists.pid.isin(df_playlists_challenge.pid)]
    df_tracks_training= df_tracks[~df_tracks.pid.isin(df_playlists_challenge.pid)]
    
    # Make incomplete playlists
    df_tracks_challenge_incomplete = pd.DataFrame()
    
    # Predict tracks for a playlist given 200 random tracks
    index = 0
    for pid in pid_list[index]:
        df_temp = df_tracks_challenge[df_tracks_challenge.pid == pid].sample(n=criteria_list[index])
        df_tracks_challenge_incomplete = pd.concat([df_tracks_challenge_incomplete,df_temp])
        
        # Write some test here
        assert (df_temp.shape[0] < df_tracks_challenge[df_tracks_challenge.pid == pid].shape[0])
        assert (df_temp.pos.isin(df_tracks_challenge[df_tracks_challenge.pid == pid].pos).all())
    
    
    # Predict tracks for a playlist given first 200 tracks
    index = 1
    for pid in pid_list[index]:
        df_temp = df_tracks_challenge[df_tracks_challenge.pid == pid].head(criteria_list[index])
        df_tracks_challenge_incomplete = pd.concat([df_tracks_challenge_incomplete,df_temp])
        
        # Write some test here
        assert (df_temp.shape[0] < df_tracks_challenge[df_tracks_challenge.pid == pid].shape[0])
        assert (df_temp.pos.isin(df_tracks_challenge[df_tracks_challenge.pid == pid].pos).all())
    
    
    # Predict tracks for a playlist given 100 random tracks
    index = 2
    for pid in pid_list[index]:
        df_temp = df_tracks_challenge[df_tracks_challenge.pid == pid].sample(n=criteria_list[index])
        df_tracks_challenge_incomplete = pd.concat([df_tracks_challenge_incomplete,df_temp])
        
        # Write some test here
        assert (df_temp.shape[0] < df_tracks_challenge[df_tracks_challenge.pid == pid].shape[0])
        assert (df_temp.pos.isin(df_tracks_challenge[df_tracks_challenge.pid == pid].pos).all())
    
    
    # Predict tracks for a playlist given first 100 tracks
    index = 3
    for pid in pid_list[index]:
        df_temp = df_tracks_challenge[df_tracks_challenge.pid == pid].head(criteria_list[index])
        df_tracks_challenge_incomplete = pd.concat([df_tracks_challenge_incomplete,df_temp])
        
        # Write some test here
        assert (df_temp.shape[0] < df_tracks_challenge[df_tracks_challenge.pid == pid].shape[0])
        assert (df_temp.pos.isin(df_tracks_challenge[df_tracks_challenge.pid == pid].pos).all())
    
    
    # Predict tracks for a playlist given 50 random tracks
    index = 4
    for pid in pid_list[index]:
        df_temp = df_tracks_challenge[df_tracks_challenge.pid == pid].sample(n=criteria_list[index])
        df_tracks_challenge_incomplete = pd.concat([df_tracks_challenge_incomplete,df_temp])
        
        # Write some test here
        assert (df_temp.shape[0] < df_tracks_challenge[df_tracks_challenge.pid == pid].shape[0])
        assert (df_temp.pos.isin(df_tracks_challenge[df_tracks_challenge.pid == pid].pos).all())
    
    
    # Predict tracks for a playlist given first 50 tracks
    index = 5
    for pid in pid_list[index]:
        df_temp = df_tracks_challenge[df_tracks_challenge.pid == pid].head(criteria_list[index])
        df_tracks_challenge_incomplete = pd.concat([df_tracks_challenge_incomplete,df_temp])
        
        # Write some test here
        assert (df_temp.shape[0] < df_tracks_challenge[df_tracks_challenge.pid == pid].shape[0])
        assert (df_temp.pos.isin(df_tracks_challenge[df_tracks_challenge.pid == pid].pos).all())
        
    
    # Predict tracks for a playlist given 25 random tracks
    index = 6
    for pid in pid_list[index]:
        df_temp = df_tracks_challenge[df_tracks_challenge.pid == pid].sample(n=criteria_list[index])
        df_tracks_challenge_incomplete = pd.concat([df_tracks_challenge_incomplete,df_temp])
        
        # Write some test here
        assert (df_temp.shape[0] < df_tracks_challenge[df_tracks_challenge.pid == pid].shape[0])
        assert (df_temp.pos.isin(df_tracks_challenge[df_tracks_challenge.pid == pid].pos).all())
    
    
    # Predict tracks for a playlist given first 25 tracks
    index = 7
    for pid in pid_list[index]:
        df_temp = df_tracks_challenge[df_tracks_challenge.pid == pid].head(criteria_list[index])
        df_tracks_challenge_incomplete = pd.concat([df_tracks_challenge_incomplete,df_temp])
        
        # Write some test here
        assert (df_temp.shape[0] < df_tracks_challenge[df_tracks_challenge.pid == pid].shape[0])
        assert (df_temp.pos.isin(df_tracks_challenge[df_tracks_challenge.pid == pid].pos).all())
        
    # Predict tracks for a playlist given first 10 tracks
    index = 8
    for pid in pid_list[index]:
        df_temp = df_tracks_challenge[df_tracks_challenge.pid == pid].head(criteria_list[index])
        df_tracks_challenge_incomplete = pd.concat([df_tracks_challenge_incomplete,df_temp])
        
        # Write some test here
        assert (df_temp.shape[0] < df_tracks_challenge[df_tracks_challenge.pid == pid].shape[0])
        assert (df_temp.pos.isin(df_tracks_challenge[df_tracks_challenge.pid == pid].pos).all())
        
    # Predict tracks for a playlist given first 5 tracks
    index = 9
    for pid in pid_list[index]:
        df_temp = df_tracks_challenge[df_tracks_challenge.pid == pid].head(criteria_list[index])
        df_tracks_challenge_incomplete = pd.concat([df_tracks_challenge_incomplete,df_temp])
        
        # Write some test here
        assert (df_temp.shape[0] < df_tracks_challenge[df_tracks_challenge.pid == pid].shape[0])
        assert (df_temp.pos.isin(df_tracks_challenge[df_tracks_challenge.pid == pid].pos).all())
        
    # ==========================================================================================   
    
    # Save file
    df_playlists_challenge.to_hdf('data/df_data/my_challenge_set/df_playlists_challenge.hdf', key='abc')
    df_tracks_challenge.to_hdf('data/df_data/my_challenge_set/df_tracks_challenge.hdf', key='abc')
    df_tracks_challenge_incomplete.to_hdf('data/df_data/my_challenge_set/df_tracks_challenge_incomplete.hdf', key='abc')
    
    
#    # Train-Test Split and validation
#        
#    # List of pid list in track incomplete
#    pid_list = df_tracks_challenge_incomplete.pid.unique()
#    
#    # Filter data that have pid in in pid_list
#    df_filter = df_tracks[~df_tracks.pid.isin(pid_list)]
#    
#    # train file = df_filter + df_tracks_incomplete
#    df_train = pd.concat([df_filter,df_tracks_challenge_incomplete])
#    
#    # test file truth
#    df_test = df_tracks_challenge_incomplete.copy()
#    
#    # test file truth
#    df_test_truth = df_tracks_challenge.copy()
#    
#    # Complete file
#    
#    df_train.to_hdf('data/df_data/df_train_new.hdf', key='abc')
#    df_test.to_hdf('data/df_data/df_test_new.hdf', key='abc')
#    df_test_truth.to_hdf('data/df_data/df_test_truth_new.hdf', key='abc')
    
if __name__ =="__main__":
    
    try:
        main()
        alertFinishJob("Done")
    except Exception as e:
        alertError(e)

#print("finish")
# Write a small test
#assert(df_playlists_challenge.pid[:list_size].isin(pid_list[0]).all())
    
    