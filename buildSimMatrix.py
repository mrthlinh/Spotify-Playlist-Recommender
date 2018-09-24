#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 11:21:52 2018

@author: bking
"""
from multiprocessing import Pool
import time

import pandas as pd
import argparse
import sys
#from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances,pairwise_distances
#from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances,pairwise_distances
from helper import alertError,alertFinishJob
import gc
import numpy as np
from tqdm import tqdm



def cosine_sim_matrix(df):
    pid_list = list(df.index.values)
    length = len(pid_list)
    record = []
    
    for i in tqdm(range(length)):
        p1 = pid_list[i]
        
        for j in range(i+1,length):
            
            p2 = pid_list[j]
#            pair = [p1,p2]
            
            vector1 = df.tid.iloc[i]
            vector2 = df.tid.iloc[j]
            sim = cosine_sim(vector1,vector2)
        
            record.append([p1,p2,sim])        
    df_sim = pd.DataFrame.from_records(record,columns=['pair_1','pair_2','similarity']) 
    return df_sim
#    return 0
    
def cosine_sim(vector1,vector2):
    set_vector1 = set(vector1)
    set_vector2 = set(vector2)
    
    intersect = len(set_vector1.intersection(set_vector2))
    
    length_vector1 = np.sqrt(len(vector1))
    length_vector2 = np.sqrt(len(vector2))
    
    cosine_sim = intersect / (length_vector1 * length_vector2)
    return cosine_sim.round(3)

#def corr_sim(vector1,vector2):
    
def main(argv):
    args = parser.parse_args(argv[1:])
    sim_metric = args.sim_metric
    
    print("Load data")

    df_ps_train = pd.read_hdf('data/df_data/df_playlistSong/df_ps_train.hdf')
#    df_ps_test = pd.read_hdf('data/df_data/df_playlistSong/df_ps_test.hdf')
#    df_ps_test_truth = pd.read_hdf('data/df_data/df_playlistSong/df_ps_test_truth.hdf')
    
#    print("Reading Data")
#    df_train = pd.read_hdf('data/df_data/df_train.hdf')
#    df_test = pd.read_hdf('data/df_data/df_test.hdf')
#    df_train['appearance'] = 1
#    df_ps_train = pivot_table(df_train, index = 'pid', columns = 'tid',values='appearance',fill_value=0)
   
    # Build playlist-song matrix
#    print("Build playlist-song matrix for train set")
#    tid = df_train.groupby(by='pid')['tid'].apply(list)
#    pos = df_train.groupby(by='pid')['pos'].apply(list)
#    df_ps_train = pd.concat([tid,pos],axis=1)
      
    
#    df_test['appearance'] = 1
#    df_ps_test = pivot_table(df_test, index = 'pid', columns = 'tid',values='appearance',fill_value=0)
#    print("Build playlist-song matrix for test set")
#    tid = df_test.groupby(by='pid')['tid'].apply(list)
#    pos = df_test.groupby(by='pid')['pos'].apply(list)
#    df_ps_test = pd.concat([tid,pos],axis=1)
    
    
    if sim_metric == 'cosine':
        print("Cosine Similarity Matrix")
        playlist_similarity = cosine_sim_matrix(df_ps_train)
#        songs_similarity = cosine_sim_matrix(df_ps_test)
#        playlist_similarity = cosine_similarity(df_ps_train)
 
#        songs_similarity = cosine_similarity(df_ps_train.T)
    
#    if sim_metric == 'correlation':
#        playlist_similarity = df_ps_train.T.corr()
#        songs_similarity = df_ps_train.corr()
    
#    if sim_metric == 'euclidean':
#        playlist_similarity = 1-pairwise_distances(df_ps)
#        songs_similarity = 1-pairwise_distances(df_ps.T)
        
#    print("save matrix Playlist-Songs")
#    df_ps_train.to_hdf("data/df_data/df_ps_train.hdf",key='abc')
#    df_ps_test.to_hdf("data/df_data/df_ps_test.hdf",key='abc')
#    
#    del df_ps_train
#    del df_ps_test
#    gc.collect()
    
    print("save similarity matrix")
    playlist_similarity.to_hdf("data/df_data/playlists_sim_"+sim_metric+".hdf",key='abc')
#    songs_similarity.to_hdf("data/df_data/songs_sim_"+sim_metric+".hdf",key='abc') 
    
    del playlist_similarity
    gc.collect()
    
    
#    Test
        
#    df_train = pd.read_hdf('data/df_data/df_tracks.hdf')
#    tracks_small = df_train[df_train.pid.isin(range(100))]
#    tracks_small['appearance'] = 1            
#    df_ps = pivot_table(tracks_small, index = 'pid', columns = 'tid',values='appearance',fill_value=0)
#    
#    playlist_similarity = 1-pairwise_distances(df_ps)
#    songs_similarity = 1-pairwise_distances(df_ps.T)
        
#    


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim_metric', default='cosine', type=str, help='Similarity Metrics')
    main(sys.argv)
#    try:
#        main(sys.argv)
#        alertFinishJob("Done")
#    except Exception as e:
#        alertError(str(e))
