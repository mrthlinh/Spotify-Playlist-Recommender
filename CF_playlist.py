#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 22:29:48 2018

@author: bking
"""

import pandas as pd
import numpy as np
from helper import cosine_sim
from scipy.sparse import dok_matrix
from helper import my_evaluation
import argparse
import sys
import pickle
#import time

def RMS(rating,rating_truth):
    '''
        rating: np array
        rating_truth: list
    '''
    rating_truth_ = dok_matrix((len(rating),1), dtype=np.float32)
    rating_truth_[rating_truth] = 1
    
    rating = np.array(rating,ndmin=2).T
    
    diff = rating_truth_ - rating
    rms = np.sqrt(np.mean(np.square(diff)))
    
    return rms
    
def main(argv):
    args = parser.parse_args(argv[1:])
    mode = args.mode    
    K = 500
    path = "data/df_data/"
    
    if mode == '1':
        K = 20
        path = "data/df_data/df_small/"
        
    # Playlist-Song Matrix
    print("Loading data")
    df_ps_train = pd.read_hdf(path+"/df_playlistSong/df_ps_train.hdf")
    df_ps_test = pd.read_hdf(path+"/df_playlistSong/df_ps_test.hdf")
    df_sp_train = pd.read_hdf(path+"/df_playlistSong/df_sp_train.hdf")
    df_ps_test_truth = pd.read_hdf(path+"/df_playlistSong/df_ps_test_truth.hdf")
    
    # Reset the index
    if mode == '1':

        df_ps_train = df_ps_train.set_index(keys="pid")
        df_ps_test = df_ps_test.set_index(keys="pid")
        df_ps_test_truth = df_ps_test_truth.set_index(keys="pid")
          
        
    
    # Get tid list
    tid_list = list(df_sp_train.index)
    
    # get pid list in test set
    pid_list_test = list(df_ps_test.index)
    
    # get pid list in train set
    pid_list_train = list(df_ps_train.index)
    
#    num_tid = len(tid_list)
#    num_pid = len(pid_list_train)
    
    # Create rating matrix -> Load Rating Matrix from dump file
    print("Create rating matrix")
    with open('data/giantMatrix','rb') as f:
        ps_matrix = pickle.load(f)    
#    ps_matrix = dok_matrix((num_pid, num_tid), dtype=np.float32)
#    for pid in pid_list_train:
##        print(pid)
#        tid = df_ps_train.loc[pid,'tid']
#        
#        # Create index
#        index_pid = pid_list_train.index(pid)
#        index_tid = [tid_list.index(t) for t in tid ]
#        
#        ps_matrix[index_pid,index_tid]=1
       
    rms = 0
    record = []
    
    print("Inference")
    for pid in pid_list_test:
        
        # tid vector for current playlist
        vector1 = df_ps_train.loc[pid,'tid']
        
        # get other playlist id
        other_pid = [id for id in pid_list_train if id != pid]
        
        # get tid for other playlist id
        vector2_list = [df_ps_train.loc[i,'tid'] for i in other_pid]
        vector2_length = len(vector2_list)
        
        # Calculate the cosine similarity between vector1 and each of vector2_list
        sim_vector = list(map(cosine_sim,[vector1]*vector2_length,vector2_list))  
        sim_vector = np.array(sim_vector) # [1 x N]
        
        # Normalization
        norm = np.sum(sim_vector)
        
        # Try to save r_list to dense
        index_other_pid = [pid_list_train.index(t) for t in other_pid ]
        ps = ps_matrix[index_other_pid,:]
        
        rating = ps.T.dot(sim_vector)
        rating = rating / norm
        
        rating_truth = df_ps_test_truth.loc[pid,'tid']
        

        
        # Enumerate index and rating
        counter_list = list(enumerate(rating, 0))

        # Sort by rating
        sortedList = sorted(counter_list, key=lambda x:x[1],reverse=True)
        
        # Filter elements in vector 1 - current songs
        sortedList_filter = [tid_list.index(x) for x,_ in sortedList if x not in vector1]
    #    sortedList_filter = [(x,y) for x,y in sortedList if x not in vector1]
           
        add_tid = sortedList_filter[:K-len(vector1)]
        
        new_tid = vector1 + add_tid
        
        record.append(new_tid)
        
        temp = RMS(rating,rating_truth)
        rms += temp
        print("pid: {} \t RMS: {}".format(pid,temp))
     
    print("Root Mean Square: {}".format(rms))  
    
    print("Create new dataframe")
    df_ps_test['tid'] = record
    
    print("Save test data")
    df_ps_test.to_hdf(path+'df_ps_test_complete_CF_playlist.hdf', key='abc')
    
    print("Evaluation")
    result = my_evaluation(df_ps_test,df_ps_test_truth)
    print(result.aggregate_metric())
    

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default= '0', type=str, help='Mode Test On/Off') 

    print(
    '''
    ---------------------- Algorithm ---------------------------
    For each playlist in test set
    Find similarity between current playlist and other playlists
    Form a rating matrix between playlist and songs
    Fill out value of rating
    ------------------------------------------------------------
    '''
        )
 
    
    main(sys.argv)
#    print(result)


 
