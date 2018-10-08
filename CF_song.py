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
from collections import defaultdict
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

def transform(row):
    pid = row[0]
    tid_list = row[1]
    new_dict = {key: pid for key in tid_list}
    return new_dict
   
def main(argv):
    args = parser.parse_args(argv[1:])
    mode = args.mode    
    K = 500
    path = "data/df_data/"
    THRESHOLD = 0.2
    if mode == '1':
        K = 20
        path = "data/df_data/df_small/"
        
    # Playlist-Song Matrix
    print("Loading data")
    df_sp_train = pd.read_hdf(path+"/df_playlistSong/df_sp_train.hdf")
    df_sp_test = pd.read_hdf(path+"/df_playlistSong/df_sp_test.hdf")    
    df_sp_test_truth = pd.read_hdf(path+"/df_playlistSong/df_sp_test_truth.hdf")
    df_ps_train = pd.read_hdf(path+"/df_playlistSong/df_ps_train.hdf")
    df_ps_test = pd.read_hdf(path+"/df_playlistSong/df_ps_test.hdf")
    df_ps_test_truth = pd.read_hdf(path+"/df_playlistSong/df_ps_test_truth.hdf")
#    df_sp_train = pd.read_hdf(path+"/df_playlistSong/df_sp_train.hdf")
    
    # Reset the index
    if mode == '1':

        df_ps_train = df_ps_train.set_index(keys="pid")
        df_ps_test = df_ps_test.set_index(keys="pid")
        df_ps_test_truth = df_ps_test_truth.set_index(keys="pid")
          
        
    
    # Get tid list
    tid_list = list(df_sp_train.index)
    
    # get tid list in test set
    tid_list_test = list(df_sp_test.index)
    
    # get tid list in test set
    pid_list_test = list(df_ps_test.index)
    
    # get pid list in train set
    pid_list = list(df_ps_train.index)
    
    num_tid = len(tid_list)
    num_pid = len(pid_list)
    
    # Create rating matrix -> Load Rating Matrix from dump file
    print("Create rating matrix")
#    with open('data/giantMatrix.pickle','rb') as f:
#        ps_matrix = pickle.load(f)    
#        
    # Delete this part when Giant Matrix is ready
    ps_matrix = dok_matrix((num_pid, num_tid), dtype=np.float32)
    for pid in pid_list:
#        print(pid)
        tid = df_ps_train.loc[pid,'tid']
        
        # Create index
        index_pid = pid_list.index(pid)
        index_tid = [tid_list.index(t) for t in tid ]
        
        ps_matrix[index_pid,index_tid]=1

       
        
    with open('data/giantMatrix_small.pickle', 'wb') as f:
        pickle.dump(ps_matrix, f)    


        
       
    rms = 0
#    record = []
    
    print("Inference")
    for tid in tid_list_test:
        
        # pid vector for current song
        vector1 = df_sp_train.loc[tid,'pid']
        
        # get other song tid
        other_tid = [id for id in tid_list if id != tid]
        
        # get tid for other playlist id
        vector2_list = [df_sp_train.loc[i,'pid'] for i in other_tid]
        vector2_length = len(vector2_list)
        
        # Calculate the cosine similarity between vector1 and each of vector2_list
        sim_vector = list(map(cosine_sim,[vector1]*vector2_length,vector2_list))  
        sim_vector = np.array(sim_vector) # [1 x N]
        
        # Normalization
        norm = np.sum(sim_vector)
        
        # Try to save r_list to dense
        index_other_tid = [tid_list.index(t) for t in other_tid ]
        ps = ps_matrix[:,index_other_tid]
        
        rating = ps.dot(sim_vector)
        rating = rating / norm
        
        rating_truth = df_sp_test_truth.loc[tid,'pid']
        
        ################ These need to be changed  ############################
        
        # If rating is above THRESHOLD set it to 1 and 0 otherwise
        rating_transform = []
        for i in range(len(rating)):
            if rating[i] >= THRESHOLD:
                rating_transform.append(i)
                
        
        # If element of rating_transform is also in pid_list_test, add these tid to current list of songs
        pid_add = [i for i in rating_transform if i in pid_list_test]
        
        if len(pid_add) != 0:
            df_ps_test.loc[pid_add,'tid'] += [tid]
        
        
        
#        # Enumerate index and rating
#        counter_list = list(enumerate(rating, 0))
#
#        # Sort by rating
#        sortedList = sorted(counter_list, key=lambda x:x[1],reverse=True)
#        
#        # Filter elements in vector 1 - current songs
#        sortedList_filter = [pid_list.index(x) for x,_ in sortedList if x not in vector1]
#    #    sortedList_filter = [(x,y) for x,y in sortedList if x not in vector1]
#           
#        add_tid = sortedList_filter[:K-len(vector1)]
#        
#        new_tid = vector1 + add_tid
#        
#        record.append(new_tid)
        
        temp = RMS(rating,rating_truth)
        rms += temp
        print("tid: {} \t RMS: {}".format(tid,temp))
     
    print("Root Mean Square: {}".format(rms))  
    
#    print("Create new dataframe")
#    df_sp_test['pid'] = record
    
    print("Save test data")
    df_ps_test.to_hdf(path+'df_ps_test_complete_CF_song.hdf', key='abc')
    
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
    Find similarity between current songs and other songs
    Form a rating matrix between playlist and songs
    Fill out value of rating
    ------------------------------------------------------------
    '''
        )
 
    
    main(sys.argv)
#    print(result)


 
