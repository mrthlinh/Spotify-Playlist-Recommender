#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 22:29:48 2018

@author: bking
"""

import pandas as pd
import numpy as np
from helper import cosine_sim
from scipy.sparse import dok_matrix,csc_matrix
from helper import my_evaluation
import argparse
import sys
import pickle
from multiprocessing import Pool,Value
import time

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default= '0', type=str, help='Mode Test On/Off') 
parser.add_argument('--proc', default= 32, type=int, help='Number of proccessor') 
args = parser.parse_args()
mode = args.mode
proc = args.proc


print(
'''
------------------------------------ Algorithm ------------------------------------------
For each playlist in test set
Find similarity between current playlist and other playlists
Form a rating matrix between playlist and songs
Fill out value of rating
    Now: get top K song with highest rating and add to current list -> we take care order
    Next: Set a threshold and pick songs that give 1 -> no order
-----------------------------------------------------------------------------------------
'''
    )
    
pickle_path = 'data/giantMatrix_new.pickle'
if mode == '1':
    pickle_path = 'data/giantMatrix_small.pickle'

print("Load rating matrix")
with open(pickle_path,'rb') as f:
    ps_matrix = pickle.load(f)      

# Change to column sparse matrix because it is much faster to get column 12s -> 0.08s
ps_matrix = ps_matrix.tocsc() 



print("Load Similarity Matrix")
sim_path = 'data/cosineSimMatrix_playlist.pickle'
with open(sim_path,'rb') as f:
    sim_matrix = pickle.load(f)      
    
# Change to column sparse matrix because it is much faster to get column 12s -> 0.08s
sim_matrix = sim_matrix.tocsr()   


K = 500
path = "data/df_data/"

if mode == '1':
    K = 20
    path = "data/df_data/df_small/"

#progress_tid = 0
#progress_tid_start = 0
#num_tid = 0

#progress_tid = Value('i',0)

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

def getRating(data):
    '''
        Rating for a song in a playlist
    '''
    sim_vector = data[0] # P x 1
    tid = data[1]
    ps = ps_matrix.getcol(tid)
    rating = ps.T.dot(sim_vector)
    
    return rating[0]

def main(argv):
    
    # pid list 0 -> 1 000 000
    # tid list -> not in order
    
    # Playlist-Song Matrix
    print("Loading data")
    df_ps_train = pd.read_hdf(path+"/df_playlistSong/df_ps_train_new.hdf")
    df_ps_test = pd.read_hdf(path+"/df_playlistSong/df_ps_test_new.hdf")
    df_sp_train = pd.read_hdf(path+"/df_playlistSong/df_sp_train_new.hdf")
    df_ps_test_truth = pd.read_hdf(path+"/df_playlistSong/df_ps_test_truth_new.hdf")
    
    # Reset the index
    if mode == '1':
        df_ps_train = df_ps_train.set_index(keys="pid")
        df_ps_test = df_ps_test.set_index(keys="pid")
        df_ps_test_truth = df_ps_test_truth.set_index(keys="pid")
    
    print("Build Playlist and Song List")
    # Get tid list
    tid_list = list(df_sp_train.index)
    num_tid = len(tid_list)   
    tid_index = list(np.arange(0,num_tid))
    
    # get pid list in train set
    pid_list = list(df_ps_train.index)  
    
    # get pid list in test set
    pid_list_test = list(df_ps_test.index)
    
#    num_pid = len(pid_list)
      
    rms = 0
    record = []
    
    print("Inference")
#    for pid in pid_list_test:
    for i in range(len(pid_list_test)):
        
        pid = pid_list_test[i]
        start = time.time()
#        print (pid,end='')
        
        # tid vector for current playlist
#        print(" -- Get vectors",end='')
        vector1 = df_ps_test.loc[pid,'tid']
        
#        # get other playlist id
#        other_pid = pid_list.copy()
#        other_pid.remove(pid)
#        
#        # get tid vectors for other playlist id
#        vector2_list = [df_ps_train.loc[i,'tid'] for i in other_pid]
#        vector2_length = len(vector2_list)
#        
#        vector2_list = [df_ps_train.loc[i,'tid'] for i in pid_list]
#        vector2_length = len(vector2_list)
        
        
        # Calculate the cosine similarity between vector1 and each of vector2_list
#        print(" -- Calculate the similarity",end='')
        
        
        
#        sim_vector = list(map(cosine_sim,[vector1]*vector2_length,vector2_list))  
#        sim_vector = np.array(sim_vector) # [1 x N]
        
#        proc = 32
#        with Pool(proc) as p:
#            sim_vector = p.map(cosine_sim, zip([vector1]*vector2_length,vector2_list))
#        sim_vector = np.array(sim_vector) # [1 x N]
        
#        sim_vector = sim_matrix[pid_list_test,]
        sim_vector_ = sim_matrix[i,:]
#        sim_vector_array = sim_vector_.toarray()[0]        
        norm = np.sum(sim_vector_)
        
        
        # Normalization
#        norm = np.sum(sim_vector)
        
        
#        print(" -- Get rating value from giant matrix")

        
#        start = time.time()
#        with Pool(proc) as p:           
#            rating = p.map(getRating, zip([sim_vector_array] * num_tid ,tid_index))
#        print("Time taken = {0:.5f}".format(time.time() - start))
        
        # Batch procrssing
#        start = time.time()
        rating = sim_vector_.dot(ps_matrix)
#        print("Time taken = {0:.5f}".format(time.time() - start))        
        
        rating = rating / norm
        
        rating_array = rating.toarray()[0]
        rating_truth = df_ps_test_truth.loc[pid,'tid']
        
        # Enumerate index and rating
        counter_list = list(enumerate(rating_array, 0))
        
        # Filter song of vector 1 from counter_list
        counter_list_filter = [(x,y) for x,y in counter_list if x not in vector1]
    
        # Sort by rating
        sortedList = sorted(counter_list_filter, key=lambda x:x[1],reverse=True)
        
           
        add_tid = [i for i,_ in sortedList[:(K-len(vector1))]]
        
        new_tid = vector1 + add_tid
        
        record.append(new_tid)
        
        temp = RMS(rating_array,rating_truth)
        rms += temp
        print("pid: {} \t RMS: {}".format(pid,temp),end='')
        print("\t Time taken = {0:.5f}".format(time.time() - start))
     
    print("Root Mean Square: {}".format(rms))  
    
    print("Create new dataframe")
    df_ps_test['tid'] = record
    
    print("Save test data")
    df_ps_test.to_hdf(path+'df_ps_test_complete_CF_playlist.hdf', key='abc')
    
    print("Evaluation")
    result = my_evaluation(df_ps_test,df_ps_test_truth)
    print(result.aggregate_metric())
    

if __name__ =="__main__":


    start = time.time()
    main(sys.argv)
    print("Total time taken = {0:.5f}".format(time.time() - start))
