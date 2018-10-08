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
from multiprocessing import Pool,Value
import os
#import time

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
    
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default= '0', type=str, help='Mode Test On/Off') 
parser.add_argument('--proc', default='16', type=str, help='Number of proccessor') 
parser.add_argument('--threshold', default='0.5', type=str, help='Threshold') 
args = parser.parse_args()
mode = args.mode
THRESHOLD = float(args.threshold)
proc = int(args.proc)

progress_pid_start = Value('i',0)
progress_pid_finish = Value('i',0)
rms = Value('f',0)

pickle_path = 'data/giantMatrix.pickle'
if mode == '1':
    pickle_path = 'data/giantMatrix_small.pickle'

print("Load rating matrix")
with open(pickle_path,'rb') as f:
    ps_matrix = pickle.load(f)   
    
#K = 500
path = "data/df_data/"
if mode == '1':
#    K = 20
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

print("Build Playlist and Song List")
# Get tid list
tid_list = list(df_sp_train.index)
num_tid = len(tid_list)   
tid_index = list(np.arange(0,num_tid))

# get pid list in train set
pid_list = list(df_ps_train.index)  

# get pid list in test set
pid_list_test = list(df_ps_test.index)


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
    pid = data[2]
        
    ps = ps_matrix.getcol(tid)
    ps_nonzero_index = ps.nonzero()[0]
    
    # For i<= tid, keep the same
    ps_1 = [i for i in ps_nonzero_index if i < pid]
    # For i > tid decrease by 1
    ps_2 = [i-1 for i in ps_nonzero_index if i >= pid]
    # Merge ps_1 and 2
    ps_nonzero_index = ps_1 + ps_2
    
    # Remove element 
    ps_reshape = dok_matrix((len(sim_vector),1), dtype=np.float32)
    ps_reshape[ps_nonzero_index]=1
       
    # 1 x P * P x 1
     
    rating = ps_reshape.T.dot(sim_vector)
#    global progress_tid
#    progress_tid += 1
#    print("tid: {} - Progress: {}".format(tid,progress_tid))
    
    return rating[0]

def iteration(pid):
#    pid = data[0]
    global progress_pid_start
    with progress_pid_start.get_lock():
        progress_pid_start.value += 1        
        print ("start pid: {} -- progress: {}".format(pid,progress_pid_start.value),end='')
    
    # tid vector for current playlist
#    print(" -- Get vectors",end='')
    vector1 = df_ps_test.loc[pid,'tid']
    
    # get other playlist id
    other_pid = pid_list.copy()
    other_pid.remove(pid)
    
    # get tid vectors for other playlist id
    vector2_list = [df_ps_train.loc[i,'tid'] for i in other_pid]
    vector2_length = len(vector2_list)
    
    # Calculate the cosine similarity between vector1 and each of vector2_list
#    print(" -- Calculate the similarity",end='')

    sim_vector = list(map(cosine_sim, zip([vector1]*vector2_length,vector2_list)))
#    with Pool(proc) as p:
#        sim_vector = p.map(cosine_sim, zip([vector1]*vector2_length,vector2_list))
    sim_vector = np.array(sim_vector) # [1 x N]
    
    # Normalization
    norm = np.sum(sim_vector)
    
    # Multi-processing  " 1 x S = 1 x P * P x S" 
    # sim_vector between playlist -> same for every song
    print(" -- Get rating value from giant matrix")
    rating = list(map(getRating, zip([sim_vector] * num_tid ,tid_index,[pid] * num_tid)))
    
    
#    with Pool(proc) as p:
#    #            rating = p.map(getRating, zip(index_other_pid,sim_vector))
#        rating = p.map(getRating, zip([sim_vector] * num_tid ,tid_index,[pid] * num_tid))
            
    rating = rating / norm
    
    rating_truth = df_ps_test_truth.loc[pid,'tid'] # Have size of tid
    
    rating_threshold = np.array([int(r >= THRESHOLD) for r in rating])
    
    
    add_tid = list(rating_threshold.nonzero()[0])
    add_tid_filter = [i for i in add_tid if i not in vector1]
    
    new_tid = vector1 + add_tid_filter
    
#    record.append(new_tid)
    
    temp = RMS(rating,rating_truth)
    
    global rms
    with rms.get_lock():
        rms.value += temp
    
    global progress_pid_finish
    with progress_pid_finish.get_lock():
        progress_pid_finish.value += 1        
        print("finish pid: {} -- progress: {} \t RMS: {}".format(pid,progress_pid_finish.value,rms.value))  
           
    return new_tid

def main():
        
    print("Inference")
    
    with Pool(proc,initargs = (progress_pid_finish,progress_pid_start,rms)) as p:
        record = p.map(iteration,pid_list_test)

    print("Root Mean Square: {}".format(rms.value))  
    
    print("Create new dataframe")
    df_ps_test['tid'] = record
    
    print("Save test data")  
    out_path = "data/df_data/df_result/"
    if (os.path.exists(out_path) == False):
        os.makedirs(out_path)
    filename = out_path+'df_ps_CF_playlist_threshold_'+str(THRESHOLD)+'.hdf'
    df_ps_test.to_hdf(filename, key='abc')
    
    print("Evaluation")
    result = my_evaluation(df_ps_test,df_ps_test_truth)
    print(result.aggregate_metric())
    

#if __name__ =="__main__":
 
main()

