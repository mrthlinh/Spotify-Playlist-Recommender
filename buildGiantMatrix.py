#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 22:18:58 2018

@author: bking
"""

import pandas as pd
import numpy as np
from helper import cosine_sim
from scipy.sparse import dok_matrix,save_npz,csc_matrix
from helper import my_evaluation
import argparse
import sys
from multiprocessing import Pool
import os
import time
import pickle
from tqdm import tqdm
print(
'''
------------------------------------ Build Playlist-Song Matrix ------------------------------------------

----------------------------------------------------------------------------------------------------------
'''
    )

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default= 0, type=int, help='Mode Test On/Off') 
args = parser.parse_args()
mode = args.mode

def main():
    print("Loading data")
    path = "data/df_data/"
    out_filename = "giantMatrix_new.pickle"
    if mode == 1:
        path = "data/df_data/df_small/"
        out_filename = "giantMatrix_small_new.pickle"
    
    
    df_ps_train = pd.read_hdf(path+"/df_playlistSong/df_ps_train_new.hdf")
#    df_ps_test = pd.read_hdf(path+"/df_playlistSong/df_ps_test_new.hdf")
    
#    df_ps_test_truth = pd.read_hdf(path+"/df_playlistSong/df_ps_test_truth_new.hdf")
    
    df_sp_complete = pd.read_hdf(path+"/df_playlistSong/df_sp_complete_new.hdf")
    
    # Get tid list
    tid_list = list(df_sp_complete.index)
    num_tid = len(tid_list)
    
    dict_index = {k:v for k,v in zip(tid_list,range(0,num_tid))}

    
    # get pid list in train set
    pid_list_train = list(df_ps_train.index)
    num_pid = len(pid_list_train)
    
    print("Create rating matrix")
    ps_matrix = dok_matrix((num_pid, num_tid), dtype=np.float32)
    
    del df_sp_complete
    
    for i in tqdm(range(num_pid)):
        pid = pid_list_train[i]
        tid = df_ps_train.loc[pid,'tid']
        index_pid = pid
        
        index_tid = [dict_index.get(i) for i in tid]
        
        ps_matrix[index_pid,index_tid]=1 
    
    
    print("Save file")
    with open("data/"+out_filename, 'wb') as f:
        pickle.dump(ps_matrix, f)    

if __name__ =="__main__":
    start = time.time()
    main()
    print("Total time taken = {0:.5f}".format(time.time() - start))
