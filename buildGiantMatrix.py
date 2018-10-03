#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 22:18:58 2018

@author: bking
"""

import pandas as pd
import numpy as np
from helper import cosine_sim
from scipy.sparse import dok_matrix
from helper import my_evaluation
import argparse
import sys
from multiprocessing import Pool
import os
import time
import pickle
print("Loading data")
path = "data/df_data/"

df_ps_train = pd.read_hdf(path+"/df_playlistSong/df_ps_train.hdf")
df_ps_test = pd.read_hdf(path+"/df_playlistSong/df_ps_test.hdf")
df_sp_train = pd.read_hdf(path+"/df_playlistSong/df_sp_train.hdf")
df_ps_test_truth = pd.read_hdf(path+"/df_playlistSong/df_ps_test_truth.hdf")
# Get tid list
tid_list = list(df_sp_train.index)

# get pid list in test set
pid_list_test = list(df_ps_test.index)

# get pid list in train set
pid_list_train = list(df_ps_train.index)

num_tid = len(tid_list)
num_pid = len(pid_list_train)

print("Create rating matrix")
ps_matrix = dok_matrix((num_pid, num_tid), dtype=np.float32)

def assignValue(pid):
#    print(pid)
    tid = df_ps_train.loc[pid,'tid']
    start = time.time()
    # Create index
    index_pid = pid_list_train.index(pid)
    index_tid = [tid_list.index(t) for t in tid ]
    
    ps_matrix[index_pid,index_tid]=1   
    
    print("pid: {} --- ProcessID: {}".format(pid,os.getpid()))
    print("Time taken = {0:.5f}".format(time.time() - start))
    
    
def main(argv):
    args = parser.parse_args(argv[1:])
    proc = int(args.proc)

    

    
#    # Create rating matrix

#    for pid in pid_list_train:
##        print(pid)
#        tid = df_ps_train.loc[pid,'tid']
#        
#        # Create index
#        index_pid = pid_list_train.index(pid)
#        index_tid = [tid_list.index(t) for t in tid ]
#        
#        ps_matrix[index_pid,index_tid]=1
        
    # Multiprocessing
    with Pool(proc) as p:
        p.map(assignValue, pid_list_train)
        
    print("Save file")
    with open('data/giantMatrix.pickle', 'wb') as f:
        pickle.dump(ps_matrix, f)    

if __name__ =="__main__":
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--proc', default='16', type=str, help='Number of proccessor') 
    main(sys.argv)
    print("Total time taken = {0:.5f}".format(time.time() - start))
        
        