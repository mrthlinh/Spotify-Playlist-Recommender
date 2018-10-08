#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 10:47:29 2018

@author: bking
"""

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
------------------ Build Playlist-Song Matrix Ground Truth for testing -----------------------------------

----------------------------------------------------------------------------------------------------------
'''
    )

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default= 0, type=int, help='Mode Test On/Off') 
#parser.add_argument('--proc', default='16', type=str, help='Number of proccessor') 
args = parser.parse_args()
mode = args.mode
#proc = int(args.proc)
def main():
    print("Loading data")
    path = "data/df_data/"
    out_filename = "giantMatrix_truth_new.pickle"
    if mode == 1:
        path = "data/df_data/df_small/"
        out_filename = "giantMatrix_truth_small_new.pickle"
    
   
    df_sp_complete = pd.read_hdf(path+"/df_playlistSong/df_sp_complete_new.hdf")    
    df_ps_test_truth = pd.read_hdf(path+"/df_playlistSong/df_ps_test_truth_new.hdf")
    
    # Get tid list
    tid_list = list(df_sp_complete.index)
    num_tid = len(tid_list)
    
    dict_index = {k:v for k,v in zip(tid_list,range(0,num_tid))}
           
    # get pid list in test set
    pid_list_test = list(df_ps_test_truth.index)
       
    num_pid_test = len(pid_list_test)
    
    print("Create rating matrix")
    ps_matrix = dok_matrix((num_pid_test, num_tid), dtype=np.float32)
#    a = 10
    for k in tqdm(range(num_pid_test)):
#    for k in tqdm(range(100)):
        pid = pid_list_test[k]
#        print(pid)
        tid = df_ps_test_truth.loc[pid,'tid']     
        index_tid = [dict_index.get(i) for i in tid]
        
        ps_matrix[k,index_tid]=1 
    
    
    print("Save file")
    with open("data/"+out_filename, 'wb') as f:
        pickle.dump(ps_matrix, f)    

if __name__ =="__main__":
    start = time.time()
    main()
    print("Total time taken = {0:.5f}".format(time.time() - start))


