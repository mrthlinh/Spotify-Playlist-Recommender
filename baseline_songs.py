#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 09:58:56 2018

@author: bking
"""
import sys
import pandas as pd
from helper import findKRelevant_song,my_evaluation,alertFinishJob,alertError
import time
import argparse
import sys
from multiprocessing import Pool


print("Load Song-Playlist matrix")    
df_sp_train = pd.read_hdf('data/df_data/df_songPlaylist/df_sp_train.hdf')

df_ps_test = pd.read_hdf('data/df_data/df_playlistSong/df_ps_test.hdf')
df_ps_test_truth = pd.read_hdf('data/df_data/df_playlistSong/df_ps_test_truth.hdf')
#
#df_sp_test = pd.read_hdf('data/df_data/df_songPlaylist/df_sp_test.hdf')
#df_sp_test_truth = pd.read_hdf('data/df_data/df_songPlaylist/df_sp_test_truth.hdf')

def my_function(data):
    current_list = data[0]
    K = data[1]
#    start = time.time()
#    start = time.time()
    print("K: ",K)
    
#    findKRelevant_song(curr_song_list,df,K):
    topK_song = findKRelevant_song(current_list,df_sp_train,K)
#    print("Time taken = {0:.5f}".format(time.time() - start))
    return topK_song

def main(argv):
    args = parser.parse_args(argv[1:])
    proc = int(args.proc)
      
    pid_list_pred = list(df_ps_test.index)    
    current_list = list(df_ps_test.loc[pid_list_pred].tid)
    
    current_len = [len(i) for i in current_list] 
    
#    total_len = list(df_ps_test_truth.loc[pid_list_pred].tid)
#    total_len = [len(i) for i in total_len] 
#    
    K_list = [500 - current_len[i] for i in range(len(current_len))]
  
    
    # Multiprocessing
    with Pool(proc) as p:
        new_list = p.map(my_function, zip(current_list,K_list))
    
    
    df_ps_pred = pd.DataFrame({'pid':pid_list_pred})
    df_ps_pred['tid'] = new_list
    
    
    result = my_evaluation(df_ps_pred,df_ps_test_truth)
    
    return result.aggregate_metric()

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--proc', default='16', type=str, help='Number of proccessor') 
#    main(sys.argv)
    start = time.time()
    try:
        result = main(sys.argv)
        message1 = "Time taken = {0:.5f}".format(time.time()-start)
        message2 = " -- Result: "+str(result)
        message = message1 + message2
        print(message)
        alertFinishJob(message)
    except Exception as e:
        alertError(str(e))

