#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 00:06:19 2018

@author: bking
"""
import sys
import pandas as pd
from helper import findKRelevant_simple,my_evaluation,alertFinishJob,alertError
import time
import argparse
import sys
from multiprocessing import Pool
# https://www.geeksforgeeks.org/multithreading-python-set-1/

K = 10
#proc = 8
MAX_tid = 500


print("Load Playlist-Song matrix")    
df_ps_train = pd.read_hdf('data/df_data/df_playlistSong/df_ps_train.hdf')
df_ps_test = pd.read_hdf('data/df_data/df_playlistSong/df_ps_test.hdf')
df_ps_test_truth = pd.read_hdf('data/df_data/df_playlistSong/df_ps_test_truth.hdf')


def my_function(data):
    pid = data[0]
    current_list = data[1]
    
    start = time.time()
    print("Pid: ",pid)

    topK_pid = findKRelevant_simple(pid,df_ps_train,K)
    
    n = 0
    
    while(1):
        top_pid = topK_pid[n]
        
        add_tid_list = df_ps_train.loc[top_pid].tid
                
        # Form new list
        new_tid_list = current_list + add_tid_list
            
        # Check number of songs and Add to data for prediction
        total_song = len(new_tid_list)
#            print("n: {}\t total_song: {}".format(n,total_song))
        if (total_song > MAX_tid):
            new_tid_list = new_tid_list[:MAX_tid]            
            # Add
            current_list = new_tid_list
            break
        else:
            current_list = new_tid_list
            
        n += 1
        if (n>=K):
            break
        
#    SIZE = SIZE - 1
    print("Time taken = {0:.5f}".format(time.time() - start))
    
    return [pid,current_list]
    
    

def main(argv):
    args = parser.parse_args(argv[1:])
    proc = int(args.proc)
    
    
    pid_list_pred = list(df_ps_test.index)    
    current_list = list(df_ps_test.loc[pid_list_pred].tid)
    
    # Multiprocessing
    with Pool(proc) as p:
        new_list = p.map(my_function, zip(pid_list_pred,current_list))
    
    #- Build similarity matrix between playlists (cosine, euclidean, Pearson correlation)
    #- For each playlist Px:
    #    n = 1
    #    While total_track is not 500:
    #      Find n-th most relevant playlist of Px, called Pr
    #      Add K (or all) songs in Pr to Px
    #      Increment n by 1
    
    df_ps_pred = pd.DataFrame.from_records(new_list,columns=['pid','tid'])
    df_ps_pred = df_ps_pred.set_index('pid')
    
    result = my_evaluation(df_ps_pred,df_ps_test_truth)
    
    return result.aggregate_metric()
    
if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--proc', default='16', type=str, help='Number of proccessor') 
    start = time.time()
#    main(sys.argv)
    
#    print("Time taken = {0:.5f}".format(time.time()-start))
    
#    message2 = " -- Result: "+str(result)
#    message = message1 + message2
#    alertFinishJob(message)
    try:
        result = main(sys.argv)
        message1 = "Time taken = {0:.5f}".format(time.time()-start)
        message2 = " -- Result: "+str(result)
        message = message1 + message2
        print(message)
        alertFinishJob(message)
    except Exception as e:
        alertError(str(e))



#result.r_precision(df_ps_pred.loc[pid].tid,df_ps_test_truth.loc[pid].tid,205)
#
#result.ndcg(df_ps_pred.loc[pid].tid,df_ps_test_truth.loc[pid].tid,205)
#
#result.song_clicks(df_ps_pred.loc[pid].tid,df_ps_test_truth.loc[pid].tid,205)
