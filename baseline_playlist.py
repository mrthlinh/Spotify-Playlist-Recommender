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
import pickle
#import sys
from multiprocessing import Pool

# https://www.geeksforgeeks.org/multithreading-python-set-1/

parser = argparse.ArgumentParser()
parser.add_argument('--proc', default= 16, type=int, help='Mode Test On/Off') 
args = parser.parse_args()
proc = args.proc

K = 100
#proc = 8
MAX_tid = 500


print("Load Playlist-Song matrix")    
df_ps_train = pd.read_hdf('data/df_data/df_playlistSong/df_ps_train_new.hdf')
df_ps_test = pd.read_hdf('data/df_data/df_playlistSong/df_ps_test_new.hdf')
df_ps_test_truth = pd.read_hdf('data/df_data/df_playlistSong/df_ps_test_truth_new.hdf')

pid_list_pred = list(df_ps_test.index) 

pid_test_length = len(pid_list_pred)
dict_index = {k:v for k,v in zip(range(0,pid_test_length),pid_list_pred)}
    

print("Load Similarity Matrix")
sim_path = 'data/cosineSimMatrix_playlist.pickle'
with open(sim_path,'rb') as f:
    sim_matrix = pickle.load(f)      
    
# Change to column sparse matrix because it is much faster to get column 12s -> 0.08s
sim_matrix = sim_matrix.tocsr()   


def my_function(data):
    
    index_pid = data[0]
    pid = dict_index.get(index_pid)
    current_list = data[1]
    
    start = time.time()
    print("Pid: ",pid)

#    topK_pid = findKRelevant_simple(pid,df_ps_train,K)
    sim_vector = sim_matrix.getrow(index_pid).toarray()[0].tolist()
    
    # Enumerate index and rating
    counter_list = list(enumerate(sim_vector, 0))

    # Sort by rating
    sortedList = sorted(counter_list, key=lambda x:x[1],reverse=True)
    
    topK_pid = [i for i,_ in sortedList[1:K+1]]
    
    n = 0
    
    while(1):
        
        top_pid = topK_pid[n]
        
#        top_pid = dict_index.get(index_top_pid)
        
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
#        print(n)
        n += 1
        print(n)
        if (n == K):
            break
        
#    SIZE = SIZE - 1
    print("Time taken = {0:.5f}".format(time.time() - start))
    
    return [pid,current_list]
    
    

def main():
    
#    index_pid_list_pred = list(range(pid_test_length))
    
    current_list = list(df_ps_test.loc[pid_list_pred].tid)
    
    
    
    # Multiprocessing
    with Pool(proc) as p:
        new_list = p.map(my_function, zip(range(0,pid_test_length),current_list))

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
    
    print(result.aggregate_metric())

    
if __name__ =="__main__":
    start = time.time()
    main()
    print("Time taken = {0:.5f}".format(time.time()-start))
    
#    message2 = " -- Result: "+str(result)
#    message = message1 + message2
#    alertFinishJob(message)
#    try:
#        result = main(sys.argv)
#        message1 = "Time taken = {0:.5f}".format(time.time()-start)
#        message2 = " -- Result: "+str(result)
#        message = message1 + message2
#        print(message)
#        alertFinishJob(message)
#    except Exception as e:
#        print(str(e))
##        alertError(str(e))



