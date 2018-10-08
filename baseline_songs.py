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
import pickle
import sklearn.preprocessing as pp
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity,paired_cosine_distances
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default= 0, type=int, help='Mode Test On/Off') 
parser.add_argument('--proc', default= 32, type=int, help='Number of proccessor') 
args = parser.parse_args()
mode = args.mode
proc = args.proc


path = "data/df_data/df_playlistSong/"
if mode == 1:
    print("Mode test: ON")
    path = "data/df_data/df_small/df_playlistSong/"
print(path)
print("Load Song-Playlist matrix")    

df_sp_train = pd.read_hdf(path+'df_sp_train_new.hdf')

df_ps_test = pd.read_hdf(path+'df_ps_test_new.hdf')
df_ps_test_truth = pd.read_hdf(path+'df_ps_test_truth_new.hdf')

df_sp_complete = pd.read_hdf(path+"df_sp_complete_new.hdf")

df_sp_test = pd.read_hdf(path+'df_sp_test_new.hdf')

pickle_path = 'data/giantMatrix_new.pickle'
if mode == 1:
    pickle_path = 'data/giantMatrix_small.pickle'

print("Load rating matrix")
with open(pickle_path,'rb') as f:
    ps_matrix = pickle.load(f)      

# Change to column sparse matrix because it is much faster to get column 12s -> 0.08s
ps_matrix = ps_matrix.tocsc() 


print("Load Similarity Matrix")
sim_path = 'data/cosineSimMatrix_song.pickle'
with open(sim_path,'rb') as f:
    sim_matrix = pickle.load(f)      
    
# Change to column sparse matrix because it is much faster to get column 12s -> 0.08s
sim_matrix = sim_matrix.tocsr()  

tid_list = list(df_sp_complete.index)
num_tid = len(tid_list)

dict_index = {k:v for k,v in zip(tid_list,range(0,num_tid))}
dict_index2tid = {k:v for k,v in zip(range(0,num_tid),tid_list)}

#tid_list = list(df_sp_test.index)
#num_tid = len(tid_list)
#
#dict_index = {k:v for k,v in zip(tid_list,range(0,num_tid))}
    
#    # Compute centroid of tid list
#    curr_pid_vector = list(df.loc[curr_song_list].pid)    
##    centroid = computeCentroid(curr_pid_vector)
#    
#    # Filter out songs that already in the list  
#    other_song_vector = df.loc[~(df.index.isin(curr_song_list))].pid
#    
#    tid_list = list(other_song_vector.index)
#    sim = []
#    
#    for v in other_song_vector.values:
#        sim.append(cosine_sim_song([centroid,v]))
#        
#    topK = [x for _,x in sorted(zip(sim,tid_list),reverse=True)]   
#
#    return topK[:K]

def cosine_similarities_song(centroid,ps_matrix):
    ps_matrix_norm = pp.normalize(ps_matrix, axis=0)
    centroid_norm = pp.normalize(centroid, axis=0)
    
    vector_sim = centroid_norm.T * ps_matrix_norm
    
    return vector_sim

def my_function(data):
    current_list_ = data[0]
    K = data[1]
#    start = time.time()
    start = time.time()
#    print("K: ",K)
    
    index_current_list = [dict_index.get(i) for i in current_list_]
    current_vectors = [ps_matrix.getcol(i) for i in index_current_list]

    
    centroid = np.sum(current_vectors) / len(current_vectors)
        
    sim_vector = cosine_similarities_song(centroid,ps_matrix)
    
    index_nonzero = np.sort(sim_vector.nonzero()[1])
    
    # Filter current tid
    index_filter = [i for i in index_nonzero if i not in index_current_list]
    
    value_filter = sim_vector[:,index_filter].toarray()[0]
    
    # convert back index to tid
    tid_list = [dict_index2tid[i] for i in index_filter]
    
    counter_list = list(zip(value_filter, tid_list))

    # Sort by rating
    sortedList = sorted(counter_list, key=lambda x:x[0],reverse=True)
    
    topK_tid = [i for _,i in sortedList[:K]]
    
    new_list = current_list_ + topK_tid
    print("Time taken = {0:.5f}".format(time.time() - start))
    
    return new_list


def main():
    
    pid_list_pred = list(df_ps_test.index)    
    current_list = list(df_ps_test.loc[pid_list_pred].tid)
    
    current_len = [len(i) for i in current_list] 
    
#    total_len = list(df_ps_test_truth.loc[pid_list_pred].tid)
#    total_len = [len(i) for i in total_len] 
    
    K_list = [500 - current_len[i] for i in range(len(current_len))]
     
    # Multiprocessing
    with Pool(proc) as p:
        new_list = p.map(my_function, zip(current_list,K_list))
        
    df_ps_pred = pd.DataFrame({'pid':pid_list_pred})
    df_ps_pred['tid'] = new_list
    
    df_ps_pred = df_ps_pred.set_index(keys='pid')
    
    result = my_evaluation(df_ps_pred,df_ps_test_truth)
    
    print(result.aggregate_metric())

if __name__ =="__main__":

    start = time.time()
    main()
    print("Total Time taken = {0:.5f}".format(time.time() - start))
