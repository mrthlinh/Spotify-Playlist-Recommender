#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 20:44:11 2018

@author: bking
"""

import sklearn.preprocessing as pp
import scipy.sparse as sp
from scipy.sparse import dok_matrix,csc_matrix,csr_matrix,vstack
import pickle
import pandas as pd
from multiprocessing import Pool,Value
import time
import argparse
from sklearn.metrics.pairwise import cosine_similarity,paired_cosine_distances


def cosine_similarities_playlist(pid_list_test,ps_matrix):
    ps_matrix_norm = pp.normalize(ps_matrix, axis=1)
    ps_matrix_test = ps_matrix_norm[pid_list_test,:]
    return ps_matrix_test * ps_matrix_norm.T

def cosine_similarities_song(index_tid_list_test,ps_matrix):
    ps_matrix_norm = pp.normalize(ps_matrix, axis=0)
    # tid_list_test must be index
    ps_matrix_test = ps_matrix_norm[:,index_tid_list_test]
    return ps_matrix_test.T * ps_matrix_norm

def main():

    args = parser.parse_args()
    mode = args.mode
    
    print(
          '''
          ========================== Build Similarity Matrix between playlists ==============================
          1. Loading Rating Matrix [P x S]
          2. [p x P] = [p x S] * [S x P]
          ===================================================================================================
          '''
          )
            
    print("Load Rating Matrix")
    
    pickle_path = 'data/giantMatrix_new.pickle'
    if mode == 1:
        pickle_path = 'data/giantMatrix_small.pickle'

    with open(pickle_path,'rb') as f:
        ps_matrix = pickle.load(f)      
        # Change to column sparse matrix because it is much faster to get column 12s -> 0.08s
    
    ps_matrix_col = ps_matrix.tocsc() 
    ps_matrix_row = ps_matrix.tocsr()
    
    print("Loading data")
    
    path = "data/df_data/"

    if mode == 1:
        path = "data/df_data/df_small/"
    
    df_ps_test_truth = pd.read_hdf(path+"df_playlistSong/df_ps_test_truth_new.hdf")
    df_sp_train = pd.read_hdf(path+"df_playlistSong/df_sp_train_new.hdf")
    df_sp_test = pd.read_hdf(path+"df_playlistSong/df_sp_test_new.hdf")
    
    pid_list_test = list(df_ps_test_truth.index) 
    
    print("Build cosine similarity playlists")
    ps_sim_playlist = cosine_similarities_playlist(pid_list_test,ps_matrix_row)

    print("Save similarity matrix playlist")
    
    out_filename = "cosineSimMatrix_playlist.pickle"
    
    with open("data/"+out_filename, 'wb') as f:
        pickle.dump(ps_sim_playlist, f,protocol=4)  
        
    # Get tid list
    tid_list = list(df_sp_train.index)
    num_tid = len(tid_list)
    dict_index = {k:v for k,v in zip(tid_list,range(0,num_tid))}
    tid_list_test = list(df_sp_test.index)
    index_tid_list_test = [dict_index.get(i) for i in tid_list_test]
    
    print("Build cosine similarity songs")
    ps_sim_song = cosine_similarities_song(index_tid_list_test,ps_matrix_col)    
    
    print("Save similarity matrix song")

    out_filename = "cosineSimMatrix_song.pickle"
    
    with open("data/"+out_filename, 'wb') as f:
        pickle.dump(ps_sim_song, f,protocol = 4)      
    
if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default= 0, type=int, help='Mode Test On/Off') 
    #parser.add_argument('--proc', default='16', type=str, help='Number of proccessor') 
    
    start = time.time()
    main()
    print("Total time taken = {0:.5f}".format(time.time() - start))    
    
    
    
    
    