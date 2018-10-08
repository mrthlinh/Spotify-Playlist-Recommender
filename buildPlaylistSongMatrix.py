#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 22:57:56 2018

@author: bking
"""

import pandas as pd
#import argparse
import sys
from helper import alertError,alertFinishJob
import gc
import time
    
def main(argv):
    
    print("Reading Data")
    df_train = pd.read_hdf('data/df_data/df_train_new.hdf')
    df_test = pd.read_hdf('data/df_data/df_test_new.hdf')
    df_test_truth = pd.read_hdf('data/df_data/df_test_truth_new.hdf')
    
    df_tracks = pd.read_hdf('data/df_data/df_tracks.hdf')

#    df_train = pd.read_hdf('data/df_data/df_train.hdf')
#    df_test = pd.read_hdf('data/df_data/df_test.hdf')
#    df_test_truth = pd.read_hdf('data/df_data/df_test_truth.hdf')
#    
#    df_truth = 
   
    # Build playlist-song matrix
    print("Build playlist-song matrix for train set")
    tid = df_train.groupby(by='pid')['tid'].apply(list)
    pos = df_train.groupby(by='pid')['pos'].apply(list)
    df_ps_train = pd.concat([tid,pos],axis=1)
      
    
    print("Build playlist-song matrix for test set incomplete")
    tid = df_test.groupby(by='pid')['tid'].apply(list)
    pos = df_test.groupby(by='pid')['pos'].apply(list)
    df_ps_test = pd.concat([tid,pos],axis=1)
    
    
    print("Build playlist-song matrix for test set truth")
    tid = df_test_truth.groupby(by='pid')['tid'].apply(list)
    pos = df_test_truth.groupby(by='pid')['pos'].apply(list)
    df_ps_test_truth = pd.concat([tid,pos],axis=1)
    
    print("Build Complate playlist-song matrix")
    tid = df_tracks.groupby(by='pid')['tid'].apply(list)
    pos = df_tracks.groupby(by='pid')['pos'].apply(list)
    df_ps_complete = pd.concat([tid,pos],axis=1)    
    
    print("save matrix Playlist-Songs")
    df_ps_train.to_hdf("data/df_data/df_playlistSong/df_ps_train_new.hdf",key='abc')
    df_ps_test.to_hdf("data/df_data/df_playlistSong/df_ps_test_new.hdf",key='abc')
    df_ps_test_truth.to_hdf("data/df_data/df_playlistSong/df_ps_test_truth_new.hdf",key='abc')
    df_ps_complete.to_hdf("data/df_data/df_playlistSong/df_ps_complete_new.hdf",key='abc')
    
    del df_ps_train
    del df_ps_test
    gc.collect()
        
if __name__ =="__main__":
#    parser = argparse.ArgumentParser()
#    parser.add_argument('--sim_metric', default='cosine', type=str, help='Similarity Metrics')
#    main(sys.argv)
    start = time.time()
    main(sys.argv)
    print("Time taken = {0:.5f}".format(time.time() - start))
#    try:
#        main(sys.argv)
#        alertFinishJob("Done")
#    except Exception as e:
#        alertError(str(e))
