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

    
def main(argv):
    
    print("Reading Data")
    df_train = pd.read_hdf('data/df_data/df_train.hdf')
    df_test = pd.read_hdf('data/df_data/df_test.hdf')
    df_test_truth = pd.read_hdf('data/df_data/df_test_truth.hdf')
   
    # Build playlist-song matrix
    print("Build song-playlist matrix for train set")
    pid = df_train.groupby(by='tid')['pid'].apply(list)
    pos = df_train.groupby(by='tid')['pos'].apply(list)
    df_sp_train = pd.concat([pid,pos],axis=1)
      
    
    print("Build song-playlist matrix for test set incomplete")
    pid = df_test.groupby(by='tid')['pid'].apply(list)
    pos = df_test.groupby(by='tid')['pos'].apply(list)
    df_sp_test = pd.concat([pid,pos],axis=1)
    
    
    print("Build song-playlist matrix for test set truth")
    pid = df_test_truth.groupby(by='tid')['pid'].apply(list)
    pos = df_test_truth.groupby(by='tid')['pos'].apply(list)
    df_sp_test_truth = pd.concat([pid,pos],axis=1)
    
    
    print("save matrix song-playlist")
    df_sp_train.to_hdf("data/df_data/df_songPlaylist/df_sp_train.hdf",key='abc')
    df_sp_test.to_hdf("data/df_data/df_songPlaylist/df_sp_test.hdf",key='abc')
    df_sp_test_truth.to_hdf("data/df_data/df_songPlaylist/df_sp_test_truth.hdf",key='abc')
    
        
if __name__ =="__main__":
#    parser = argparse.ArgumentParser()
#    parser.add_argument('--sim_metric', default='cosine', type=str, help='Similarity Metrics')
#    main(sys.argv)
    try:
        main(sys.argv)
        alertFinishJob("Done")
    except Exception as e:
        alertError(str(e))
