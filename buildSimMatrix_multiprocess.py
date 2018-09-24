#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 01:16:13 2018

@author: bking
"""

from multiprocessing import Pool
import time
import numpy as np
import pandas as pd
import argparse
import sys
from tqdm import tqdm
def cosine_sim(vector):
    set_vector1 = set(vector[0])
    set_vector2 = set(vector[1])
    
    intersect = len(set_vector1.intersection(set_vector2))
    
    length_vector1 = np.sqrt(len(vector[0]))
    length_vector2 = np.sqrt(len(vector[1]))
    
    cosine_sim = intersect / (length_vector1 * length_vector2)
    return cosine_sim.round(3)


       
def cosine_sim_matrix(df,proc):
    list_p1 = list(df.index.values)
    length = len(list_p1)
#    list_p2 = [[j for j in list_p1[i+1:]] for i in range(length-1)]
    
    col_p1 = []
    col_p2 = []
    col_sim = []
    for i in tqdm(range(length-1)):
#        print(i,end=" :")
#        start = time.time()
        
        p2 = [j for j in list_p1[i+1:]]
        p1 = [list_p1[i]]*len(p2)
        
#        print(p1)
#        print(p2)
         
        vector1 = list(df.tid.loc[p1])
        vector2 = list(df.tid.loc[p2])

        with Pool(proc) as p:
            sim = p.map(cosine_sim, zip(vector1,vector2))
#            sim = p.map(cosine_sim, zip([1,2,3,4],[1,2,7,8,9]))
        
        col_p1.append(p1)
        col_p2.append(p2)
        col_sim.append(sim)
        
#        print("Time taken = {0:.5f}".format(time.time() - start))
        
    df_sim = pd.DataFrame({'pid_1':col_p1, 'pid_2':col_p2, 'similarity':sim})
    return df_sim

def main(argv):
    args = parser.parse_args(argv[1:])
    proc = int(args.proc)
    
    print("Number of proccessor using: ",proc)
    print("Load data")
    df_ps_train = pd.read_hdf('data/df_data/df_playlistSong/df_ps_train.hdf')
#    df = pd.read_hdf('data/df_data/df_playlistSong/df_ps_test.hdf')
    print("Build Sim matrix")
    df_sim = cosine_sim_matrix(df_ps_train,proc)
    
    print("save similarity matrix")
    df_sim.to_hdf("data/df_data/playlists_sim_cosine.hdf",key='abc')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--proc', default='8', type=str, help='Number of proccessor')
    main(sys.argv)
    
