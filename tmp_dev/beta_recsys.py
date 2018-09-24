#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 09:54:59 2018

@author: bking
"""

import pandas as pd
import numpy as np


df_test = pd.read_hdf('data/df_data/df_test.hdf')
df_train = pd.read_hdf('data/df_data/df_train.hdf')
# Build playlist-song matrix
tid = df_train.groupby(by='pid')['tid'].apply(list)
pos = df_train.groupby(by='pid')['pos'].apply(list)
df_ps_train = pd.concat([tid,pos],axis=1)

# Build playlist-song matrix
tid = df_test.groupby(by='pid')['tid'].apply(list)
pos = df_test.groupby(by='pid')['pos'].apply(list)
df_ps_test = pd.concat([tid,pos],axis=1)



def cosine_sim(vector1,vector2):
    set_vector1 = set(vector1)
    set_vector2 = set(vector2)
    
    intersect = len(set_vector1.intersection(set_vector2))
    
    length_vector1 = np.sqrt(len(vector1))
    length_vector2 = np.sqrt(len(vector2))
    
    cosine_sim = intersect / (length_vector1 * length_vector2)
    return cosine_sim.round(5)
    
vector1 = df_ps_test.tid.iloc[0]
vector2 = df_ps_test.tid.iloc[1]

cosine_sim(vector1,vector2)

record = []
for i in range(100):
    for j in range(100):
        pair = [i,j]
        vector1 = df_ps_test.tid.iloc[i]
        vector2 = df_ps_test.tid.iloc[j]
        sim = cosine_sim(vector1,vector2)
        
        record.append([pair,sim])
        
my_df = pd.DataFrame.from_records(record,columns=['pair','similarity'])        
        


