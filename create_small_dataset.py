#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 13:45:37 2018

@author: bking
"""

import pandas as pd
import numpy as np
import os
#from functools import reduce
#from itertools import chain
from collections import defaultdict
#from scipy.sparse import dok_matrix
#import pickle


pid_size = 100
tid_size = 150
test_size = 10
num_holdout = 10

pid = list(np.arange(0,pid_size))
np.random.seed(100)
tid_complete = [list(np.random.randint(0,tid_size, 2 * num_holdout)) for i in range(pid_size)]

tid_test = [tid_complete[i][:num_holdout] for i in range(pid_size-test_size,pid_size)]

tid_train = [tid_complete[i] for i in range(pid_size-test_size)] + tid_test
tid_test_truth = tid_complete[-test_size:]


df_ps_train = pd.DataFrame({'pid':pid,'tid':tid_train})


df_ps_test = pd.DataFrame({'pid':pid[-test_size:],'tid':tid_test})


df_ps_test_truth = pd.DataFrame({'pid':pid[-test_size:],'tid':tid_test_truth})

def transform(row):
    pid = row[0]
    tid_list = row[1]
    new_dict = {key: pid for key in tid_list}
    return new_dict
#    return pid
#######################################
    
df_sp_train = df_ps_train.copy()
#df_sp_train['test'] = df_sp_train.apply(transform,axis=1)
tid_dict = list(df_ps_train.apply(transform,axis=1))
sp = defaultdict(list)
for d in tid_dict:
    for k, v in d.items():
#        d_[k]+= v
        sp[k].append(v)

df_sp_train = pd.DataFrame({'tid':list(sp.keys()),'pid':list(sp.values())})
df_sp_train = df_sp_train.set_index('tid').sort_index()

########################################

df_sp_test = df_ps_test.copy()
tid_dict = list(df_ps_test.apply(transform,axis=1))
sp = defaultdict(list)

for d in tid_dict:
    for k, v in d.items():
#        d_[k]+= v
        sp[k].append(v)

df_sp_test = pd.DataFrame({'tid':list(sp.keys()),'pid':list(sp.values())})
df_sp_test = df_sp_test.set_index('tid').sort_index()

#######################################

df_sp_test_truth = df_ps_test_truth.copy()
tid_dict = list(df_ps_test_truth.apply(transform,axis=1))
sp = defaultdict(list)

for d in tid_dict:
    for k, v in d.items():
#        d_[k]+= v
        sp[k].append(v)

df_sp_test_truth = pd.DataFrame({'tid':list(sp.keys()),'pid':list(sp.values())})
df_sp_test_truth = df_sp_test_truth.set_index('tid').sort_index()



path = "data/df_data/df_small/df_playlistSong/"
if (os.path.exists(path) == False):
    os.makedirs(path)

df_sp_train.to_hdf(path+"df_sp_train.hdf",key="abc")
df_sp_test.to_hdf(path+"df_sp_test.hdf",key="abc")
df_sp_test_truth.to_hdf(path+"df_sp_test_truth.hdf",key="abc")

df_ps_train = df_ps_train.set_index('pid').sort_index()
df_ps_test = df_ps_test.set_index('pid').sort_index()
df_ps_test_truth = df_ps_test_truth.set_index('pid').sort_index()

df_ps_train.to_hdf(path+"df_ps_train.hdf",key="abc")
df_ps_test.to_hdf(path+"df_ps_test.hdf",key="abc")
df_ps_test_truth.to_hdf(path+"df_ps_test_truth.hdf",key="abc")


