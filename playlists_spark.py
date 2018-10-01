#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 22:36:17 2018

@author: bking
"""

from pyspark import SparkContext
import pandas as pd
from pyspark.mllib.feature import Word2Vec
from helper import findK_relevant
import time
from helper import my_evaluation

vector_size = 5
min_count = 5

sc = SparkContext("local", "Simple App")
sc.setLogLevel("ERROR")

path = "data/df_data/df_small/df_playlistSong/"
df_ps_train = pd.read_hdf(path+'df_ps_train.hdf')
df_ps_test = pd.read_hdf(path+'df_ps_test.hdf')
df_ps_test_truth = pd.read_hdf(path+'df_ps_test_truth.hdf')
df_sp_train = pd.read_hdf(path+'df_sp_train.hdf')


data_str = [list(map(str,item)) for item in df_sp_train.tid.values]
doc = sc.parallelize(data_str)

model = Word2Vec().setVectorSize(vector_size).setSeed(3).setMinCount(min_count).fit(doc)


pid_list_pred = list(df_ps_test.index)    
current_list = list(df_ps_test.loc[pid_list_pred].tid)
current_len = [len(i) for i in current_list] 
MAX_LEN=100    
K_list = [MAX_LEN - current_len[i] for i in range(len(current_len))]

current_list_str = [list(map(str,item)) for item in current_list]
i = 0
record = []
for data_list in current_list_str:
    print("pid: {} {}".format(pid_list_pred[i],data_list))
    start = time.time()
    # Find the centroid of data_list
    record.append(findK_relevant(model,K_list[i],data_list,sc))
    i += 1
    print("Time taken = {0:.5f}".format(time.time() - start))
    
df_ps_test['new_tid'] = record

df_ps_test['tid']=df_ps_test.apply(lambda x: x[1]+ x[2],axis=1)
df_ps_test=df_ps_test.drop(columns='new_tid')

result = my_evaluation(df_ps_test,df_ps_test_truth)
print(result.aggregate_metric())



