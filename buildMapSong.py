#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 00:35:54 2018

@author: bking
"""
import pandas as pd
import pickle
path = "data/df_data/df_playlistSong/"
df_sp_complete = pd.read_hdf(path+"df_sp_complete_new.hdf")

tid_list = list(df_sp_complete.index)
num_tid = len(tid_list)

dict_tid2index = {k:v for k,v in zip(tid_list,range(0,num_tid))}
dict_index2tid = {k:v for k,v in zip(range(0,num_tid),tid_list)}

out_dict = {'tid2index':dict_tid2index,'index2tid':dict_index2tid}
out_filename = "tid_map.pickle"
print("Save file")
with open("data/"+out_filename, 'wb') as f:
    pickle.dump(out_dict, f)  