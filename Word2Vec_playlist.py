#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 22:04:01 2018

@author: bking
"""

from pyspark import SparkContext,StorageLevel
import pandas as pd
from pyspark.mllib.feature import Word2Vec,Word2VecModel
from helper import findK_relevant
import time
from helper import my_evaluation
import argparse
import sys
import pickle
import os.path
import glob

from multiprocessing import Pool
# How to write spark-submit
#https://www.alibabacloud.com/help/doc-detail/28124.htm

#https://spark.apache.org/docs/latest/configuration.html
# --conf spark.driver.maxResultSize=3g

# result
#{'r-precision': 0.0030150149552267557, 'ndcg': 0.004065222638120043, 'song clicks': 10.357300000000002}




# Initialize df_ps_train  
#df_ps_train = pd.DataFrame()

#def my_function(data):
#    pid = data[0]
#    current_list = data[1]
#
#    
#    start = time.time()
#    print("Pid: ",pid)
#    
#    
##    topK_pid = findKRelevant_simple(pid,df_ps_train,K)
#    syms = model.findSynonyms(str(pid),K)
#    
#    topK_pid = [s[0] for s in syms]
#    n = 0
#    
#    # Need to convert str to Int here
#    topK_pid = [int(i) for i in topK_pid]
#    
#    while(1):
#        # Get the top 1 pid 
#        top_pid = topK_pid[n]
#        
#        # Retrieve tid from the top 1 pid
#        add_tid_list = df_ps_train.loc[top_pid].tid
#                
#        # Form new list
#        new_tid_list = current_list + add_tid_list
#        
#        # Check duplicate lists
#        new_tid_list = [tid for tid in new_tid_list if tid not in current_list]
#             
#        # Check number of songs and Add to data for prediction
#        total_song = len(new_tid_list)
#        
##            print("n: {}\t total_song: {}".format(n,total_song))
#        
#        
#        if (total_song > MAX_tid):
#            current_list = new_tid_list[:MAX_tid]            
#            break
#        else:
#            current_list = new_tid_list
#            
#        n += 1
#        if (n>=K):
#            break
#        
##    SIZE = SIZE - 1
#    print("Time taken = {0:.5f}".format(time.time() - start))
#    
#    return current_list
#    return [pid,current_list]

def main(argv):
    sc = SparkContext("local", "Simple App")
    sc.setLogLevel("ERROR")
    
    args = parser.parse_args(argv[1:])
    vector_size = int(args.vector_size)
    min_count = int(args.min_count)    
    test = int(args.mode)
    resume = int(args.resume)
#    proc = int(args.proc)
    
    MAX_LEN = 500     
    K=10

#    vector_size = 5
#    min_count = 5
    
    # Check the existence of word2vec_model folder
    model_name = "word2vec_model_playlist"
    model_folder = glob.glob(model_name+"*")
    model_num = len(model_folder)
    
 
    
    path = "data/df_data/df_playlistSong/"
    if test == 1:
        print("Mode test: ON")
        path = "data/df_data/df_small/df_playlistSong/"
        MAX_LEN = 100 
    print(path)
    print("Load Song-Playlist matrix")    
    
#    path = "data/df_data/df_small/df_playlistSong/"
    
    df_ps_train = pd.read_hdf(path+'df_ps_train.hdf')
    
    df_ps_test = pd.read_hdf(path+'df_ps_test.hdf')
    df_ps_test_truth = pd.read_hdf(path+'df_ps_test_truth.hdf')
       
    df_sp_train = pd.read_hdf(path+'df_sp_train.hdf')
    
    data_str = [list(map(str,item)) for item in df_sp_train.pid.values]

    pid_list_pred = list(df_ps_test.index)    
    current_list = list(df_ps_test.loc[pid_list_pred].tid)
    current_len = [len(i) for i in current_list]    
    
#    K_list = [MAX_LEN - current_len[i] for i in range(len(current_len))]
    
    current_list_str = [list(map(str,item)) for item in current_list]
    
    record = []
    index = 0
    
    # Resume or not
    if resume == 0:
        print("Serialize data")
        doc = sc.parallelize(data_str).persist(StorageLevel.DISK_ONLY)
        
        print("Train Word2Vec model")
        model = Word2Vec().setVectorSize(vector_size).setSeed(3).setMinCount(min_count).fit(doc)
    
        print("Get vocabulary")
        vocab = model.getVectors().keySet()
    
        print("Save model")
        model_name = model_name + str(model_num)
        model.save(sc, model_name)
        
    elif resume == 1:
        print("load recent model")
        model_name = model_name + str(model_num-1)
        model = Word2VecModel.load(sc, model_name)
        
        print("Get vocabulary")
        vocab = model.getVectors().keySet()
        
        first_key = list(vocab)[0]
        vector_size = len(model.getVectors()[first_key])
        
        print("Check resume file: ",end='')
        
        if(os.path.exists("resumefile")):
            print("Exist")
            with open ('resumefile', 'rb') as fp:
                resumefile = pickle.load(fp)
    
            pid,record = resumefile.get('pid'), resumefile.get('data')
            index = current_list_str.index(pid)
           
            print("Resume at point pid: {} \t index: {}".format(pid,index))
        else:
            print("Not exist")
    

        
    print("Find K Relevant Songs")
    try:
      
        
        i = 0
        for data_list in current_list_str[index:]:
            pid = pid_list_pred[i]
            print("Iter: {} \t pid: {} ".format(str(i+1),pid))
            start = time.time()

            ######################## START CHANGING HERE ################################
            
            syms = model.findSynonyms(str(pid),K)
            topK_pid = [s[0] for s in syms]
    
            # Need to convert str to Int here
            topK_pid = [int(i) for i in topK_pid]
               
            n = 0
            while(1):
                # Get the top 1 pid 
                top_pid = topK_pid[n]
                
                # Retrieve tid from the top 1 pid
                add_tid_list = df_ps_train.loc[top_pid].tid
                        
                # Form new list
                new_tid_list = data_list + add_tid_list
                
                # Check duplicate lists
                new_tid_list = [tid for tid in new_tid_list if tid not in data_list]
                     
                # Check number of songs and Add to data for prediction
                total_song = len(new_tid_list)
        
    
                if (total_song > MAX_LEN):
                    new_list = new_tid_list[:MAX_LEN]            
                    break
                else:
                    new_list = new_tid_list
                    
                n += 1
                if (n>=K):
                    break
                
            record.append(new_list)
            i += 1
            print("Time taken = {0:.5f}".format(time.time() - start))
            
        

        print("Create new dataframe")
        df_ps_test['new_tid'] = record
        
        df_ps_test['tid']=df_ps_test.apply(lambda x: x[1]+ x[2],axis=1)
        df_ps_test=df_ps_test.drop(columns='new_tid')
        
        
#        df_ps_pred = pd.DataFrame.from_records(new_list,columns=['pid','tid'])
#        df_ps_pred = df_ps_pred.set_index('pid')
        
        print("Save test data")
        df_ps_test.to_hdf(path+'df_ps_test_complete.hdf', key='abc')
        
        print("Evaluation")
        result = my_evaluation(df_ps_test,df_ps_test_truth)
        print(result.aggregate_metric())
    except Exception as e:
        print(e)
        print("Create a resume point")
        resume_dict = {'pid':pid,'data':record}
        with open('resumefile', 'wb') as fp:
            pickle.dump(resume_dict, fp)
    


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vector_size', default='100', type=str, help='Vector Size in Word2Vec') 
    parser.add_argument('--min_count', default= '5', type=str, help='Minimum frequency') 
    parser.add_argument('--mode', default= '0', type=str, help='Mode Test On/Off') 
    parser.add_argument('--resume', default= '0', type=str, help='Load model and resume') 
#    parser.add_argument('--proc', default= '8', type=str, help='Number of processor') 
    
    
    main(sys.argv)
#    print(result)

