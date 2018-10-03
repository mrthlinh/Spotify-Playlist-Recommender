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
# How to write spark-submit
#https://www.alibabacloud.com/help/doc-detail/28124.htm

#https://spark.apache.org/docs/latest/configuration.html
# --conf spark.driver.maxResultSize=3g

# result
#{'r-precision': 0.0030150149552267557, 'ndcg': 0.004065222638120043, 'song clicks': 10.357300000000002}


def main(argv):
    sc = SparkContext("local", "Simple App")
    sc.setLogLevel("ERROR")
    
    args = parser.parse_args(argv[1:])
    vector_size = int(args.vector_size)
    min_count = int(args.min_count)    
    test = int(args.mode)
    resume = int(args.resume)
    MAX_LEN = 500 
#    vector_size = 5
#    min_count = 5
    
    # Check the existence of word2vec_model folder
    model_name = "word2vec_model"
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
       
    data_str = [list(map(str,item)) for item in df_ps_train.tid.values]

    pid_list_pred = list(df_ps_test.index)    
    current_list = list(df_ps_test.loc[pid_list_pred].tid)
    current_len = [len(i) for i in current_list]    
    K_list = [MAX_LEN - current_len[i] for i in range(len(current_len))]
    
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
    #        print("pid: {} {}".format(pid_list_pred[i],data_list))
            pid = pid_list_pred[i]
            print("Iter: {} \t pid: {} ".format(str(i+1),pid))
            start = time.time()
            
            # Filter data not in vocabulary
            data_list_filter = [value for value in data_list if value in vocab]
            
    #        topK = [value for value in topK if value not in data_list]
            
            # Find the centroid of data_list
            record.append(findK_relevant(model,K_list[i],data_list_filter,sc,vector_size))
            i += 1
            print("Time taken = {0:.5f}".format(time.time() - start))
            
        print("Create new dataframe")
        df_ps_test['new_tid'] = record
        
        df_ps_test['tid']=df_ps_test.apply(lambda x: x[1]+ x[2],axis=1)
        df_ps_test=df_ps_test.drop(columns='new_tid')
        
        print("Save test data")
        df_ps_test.to_hdf(path+'df_ps_test_complete.hdf', key='abc')
        
        print("Evaluation")
        result = my_evaluation(df_ps_test,df_ps_test_truth)
        print(result.aggregate_metric())
    except:
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
    
    
    main(sys.argv)
#    print(result)

