#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 16:54:47 2018

@author: bking
"""

import smtplib
from email.mime.text import MIMEText
import numpy as np
from multiprocessing import Pool
import pandas as pd


def sendGmail(to,body,subject):
    gmail_user = 'mrthlinh@gmail.com'  
    gmail_password = 'Inevergiveup1992'

    fromx = 'mrthlinh@gmail.com'
#    to  = 'mrthlinh@gmail.com'
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = fromx
    msg['To'] = to
    
    try:  
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.ehlo()
        server.login(gmail_user, gmail_password)
        server.sendmail(fromx, to, msg.as_string())
        server.close() 
        print ('Email sent!')
    except:  
        print ('Something went wrong...')

def alertFinishJob(message):
    subject="Finished"
    sendGmail('mrthlinh@gmail.com',message,subject=subject) 
    
def alertError(message):
    subject="Failed"
    sendGmail('mrthlinh@gmail.com',message,subject=subject) 

def computeCentroid(vector_list):
    """
        Description: 
        Usage:
         
    """
    
    size = len(vector_list)
    unfold_vector = [item for sublist in vector_list for item in sublist]
    
    series = pd.Series(unfold_vector)
    freq = series.value_counts()
    centroid = freq / size
    
    return dict(centroid)

def cosine_sim_song(vector):
    """
        Description: Compute the cosine similarity between 1 centroid and 1 vector
        Usage:
         
    """
    centroid = vector[0]
    compare_vector = vector[1]
    
    set_vector1 = set(centroid)
    set_vector2 = set(compare_vector)
    
    intersect = set_vector1.intersection(set_vector2)
    value = [centroid.get(i) for i in intersect]
    A = sum(value)
    
    length_vector1 = np.sqrt(sum(np.square(list(centroid.values()))))
    
    length_vector2 = np.sqrt(len(compare_vector))
    
    B = length_vector1 * length_vector2
    
    cosine_sime = A / B
    
    return cosine_sime.round(2)

#def cosine_sim(vector1,vector2):
#    """
#        Description: Compute the cosine similarity between 2 vectors
#        Usage:
#         
#    """
#
#   
#    set_vector1 = set(vector1)
#    set_vector2 = set(vector2)
#    
#    intersect = set_vector1.intersection(set_vector2)
#    value = [centroid.get(i) for i in intersect]
#    A = sum(value)
#    
#    length_vector1 = np.sqrt(sum(np.square(list(centroid.values()))))
#    
#    length_vector2 = np.sqrt(len(compare_vector))
#    
#    B = length_vector1 * length_vector2
#    
#    cosine_sime = A / B
#    
#    return cosine_sime.round(2)

def findKRelevant_song(curr_song_list,df,K):
    """
        Description: Find K most relevent objects of centroid
        Usage:
            - curr_song_list = a list current songs in playlists
            - tid: int
            - df: Song-Playlist matrix column = [pid,pos], index = [tid]
            - K: int
        Return:
            list of tid 
    """
    
    
    # Compute centroid of tid list
    curr_pid_vector = list(df.loc[curr_song_list].pid)    
    centroid = computeCentroid(curr_pid_vector)
    
    # Filter out songs that already in the list  
    other_song_vector = df.loc[~(df.index.isin(curr_song_list))].pid
    
    tid_list = list(other_song_vector.index)
    sim = []
    
    for v in other_song_vector.values:
        sim.append(cosine_sim_song([centroid,v]))
        
    topK = [x for _,x in sorted(zip(sim,tid_list),reverse=True)]   

    return topK[:K]





#def cosine_sim(vector1,vector2):
def cosine_sim(vector):
    """
        Description: Compute the cosine similarity between 2 vectors with multiprocessing
        Usage:
         
    """
    vector1 = vector[0]
    vector2 = vector[1]
#    print("length: {}".format(len(vector1)))
    set_vector1 = set(vector1)
    set_vector2 = set(vector2)
    
    intersect = len(set_vector1.intersection(set_vector2))
    
    length_vector1 = np.sqrt(len(set_vector1))
    length_vector2 = np.sqrt(len(set_vector2))
    
    cosine_sim = intersect / (length_vector1 * length_vector2)
    return cosine_sim.round(2)

def findKRelevant(pid,df,K,proc):
    """
        Description: Find K most relevent objects, with multiprocessing 
        Usage:
            - pid: int
            - df: column = [tid,pos], index = [pid]
            - K: int
            - proc: int, number of processing            
    """
    
    pid_list = list(df.index.values)
    length = len(pid_list)
    
    p1 = [pid]*(length-1)
    
    p2 = pid_list.copy()
    p2.remove(pid)
    
    vector1 = list(df.tid.loc[p1])
    vector2 = list(df.tid.loc[p2])
    
    with Pool(proc) as p:
        sim = p.map(cosine_sim, zip(vector1,vector2))
       
    topK = [x for _,x in sorted(zip(sim,p2),reverse=True)]
    return topK[:K]

def findKRelevant_simple(pid,df,K):
    """
        Description: Find K most relevent objects, with multiprocessing 
        Usage:
            - pid: int
            - df: column = [tid,pos], index = [pid]
            - K: int
            - proc: int, number of processing            
    """
    
    pid_list = list(df.index.values)
    length = len(pid_list)
    
    p1 = [pid]*(length-1)
    
    p2 = pid_list.copy()
    p2.remove(pid)
    
    vector1 = list(df.tid.loc[p1])
    vector2 = list(df.tid.loc[p2])
    
    sim = []
    for i in range(length-1):
        v1 = vector1[i]
        v2 = vector2[i]
        sim.append(cosine_sim([v1,v2]))
        
#    with Pool(proc) as p:
#        sim = p.map(cosine_sim, zip(vector1,vector2))
       
    topK = [x for _,x in sorted(zip(sim,p2),reverse=True)]
    return topK[:K]

# Function forSpark
def centroid(model,data,sc,vector_size):
    if len(data) == 0:
        print("All data points are not in vocab")
        print(vector_size)
        from pyspark.mllib.linalg import Vectors
        return Vectors.dense(Vectors.zeros(vector_size))
        
    vectorize_list = list(map(model.transform, data))
    centroid = sc.parallelize(vectorize_list).reduce(lambda x,y: x+y)
    centroid = centroid / len(vectorize_list)
    return centroid

# Function forSpark
def findK_relevant(model,K,data_list,sc,vector_size):
    # Find the centroid of data_list
    vec  = centroid(model,data_list,sc,vector_size)
    # Define empty list
    topK = []
    
    # Define multiplity constant
    constant = 0
    
    # Loop until get all K element
    while(1):
#         print(constant)
        # Get nearest vectors
        syms = model.findSynonyms(vec, int(K*(1.1 + constant / 10)))
        # Find top K
        topK = [s[0] for s in syms][1:]        
        # Filter out duplication
        topK = [value for value in topK if value not in data_list]
        
        if (len(topK) >= K):
            break
        
        if (round(constant) == 10):
            break
        
        constant += 2
        
    return topK[:K]   




class my_evaluation:
#    df_predictions = pd.DataFrame()
#    df_truth = pd.DataFrame()
    
    def __init__(self,df_predictions,df_truth):
         self.df_predictions = df_predictions
         self.df_truth = df_truth
         
             
    def aggregate_metric(self):
        '''
            Some description
            
            predictions:
            truth:
        '''
        df_predictions = self.df_predictions
        df_truth = self.df_truth
        
        r_precision_list = []
        ndcg_list = []
        song_clicks_list =[]
        
#        pid_list = df_truth.pid.unique()
        
        pid_list = df_truth.index.unique()
        
        # Iterate through each pid to get value of each metric
        for pid in pid_list:
            # r_precision
            
            # Filter out data with specific pid
            predictions = df_predictions.loc[pid].tid
            truth = df_truth.loc[pid].tid
            
            # Get n_track
#            n_track = len(truth)
            
#            truth = truth.tid
#            predictions = predictions.tid        
            r_precision_list.append(self.r_precision(predictions,truth,500))
            ndcg_list.append(self.ndcg(predictions,truth,500))
            song_clicks_list.append(self.song_clicks(predictions,truth,500))
        
        r_precision_value = np.array([r_precision_list]).mean()
        ndcg_value = np.array([ndcg_list]).mean()
        song_clicks_value = np.array([song_clicks_list]).mean()
    
        return {'r-precision':r_precision_value,'ndcg':ndcg_value, 'song clicks':song_clicks_value}
         
    def r_precision(self,predictions,truth,n_track):
        '''
            some description
            predictions:
            truth:
        '''
        
         # Calculate metric
        truth_set = set(truth)
        prediction_set = set(predictions[:n_track])
        
        intersect = prediction_set.intersection(truth_set)
        
        return float(len(intersect)) / len(truth_set)
            
        
    def ndcg(self,predictions,truth,n_tracks):
        '''
            some description
            predictions:
            truth:
        '''
        predictions = list(predictions[:n_tracks])
        truth = list(truth)
    
        # Computes an ordered vector of 1.0 and 0.0
        
        # Sum ( rel / log2 (i+1) )
        
        score = [float(element in truth) for element in predictions]    
        dcg  = np.sum(score / np.log2(1 + np.arange(1, len(score) + 1)))
        
        ones = np.ones([1,len(truth)])
        idcg = np.sum(ones / np.log2(1 + np.arange(1, len(truth) + 1)))
    
        return (dcg / idcg)
    
    
    def song_clicks(self,predictions,truth,n_tracks):
        '''
            Minumum clicks until a relevant track is found
            The Lower, the better
            predictions:
            truth:
        '''
        predictions = predictions[:n_tracks]
    
        # Calculate metric
        i = set(predictions).intersection(set(truth))
        
        for index, t in enumerate(predictions):
            for track in i:
                if t == track:
                    return float(int(index / 10))
                
        return float(n_tracks / 10.0 + 1)
    

