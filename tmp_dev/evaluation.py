#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 23:17:02 2018

@author: bking
"""
import numpy as np
#import pandas as pd

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
        
        pid_list = df_truth.pid.unique()
        # Iterate through each pid to get value of each metric
        for pid in pid_list:
            # r_precision
            
            # Filter out data with specific pid
            predictions = df_predictions[df_predictions.pid==pid]
            truth = df_truth[df_truth.pid==pid]
            
            # Get n_track
            n_track = truth.shape[0]
            
            truth = truth.tid
            predictions = predictions.tid        
            r_precision_list.append(self.r_precision(predictions,truth,n_track))
            ndcg_list.append(self.ndcg(predictions,truth,n_track))
            song_clicks_list.append(self.song_clicks(predictions,truth,n_track))
        
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
        score = [float(element in truth) for element in predictions]    
        dcg  = np.sum(score / np.log2(1 + np.arange(1, len(score) + 1)))
        
        ones = np.ones([1,len(truth)])
        idcg = np.sum(ones / np.log2(1 + np.arange(1, len(truth) + 1)))
    
        return (dcg / idcg)
    
    
    def song_clicks(self,predictions,truth,n_tracks):
        '''
            Minumum clicks until a relevant track is found.
            
            
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
    
    
#
#test = my_df_tracks_test.iloc[:500,:]
#
#result = my_evaluation(test,test)   
#result.aggregate_metric()
# Test
#pid = 980
#predictions = my_df_tracks_test[my_df_tracks_test.pid==pid]
#predictions = pd.concat([predictions,predictions])
#predictions['tid'] = np.arange(1,401)
#
#truth = my_df_tracks_truth[my_df_tracks_truth.pid==pid]
#n_tracks=truth.shape[0]
#
#r_precision(predictions.tid,truth.tid,n_tracks)
#ndcg(predictions.tid,truth.tid,n_tracks)
#rec_song_clicks(predictions.tid,truth.tid,n_tracks)
#
#
#
#test = my_df_tracks_test.iloc[:500,:]
#aggregate_metric(pd.concat([test,test]),test)




  