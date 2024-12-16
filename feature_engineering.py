# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 01:10:55 2024

@author: Saleh
"""

import numpy as np

class preprocess:
    
    def __init__(self, data):
        
        self.data = data
        
        self.features = []
        
    def normalize(self, data):
        
        for i in range(data.shape[1]):
            
            data[:,i] = (data[:,i] - np.mean(data[:,i]))/(np.std(data[:,i]))
            
        return data
        
    def integral(self,data):
        
        Temp = 0
        
        time_step = 0.2
            
        for i in range(len(data)):
                
            Temp += time_step*data[i]
                
        return Temp
        
    def feature_extraction_I(self):
        
        for i in range(len(self.data)):
            
            list_features = []
            
            data = self.data[i]
            
            for j in range(data.shape[1]):
                
                features_data = data[:,j]
                
                features_data = np.trim_zeros(features_data)
                
                if j < 4:
                    
                    feature = np.mean(features_data)
                    
                    list_features.append(feature)
                elif j == 4:
                    
                    feature = self.integral(features_data)
                    
                    list_features.append(feature)
                    
            self.features.append(list_features)
            
        return self.normalize(np.array(self.features, dtype = np.float16))
                    
    def feature_extraction_II(self):
        
        for i in range(len(self.data)):
            
            list_features = []
            
            data = self.data[i]
            
            for j in range(data.shape[1]):
                
                features = data[:,j]
                
                features = np.trim_zeros(features)
                
                if j < 4:
                    
                    feature = np.var(features)
                    
                    list_features.append(feature)
                elif j == 4:
                    
                    feature = self.integral(features)
                    
                    list_features.append(feature)
                    
            self.features.append(list_features)
            
        return self.normalize(np.array(self.features, dtype = np.float16))
                
    def feature_extraction_III(self):
        
        for i in range(len(self.data)):
            
            list_features = []
            
            data = self.data[i]
            
            for j in range(data.shape[1]):
                
                features = data[:,j]
                
                features = np.trim_zeros(features)
                
                if j < 4:
                    
                    feature = np.mean(features)
                    
                    list_features.append(feature)
                elif j == 4:
                    
                    feature = np.mean(features)
                    
                    list_features.append(feature)
                    
            self.features.append(list_features)
            
        return self.normalize(np.array(self.features, dtype = np.float16))
                    
    def feature_extraction_IV(self):
        
        for i in range(len(self.data)):
            
            list_features = []
            
            data = self.data[i]
            
            for j in range(data.shape[1]):
                
                features = data[:,j]
                
                features = np.trim_zeros(features)
                
                if j < 4:
                    
                    feature = np.var(features)
                    
                    list_features.append(feature)
                elif j == 4:
                    
                    feature = np.mean(features)
                    
                    list_features.append(feature)
                    
            self.features.append(list_features)
            
        return self.normalize(np.array(self.features, dtype = np.float16))
    
    

def feature_extraction(data, policy):
    
    features_obj = preprocess(data)
    
    if policy == 2:
        
        features = features_obj.feature_extraction_I()
        
    elif policy == 3:
        
        features = features_obj.feature_extraction_II()
        
    elif policy == 4:
        
        features = features_obj.feature_extraction_III()
        
    elif policy == 5:
        
        features = features_obj.feature_extraction_IV()
        
    del features_obj
        
    return features


                
        


       