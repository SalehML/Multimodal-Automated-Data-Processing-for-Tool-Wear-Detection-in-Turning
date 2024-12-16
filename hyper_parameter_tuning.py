#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 13:53:18 2024

@author: svalizad
"""

import numpy as np
import pandas as pd
import os

directory = '/content/gdrive/MyDrive/ADP_Journal Paper'

hyper_parameters_name = 'hyper.csv'

def check_for_file():

    
    if hyper_parameters_name not in os.listdir(directory):
    
        hyper_parameters = np.zeros(shape = [42,10])
        
        hyper_parameters[:,-1] = -10
        
        hyper_parameters_pd = pd.DataFrame(hyper_parameters)
        
        hyper_parameters_pd.to_csv(os.path.join(directory, hyper_parameters_name))
    
    elif hyper_parameters_name in os.listdir(directory):
        
        hyper_parameters_pd = pd.read_csv(os.path.join(directory, hyper_parameters_name))
        
        hyper_parameters = np.array(hyper_parameters_pd)
        
        hyper_parameters = hyper_parameters[:,1:]
        
    return hyper_parameters


def set_hyperparameters(policy_num):
    
    """
    
    Run check_for_file if the hyper.csv is not in the directory
    
    hyper_parameters = check_for_file()
    
    """
    
    hyper_parameters_pd = pd.read_csv(os.path.join(directory, hyper_parameters_name))
    
    hyper_parameters = np.array(hyper_parameters_pd)
    
    hyper_parameters = hyper_parameters[:,1:]
    
    reward_threshold = 0.05
    
    lr_options = [0.0002, 0.001]

    dropout_choice = [0.25, 0.35]

    l2_choice = [0.01]
    
    possible_values_batch = [16, 32]
    
    possible_values_beta_1 = [0.9]
    
    possible_values_beta_2 = [0.999]
    
    k_fold_options = [3]
    
    layers_options = [1, 2]
    
    neurons_options = [20]
    
    parameters = hyper_parameters[policy_num]
    
    reward = parameters[-1]
    
    if reward < reward_threshold:
        
        parameters[0] = np.random.choice(lr_options)
        
        parameters[1] = np.random.choice(possible_values_batch)
        
        parameters[2] = np.random.choice(possible_values_beta_1)
        
        parameters[3] = np.random.choice(possible_values_beta_2)
        
        parameters[4] = np.random.choice(dropout_choice)
        
        parameters[5] = np.random.choice(l2_choice)
        
        parameters[6] = np.random.choice(k_fold_options)
        
        parameters[7] = np.random.choice(neurons_options)
        
        parameters[8] = np.random.choice(layers_options)
        
        hyper_parameters[policy_num] = parameters
        
        hyper_parameters_pd = pd.DataFrame(hyper_parameters)
        
        hyper_parameters_pd.to_csv(os.path.join(directory, hyper_parameters_name))
        
        return parameters[:-1]
        
    elif reward > reward_threshold:
        
        return parameters[:-1]
    
def set_reward(policy_num, reward):
    
    """
    
    Run check_for_file if the hyper.csv is not in the directory
    
    hyper_parameters = check_for_file()
    
    """
    
    hyper_parameters_pd = pd.read_csv(os.path.join(directory, hyper_parameters_name))
    
    hyper_parameters = np.array(hyper_parameters_pd)
    
    hyper_parameters = hyper_parameters[:,1:]
    
    hyper_parameters[policy_num][-1] = reward
    
    hyper_parameters_pd = pd.DataFrame(hyper_parameters)
    
    hyper_parameters_pd.to_csv(os.path.join(directory, hyper_parameters_name))
    
    
    