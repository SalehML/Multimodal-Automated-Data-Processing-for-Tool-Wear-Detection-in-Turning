"""
Automated ML - Porject I
Reinforcement Learning Unit

Program Developer: Saleh ValizadehSotubadi
Advisor: Dr. Vinh Nguyen

Department of Mechanical Engineering-Engineering Mechanics (MEEM)
Michigan Technological University

Date: 01/09/2023

"""

import numpy as np
import ypstruct 
from ypstruct import structure
from matplotlib import pyplot as plt
#from MachineLearningUnit_journal import AutoML_Model
from MachineLearningUnit_journal_Version_2 import AutoML_Model
import pickle
import os
import time
import gc

save_directory = '/content/gdrive/MyDrive/ADP_Journal Paper'


Total_Episodes = 15

count = 0

Iterations = 3000

alpha = 0.2

c = 2

options = 49

Q_Values = np.random.uniform(0, 1, size=(Total_Episodes, options)) - 0.5

#Q_Values = np.ones([Total_Episodes, options])

selected_actions = np.zeros([Total_Episodes, Iterations])

Best_Action = np.zeros([Total_Episodes,1])  

Reward_History = np.zeros([Total_Episodes, Iterations])

Policy_Reward_History = np.zeros([options, Iterations, Total_Episodes])

def action_selection(Q, counter, t):
    
    t += 1
    
    counter[np.where(counter == 0)] = 1e-10
    
    action_values = Q.reshape(options,1) + c * np.sqrt(np.log(t)/counter)
    
    return action_values 

for episode in range(Total_Episodes):
    
    epsilon = 0.1
    
    i_th_Q_Values = Q_Values[episode, :]
    
    action_counter = np.zeros([options,1])
    
    for iteration in range(Iterations):
        
        print('Episode: '+ str(episode + 1) + ', Iteration: ' + str(iteration + 1))
        
    
        random_event = np.random.random()
        
        Action_selections = action_selection(i_th_Q_Values, action_counter, iteration)
    
        if random_event <= epsilon:
            
            policy = np.random.randint(options)
            
        elif random_event > epsilon:
            
            max_i_th_Q = np.argmax(Action_selections)
            
            pol = max_i_th_Q
            
            policy = pol
            
        selected_actions[episode, iteration] =  policy  
        
        action_counter[policy] += 1
        
        print("Policy is: " + str(policy))
        print('------------------------')
            
        Reward = AutoML_Model(policy)
        
        Reward *= 1
        
        i_th_Q_Values[policy] = ((1 - alpha) * (i_th_Q_Values[policy])) + alpha * Reward
        
        Reward_History[episode, iteration] = Reward
        
        count += 1
        
        Policy_Reward_History[policy, iteration, episode] = Reward
        
        
        gc.collect()
        
    Best_Action[episode] = np.argmax(action_counter)
        

pickle.dump(Best_Action, open(os.path.join(save_directory, 'RL_Best_Actions'), 'wb'))

pickle.dump(Reward_History, open(os.path.join(save_directory, 'RL_Reward_History'), 'wb'))

pickle.dump(selected_actions, open(os.path.join(save_directory, 'RL_selected_actions'), 'wb'))

pickle.dump(Policy_Reward_History, open(os.path.join(save_directory, 'RL_Policy_Reward_History'), 'wb'))


"""
load_directory = 'C:\\Users\\Saleh\\OneDrive\\Desktop\\AutoML\\Variables_after_code_running'

os.listdir(load_directory)[0]


action_counter_for_each_episode = []


best_actions = pickle.load(open(os.path.join(load_directory, os.listdir(load_directory)[0]), 'rb'))
Policy_Reward_History = pickle.load(open(os.path.join(load_directory, os.listdir(load_directory)[1]), 'rb'))
Q_Values = pickle.load(open(os.path.join(load_directory, os.listdir(load_directory)[2]), 'rb'))

for i in range(len(options)):
    
    max_action_for_each_episode = len(np.where(best_actions == i)[0]) 
    
    action_counter_for_each_episode.append(max_action_for_each_episode)

reW = np.mean(Policy_Reward_History, axis = 1)
"""