# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 21:23:00 2023

@author: Saleh
"""

import numpy as np
import ypstruct 
from ypstruct import structure
from matplotlib import pyplot as plt
from MachineLearningUnit_journal_Version_2 import AutoML_Model
import pickle
import os
import time

save_directory = '/content/gdrive/MyDrive/ADP_Journal Paper/BGA_Results'

#save_directory = 'C:\\Users\Saleh\OneDrive\Desktop\PhD_Works'

number_of_policies = 42

# Problem definition

problem = structure()

problem.nVar = 6

best_model = ''

best_models = []


# GA parameters

params = structure()

params.Max_Iteration = 10
params.num_pop = 20
params.pC = 1
params.beta = 1
params.mu = 0.05
params.crs_mu = 0.85

Iterations = params.Max_Iteration

rewards = np.zeros([number_of_policies, Iterations])

# Run GA

def decimal_to_binary(decimal):
    binary = ""
    if decimal == 0:
        binary = "0"
    while decimal > 0:
        binary = str(decimal % 2) + binary
        decimal //= 2
    return int(binary)

def Binary_to_Decimal(binary_num):
    
    array_string = ''.join(map(str, binary_num))
    
    binary_num = '0b' + array_string
    
    return int(binary_num, 2)

def roulette_wheel_selection(p):
    c = np.cumsum(p)
    r = sum(p)*np.random.rand()
    ind = np.argwhere(r <= c)
    return ind[0][0]

def crossover(p1, p2, crs_mu):
    
    nvar = np.size(p1['position'])
    
    j = np.random.randint(nvar - 1)
    
    c1 = p1.deepcopy()
    
    c2 = p2.deepcopy()
    
    rnd_num = np.random.random()
    
    if rnd_num < crs_mu:
    
        c1_pos_1, c1_pos_2 = list(p1['position'][:j]), list(p2['position'][j:])
    
        c1_pos = c1_pos_1 + c1_pos_2
    
        c2_pos_1, c2_pos_2 = list(p2['position'][:j]), list(p1['position'][j:])
    
        c2_pos = c2_pos_1 + c2_pos_2
        
        c1.position = np.array(c1_pos)
        
        c2.position = np.array(c2_pos)
        
        return c1, c2
    else:
        
        c1_pos, c2_pos = p1, p2
        
        print("c1_pos: " + str(c1_pos))
        print("c2_pos: " + str(c2_pos))
        
        return p1, p2

    
def mutate(c, mu):
    
    nvar = np.size(c['position'])
    
    flag = (np.random.random(nvar) < mu)
    
    c['position'][flag] = 1 - c['position'][flag]
    
    return c

def Binary_Genetics(problem, params):
    
    #print("Binary_GA")
    
    # Problem
    
    num_var = problem.nVar
    
    # Parameters
    
    Max_Iteration = params.Max_Iteration
    num_pop = params.num_pop
    pC = params.pC
    nC = int(np.round(pC * num_pop/2) * 2)
    beta = params.beta
    mu = params.mu
    crs_mu = params.crs_mu
    
    # Empty Individual Template
    empty_individual = structure()
    empty_individual.position = None
    empty_individual.cost = None

    # Best Solution Ever Found
    bestsol = empty_individual.deepcopy()
    bestsol.cost = -100.0
    
    # Initialize Population
    pop = empty_individual.repeat(num_pop)
    
    print("initialization")
    
    
    start_time = time.time()
    
    for i in range(num_pop):
        
        pop[i].position = np.random.randint(0, 2, num_var)
        
        policy = Binary_to_Decimal(pop[i].position)
        
        if policy < number_of_policies:
        
            pop[i].cost = np.random.uniform(low=-0.5, high=-0.2, size=1) 
        else: 
            
            pop[i].cost = -10.0
        
        #rewards[index, 1] = pop[i].cost
        
#    return pop[i].cost
        
#out = Binary_Genetics(problem, params)

        if pop[i].cost > bestsol.cost:
            bestsol = pop[i].deepcopy()
            
    # Best Cost of Iterations
    bestcost = np.empty(Max_Iteration)
    
    print("initialization done!")
    
    for it in range(Max_Iteration):
        
        #costs = np.array([x.cost for x in pop])
        #avg_cost = np.mean(costs)
        #if avg_cost != 0:
            #costs = costs/avg_cost
        #probs = np.exp(-beta*costs)
        
        #pop_c = empty_individual.repeat(nC)
        
        #pop_c = np.array(pop_c).reshape(int(nC/2) , 2)
        
        pop_C = []
    
        for _ in range(int(nC//2)):
            
            
            #Select Parents
            q = np.random.permutation(num_pop)
            p1 = pop[q[0]]
            p2 = pop[q[1]]
            
            try:
                p1_decimal = Binary_to_Decimal(p1.position)
            except:
                print(pop_C)
            
            try:
                p2_decimal = Binary_to_Decimal(p2.position)
            except:
                print(pop_C)
            
            if p1_decimal == p2_decimal:
              q = np.random.permutation(num_pop)
              p2 = pop[q[2]]
            else:
              print("Parents Different")
            
            #p1 = pop[roulette_wheel_selection(probs)]
            #p2 = pop[roulette_wheel_selection(probs)]
            
            
            # crossover
            c1, c2 = crossover(p1, p2, crs_mu)
            
            # mutation
            c1 = mutate(c1, mu)
            c2 = mutate(c2, mu)
            
            policy_1 = Binary_to_Decimal(c1.position)
            policy_2 = Binary_to_Decimal(c2.position)
            
            print("Policy 1: " + str(policy_1) + " Policy 2: " + str(policy_2))
            print("---------------")
            
            # Evaluate First Offspring

            if policy_1 < 42:
              c1.cost = AutoML_Model(policy_1)
              #c1.cost = np.random.random() * 10
              if c1.cost > bestsol.cost:
                  bestsol = c1.deepcopy()
            else:

              if np.random.uniform(low = 0.0, high = 1.0, size = 1) > 0.7:
                choices =  [35, 33, 29, 26]

                c1_decimal = np.random.choice(choices)

                if c1_decimal == 35:
                    c1.position = np.array([1, 0, 0, 0, 1, 1])
                elif c1_decimal == 33:
                    c1.position = np.array([1, 0, 0, 0, 0, 1])

                elif c1_decimal == 29:
                    c1.position = np.array([0, 1, 1, 1, 0, 1])

                elif c1_decimal == 26:
                    c1.position = np.array([0, 1, 1, 0, 1, 0])

                policy_1 = c1_decimal

                print("Policy 1: " + str(policy_1))
                print("---------------")

                c1.cost = AutoML_Model(policy_1)
                #c1.cost = np.random.random() * 10
                if c1.cost > bestsol.cost:
                    bestsol = c1.deepcopy()
              else:
                c1.cost = -10.0

            # Evaluate Second Offspring

            if policy_2 < 42:
              c2.cost = AutoML_Model(policy_2)
              #c2.cost = np.random.random() * 10
              if c2.cost > bestsol.cost:
                  bestsol = c2.deepcopy()
            else:

              if np.random.uniform(low = 0.0, high = 1.0, size = 1) > 0.7:
                choices = [23, 24, 22, 19, 14, 13]
                
                c2_decimal = np.random.choice(choices)
                
                if c2_decimal ==23:
                    c2.position = np.array([0, 1, 0, 1, 1, 1])
                elif c2_decimal == 24:
                    c2.position = np.array([0, 1, 1, 0, 0, 0])
                elif c2_decimal== 22:
                    c2.position = np.array([0, 1, 0, 1, 1, 0])
                elif c2_decimal== 19:
                    c2.position = np.array([0, 1, 0, 0, 1, 1])
                elif c2_decimal== 13:
                    c2.position = np.array([0, 0, 1, 1, 0, 1])
                elif c2_decimal== 14:
                    c2.position = np.array([0, 0, 1, 1, 1, 0])

                policy_2 = c2_decimal

                print("Policy 2: " + str(policy_2))
                print("---------------")

                c2.cost = AutoML_Model(policy_2)
                #c2.cost = np.random.random() * 10
                if c2.cost > bestsol.cost:
                    bestsol = c2.deepcopy()
              else:
                c2.cost = -10.0
        
            indx_1, indx_2, it_1 = policy_1, policy_2, it
            
            if policy_1 < number_of_policies and policy_2 < number_of_policies:
                
                rewards[indx_1, it_1] = c1.cost
                
                rewards[indx_2, it_1] = c2.cost
            
            pop_C.append(c1)
            pop_C.append(c2)
            
        # Merge, Sort and Select
        pop += pop_C
        pop = sorted(pop, key=lambda x: x.cost)
        pop = pop[0:num_pop]

        # Store Best Cost
        bestcost[it] = bestsol.cost

        # Show Iteration Information
        print("Iteration {}: Best Cost = {}".format(it, bestcost[it]))
        
        if np.mod(it+1, 5) == 0:
            
            out = structure()
            out.pop = pop
            out.bestsol = bestsol
            out.bestcost = bestcost
            out.best_model = best_model
            out.best_models = best_models
            out.rewards = rewards
            
            out_keys = list(out.keys())

            out_pos = {'position':out[out_keys[0]]['position'],
                       'cost': out[out_keys[0]]['cost']}

            out_best_costs = out[out_keys[1]]

            out_rewards = out[out_keys[-1]]
            
            name_1 = 'BGA_Results_Out_Pos_Version_' + str(it+44)
            
            name_2 = 'BGA_Results_Out_best_costs_Version_' + str(it + 44)
            
            name_3 = 'BGA_Results_Out_rewards_Version_' + str(it + 44)

            pickle.dump(out_pos, open(os.path.join(save_directory, name_1), 'wb'))

            pickle.dump(out_best_costs, open(os.path.join(save_directory, name_2), 'wb'))

            pickle.dump(out_rewards, open(os.path.join(save_directory, name_3), 'wb'))
                
    # Output
    out = structure()
    out.pop = pop
    out.bestsol = bestsol
    out.bestcost = bestcost
    out.best_model = best_model
    out.best_models = best_models
    out.rewards = rewards
    
    end_time = time.time()
    
    execusion_time = end_time - start_time
    
    return out, execusion_time


out, execusion_time = Binary_Genetics(problem, params)

out_keys = list(out.keys())

out_pos = {'position':out[out_keys[0]]['position'],
           'cost': out[out_keys[0]]['cost']}

out_best_costs = out[out_keys[1]]

out_rewards = out[out_keys[-1]]

pickle.dump(out_pos, open(os.path.join(save_directory, 'BGA_Results_Out_Pos_Version'), 'wb'))

pickle.dump(out_best_costs, open(os.path.join(save_directory, 'BGA_Results_Out_best_costs_Version'), 'wb'))

pickle.dump(out_rewards, open(os.path.join(save_directory, 'BGA_Results_Out_rewards_Version'), 'wb'))
    
x_lim = np.linspace(1,params.Max_Iteration,params.Max_Iteration)

# Results
plt.plot(x_lim, out.bestcost)
# plt.semilogy(out.bestcost)
plt.xlim(0, params.maxit)
plt.xlabel('Iterations')
plt.ylabel('Best Cost')
plt.title('Genetic Algorithm (GA)')
plt.grid(True)
plt.show()

"""

#Brute force

print("Brute force method is used")

iterations = 20

brute_force_rewards = np.zeros([number_of_policies,iterations])

start_time_brute = time.time()

for it in range(iterations):

    for i in range(number_of_policies):
        
        policy = i
        
        brute_force_rewards[i, it] = AutoML_Model(policy)
    
end_time_brute = time.time()

entire_time= end_time_brute - start_time_brute

pickle.dump(brute_force_rewards, open(os.path.join(save_directory, 'Brute_force_rewards_Version_III'), 'wb'))
   
""" 