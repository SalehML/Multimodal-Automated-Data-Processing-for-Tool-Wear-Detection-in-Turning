# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 21:23:00 2023

@author: Saleh
"""

import numpy as np
import ypstruct 
from ypstruct import structure
from matplotlib import pyplot as plt
from MachineLearningUnit_journal import AutoML_Model

# Problem definition

problem = structure()

problem.nVar = 12

best_models = []


# GA parameters

params = structure()

params.Max_Iteration = 250
params.num_pop = 20
params.pC = 1
params.beta = 1
params.mu = 0.08

# Run GA

def Binary_to_Decimal(binary_num):
    
    array_string = ''.join(map(str, binary_num))
    
    binary_num = '0b' + array_string
    
    return int(binary_num, 2)

def unify(x):
    
    y = ''.join(x.astype(str))
    
    y = int(y)
    
    return y

def roulette_wheel_selection(p):
    c = np.cumsum(p)
    r = sum(p)*np.random.rand()
    ind = np.argwhere(r <= c)
    return ind[0][0]

def crossover(p1, p2):
    
    nvar = np.size(p1['position'])
    
    j = np.random.randint(nvar - 1)
    
    c1 = p1.deepcopy()
    
    c2 = p2.deepcopy()
    
    c1_pos_1, c1_pos_2 = list(p1['position'][:j]), list(p2['position'][j:])
    
    c1_pos = c1_pos_1 + c1_pos_2
    
    c2_pos_1, c2_pos_2 = list(p2['position'][:j]), list(p1['position'][j:])
    
    c2_pos = c2_pos_1 + c2_pos_2
    
    c1.position = np.array(c1_pos)
    
    c2.position = np.array(c2_pos)
    
    return c1, c2
    
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
    
    # Empty Individual Template
    empty_individual = structure()
    empty_individual.position = None
    empty_individual.cost = None

    # Best Solution Ever Found
    bestsol = empty_individual.deepcopy()
    bestsol.cost = -1000000.0
    
    # Initialize Population
    pop = empty_individual.repeat(num_pop)
    
    for i in range(num_pop):
        
        pop[i].position = np.random.randint(0, 2, num_var)
        
        policy = Binary_to_Decimal(pop[i].position)
        
        pop[i].cost, _  = AutoML_Model(policy)
        
#    return pop[i].cost
        
#out = Binary_Genetics(problem, params)

        if pop[i].cost > bestsol.cost:
            bestsol = pop[i].deepcopy()
            
    # Best Cost of Iterations
    bestcost = np.empty(Max_Iteration)
    
    print("initialization done!")
    
    made_policies = []
    
    for it in range(Max_Iteration):
        
        count = len(made_policies)
        
        if count < 2400:
        
            costs = np.array([x.cost for x in pop])
            avg_cost = np.mean(costs)
            if avg_cost != 0:
                costs = costs/avg_cost
            probs = np.exp(-beta*costs)
            
            #pop_c = empty_individual.repeat(nC)
            
            #pop_c = np.array(pop_c).reshape(int(nC/2) , 2)
            
            pop_C = []
            
            for _ in range(int(nC//2)):
                
                
                #Select Parents
                q = np.random.permutation(num_pop)
                p1 = pop[q[0]]
                p2 = pop[q[1]]
                
                
                #p1 = pop[roulette_wheel_selection(probs)]
                #p2 = pop[roulette_wheel_selection(probs)]
                
                
                # crossover
                c1, c2 = crossover(p1, p2)
                
                # mutation
                c1 = mutate(c1, mu)
                c2 = mutate(c2, mu)
                
                c1_unified = unify(c1.position)
                c2_unified = unify(c2.position)
                
                if c1_unified not in made_policies:
                    
                    made_policies.append(c1_unified)
                    
                    policy_1 = Binary_to_Decimal(c1.position)
                    
                    # Evaluate First Offspring
                    c1.cost, model_1 = AutoML_Model(policy_1)
                    if c1.cost > bestsol.cost:
                        bestsol = c1.deepcopy()
                        best_models.append(model_1)
                        
                elif c1_unified in made_policies:
                    
                    pass
                
                
                if c2_unified not in made_policies:
                    
                    made_policies.append(c1_unified)
                
                    policy_2 = Binary_to_Decimal(c2.position)
                
                
    
                    # Evaluate Second Offspring
                    c2.cost, model_2 = AutoML_Model(policy_2)
                    if c2.cost > bestsol.cost:
                        bestsol = c2.deepcopy()
                        best_models.append(model_2)
                        
                elif c2_unified in made_policies:
                    
                    pass
            
                
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
            
        elif count == 2400:
            
            pass
        
    # Output
    out = structure()
    out.pop = pop
    out.bestsol = bestsol
    out.bestcost = bestcost
    return out


out = Binary_Genetics(problem, params)
    

# Results
plt.plot(out.bestcost)
# plt.semilogy(out.bestcost)
plt.xlim(0, params.maxit)
plt.xlabel('Iterations')
plt.ylabel('Best Cost')
plt.title('Genetic Algorithm (GA)')
plt.grid(True)
plt.show()
