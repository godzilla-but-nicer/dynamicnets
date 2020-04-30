# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 13:50:37 2019

@author: Lauren Benson
"""

import CTRNN_Lauren
import matplotlib.pyplot as plt
import numpy as np


#Global parameters
size = 100
duration = 30
stepsize = 0.1
repetitions = 1000

possible_connections = np.arange(0,size**2,1)
possible_connections_ordered = np.flip(possible_connections)

MeansList = np.zeros(len(possible_connections))
StdList = np.zeros(len(possible_connections))

k=0
Reps = np.zeros((len(possible_connections),repetitions))
for c in possible_connections_ordered:
    for r in range(repetitions):
        time = np.arange(0.0,duration,stepsize)
        nn = CTRNN_Lauren.CTRNN(size)
        nn.randomizeParameters()
        nn.setweightmatrix3(c)
        nn.initializeState(np.zeros(size))
        outputs1 = np.zeros((len(time)+1,size))
        outputs1[0] = nn.Output
        TimeSum = 0
        step = 1
        for t in time:
            nn.step(stepsize)
            outputs1[step] = nn.Output
            NewOutput = np.absolute(outputs1[step] - outputs1[step-1])
            SumNewOutput = np.sum(NewOutput)
            TimeSum = TimeSum + SumNewOutput
            step = step+1
        NewTimeSum = (TimeSum) / (duration * size)
        Reps[k,r] = NewTimeSum #iterates over repetitions 
    k = k+1 #iterates over possible connections

np.save('Reps.npy',Reps)
Avg = np.mean(Reps,axis=1)
Std = np.std(Reps,axis=1)


    
sem = Std / np.sqrt(repetitions)
plt.plot(possible_connections, Avg)
plt.fill_between(possible_connections, Avg-sem, Avg+sem, alpha=0.2)
plt.xlabel("Number of connections")
plt.ylabel("Change in Activity")
plt.title("Total Neural activity")
plt.show()

   #     TotalAvg = NewTimeSum + TotalAvg
  #  NewTotalAvg = TotalAvg / repetitions #this is the average neural output over all reps (one value)
  #  NewTotalAvgList.append(NewTotalAvg) #This is an array of each average neural output for each number of connections
   # NewTotalAvgListStd.append(np.std(NewTotalAvgList))
   # np.save('avg_N.O_Per_Connection', NewTotalAvgList)
   # np.save('Std_per_connection', NewTotalAvgListStd)
   # Errors = NewStdV2/(np.sqrt(repetitions))
   # np.save('ErrorBars_Per_Connection',Errors)
    
    
#Research Question: How does changing the number of connections in the network affect neural output?
# =============================================================================
# 
# 
# plt.plot(possible_connections, NewTotalAvgList)
# plt.errorbar(possible_connections,NewTotalAvgList,yerr=Errors, label = 'average over each possible connection', fmt='o')
# plt.xlabel("Number of connections")
# plt.ylabel("Output")
# plt.title("Neural activity")
# plt.legend(loc='lower right')
# plt.show()        
# =============================================================================
