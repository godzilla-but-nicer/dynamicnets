# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 16:13:40 2019

@author: benso
"""

import mga
import numpy as np
import matplotlib.pyplot as plt

popsize = 100
genesize = 10
recombProb = 0.75
mutatProb = 0.01
generations = 50
tournaments = generations * popsize
repetitions = 5

def fitnessFunction(genotype):
    return np.sum(genotype)

avghist=[]
besthist=[]
for i in range(repetitions):
    ga = mga.Microbial(fitnessFunction, popsize, genesize, recombProb, mutatProb)
    ga.run(tournaments)
    avghist.append(ga.avgHistory)
    besthist.append(ga.bestHistory)

c1b = np.mean(np.array(besthist),axis=0)
c1a = np.mean(np.array(avghist),axis=0)

recombProb = 0.5

avghist=[]
besthist=[]
for i in range(repetitions):
    ga = mga.Microbial(fitnessFunction, popsize, genesize, recombProb, mutatProb)
    ga.run(tournaments)
    avghist.append(ga.avgHistory)
    besthist.append(ga.bestHistory)

c2b = np.mean(np.array(besthist),axis=0)
c2a = np.mean(np.array(avghist),axis=0)

plt.plot(c1b)
plt.plot(c1a)
plt.plot(c2b)
plt.plot(c2a)
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.title("Best and average fitness")
plt.show()

#ga.showFitness()
#avgfit, bestfit, bestind = ga.fitStats()
#print(bestind)