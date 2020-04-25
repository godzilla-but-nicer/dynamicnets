# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 11:52:41 2019

@author: Lauren Benson
"""

import random
import numpy as np
import matplotlib.pyplot as plt


class Microbial():
    def __init__(self, fitnessFunction, popsize, genesize, recombProb,
                 mutatProb):
        self.fitnessFunction = fitnessFunction
        self.popsize = popsize
        self.genesize = genesize
        self.recombProb = recombProb
        self.mutatProb = mutatProb
        self.pop = np.random.rand(popsize, genesize)*2 - 1
        self.avgHistory = []
        self.bestHistory = []

    def showFitness(self):
        plt.plot(self.bestHistory)
        plt.plot(self.avgHistory)
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.title("Best and average fitness")
        plt.show()

    def fitStats(self):
        bestfit = 0
        bestind = -1
        avgfit = 0
        for i in self.pop:
            fit = self.fitnessFunction(i)
            avgfit += fit
            if (fit > bestfit):
                bestfit = fit
                bestind = i
        return avgfit/self.popsize, bestfit, bestind

    def run(self, tournaments):

        # Evolutionary loop
        for i in range(tournaments):

            # Report statistics every generation
            if (i % self.popsize == 0):
                # print(i/self.popsize)
                af, bf, bi = self.fitStats()
                self.avgHistory.append(af)
                self.bestHistory.append(bf)

            # Step 1: Pick 2 individuals
            a = random.randint(0, self.popsize-1)
            b = random.randint(0, self.popsize-1)
            while (a == b):   # Make sure they are two different individuals
                b = random.randint(0, self.popsize-1)

            # Step 2: Compare their fitness
            if (self.fitnessFunction(self.pop[a]) >
                    self.fitnessFunction(self.pop[b])):
                winner = a
                loser = b
            else:
                winner = b
                loser = a

            # Step 3: Transfect loser with winner
            for l in range(self.genesize):
                if (random.random() < self.recombProb):
                    self.pop[loser][l] = self.pop[winner][l]

            # Step 4: Mutate loser and Make sure new organism stays in bounds
            for l in range(self.genesize):
                self.pop[loser][l] += random.gauss(0.0, self.mutatProb)
                if self.pop[loser][l] > 1.0:
                    self.pop[loser][l] = 1.0
                if self.pop[loser][l] < -1.0:
                    self.pop[loser][l] = -1.0
