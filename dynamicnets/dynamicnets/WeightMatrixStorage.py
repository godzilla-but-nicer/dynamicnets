import numpy as np


class WeightMatrix:
    def __init__(self, nodes, weight_matrix, edgelist, fitness):
        """ this class just makes an object that lets us access weight matrices
        from previous evolution runs easily """
        self.size = nodes
        self.weight_matrix = weight_matrix
        self.edgelist = edgelist
        self.id = edgelist.split('/')[-1].split('_')[2].split('.')[0]
        self.fitness = fitness
    
    def print_info(self):
        """ just print out some details"""
        print('Weight Matrix Details:')
        print('Size:', self.size)
        print('id:', self.id)
        print('Fitness:', self.fitness)
        print('Edgelist:', self.edgelist)
        print('use "this.weight_matrix" for numpy array')


class WeightMatrixStorage:
    def __init__(self):
        """ this class lets us organize the weight matrices so that we can find
        them later """
        self.weight_matrices = []

    def get_matrices(self, nodes, id):
        ret_list = []
        for wm in self.weight_matrices:
            if wm.size == nodes and wm.id == id:
                ret_list.append(wm)

        return ret_list

    def get_matrices_by_fitness(self, fitness, cond='gt'):
        """ cond can be 'gt' for greater than or 'lt' for less than"""
        ret_list = []
        if cond == 'gt':
            for wm in self.weight_matrices:
                if wm.fitness > fitness:
                    ret_list.append(wm)
        
        elif cond == 'lt':
            for wm in self.weight_matrices:
                if wm.fitness < fitness:
                    ret_list.append(wm)
        
        return ret_list

    def add_matrix(self, wm):
        self.weight_matrices.append(wm)
