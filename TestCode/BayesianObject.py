from random import random, shuffle
from time import time
from copy import deepcopy
from math import sqrt
from ParameterObject import ParameterData

try:
    from hyperopt import hp, fmin, tpe, space_eval, Trials
except:
    raise ModuleNotFoundError("The HyperOpt Python Package is Required for this Sampling Approach. Trying using 'pip install hyperopt' to install.")
    
import sys
import numpy as np
#================================================
class MissingParameters(Exception):
    pass
#================================================
class InvalidParameterBounds(Exception):
    pass
#================================================
class BayesianData(ParameterData):
    '''
     Default Hypersphere Search for a cMCTS algorithm. 
    '''
    #------------------------------------------------
    def __init__(self, parameters, lossfunction, lbounds, ubounds, depthscale=None, options={}, parentscale=1.0):
        '''
         Input Variables:
           parameters => List or numpy array of the trial parameters for this data obbject
           lossfunction => C
           lbounds => Lower bounds for the parameter search. 
           ubounds => Upper bounds for the parameter search. 
           depthscale => The maximum distance that rollouts and child nodes are allowed
           options => Additional non-manditory control values that can be set.
        '''
        super().__init__(parameters, lossfunction, lbounds, ubounds, depthscale, parentscale, options)
        print("BAYESIAN")
        self.bay_trials = None
    #------------------------------------------------
    def newdataobject(self):
        newobj = BayesianData(parameters=self.parameters, lossfunction=self.lossfunction, lbounds=self.lbounds, ubounds=self.ubounds,
                              depthscale=self.depthscale, options=self.options)
        return newobj
    #------------------------------------------------------------------------------
    def runsim(self, playouts=1, node=None):
        tries = 0
        structlist = []
        energylist = []

        searchmax = self.getsearchmax(node)
        self.computechildbounds(node)
        space = []
        i = 0
        for lower, upper in zip(self.childlb, self.childub):
            space.append(hp.uniform(str(i), lower, upper))
            i += 1

        if self.bay_trials is None:
            trials = Trials()
            nprev = 0
        else:
            trials = self.bay_trials
            nprev = len(trials.trials)


        results = fmin(self.lossfunction, space=space, algo=tpe.suggest, max_evals=nprev+playouts, trials=trials, verbose=0)
            
        self.bay_trials = trials
        cnt = nprev

        for trial, result in zip(trials.trials[nprev:], trials.results[nprev:]):
            newlist = [trial['misc']['vals'][str(i)][0] for i in range(len(self.parameters))]
            energy = result['loss']
            cnt += 1
            if self.verbose:
                print("Playout %s Result: %s"%(cnt, energy))
            structlist.append(newlist)
            energylist.append(energy)
        newlist = [results[str(x)] for x in range(self.dimensions)]
#        for energy, struct in zip(energylist, structlist):
#            self.addhist(struct)
#            self.addscorehist(energy, struct)
        energy = min(trials.losses())
        return energylist, structlist     
    #----------------------------------------------------
#================================================
