from ScriptTemplate import InputTemplate
from ObjectiveClass import HeriacleObjective
from time import time
import os
import numpy as np

#==============================================================================
class HeadObjective(HeriacleObjective):
    '''
    This is the head objective class. It is the top level objective which 
    performs tasks such as cleaning up files from previous runs, initializes parameters,
    or performs other tasks that need to be done before the child objectives are called.
    This does not perform any direct calculations, but is simply for organizing the
    hierarchy of objectives.
    '''
    #----------------------------------------------------------
    def __init__(self, parentobj=None, nullscore=1.0):
        super().__init__(parentobj=parentobj, nullscore=nullscore)


    #----------------------------------------------------------
    def __call__(self, parameters, depth=0, **kwargs):
        starttime = time()
        print("Starting Chain, Depth: %s"%(depth))
        score = 0.0
        score += self.getchildscores(parameters=parameters, depth=depth, **kwargs)
        if np.isnan(score):
            score = 1e300
        print("Total Score: %s"%(score))

        return score
    #----------------------------------------------------------
#==============================================================================
