from ObjectiveClass import HeriacleObjective
from time import time
import os
import numpy as np

#==============================================================================
class RMSE_Fragment(HeriacleObjective):
    '''
    This is a simple objective function that calculates the Root Mean Square Error
    of a data set using a fragment of the data set. The goal of this is to break
    down the objective function into smaller pieces to force the optimizer
    to obtain a reasonable value for each and every chunk of data.
    '''
    #----------------------------------------------------------
    def __init__(self, X_train, Y_train, parentobj=None, pointtol=1.0, meantol=0.1, nullscore=1.0, **kwargs):
        # Initialize the parent class
        super(RMSE_Fragment, self).__init__(parentobj=parentobj, nullscore=nullscore)
        
        # The name of the objective function
        self.name = 'RMSE_Fragment'
        
        # The data set to be used for the objective function
        self.X_train = X_train
        self.Y_train = Y_train
        
        # The tolerances for the objective function
        #Pointtol is the maximum error allowed for a single point
        #Meantol is the maximum error allowed for the mean of the data set
        self.pointtol = pointtol
        self.meantol = meantol
        


    #----------------------------------------------------------
    def __call__(self, parameters, depth=0, **kwargs):
        # Initialize the score
        score = 0.0
        model = kwargs['model']
        
        # Calculate the RMSE of the data set
        Y_pred = model.predict(parameters)
        Y_rmse = self.rmse(self.Y_train, Y_pred)
        
        #Compute the mean and maximum disagreement. If the mean is less than the
        #meantol or the maximum is less than the pointtol, then the score is
        #the mean of the RMSE. Otherwise, the score is the nullscore.
        rmse_mean = np.sqrt(np.mean(Y_rmse))
        rmse_max = np.max(Y_rmse)
        score += rmse_mean
        print('Depth:%s, RMSE Mean:%s , RMSE Max:%s'%(depth,rmse_mean, rmse_max))
        if rmse_mean < self.meantol and rmse_max < self.pointtol:
            score += self.getchildscores(parameters=parameters, depth=depth, **kwargs)
        else:
            score -= self.nullscore
            score += self.getnullscores(depth=depth)
        return score
    #----------------------------------------------------------
    def rmse(self, predict, target):
        '''
        This function calculates the root mean square error of the data set.
        It returns an array of the RMSE for each point in the data set.
        This is done so both the max and mean can be calculated.
        
        data: The data set to be used for the objective function
        target: The target data set to be used for the objective function
        '''
        return np.mean(predict - target)**2
    #----------------------------------------------------------
#==============================================================================
