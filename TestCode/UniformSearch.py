import numpy as np
from ParameterObject import ParameterData

#================================================
class UniformSearch(ParameterData):
    '''
     Uniform Search Mode.  This replaces the hyper-sphere expansion style playouts with uniformly generated points.
     This search mode works fairly well for low dimensional search spaces, but tends to struggle as the size of
     the search space increases.  Uniform search modes tend to produce large displacement vectors for a higher number
     of parameters.
    '''
    #------------------------------------------------
    def __init__(self, parameters, lossfunction, lbounds, ubounds, depthscale=None, options={}):
        super().__init__(parameters, lossfunction, lbounds, ubounds, depthscale, options)

    #------------------------------------------------
    def newdataobject(self):
        newobj = UniformSearch(parameters=self.parameters, lossfunction=self.lossfunction, lbounds=self.lbounds, ubounds=self.ubounds,
                              depthscale=self.depthscale, options=self.options)
        return newobj

    #------------------------------------------------
    def perturbate(self, node=None, parOnly=False):
        '''
         Function that when called produces a new parameter set by taking this data object's parameters
         and displacing it.
        '''
        self.computechildbounds(node)
        newlist = None
        while newlist is None:
            newlist = self.localshift(node=node)
        if not parOnly:
            newobj = UniformSearch(parameters=newlist, lossfunction=self.lossfunction, lbounds=self.lbounds, ubounds=self.ubounds,
                                   depthscale=self.depthscale, options=self.options)
            return newobj
        else:
            return newlist
    #------------------------------------------------
    def localshift(self, node=None):
        self.computechildbounds(node)
        newlist = np.random.uniform(self.childlb, self.childub)
#        newlist = np.multiply(self.ubounds-self.lbounds, x_r) + self.lbounds
        return newlist
    #----------------------------------------------------
#================================================
