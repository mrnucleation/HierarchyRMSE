#=============================================================================================
from HeadObjective import HeadObjective
from RMSE_Fragment import RMSE_Fragment
#=============================================================================================
class HierObjective():
    '''
    This class is used to create a symbolic objective function that can be used as a function
    for optimization. The objective function is a chain of heriacle objects. 
    '''
    #---------------------------------------------------------------------
    def __init__(self, datasets, model, dumpfilename='dumpfile.dat'):
        #Create the initial list.
        objstack = [ HeadObjective(model, nullscore=0.0) ]
        
        #Create the list of objects to be used in the objective function.
        for dataset in datasets:
            X_train, Y_train = dataset
            rmseobj = RMSE_Fragment(X_train, Y_train, parentobj=None, pointtol=0.05, meantol=0.5, nullscore=0.7)
            objstack.append(rmseobj)

        #We now embed all the objects into a chain of heriacle objects.
        if len(objstack) > 1:
            for i, obj in enumerate(objstack):
                if i == 0:
                    continue
                print(obj)
                objstack[i-1].addchild(obj)
        self.heracleobj = objstack[0]
        self.heracleobj.printinfo()
        
        #Create a dump file to store the results of the optimization.
        self.dumpfilename = dumpfilename
        if self.dumpfilename is not None:
            self.dumpfile = open(self.dumpfilename, 'w')

    #---------------------------------------------------------------------
    def __call__(self, parameters, verbose=False, **kwargs):
        score = self.heracleobj(parameters=parameters, verbose=verbose, **kwargs)
        if not verbose or not self.dumpfilename is None:
            outstr = ' '.join([str(x) for x in parameters])
            self.dumpfile.write('%s | %s \n'%(outstr, score))

        return score
    #---------------------------------------------------------------------
    def __del__(self):
        try:
            self.dumpfile.close()
        except AttributeError:
            pass
    #---------------------------------------------------------------------
#============================================================================================
if __name__ == "__main__":
    import sys
    filename = sys.argv[1]
    par = []
    with open(filename, 'r') as parfile:
        for line in parfile:
            newpar = float(line.split()[0])
            par.append(newpar)
    objective = SymbolicObjective(datasets)
    score = objective(par, verbose=True)
    print("Test Score: %s"%score)

