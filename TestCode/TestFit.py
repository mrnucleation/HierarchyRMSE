import sys
sys.path.append('../')
from Hierch_RMSE import HierObjective
from NeuralNet import NeuralNet
import numpy as np
from MCTSOpt import Tree
from LogisticSearch import LogisticSearch
#from BayesianObject import BayesianData as LogisticSearch
from SelectionRule_UBUnique import UBUnique
from math import sqrt

#Create a data set using numpy which plots the function y = sin(x).
#The data set is a 2D array with 100 rows and 2 columns.

datasets = []
for i in range(3):
    X = np.random.uniform(0, 2*np.pi, size=30)
    Y = np.sin(X) + np.random.normal(0.0, 0.1, size=X.shape[0])
    datasets.append((X, Y))

model = NeuralNet(1, 1)

weights = model.get_weights()
nweights = 0
for layer in weights:
    nweights += layer.size

print(nweights)

objective = HierObjective(datasets, model)

depthscale = [10.0,  0.4, 0.2, 0.1, 0.05, 0.01]
depthscale = [sqrt(nweights)*x for x in depthscale]

startset = np.random.normal(0, 0.001, size=nweights)
ubounds = np.full(nweights, 5.5)
lbounds = np.full(nweights, -5.5)

options ={'verbose':2}




indata = LogisticSearch(parameters=startset, ubounds=ubounds, lbounds=lbounds, lossfunction=objective, depthscale=depthscale, options=options)
tree = Tree(seeddata=indata, 
        playouts=10, 
        selectfunction=UBUnique, 
        headexpansion=15,
        verbose=True)
tree.expandfromdata(newdata=indata)
depthlimit = 25
for iLoop in range(1,500):
    print("Loop Number: %s"%(iLoop))
    tree.playexpand(nExpansions=1, depthlimit=depthlimit)
    tree.simulate(nSimulations=1)
    tree.autoscaleconstant(scaleboost=0.5)
    minval = tree.getbestscore()
    if minval < 5e-4:
        break
    print(tree)

