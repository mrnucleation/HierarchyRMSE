import sys
sys.path.append('../')
from Hierch_RMSE import HierObjective
from NeuralNet import NeuralNet, kerasmodel
import numpy as np
from MCTSOpt import Tree
from LogisticSearch import LogisticSearch
#from UniformSearch import UniformSearch as LogisticSearch
#from BayesianObject import BayesianData as LogisticSearch
from SelectionRule_UBUnique import UBUnique
from math import sqrt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#Create a data set using numpy which plots the function y = sin(x).
#The data set is a 2D array with 100 rows and 2 columns.

datasets = []
for i in range(1):
    X = np.random.uniform(0, 1.0, size=50)
    Y = np.sin(2.0*np.pi*X) 
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    datasets.append((X, Y))

print(X)
print(Y)

#model = NeuralNet(1, 1)
model = kerasmodel()
model.summary()
nweights = 0
for layer in model.layers: 
    print(layer.get_config())
    weights = layer.get_weights()[0]
    biasweights = layer.get_weights()[1]
    print(weights)
    print(biasweights)
    nweights += weights.size + biasweights.size
#    nweights += weights.size 

print(nweights)


objective = HierObjective(datasets, model)

depthscale = [10.0, 0.5, 0.4] + [0.3 for i in range(3)] 
#depthscale = [sqrt(nweights)*x for x in depthscale]

startset = np.random.normal(0, 0.01, size=nweights)
ubounds = np.full(nweights, 3.7)
lbounds = np.full(nweights, -3.7)

options ={'verbose':2}




indata = LogisticSearch(parameters=startset, ubounds=ubounds, lbounds=lbounds, lossfunction=objective, depthscale=depthscale, options=options)
tree = Tree(seeddata=indata, 
        playouts=10, 
        selectfunction=UBUnique, 
        headexpansion=10,
        verbose=True)
tree.expandfromdata(newdata=indata)
tree.expand(nExpansions=1)
depthlimit = 35000
for iLoop in range(1,500):
    print("Loop Number: %s"%(iLoop))
    tree.playexpand(nExpansions=1, depthlimit=depthlimit)
    tree.simulate(nSimulations=1)
    tree.autoscaleconstant(scaleboost=0.5)
    minval = tree.getbestscore()
    if minval < 5e-4:
        break
    print(tree)

