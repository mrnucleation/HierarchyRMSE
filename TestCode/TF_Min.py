import sys
sys.path.append('../')
import os
from math import log, fabs

from time import time
from Hierch_RMSE import HierObjective
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from NeuralNet import kerasmodel
import numpy as np

datasets = []
for i in range(3):
    X = np.random.uniform(0, 1.0, size=50)
    Y = np.sin(2*np.pi*X) 
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    datasets.append((X, Y))

print(X)
print(Y)

#model = NeuralNet(1, 1)
model = kerasmodel()

objective = HierObjective(datasets, model)
objective = objective.heracleobj


curweight = model.get_weights()
cnt = 0
for i, row in enumerate(curweight):
    for j, x in np.ndenumerate(row):
        cnt += 1
nPar = cnt

print(nPar)

parameters = list(np.random.uniform(low=-1.5, high=1.5, size=nPar+1))
print(len(parameters))

curweight = model.get_weights()
cnt = -1
for i, row in enumerate(curweight):
    for j, x in np.ndenumerate(row):
        cnt += 1
        curweight[i][j] = parameters[cnt]
model.set_weights(curweight)

model.summary()


#Sample training loop pulled from the Tensorflow webpage.
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))



# Instantiate an optimizer.
optimizer = keras.optimizers.Adam(learning_rate=4e-2)
#optimizer = keras.optimizers.SGD(learning_rate=9e-9)

# Instantiate a loss function.
epochs = 2000



for epoch in range(epochs):
    # Iterate over the batches of the dataset.
    # Open a GradientTape to record the operations run
    # during the forward pass, which enables auto-differentiation.
    timestart = time()
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        loss_value = objective(parameters=None)
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    timeend = time()
    print("Training loss at step %d: %.10f, time:%s"
                % (epoch, float(loss_value),timeend-timestart)
            )


curweight = model.get_weights()
cnt = -1
endparameters = []
for i, row in enumerate(curweight):
    for j, x in np.ndenumerate(row):
        endparameters.append(curweight[i][j])

with open('dumpfile.dat', 'w') as outfile:
    outstr = ' '.join([str(x) for x in endparameters])
    outfile.write('%s | %s \n'%(outstr, 1.0))

model.save('finalmodel.keras')
