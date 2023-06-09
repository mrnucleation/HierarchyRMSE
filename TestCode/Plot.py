from NeuralNet import NeuralNet, kerasmodel
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('finalmodel.keras')
X_Test = np.linspace(0, 1.0, 100)
Y_Test = np.sin(X_Test*2*np.pi) 
Y_Pred = model.predict(X_Test)
print(Y_Pred)
'''
minscore = 1e300
minparams = None
with open('dumpfile.dat', "r") as f:
    for line in f:
        col = line.split('|')
        parameters = np.array([float(x) for x in col[0].split()])
        score = float(col[1])
        if score < minscore:
            minscore = score
            minparams = parameters
            print(minscore)
print(minparams)
weights = model.get_weights()
nweights = 0
for i, layer in enumerate(weights):
    for j, x in np.ndenumerate(layer):
        weights[i][j] = minparams[nweights]
        nweights += 1
model.set_weights(weights)
    
'''


#A function which plots the data set and the function y = sin(x) using matplotlib.
def plot(X, Y, Y_Pred):
    import matplotlib.pyplot as plt
    plt.plot(X, Y, 'o')
    plt.plot(X, Y_Pred)
    plt.show()
#    plt.savefig('fig.png')
    
plot(X_Test, Y_Test, Y_Pred)
