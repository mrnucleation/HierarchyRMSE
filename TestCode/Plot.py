from NeuralNet import NeuralNet
import numpy as np

model = NeuralNet(1, 1)
X_Test = np.random.uniform(0, 2*np.pi, size=100)
Y_Test = np.sin(X_Test) + np.random.normal(-0.01, 0.01, size=X_Test.shape[0])
Y_Pred = model.predict(X_Test)
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
model = NeuralNet(1, 1)
weights = model.get_weights()
nweights = 0
for i, layer in enumerate(weights):
    for j, x in np.ndenumerate(layer):
        weights[i][j] = minparams[nweights]
        nweights += 1
model.set_weights(weights)
    


#A function which plots the data set and the function y = sin(x) using matplotlib.
def plot(X, Y, Y_Pred):
    import matplotlib.pyplot as plt
    plt.plot(X, Y, 'o')
    plt.plot(X, Y_Pred)
    plt.show()
    
plot(X_Test, Y_Test, Y_Pred)