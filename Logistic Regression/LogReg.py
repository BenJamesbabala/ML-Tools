import pandas as pd
import numpy as np
import sys

arguments = sys.argv
trainFile = str(arguments[1])
testFile = str(arguments[2])
lr = float(arguments[3])
reg = float(arguments[4])
modelF = str(arguments[5])

data = pd.read_csv(trainFile)
testdata = pd.read_csv(testFile)
modelFile = modelF


Y = data[data.columns[-1]].as_matrix()


testY = testdata[testdata.columns[-1]].as_matrix()


Xframe = data.drop(data.columns[-1], axis=1)
X = Xframe.as_matrix()
X = np.append(np.ones((X.shape[0],1)),X,axis=1)


testXframe = testdata.drop(data.columns[-1],axis=1)
testX = testXframe.as_matrix()
testX = np.append(np.ones((testX.shape[0],1)),testX,axis=1)

def sigmoid(x):
    g = 1 / (1+np.exp(-x))
    return g


w = np.zeros(X.shape[1])
##lr = 1;
##reg = 1;
m = len(Y)

gradient = np.array([1,2])
while np.sqrt(gradient.dot(gradient)) > .001:
    a = sigmoid(np.dot(X,w))
    cost = (1/m) * sum((-Y * np.log(a)) - ((1-Y) * np.log(1-a))) + (reg/(2*m))*sum(np.square(w))
    gradient = np.sum(np.multiply((a-Y),X.T),axis=1)/m + (reg/m)*w
    w = w - (lr*gradient)



index = 0
with open(modelFile,"w") as modelSave:
    modelSave.write('bias :{0}\n'.format(w[0]))
    for wi in w[1:]:
        modelSave.write('{0}:{1}\n'.format(data.columns[index], wi))
        index+=1


def getAccuracy(myX,myY):
    prediction = sigmoid(np.dot(myX,w))
    prediction = prediction>=.5
    accuracy = myY[myY==prediction].shape[0]/float(myY.shape[0])
    return accuracy


predictions = sigmoid(np.dot(testX,w))
for prediction in predictions:
    print "{0:.3f}".format(prediction)


print('Training Accuracy: ', getAccuracy(X,Y))
print('Test Accuracy: ',getAccuracy(testX,testY))
