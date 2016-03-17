import pandas as pd
import numpy as np
import sys

arguments = sys.argv
trainFile = str(arguments[1])
testFile = str(arguments[2])
modelF = str(arguments[3])

data = pd.read_csv(trainFile)
testdata = pd.read_csv(testFile)
modelFile = modelF

#data = pd.read_csv('spambase-train.csv')
#testdata = pd.read_csv('spambase-test.csv')

Y = data[data.columns[-1]].as_matrix()
Y[Y==0] = -1

testY = testdata[testdata.columns[-1]].as_matrix()
testY[testY==0] = -1

Xframe = data.drop(data.columns[-1], axis=1)
X = Xframe.as_matrix()

testXframe = testdata.drop(data.columns[-1],axis=1)
testX = testXframe.as_matrix()

b = 0
w = np.zeros(X.shape[1])

def getAccuracy():
    prediction = np.sign(np.dot(X,w) + b)
    trainAccuracy = Y[Y==prediction].shape[0]/Y.shape[0]
    return trainAccuracy

def getTestAccuracy():
    testPreda = np.dot(testX,w) + b
    testPred = np.sign(testPreda)
    testAccuracy = testY[testY==testPred].shape[0]/testY.shape[0]
    return testAccuracy,testPreda

for n in range(0,100):
    p = np.random.permutation(len(Y))
    Y = Y[p]
    X = X[p]
    for i in range(0,len(Y)):
        a = np.dot(X[i,:],w) + b;
        if (Y[i]*a) <= 0:
            b = b + Y[i]
            w = w + (Y[i]*X[i,:])
    if getAccuracy() == 1:
        break
#print(w,b)


print('Training Accuracy: ', getAccuracy())


index = 0
for ex in getTestAccuracy()[1]:
    print('Example',index, ': ', ex)
    index+=1

print('Test Accuracy: ',getTestAccuracy()[0])

with open(modelFile, "w") as modelSave:
    print('bias: ',b, file=modelSave)
    index = 0
    for wi in w:
        print(data.columns[index], ': ', wi, file=modelSave)
        index+=1