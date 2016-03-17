import pandas as pd
import numpy as np
import scipy.sparse
import sys

arguments = sys.argv
trainFile = str(arguments[1])
testFile = str(arguments[2])
learningRate = float(arguments[3])
lam = float(arguments[4])
iterations = int(arguments[5])
#modelF = str(arguments[5])

data = pd.read_csv(trainFile)
testdata = pd.read_csv(testFile)
#modelFile = modelF


y = data[data.columns[-1]].as_matrix()
testY = testdata[testdata.columns[-1]].as_matrix()


Xframe = data.drop(data.columns[-1], axis=1)
x = Xframe.as_matrix()

testXframe = testdata.drop(data.columns[-1],axis=1)
testX = testXframe.as_matrix()



def getLoss(w,x,y,lam):
    m = x.shape[0]
    y_mat = scipy.sparse.csr_matrix((np.ones(m), (y, np.array(range(m)))))
    y_mat = np.array(y_mat.todense())
    y_mat = y_mat.T
    scores = np.dot(x,w)
    prob = softmax(scores)
    loss = (-1 / m) * np.sum(y_mat * np.log(prob)) + (lam/2)*np.sum(w*w)
    grad = (-1 / m) * np.dot(x.T,(y_mat - prob)) + lam*w
    return loss,grad

def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm


def getProbsAndPreds(someX):
    probs = softmax(np.dot(someX,w))
    preds = np.argmax(probs,axis=1)
    return probs,preds


w = np.zeros([x.shape[1],len(np.unique(y))])
for i in range(0,iterations):
    loss,grad = getLoss(w,x,y,lam)
    w = w - (learningRate * grad)


def getAccuracy(someX,someY):
    prob,prede = getProbsAndPreds(someX)
    accuracy = sum(prede == someY)/(float(len(someY)))
    return accuracy

print('Training Accuracy: ', getAccuracy(x,y))
print('Test Accuracy: ', getAccuracy(testX,testY))

