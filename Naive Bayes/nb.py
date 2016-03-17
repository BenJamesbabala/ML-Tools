import pandas as pd
import numpy as np
import sys

arguments = sys.argv
trainFile = str(arguments[1])
testFile = str(arguments[2])
beta = int(arguments[3])
modelF = str(arguments[4])

data = pd.read_csv(trainFile)
testdata = pd.read_csv(testFile)
modelFile = modelF

Y = data[data.columns[-1]].as_matrix()
testY = testdata[testdata.columns[-1]].as_matrix()

Xframe = data.drop(data.columns[-1], axis=1)
X = Xframe.as_matrix()

testXframe = testdata.drop(data.columns[-1],axis=1)
testX = testXframe.as_matrix()

dataArray = data.as_matrix()

#Calculate Conditional Probabilities
pY = {}
pX = {}
Yu = np.unique(dataArray[:,-1])
for y in Yu:
    ySubset = (dataArray[dataArray[:,-1]==y])[:,:-1]
    ylength = float(len(Y[Y==y]))
    yother = float(len(Y[Y!=y]))
    pY[y] = (beta + ylength - 1)/((beta + ylength -1)+(beta + yother -1))
    xindex = 0
    for x in ySubset.T:
        xindex+=1
        xcount = 0
        Xu = np.unique(x)
        for xValue in Xu:
            xSubset = float(len(ySubset[ySubset[:,xindex-1]==xValue]))
            xother = float(len(ySubset[ySubset[:,xindex-1]!=xValue]))
            pX[xindex,xValue,y] = (beta + xSubset - 1)/((beta + xSubset -1)+(beta + xother -1))


#Returns probabilities for class output for a given example
def getPredicted(xrow):
    probs = {}
    for Yvalue in Yu:
        probs[Yvalue] = 1
        xindex = 0
        for xn in xrow:
            xindex+=1
            probs[Yvalue] = probs[Yvalue] * pX[xindex,xn,Yvalue]
        probs[Yvalue] = probs[Yvalue] * pY[Yvalue]
    return probs

#Returns dataset accuracy
def getAccuracy(anX, aY):
    predictions = []
    for x in anX:
        proba = getPredicted(x)
        predictions.append(np.argmax(proba.values()))
    accuracy = 1- sum(abs(predictions-aY))/float(len(aY))   
    return accuracy

#Calculate Log Odds
xindex = 1
logOdds = 0
wOdds = []
for x in X.T:
    logOdds+= np.log(pX[xindex,0,1]/pX[xindex,0,0])
    wOdds.append(np.log(pX[xindex,1,1]/pX[xindex,1,0]) - np.log(pX[xindex,0,1])/pX[xindex,0,1])
    xindex+=1

baseOdds = np.log(pY[1]/pY[0]) + logOdds

with open(modelFile,"w") as modelSave:
    modelSave.write ('Base Log Odds:{}\n'.format(baseOdds))
    wC = 1
    for w in wOdds:
        modelSave.write('Feature{0} Log Odds:{1}\n'.format(wC,w))
        wC+=1
    
    
for x in testX:
    print ('Predicted Values:')
    print (getPredicted(x)[1])

print('Training Accuracy: ', getAccuracy(X,Y))
print('Test Accuracy: ',getAccuracy(testX,testY))
