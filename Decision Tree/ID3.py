

import sys
import pandas as pd
import numpy as np
from scipy.stats import chisquare

arguments = sys.argv
trainFile = str(arguments[1])
testFile = str(arguments[2])
modelF = str(arguments[3])

data = pd.read_csv(trainFile)
testdata = pd.read_csv(testFile)
modelFile = modelF

#data = pd.read_csv('training_set.csv')
#testdata = pd.read_csv('test_set.csv')
#modelFile = 'output.model'


def uniqueClasses(myData):
    results={}
    for a in myData[myData.columns[-1]]:
        if a not in results: 
            results[a] = 0
        results[a]+=1
    return results


def entropy(rows):
    from math import log
    log2=lambda x:log(x)/log(2)  
    results=uniqueClasses(rows)
    ent=0.0
    for r in results.keys():
        p=float(results[r])/len(rows) 
        ent=ent-p*log2(p)
    return ent


def getTopClass(dataSet):
    compare = uniqueClasses(dataSet)
    topClass = sorted(compare, key=compare.get)
    return topClass[-1:]


def getTopIG(someData):
    IG = {}
    for column in someData.ix[:,:-1]:
        ENT = (len(someData[someData[column]==1])/len(someData))*entropy(someData[someData[column]==1]) + (len(someData[someData[column]==0])/len(someData))*entropy(someData[someData[column]==0])
        IG[column] = entropy(someData) - ENT
    return IG


class TreeNode:
    def __init__(self,myData,parents,decision, parameter, IG):
        self.myData = myData
        self.parents = parents
        self.parameter = parameter
        self.decision = decision
        self.output = []
        self.index = []
        self.IG = IG

def iterateTree(node):
    gainList = getTopIG(node.myData)
    sortedIG = sorted(gainList, key=gainList.get)
    topParameter = sortedIG[-1]
    childData0 = node.myData[node.myData[topParameter]==0]
    childData1 = node.myData[node.myData[topParameter]==1]
    child0 = TreeNode(childData0,node.parents+1, 0, topParameter, gainList[topParameter])
    child1 = TreeNode(childData1,node.parents+1, 1, topParameter, gainList[topParameter])
    return child0,child1

def getChiSquare(child0, child1):
    classes0 = uniqueClasses(child0.myData)
    classes1 = uniqueClasses(child1.myData)
    if 0 not in classes0.keys():
        classes0[0] = 0
    if 1 not in classes0.keys():
        classes0[1] = 0
    if 0 not in classes1.keys():
        classes1[0] = 0
    if 1 not in classes1.keys():
        classes1[1] = 0
    child0Compare = [classes0[0],classes0[1]]
    child1Compare = [classes1[0],classes1[1]]
    [chi,p] = chisquare(child0Compare,child1Compare)
    if np.isnan(p): p = 0
    #if p > .01: print(p, ' ', child0.parameter,' ', child1.parameter)
    return p 


nodes = []
que = []
index = 0
nodes.append(TreeNode(data,1,0, "Root",1))
endCreate = False
while endCreate == False:
    child0,child1 = iterateTree(nodes[-1])
    p = getChiSquare(child0,child1)
    if len(uniqueClasses(nodes[-1].myData)) == 2 and nodes[-1].IG > 0 and p < .01:
        index = index + 1
        child0.index = index
        index = index + 1
        child1.index = index
        
        if len(uniqueClasses(child0.myData)) == 2 and child0.IG > 0:
            nodes.append(child0)
            que.append(child1)
        elif len(uniqueClasses(child1.myData)) ==2 and child1.IG > 0:
            child0.output = getTopClass(child0.myData)
            nodes.append(child0)
            nodes.append(child1)
        else:
            child0.output = getTopClass(child0.myData)
            nodes.append(child0)
            child1.output = getTopClass(child1.myData)
            nodes.append(child1)
            if len(que) == 0:
                endCreate = True
            else:
                nodes.append(que[-1])
                que.pop()
    else:
        nodes[-1].output = getTopClass(nodes[-1].myData)
        if len(que) == 0:
            endCreate = True
        else:
            nodes.append(que[-1])
            que.pop()




with open(modelFile, "w") as modelSave:
    for node in nodes:
        if node.parameter != 'Root':
            if node.output == []:
                print('| '*(node.parents-2),node.parameter, ' = ', node.decision, ' : ', file=modelSave)
            else:
                print('| '*(node.parents-2),node.parameter, ' = ', node.decision, ' : ', str(node.output).strip('[]'), file=modelSave)


def findPrediction(rowIn):
    nodeNum = 1
    foundIt = False
    while foundIt == False:
        if nodes[nodeNum].output == []:
            if rowIn[nodes[nodeNum].parameter][rowNum-1] == nodes[nodeNum].decision:
                nodeNum+=1
            else:
                for i,j in enumerate(nodes):
                    if j.index == nodes[nodeNum].index+1:
                        nodeNum = i
        else:
            if rowIn[nodes[nodeNum].parameter][rowNum-1] == nodes[nodeNum].decision:
                foundIt = True
            else:
                nodeNum+=1
    return nodes[nodeNum].output

predictions = {}
rowNum = 1

while rowNum <= len(data):
    predictions[rowNum-1] = findPrediction(data[rowNum-1:rowNum])
    rowNum+=1

testpredictions = {}
rowNum = 1

while rowNum <= len(testdata):
    testpredictions[rowNum-1] = findPrediction(testdata[rowNum-1:rowNum])
    rowNum+=1


def getAccuracy(dataSet, preds):
    error = 0
    preds, dataSet[dataSet.columns[-1]]
    for i in range(0,len(preds)-1):
        if preds[i] != dataSet[dataSet.columns[-1]][i]:
            error+=1
    return 1-(error/len(preds))

print('Training Accuracy: ', getAccuracy(data,predictions))
print('Test Accuracy: ', getAccuracy(testdata,testpredictions))

