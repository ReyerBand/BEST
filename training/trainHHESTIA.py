#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# trainHHESTIA.py /////////////////////////////////////////////////////////////////
#==================================================================================
# This program trains HHESTIA: HH Event Shape Topology Indentification Algorithm //
#==================================================================================

# modules
import ROOT as root
import numpy
import matplotlib.pyplot as plt
import copy
import random

# user modules
import tools.functions as tools

# get stuff from modules
from root_numpy import tree2array
from sklearn import svm, metrics, preprocessing, neural_network, tree
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib

# enter batch mode in root (so python can access displays)
root.gROOT.SetBatch(True)

#==================================================================================
# Load Monte Carlo ////////////////////////////////////////////////////////////////
#==================================================================================

# access the TFiles
fileJJ = root.TFile("preprocess_HHESTIA_QCD.root", "READ")
fileHH4W = root.TFile("preprocess_HHESTIA_HH.root", "READ")

# access the trees
treeJJ = fileJJ.Get("run/jetTree")
treeHH4W = fileHH4W.Get("run/jetTree")

print "Accessed the trees"

# get input variable names from branches
vars = tools.getBranchNames(treeJJ)
treeVars = vars

# create selection criteria
sel = ""
#sel = "jet1AK8_pt > 0 && jet2AK8_pt > 0"
#sel = "tau32 < 9999. && et > 500. && et < 2500. && bDisc1 > -0.05 && SDmass < 400"

# make arrays from the trees
arrayJJ = tree2array(treeJJ, treeVars, sel)
arrayJJ = tools.appendTreeArray(arrayJJ)

arrayHH4W = tree2array(treeHH4W, treeVars, sel)
arrayHH4W = tools.appendTreeArray(arrayHH4W)

# make an array with all of the datasets
arrayData = [arrayJJ, arrayHH4W]

print "Made arrays from the datasets"

#==================================================================================
# Plot Input Variables ////////////////////////////////////////////////////////////
#==================================================================================

# store the data in histograms
histsJJ = numpy.array(arrayJJ).T
histsHH4W = numpy.array(arrayHH4W).T

# plot with python
for index, hist in enumerate(histsJJ):
   plt.figure()
   plt.hist(hist, bins=100, color='b', label='QCD', histtype='step', normed=True)
   plt.hist(histsHH4W[index], bins=100, color='m', label='HH->WWWWW', histtype='step', normed=True)
   plt.xlabel(vars[index])
   plt.legend()
   plt.savefig("Hist_" + vars[index] + ".pdf")
   plt.close()

print "Plotted each of the variables"

#==================================================================================
# Train the Neural Network ////////////////////////////////////////////////////////
#==================================================================================

# randomize the datasets
trainData, targetData = tools.randomizeData(arrayData)

# standardize the datasets
scaler = preprocessing.StandardScaler().fit(trainData)
trainData = scaler.transform(trainData)
arrayJJ = scaler.transform(arrayJJ)
arrayHH4W = scaler.transform(arrayHH4W)

# number of events to train with
numTrain = 20000

# train the neural network
mlp = neural_network.MLPClassifier(hidden_layer_sizes=(40,40,40), verbose=True, activation='relu')
#mlp = tree.DecisionTreeClassifier()
mlp.fit(trainData[:numTrain], targetData[:numTrain])

print "Trained the neural network!"

#==================================================================================
# Plot Training Results ///////////////////////////////////////////////////////////
#==================================================================================

# Confusion Matrix
cm  + metrics.confusion_matrix(mlp.predict(trainData[400000:]), targetData[400000:])
plt.figure()
targetNames = ['j', 'W', 'Z', 'H', 't', 'b']
plot_confusion_matrix(cm.T, targetNames, normalize=True)
plt.savefig('confusion_matrix.pdf')
plt.close()

# score
print "Training Score: ", mlp.score(trainData[400000:], targetData[400000:])

# get the probabilities
probsJJ = mlp.predict_proba(arrayJJ)
probsHH4W = mlp.predict_proba(arrayHH4W)

# [ [probArray, label, color], .. ]
probs = [ [probsJJ, 'QCD', 'b'],
          [probsHH4W, 'QCD', 'm'] ]

# plot probability results
tools.plotProbability(probs)

# make file with probability results
joblib.dump(mlp, "HHESTIA_mlp.pkl")
joblib.dump(scaler, "HHESTIA_scaler.pkl")

print "Made weights based on probability results"
print "Program was a great success!!!"
