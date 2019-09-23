#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# trainBEST.py ////////////////////////////////////////////////////////////////////
#==================================================================================
# This program trains BEST: The Boosted Event Shape Tagger ////////////////////////
#==================================================================================

# modules
import numpy
import pandas as pd
import h5py
import matplotlib
matplotlib.use('Agg') #prevents opening displays, must use before pyplot
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import copy
import random

# get stuff from modules
from sklearn import svm, metrics, preprocessing, neural_network, tree
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

# set up keras
from os import environ
environ["KERAS_BACKEND"] = "tensorflow" #must set backend before importing keras
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.layers import Input, Activation, Dense, SeparableConv2D, Conv2D, MaxPool2D, BatchNormalization, Dropout, Flatten, MaxoutDense
from keras.layers import GRU, LSTM, ConvLSTM2D, Reshape
from keras.layers import concatenate
from keras.regularizers import l1,l2
from keras.utils import np_utils, to_categorical, plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

# set up gpu environment
from keras import backend as k
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6
k.tensorflow_backend.set_session(tf.Session(config=config))

# user modules
import tools.functions as tools

# Print which gpu/cpu this is running on
sess = tf.Session(config=config)
h = tf.constant('hello world')
print(sess.run(h))

# set options 
savePDF = False
savePNG = True 
plotInputs = True
#==================================================================================
# Load Jet Images /////////////////////////////////////////////////////////////////
#==================================================================================

# Load images from h5 file
#h5f = h5py.File("images/phiCosThetaBoostedJetImages.h5","r")

# put images and BES variables in data frames
jetImagesDF = {}
jetBESvarsDF = {}
QCD = h5py.File("images/QCD_Flat_1phiCosThetaBoostedJetImagesX10.h5","r")
jetImagesDF['QCD_H'] = QCD['QCD_Flat_1_Higgs_images'][()]
jetImagesDF['QCD_T'] = QCD['QCD_Flat_1_Top_images'][()]
jetImagesDF['QCD_W'] = QCD['QCD_Flat_1_W_images'][()]
jetImagesDF['QCD_Z'] = QCD['QCD_Flat_1_Z_images'][()]
jetBESvarsDF['QCD'] = QCD['QCD_Flat_1_BES_vars'][()]
QCD.close()
H = h5py.File("images/HH_1phiCosThetaBoostedJetImagesX10.h5","r")
jetImagesDF['H_H'] = H['HH_1_Higgs_images'][()]
jetImagesDF['H_T'] = H['HH_1_Top_images'][()]
jetImagesDF['H_W'] = H['HH_1_W_images'][()]
jetImagesDF['H_Z'] = H['HH_1_Z_images'][()]
jetBESvarsDF['H'] = H['HH_1_BES_vars'][()]
H.close()
T = h5py.File("images/TT_1phiCosThetaBoostedJetImagesX10.h5","r")
jetImagesDF['T_H'] = T['TT_1_Higgs_images'][()]
jetImagesDF['T_T'] = T['TT_1_Top_images'][()]
jetImagesDF['T_W'] = T['TT_1_W_images'][()]
jetImagesDF['T_Z'] = T['TT_1_Z_images'][()]
jetBESvarsDF['T'] = T['TT_1_BES_vars'][()]
T.close()
W = h5py.File("images/WW_1phiCosThetaBoostedJetImagesX10.h5","r")
jetImagesDF['W_H'] = W['WW_1_Higgs_images'][()]
jetImagesDF['W_T'] = W['WW_1_Top_images'][()]
jetImagesDF['W_W'] = W['WW_1_W_images'][()]
jetImagesDF['W_Z'] = W['WW_1_Z_images'][()]
jetBESvarsDF['W'] = W['WW_1_BES_vars'][()]
W.close()
Z = h5py.File("images/ZZ_1phiCosThetaBoostedJetImagesX10.h5","r")
jetImagesDF['Z_H'] = Z['ZZ_1_Higgs_images'][()]
jetImagesDF['Z_T'] = Z['ZZ_1_Top_images'][()]
jetImagesDF['Z_W'] = Z['ZZ_1_W_images'][()]
jetImagesDF['Z_Z'] = Z['ZZ_1_Z_images'][()]
jetBESvarsDF['Z'] = Z['ZZ_1_BES_vars'][()]
Z.close()
B = h5py.File("images/BB_1phiCosThetaBoostedJetImagesX10.h5","r")
jetImagesDF['B_H'] = B['BB_1_Higgs_images'][()]
jetImagesDF['B_T'] = B['BB_1_Top_images'][()]
jetImagesDF['B_W'] = B['BB_1_W_images'][()]
jetImagesDF['B_Z'] = B['BB_1_Z_images'][()]
jetBESvarsDF['B'] = B['BB_1_BES_vars'][()]
B.close()


for i in range(2, 6):
   QCD = h5py.File("images/QCD_Flat_"+str(i)+"phiCosThetaBoostedJetImagesX10.h5","r")
   jetImagesDF['QCD_H'] = numpy.concatenate([jetImagesDF['QCD_H'], QCD['QCD_Flat_'+str(i)+'_Higgs_images'][()]])
   jetImagesDF['QCD_T'] = numpy.concatenate([jetImagesDF['QCD_T'], QCD['QCD_Flat_'+str(i)+'_Top_images'][()]])
   jetImagesDF['QCD_W'] = numpy.concatenate([jetImagesDF['QCD_W'], QCD['QCD_Flat_'+str(i)+'_W_images'][()]])
   jetImagesDF['QCD_Z'] = numpy.concatenate([jetImagesDF['QCD_Z'], QCD['QCD_Flat_'+str(i)+'_Z_images'][()]])
   jetBESvarsDF['QCD'] = numpy.concatenate([jetBESvarsDF['QCD'], QCD['QCD_Flat_'+str(i)+'_BES_vars'][()]])
   QCD.close()

for i in range(2, 6):
   for heavy in ('H','T','W','Z','B'):
      if (heavy is 'W' or heavy is 'B') and i == 5: continue
      imagefile = h5py.File("images/"+heavy+heavy+"_"+str(i)+"phiCosThetaBoostedJetImagesX10.h5","r")
      jetImagesDF[heavy+'_H'] = numpy.concatenate([jetImagesDF[heavy+'_H'], imagefile[heavy+heavy+'_'+str(i)+'_Higgs_images'][()]])
      jetImagesDF[heavy+'_T'] = numpy.concatenate([jetImagesDF[heavy+'_T'], imagefile[heavy+heavy+'_'+str(i)+'_Top_images'][()]])
      jetImagesDF[heavy+'_W'] = numpy.concatenate([jetImagesDF[heavy+'_W'], imagefile[heavy+heavy+'_'+str(i)+'_W_images'][()]])
      jetImagesDF[heavy+'_Z'] = numpy.concatenate([jetImagesDF[heavy+'_Z'], imagefile[heavy+heavy+'_'+str(i)+'_Z_images'][()]])
      jetBESvarsDF[heavy] = numpy.concatenate([jetBESvarsDF[heavy], imagefile[heavy+heavy+'_'+str(i)+'_BES_vars'][()]])
      imagefile.close()

print "Accessed Jet Images and BES variables"

#h5f.close()

print "Made image dataframes"

#==================================================================================
# Train the Neural Network ////////////////////////////////////////////////////////
#==================================================================================

# Store data and truth
HiggsQCDImages = jetImagesDF['QCD_H']
HiggsQCDImages = HiggsQCDImages[:90000]
TopQCDImages = jetImagesDF['QCD_T']
TopQCDImages = TopQCDImages[:90000]
WQCDImages = jetImagesDF['QCD_W']
WQCDImages = WQCDImages[:90000]
ZQCDImages = jetImagesDF['QCD_Z']
ZQCDImages = ZQCDImages[:90000]


HiggsTImages = jetImagesDF['T_H']
HiggsTImages = HiggsTImages[:90000]
TopTImages = jetImagesDF['T_T']
TopTImages = TopTImages[:90000]
WTImages = jetImagesDF['T_W']
WTImages = WTImages[:90000]
ZTImages = jetImagesDF['T_Z']
ZTImages = ZTImages[:90000]

HiggsHImages = jetImagesDF['H_H']
HiggsHImages = HiggsHImages[:90000]
TopHImages = jetImagesDF['H_T']
TopHImages = TopHImages[:90000]
WHImages = jetImagesDF['H_W']
WHImages = WHImages[:90000]
ZHImages = jetImagesDF['H_Z']
ZHImages = ZHImages[:90000]


HiggsWImages = jetImagesDF['W_H']
HiggsWImages = HiggsWImages[:90000]
TopWImages = jetImagesDF['W_T']
TopWImages = TopWImages[:90000]
WWImages = jetImagesDF['W_W']
WWImages = WWImages[:90000]
ZWImages = jetImagesDF['W_Z']
ZWImages = ZWImages[:90000]

#TopWImages = jetImagesDF['W_T']
#WWImages = jetImagesDF['W_W']
#ZWImages = jetImagesDF['W_Z']

HiggsZImages = jetImagesDF['Z_H']
HiggsZImages = HiggsZImages[:90000]
TopZImages = jetImagesDF['Z_T']
TopZImages = TopZImages[:90000]
WZImages = jetImagesDF['Z_W']
WZImages = WZImages[:90000]
ZZImages = jetImagesDF['Z_Z']
ZZImages = ZZImages[:90000]

#TopZImages = jetImagesDF['Z_T']
#WZImages = jetImagesDF['Z_W']
#ZZImages = jetImagesDF['Z_Z']

HiggsBImages = jetImagesDF['B_H']
HiggsBImages = HiggsBImages[:90000]
TopBImages = jetImagesDF['B_T']
TopBImages = TopBImages[:90000]
WBImages = jetImagesDF['B_W']
WBImages = WBImages[:90000]
ZBImages = jetImagesDF['B_Z']
ZBImages = ZBImages[:90000]

#TopBImages = jetImagesDF['B_T']
#WBImages = jetImagesDF['B_W']
#ZBImages = jetImagesDF['B_Z']

qcdBESvars = jetBESvarsDF['QCD']
qcdBESvars = qcdBESvars[:90000]
higgsBESvars = jetBESvarsDF['H']
higgsBESvars = higgsBESvars[:90000]
topBESvars = jetBESvarsDF['T']
topBESvars = topBESvars[:90000]
WBESvars = jetBESvarsDF['W']
WBESvars = WBESvars[:90000]
ZBESvars = jetBESvarsDF['Z']
ZBESvars = ZBESvars[:90000]
bBESvars = jetBESvarsDF['B']
bBESvars = bBESvars[:90000]

jetHiggsImages = numpy.concatenate([HiggsQCDImages, HiggsHImages, HiggsTImages, HiggsWImages, HiggsZImages, HiggsBImages ])
jetTopImages = numpy.concatenate([TopQCDImages, TopHImages, TopTImages, TopWImages, TopZImages, TopBImages ])
jetWImages = numpy.concatenate([WQCDImages, WHImages, WTImages, WWImages, WZImages, WBImages ])
jetZImages = numpy.concatenate([ZQCDImages, ZHImages, ZTImages, ZWImages, ZZImages, ZBImages ])

jetLabels = numpy.concatenate([numpy.zeros(len(HiggsQCDImages) ), numpy.ones(len(HiggsHImages) ), numpy.full(len(HiggsTImages), 2), numpy.full(len(HiggsWImages), 3), numpy.full(len(HiggsZImages), 4), numpy.full(len(HiggsBImages), 5)] )

jetBESvars = numpy.concatenate([qcdBESvars, higgsBESvars, topBESvars, WBESvars, ZBESvars, bBESvars])
print "Stored data and truth information"

if plotInputs == True:
   for index, hist in enumerate(qcdBESvars):
      print index
      plt.figure()
      plt.hist(hist, bins=100, color='b', label='QCD', histtype='step', normed=True)
      plt.hist(WBESvars[index], bins=100, color='g', label='W', histtype='step', normed=True)
      plt.hist(ZBESvars[index], bins=100, color='y', label='Z', histtype='step', normed=True)
      plt.hist(higgsBESvars[index], bins=100, color='m', label='H', histtype='step', normed=True)
      plt.hist(topBESvars[index], bins=100, color='r', label='T', histtype='step', normed=True)
      plt.hist(bBESvars[index], bins=100, color='c', label='B', histtype='step', normed=True)
      plt.xlabel(index)
      print index
      plt.legend()
      if savePDF == True:
         plt.savefig("plots/Hist_"+index+".pdf")
         plt.close()
         pass
      pass

scaler = preprocessing.StandardScaler().fit(jetBESvars)
jetBESvars = scaler.transform(jetBESvars)

# split the training and testing data
trainHiggsImages, testHiggsImages, trainBESvars, testBESvars, trainTopImages, testTopImages, trainWImages, testWImages, trainZImages, testZImages, trainTruth, testTruth = train_test_split(jetHiggsImages, jetBESvars, jetTopImages, jetWImages, jetZImages, jetLabels, test_size=0.1)

#trainBESvars, testBESvars, trainTruth, testTruth = train_test_split(jetBESvars, jetLabels, test_size=0.1)
#data_tuple = list(zip(trainImages,trainTruth))
#random.shuffle(data_tuple)
#trainImages, trainTruth = zip(*data_tuple)
#trainImages=numpy.array(trainImages)
#trainTruth=numpy.array(trainTruth)

print trainBESvars[2].dtype


print "Number of QCD jets in training: ", numpy.sum(trainTruth == 0)
print "Number of H jets in training: ", numpy.sum(trainTruth == 1)
print "Number of T jets in training: ", numpy.sum(trainTruth == 2)
print "Number of W jets in training: ", numpy.sum(trainTruth == 3)
print "Number of Z jets in training: ", numpy.sum(trainTruth == 4)
print "Number of b jets in training: ", numpy.sum(trainTruth == 5)

print "Number of QCD jets in testing: ", numpy.sum(testTruth == 0)
print "Number of H jets in testing: ", numpy.sum(testTruth == 1)
print "Number of T jets in testing: ", numpy.sum(testTruth == 2)
print "Number of W jets in testing: ", numpy.sum(testTruth == 3)
print "Number of Z jets in testing: ", numpy.sum(testTruth == 4)
print "Number of b jets in testing: ", numpy.sum(testTruth == 5)

# make it so keras results can go in a pkl file
tools.make_keras_picklable()

# get the truth info in the correct form
trainTruth = to_categorical(trainTruth, num_classes=6)
testTruth = to_categorical(testTruth, num_classes=6)

print "NN image input shape: ", trainHiggsImages.shape[1], trainHiggsImages.shape[2], trainHiggsImages.shape[3]
print "NN image input shape: ", trainTopImages.shape[1], trainTopImages.shape[2], trainTopImages.shape[3]
print "NN image input shape: ", trainWImages.shape[1], trainWImages.shape[2], trainWImages.shape[3]
print "NN image input shape: ", trainZImages.shape[1], trainZImages.shape[2], trainZImages.shape[3]
print "BES input shape: ", trainBESvars.shape[1]
# Define the Neural Network Structure using functional API
# Create the Higgs image portion

HiggsImageInputs = Input( shape=(trainHiggsImages.shape[1], trainHiggsImages.shape[2], trainHiggsImages.shape[3]) )

HiggsImageLayer = SeparableConv2D(32, (11,11), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(HiggsImageInputs)
HiggsImageLayer = SeparableConv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(HiggsImageLayer)
HiggsImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(HiggsImageLayer)
HiggsImageLayer = BatchNormalization(momentum = 0.6)(HiggsImageLayer)
HiggsImageLayer = MaxPool2D(pool_size=(2,2) )(HiggsImageLayer)
HiggsImageLayer = SeparableConv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(HiggsImageLayer)
HiggsImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(HiggsImageLayer)
HiggsImageLayer = SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(HiggsImageLayer)
HiggsImageLayer = BatchNormalization(momentum = 0.6)(HiggsImageLayer)
HiggsImageLayer = MaxPool2D(pool_size=(2,2) )(HiggsImageLayer)
HiggsImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(HiggsImageLayer)
HiggsImageLayer = SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(HiggsImageLayer)
HiggsImageLayer = BatchNormalization(momentum = 0.6)(HiggsImageLayer)
HiggsImageLayer = MaxPool2D(pool_size=(2,2) )(HiggsImageLayer) 
HiggsImageLayer = Flatten()(HiggsImageLayer)
HiggsImageLayer = Dropout(0.20)(HiggsImageLayer)
HiggsImageLayer = Dense(144, kernel_initializer="glorot_normal", activation="relu" )(HiggsImageLayer)
HiggsImageLayer = Dense(72, kernel_initializer="glorot_normal", activation="relu" )(HiggsImageLayer)
HiggsImageLayer = Dense(24, kernel_initializer="glorot_normal", activation="relu" )(HiggsImageLayer)
HiggsImageLayer = Dropout(0.10)(HiggsImageLayer)

HiggsImageModel = Model(inputs = HiggsImageInputs, outputs = HiggsImageLayer)

#Top image
TopImageInputs = Input( shape=(trainTopImages.shape[1], trainTopImages.shape[2], trainTopImages.shape[3]) )

TopImageLayer = SeparableConv2D(32, (11,11), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(TopImageInputs)
TopImageLayer = SeparableConv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(TopImageLayer)
TopImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(TopImageLayer)
TopImageLayer = BatchNormalization(momentum = 0.6)(TopImageLayer)
TopImageLayer = MaxPool2D(pool_size=(2,2) )(TopImageLayer)
TopImageLayer = SeparableConv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(TopImageLayer)
TopImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(TopImageLayer)
TopImageLayer = SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(TopImageLayer)
TopImageLayer = BatchNormalization(momentum = 0.6)(TopImageLayer)
TopImageLayer = MaxPool2D(pool_size=(2,2) )(TopImageLayer)
TopImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(TopImageLayer)
TopImageLayer = SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(TopImageLayer)
TopImageLayer = BatchNormalization(momentum = 0.6)(TopImageLayer)
TopImageLayer = MaxPool2D(pool_size=(2,2) )(TopImageLayer)
TopImageLayer = Flatten()(TopImageLayer)
TopImageLayer = Dropout(0.20)(TopImageLayer)
TopImageLayer = Dense(144, kernel_initializer="glorot_normal", activation="relu" )(TopImageLayer)
TopImageLayer = Dense(72, kernel_initializer="glorot_normal", activation="relu" )(TopImageLayer)
TopImageLayer = Dense(24, kernel_initializer="glorot_normal", activation="relu" )(TopImageLayer)
TopImageLayer = Dropout(0.10)(TopImageLayer)

TopImageModel = Model(inputs = TopImageInputs, outputs = TopImageLayer)

#W Model
WImageInputs = Input( shape=(trainWImages.shape[1], trainWImages.shape[2], trainWImages.shape[3]) )

WImageLayer = SeparableConv2D(32, (11,11), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(WImageInputs)
WImageLayer = SeparableConv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(WImageLayer)
WImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(WImageLayer)
WImageLayer = BatchNormalization(momentum = 0.6)(WImageLayer)
WImageLayer = MaxPool2D(pool_size=(2,2) )(WImageLayer)
WImageLayer = SeparableConv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(WImageLayer)
WImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(WImageLayer)
WImageLayer = SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(WImageLayer)
WImageLayer = BatchNormalization(momentum = 0.6)(WImageLayer)
WImageLayer = MaxPool2D(pool_size=(2,2) )(WImageLayer)
WImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(WImageLayer)
WImageLayer = SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(WImageLayer)
WImageLayer = BatchNormalization(momentum = 0.6)(WImageLayer)
WImageLayer = MaxPool2D(pool_size=(2,2) )(WImageLayer)
WImageLayer = Flatten()(WImageLayer)
WImageLayer = Dropout(0.20)(WImageLayer)
WImageLayer = Dense(144, kernel_initializer="glorot_normal", activation="relu" )(WImageLayer)
WImageLayer = Dense(72, kernel_initializer="glorot_normal", activation="relu" )(WImageLayer)
WImageLayer = Dense(24, kernel_initializer="glorot_normal", activation="relu" )(WImageLayer)
WImageLayer = Dropout(0.10)(WImageLayer)

WImageModel = Model(inputs = WImageInputs, outputs = WImageLayer)


#Z Model
ZImageInputs = Input( shape=(trainZImages.shape[1], trainZImages.shape[2], trainZImages.shape[3]) )

ZImageLayer = SeparableConv2D(32, (11,11), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(ZImageInputs)
ZImageLayer = SeparableConv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(ZImageLayer)
ZImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(ZImageLayer)
ZImageLayer = BatchNormalization(momentum = 0.6)(ZImageLayer)
ZImageLayer = MaxPool2D(pool_size=(2,2) )(ZImageLayer)
ZImageLayer = SeparableConv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(ZImageLayer)
ZImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(ZImageLayer)
ZImageLayer = SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(ZImageLayer)
ZImageLayer = BatchNormalization(momentum = 0.6)(ZImageLayer)
ZImageLayer = MaxPool2D(pool_size=(2,2) )(ZImageLayer)
ZImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(ZImageLayer)
ZImageLayer = SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(ZImageLayer)
ZImageLayer = BatchNormalization(momentum = 0.6)(ZImageLayer)
ZImageLayer = MaxPool2D(pool_size=(2,2) )(ZImageLayer)
ZImageLayer = Flatten()(ZImageLayer)
ZImageLayer = Dropout(0.20)(ZImageLayer)
ZImageLayer = Dense(144, kernel_initializer="glorot_normal", activation="relu" )(ZImageLayer)
ZImageLayer = Dense(72, kernel_initializer="glorot_normal", activation="relu" )(ZImageLayer)
ZImageLayer = Dense(24, kernel_initializer="glorot_normal", activation="relu" )(ZImageLayer)
ZImageLayer = Dropout(0.10)(ZImageLayer)

ZImageModel = Model(inputs = ZImageInputs, outputs = ZImageLayer)

# Create the BES variable version
besInputs = Input( shape=(trainBESvars.shape[1], ) )
besLayer = Dense(40, kernel_initializer="glorot_normal", activation="relu" )(besInputs)
besLayer = Dense(40, kernel_initializer="glorot_normal", activation="relu" )(besLayer)

besModel = Model(inputs = besInputs, outputs = besLayer)
print besModel.output
# Add BES variables to the network
combined = concatenate([HiggsImageModel.output, TopImageModel.output, WImageModel.output, ZImageModel.output, besModel.output])
#Testing with just Higgs layer
#combined = concatenate([HiggsImageModel.output, besModel.output])
combLayer = Dense(40, kernel_initializer="glorot_normal", activation="relu" )(combined)
combLayer = Dense(40, kernel_initializer="glorot_normal", activation="relu" )(combLayer)
combLayer = Dropout(0.10)(combLayer)
outputBEST = Dense(6, kernel_initializer="glorot_normal", activation="softmax")(combLayer)

# compile the model
model_BEST = Model(inputs = [HiggsImageModel.input, TopImageModel.input, WImageModel.input, ZImageModel.input, besModel.input], outputs = outputBEST)
#Testing with just BEST layer
#model_BEST = Model(inputs = [HiggsImageModel.input, besModel.input], outputs = outputBEST)
#model_BEST = Model(inputs = besModel.input, outputs = outputBEST)
model_BEST.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# print the model summary
print(model_BEST.summary() )

# early stopping
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20, verbose=0, mode='auto')

# model checkpoint callback
# this saves the model architecture + parameters into dense_model.h5
model_checkpoint = ModelCheckpoint('BEST_model.h5', monitor='val_loss', 
                                   verbose=0, save_best_only=True, 
                                   save_weights_only=False, mode='auto', 
                                   period=1)

# train the neural network
history = model_BEST.fit([trainHiggsImages[:], trainTopImages[:], trainWImages[:], trainZImages[:], trainBESvars[:] ], trainTruth[:], batch_size=1000, epochs=200, callbacks=[early_stopping, model_checkpoint], validation_split = 0.15)
#Testing with just BES vars
#history = model_BEST.fit([trainHiggsImages[:], trainBESvars[:]], trainTruth[:], batch_size=1000, epochs=200, callbacks=[early_stopping, model_checkpoint], validation_split = 0.15)

print "Trained the neural network!"

# print model visualization
#plot_model(model_HHESTIA, to_file='plots/boost_CosTheta_NN_Vis.png')

# save the test data

h5f = h5py.File("images/BESTtestDataFourFrame.h5","w")
h5f.create_dataset('test_HiggsImages', data=testHiggsImages, compression='lzf')
#Testing just BES
h5f.create_dataset('test_TopImages', data=testTopImages, compression='lzf')
h5f.create_dataset('test_WImages', data=testWImages, compression='lzf')
h5f.create_dataset('test_ZImages', data=testZImages, compression='lzf')
h5f.create_dataset('test_BES_vars', data=testBESvars, compression='lzf')
h5f.create_dataset('test_truth', data=testTruth, compression='lzf')

print "Saved the testing data!"


#==================================================================================
# Plot Training Results ///////////////////////////////////////////////////////////
#==================================================================================

# Confusion Matrix
cm = metrics.confusion_matrix(numpy.argmax(model_BEST.predict([testHiggsImages[:], testTopImages[:], testWImages[:], testZImages[:], testBESvars[:] ]), axis=1), numpy.argmax(testTruth[:], axis=1) )
#Testing just BES layer
#cm = metrics.confusion_matrix(numpy.argmax(model_BEST.predict([testHiggsImages[:], testBESvars[:] ]), axis=1), numpy.argmax(testTruth[:], axis=1) )
plt.figure()

targetNames = ['QCD', 'H', 't', 'W', 'Z', 'b']
tools.plot_confusion_matrix(cm.T, targetNames, normalize=True)
if savePDF == True:
   plt.savefig('plots/boost_CosTheta_confusion_matrix_FourFrame.pdf')
if savePNG == True:
   plt.savefig('plots/boost_CosTheta_confusion_matrix_FourFrame.png')
plt.close()

# score
print "Training Score: ", model_BEST.evaluate([testHiggsImages[:], testTopImages[:], testWImages[:], testZImages[:], testBESvars[:]], testTruth[:], batch_size=100)

# performance plots
loss = [history.history['loss'], history.history['val_loss'] ]
acc = [history.history['acc'], history.history['val_acc'] ]
tools.plotPerformance(loss, acc, "boost_CosTheta")
print "plotted HESTIA training Performance"

# make file with probability results
joblib.dump(model_BEST, "BEST_keras_CosTheta_FourFrame.pkl")
#joblib.dump(scaler, "BEST_scaler.pkl")

print "Made weights based on probability results"
print "Program was a great success!!!"
