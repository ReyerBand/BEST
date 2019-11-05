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
from tools.GenerateBatch import GenerateBatch
# Print which gpu/cpu this is running on
sess = tf.Session(config=config)
h = tf.constant('hello world')
print(sess.run(h))

# set options 
savePDF = True
savePNG = False 
plotInputs = True
#==================================================================================
# Load Jet Images /////////////////////////////////////////////////////////////////
#==================================================================================

#Keeping some code from pre-generator style to get shapes correct

# put images and BES variables in data frames
arbitrary_length = 10 #Hopefully this number doesn't matter
nx = 31
ny = 31
ImageShapeHolder = numpy.zeros((arbitrary_length, nx, ny, 1))
BestShapeHolder = 44

HiggsImageInputs = Input( shape=(ImageShapeHolder.shape[1], ImageShapeHolder.shape[2], ImageShapeHolder.shape[3]) )

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
TopImageInputs = Input( shape=(ImageShapeHolder.shape[1], ImageShapeHolder.shape[2], ImageShapeHolder.shape[3]) )

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
WImageInputs = Input( shape=(ImageShapeHolder.shape[1], ImageShapeHolder.shape[2], ImageShapeHolder.shape[3]) )

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
ZImageInputs = Input( shape=(ImageShapeHolder.shape[1], ImageShapeHolder.shape[2], ImageShapeHolder.shape[3]) )

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
besInputs = Input( shape=(BestShapeHolder, ) )
besLayer = Dense(40, kernel_initializer="glorot_normal", activation="relu" )(besInputs)
besLayer = Dense(40, kernel_initializer="glorot_normal", activation="relu" )(besLayer)

besModel = Model(inputs = besInputs, outputs = besLayer)
print besModel.output
# Add BES variables to the network
#combined = concatenate([HiggsImageModel.output, TopImageModel.output, WImageModel.output, ZImageModel.output, besModel.output])
#Testing with just Higgs layer
combined = concatenate([HiggsImageModel.output, TopImageModel.output, besModel.output])

#combined = besModel.output
combLayer = Dense(40, kernel_initializer="glorot_normal", activation="relu" )(combined)
combLayer = Dense(40, kernel_initializer="glorot_normal", activation="relu" )(combLayer)
combLayer = Dropout(0.10)(combLayer)
outputBEST = Dense(6, kernel_initializer="glorot_normal", activation="softmax")(combLayer)

# compile the model
#model_BEST = Model(inputs = [HiggsImageModel.input, TopImageModel.input, WImageModel.input, ZImageModel.input, besModel.input], outputs = outputBEST)
#Testing with just Higgs
model_BEST = Model(inputs = [HiggsImageModel.input, TopImageModel.input, besModel.input], outputs = outputBEST)
#model_BEST = Model(inputs = [HiggsImageModel.input], outputs = outputBEST)

#model_BEST = Model(inputs = besModel.input, outputs = outputBEST)
#Testing with just BEST layer
#model_BEST = Model(inputs = [HiggsImageModel.input, besModel.input], outputs = outputBEST)
#model_BEST = Model(inputs = besModel.input, outputs = outputBEST)
model_BEST.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# print the model summary
print(model_BEST.summary() )

# early stopping

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0005, patience=50, verbose=0, mode='auto')

# model checkpoint callback
# this saves the model architecture + parameters into dense_model.h5
model_checkpoint = ModelCheckpoint('BEST_model.h5', monitor='val_loss', 
                                   verbose=0, save_best_only=True, 
                                   save_weights_only=False, mode='auto', 
                                   period=1)

# train the neural network
data_generator = GenerateBatch(batch_size = 1200, validation_frac = 0.2, debug_info = False, debug_plots = False)
print 'Starting training'
history = model_BEST.fit_generator(generator = data_generator.generator_train(), validation_data=data_generator.generator_valid(), steps_per_epoch = 40, epochs=200, callbacks=[early_stopping, model_checkpoint], validation_steps = 5, use_multiprocessing = True, workers=4) 
#Testing with just BES vars
#history = model_BEST.fit([trainHiggsImages[:], trainBESvars[:]], trainTruth[:], batch_size=1000, epochs=200, callbacks=[early_stopping, model_checkpoint], validation_split = 0.15)

print "Trained the neural network!"

# print model visualization
#plot_model(model_HHESTIA, to_file='plots/boost_CosTheta_NN_Vis.png')

# Evaluate on ALL the data
testHiggsImages = numpy.concatenate([data_generator.data['QCD_H'], data_generator.data['H_H'], data_generator.data['T_H'], data_generator.data['W_H'], data_generator.data['Z_H'], data_generator.data['B_H']])
#print type(testHiggsImages)
testTopImages = numpy.concatenate([data_generator.data['QCD_T'], data_generator.data['H_T'], data_generator.data['T_T'], data_generator.data['W_T'], data_generator.data['Z_T'], data_generator.data['B_T']])
#testWImages = numpy.concatenate([data_generator.data['QCD_W'], data_generator.data['H_W'], data_generator.data['T_W'], data_generator.data['W_W'], data_generator.data['Z_W'], data_generator.data['B_W']])
#testZImages = numpy.concatenate([data_generator.data['QCD_Z'], data_generator.data['H_Z'], data_generator.data['T_Z'], data_generator.data['W_Z'], data_generator.data['Z_Z'], data_generator.data['B_Z']])

testBESvars = numpy.concatenate([data_generator.data['QCD_BES'], data_generator.data['H_BES'], data_generator.data['T_BES'], data_generator.data['W_BES'], data_generator.data['Z_BES'], data_generator.data['B_BES']])
#print type(testBESvars)

testTruth = numpy.concatenate([numpy.full(len(data_generator.data['QCD_H']), 4), numpy.full(len(data_generator.data['H_H']), 1), numpy.full(len(data_generator.data['T_H']), 2), numpy.full(len(data_generator.data['W_H']), 3), numpy.full(len(data_generator.data['Z_H']), 5), numpy.full(len(data_generator.data['B_H']), 0)])

testTruth=to_categorical(testTruth, num_classes = 6)
#print len(testHiggsImages), len(testBESvars), len(testTruth)
#cm = metrics.confusion_matrix(numpy.argmax(model_BEST.predict([testHiggsImages[:], testTopImages[:], testWImages[:], testZImages[:], testBESvars[:] ]), axis=1), numpy.argmax(testTruth[:], axis=1) )
cm = metrics.confusion_matrix(numpy.argmax(model_BEST.predict([testHiggsImages[:], testTopImages[:], testBESvars[:]]), axis=1), numpy.argmax(testTruth[:], axis=1) )

plt.figure(
)
targetNames = ['B', 'H', 't', 'W', 'QCD', 'Z']
tools.plot_confusion_matrix(cm.T, targetNames, normalize=True)
if savePDF == True:
   plt.savefig('plots/ConfusionFlatPtTwoFrames40Batch.pdf')
if savePNG == True:
   plt.savefig('plots/ConfusionFlatPtTwoFrames40Batch.png')
plt.close()


loss = [history.history['loss'], history.history['val_loss'] ]
acc = [history.history['acc'], history.history['val_acc'] ]
tools.plotPerformance(loss, acc, "FlatPT")

exit()
print 'Did not exit'
joblib.dump(model_BEST, "BEST_keras_FlatPT.pkl")


h5f = h5py.File("images/BEST_FlatPT.h5","w")
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
   plt.savefig('plots/boost_CosTheta_confusion_matrix_FlatPtOneFrame.pdf')
if savePNG == True:
   plt.savefig('plots/boost_CosTheta_confusion_matrix_FlatPtOneFrame.png')
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
