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
savePDF = True
savePNG = True 
plotInputs = True
#==================================================================================
# Load Jet Images /////////////////////////////////////////////////////////////////
#==================================================================================

# Load images from h5 file
#h5f = h5py.File("images/phiCosThetaBoostedJetImages.h5","r")

# put images and BES variables in data frames
jetImagesDF = {}
QCD = h5py.File("images/QCD_Flat_1phiCosThetaBoostedJetImagesX10.h5","r")
jetImagesDF['QCD_H'] = QCD['QCD_Flat_1_Higgs_images'][()]
jetImagesDF['QCD_T'] = QCD['QCD_Flat_1_Top_images'][()]
jetImagesDF['QCD_W'] = QCD['QCD_Flat_1_W_images'][()]
jetImagesDF['QCD_Z'] = QCD['QCD_Flat_1_Z_images'][()]
jetImagesDF['QCD'] = QCD['QCD_Flat_1_BES_vars'][()]
QCD.close()
H = h5py.File("images/HH_1phiCosThetaBoostedJetImagesX10.h5","r")
jetImagesDF['H_H'] = H['HH_1_Higgs_images'][()]
jetImagesDF['H_T'] = H['HH_1_Top_images'][()]
jetImagesDF['H_W'] = H['HH_1_W_images'][()]
jetImagesDF['H_Z'] = H['HH_1_Z_images'][()]
jetImagesDF['H'] = H['HH_1_BES_vars'][()]
H.close()
T = h5py.File("images/TT_1phiCosThetaBoostedJetImagesX10.h5","r")
jetImagesDF['T_H'] = T['TT_1_Higgs_images'][()]
jetImagesDF['T_T'] = T['TT_1_Top_images'][()]
jetImagesDF['T_W'] = T['TT_1_W_images'][()]
jetImagesDF['T_Z'] = T['TT_1_Z_images'][()]
jetImagesDF['T'] = T['TT_1_BES_vars'][()]
T.close()
W = h5py.File("images/WW_1phiCosThetaBoostedJetImagesX10.h5","r")
jetImagesDF['W_H'] = W['WW_1_Higgs_images'][()]
jetImagesDF['W_T'] = W['WW_1_Top_images'][()]
jetImagesDF['W_W'] = W['WW_1_W_images'][()]
jetImagesDF['W_Z'] = W['WW_1_Z_images'][()]
jetImagesDF['W'] = W['WW_1_BES_vars'][()]
W.close()
Z = h5py.File("images/ZZ_1phiCosThetaBoostedJetImagesX10.h5","r")
jetImagesDF['Z_H'] = Z['ZZ_1_Higgs_images'][()]
jetImagesDF['Z_T'] = Z['ZZ_1_Top_images'][()]
jetImagesDF['Z_W'] = Z['ZZ_1_W_images'][()]
jetImagesDF['Z_Z'] = Z['ZZ_1_Z_images'][()]
jetImagesDF['Z'] = Z['ZZ_1_BES_vars'][()]
Z.close()
B = h5py.File("images/BB_1phiCosThetaBoostedJetImagesX10.h5","r")
jetImagesDF['B_H'] = B['BB_1_Higgs_images'][()]
jetImagesDF['B_T'] = B['BB_1_Top_images'][()]
jetImagesDF['B_W'] = B['BB_1_W_images'][()]
jetImagesDF['B_Z'] = B['BB_1_Z_images'][()]
jetImagesDF['B'] = B['BB_1_BES_vars'][()]
B.close()

#QCD kept separate because of dumb naming conventions I introduced 
for i in range(2, 6):
   QCD = h5py.File("images/QCD_Flat_"+str(i)+"phiCosThetaBoostedJetImagesX10.h5","r")
   jetImagesDF['QCD_H'] = numpy.concatenate([jetImagesDF['QCD_H'], QCD['QCD_Flat_'+str(i)+'_Higgs_images'][()]])
   jetImagesDF['QCD_T'] = numpy.concatenate([jetImagesDF['QCD_T'], QCD['QCD_Flat_'+str(i)+'_Top_images'][()]])
   jetImagesDF['QCD_W'] = numpy.concatenate([jetImagesDF['QCD_W'], QCD['QCD_Flat_'+str(i)+'_W_images'][()]])
   jetImagesDF['QCD_Z'] = numpy.concatenate([jetImagesDF['QCD_Z'], QCD['QCD_Flat_'+str(i)+'_Z_images'][()]])
   jetImagesDF['QCD'] = numpy.concatenate([jetImagesDF['QCD'], QCD['QCD_Flat_'+str(i)+'_BES_vars'][()]])
   QCD.close()

for heavy in ('H','T','W','Z','B'):
   for i in range(2, 6):
      if (heavy is 'W' or heavy is 'B') and i == 5: continue
      imagefile = h5py.File("images/"+heavy+heavy+"_"+str(i)+"phiCosThetaBoostedJetImagesX10.h5","r")
      jetImagesDF[heavy+'_H'] = numpy.concatenate([jetImagesDF[heavy+'_H'], imagefile[heavy+heavy+'_'+str(i)+'_Higgs_images'][()]])
      jetImagesDF[heavy+'_T'] = numpy.concatenate([jetImagesDF[heavy+'_T'], imagefile[heavy+heavy+'_'+str(i)+'_Top_images'][()]])
      jetImagesDF[heavy+'_W'] = numpy.concatenate([jetImagesDF[heavy+'_W'], imagefile[heavy+heavy+'_'+str(i)+'_W_images'][()]])
      jetImagesDF[heavy+'_Z'] = numpy.concatenate([jetImagesDF[heavy+'_Z'], imagefile[heavy+heavy+'_'+str(i)+'_Z_images'][()]])
      jetImagesDF[heavy] = numpy.concatenate([jetImagesDF[heavy], imagefile[heavy+heavy+'_'+str(i)+'_BES_vars'][()]])
      imagefile.close()
#Standardize the BES inputs                                                                                                                                                              
allBESinputs = numpy.concatenate([jetImagesDF['QCD'], jetImagesDF['H'], jetImagesDF['T'], jetImagesDF['W'], jetImagesDF['Z'], jetImagesDF['B']])
scaler = preprocessing.StandardScaler().fit(allBESinputs)
#Save each class
for heavy in ('QCD','H','T','W','Z','B'):
   jetImagesDF[heavy] = scaler.transform(jetImagesDF[heavy])
   h5f = h5py.File("images/Full"+heavy+".h5", "w")
   h5f.create_dataset(heavy+'_H', data=jetImagesDF[heavy+'_H'], compression='lzf')
   h5f.create_dataset(heavy+'_T', data=jetImagesDF[heavy+'_T'], compression='lzf')
   h5f.create_dataset(heavy+'_W', data=jetImagesDF[heavy+'_W'], compression='lzf')
   h5f.create_dataset(heavy+'_Z', data=jetImagesDF[heavy+'_Z'], compression='lzf')
   h5f.create_dataset(heavy+'_BES', data=jetImagesDF[heavy+''], compression='lzf')

print "Accessed Jet Images and BES variables"



print "Made image dataframes"

