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

# Print which gpu/cpu this is running on 
sess = tf.Session(config=config)
h = tf.constant('hello world')
print(sess.run(h))


# user modules
import tools.functions as tools

# Print which gpu/cpu this is running on

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
jetBESvarsDF = {}
jetLabETDF = {}
QCD = h5py.File("images/QCD_Flat_1phiCosThetaBoostedJetImagesX10.h5","r")
jetImagesDF['QCD_H'] = QCD['QCD_Flat_1_Higgs_images'][()]
jetImagesDF['QCD_T'] = QCD['QCD_Flat_1_Top_images'][()]
jetImagesDF['QCD_W'] = QCD['QCD_Flat_1_W_images'][()]
jetImagesDF['QCD_Z'] = QCD['QCD_Flat_1_Z_images'][()]
jetBESvarsDF['QCD'] = QCD['QCD_Flat_1_BES_vars'][()]
jetLabETDF['QCD'] = QCD['QCD_Flat_1_LabFrameET'][()]
QCD.close()
H = h5py.File("images/HH_1phiCosThetaBoostedJetImagesX10.h5","r")
jetImagesDF['H_H'] = H['HH_1_Higgs_images'][()]
jetImagesDF['H_T'] = H['HH_1_Top_images'][()]
jetImagesDF['H_W'] = H['HH_1_W_images'][()]
jetImagesDF['H_Z'] = H['HH_1_Z_images'][()]
jetBESvarsDF['H'] = H['HH_1_BES_vars'][()]
jetLabETDF['H'] = H['HH_1_LabFrameET'][()]
H.close()
T = h5py.File("images/TT_1phiCosThetaBoostedJetImagesX10.h5","r")
jetImagesDF['T_H'] = T['TT_1_Higgs_images'][()]
jetImagesDF['T_T'] = T['TT_1_Top_images'][()]
jetImagesDF['T_W'] = T['TT_1_W_images'][()]
jetImagesDF['T_Z'] = T['TT_1_Z_images'][()]
jetBESvarsDF['T'] = T['TT_1_BES_vars'][()]
jetLabETDF['T'] = T['TT_1_LabFrameET'][()]
T.close()
W = h5py.File("images/WW_1phiCosThetaBoostedJetImagesX10.h5","r")
jetImagesDF['W_H'] = W['WW_1_Higgs_images'][()]
jetImagesDF['W_T'] = W['WW_1_Top_images'][()]
jetImagesDF['W_W'] = W['WW_1_W_images'][()]
jetImagesDF['W_Z'] = W['WW_1_Z_images'][()]
jetBESvarsDF['W'] = W['WW_1_BES_vars'][()]
jetLabETDF['W'] = W['WW_1_LabFrameET'][()]
W.close()
Z = h5py.File("images/ZZ_1phiCosThetaBoostedJetImagesX10.h5","r")
jetImagesDF['Z_H'] = Z['ZZ_1_Higgs_images'][()]
jetImagesDF['Z_T'] = Z['ZZ_1_Top_images'][()]
jetImagesDF['Z_W'] = Z['ZZ_1_W_images'][()]
jetImagesDF['Z_Z'] = Z['ZZ_1_Z_images'][()]
jetBESvarsDF['Z'] = Z['ZZ_1_BES_vars'][()]
jetLabETDF['Z'] = Z['ZZ_1_LabFrameET'][()]
Z.close()
B = h5py.File("images/BB_1phiCosThetaBoostedJetImagesX10.h5","r")
jetImagesDF['B_H'] = B['BB_1_Higgs_images'][()]
jetImagesDF['B_T'] = B['BB_1_Top_images'][()]
jetImagesDF['B_W'] = B['BB_1_W_images'][()]
jetImagesDF['B_Z'] = B['BB_1_Z_images'][()]
jetBESvarsDF['B'] = B['BB_1_BES_vars'][()]
jetLabETDF['B'] = B['BB_1_LabFrameET'][()]
B.close()


for i in range(2, 6):
   QCD = h5py.File("images/QCD_Flat_"+str(i)+"phiCosThetaBoostedJetImagesX10.h5","r")
   jetImagesDF['QCD_H'] = numpy.concatenate([jetImagesDF['QCD_H'], QCD['QCD_Flat_'+str(i)+'_Higgs_images'][()]])
   jetImagesDF['QCD_T'] = numpy.concatenate([jetImagesDF['QCD_T'], QCD['QCD_Flat_'+str(i)+'_Top_images'][()]])
   jetImagesDF['QCD_W'] = numpy.concatenate([jetImagesDF['QCD_W'], QCD['QCD_Flat_'+str(i)+'_W_images'][()]])
   jetImagesDF['QCD_Z'] = numpy.concatenate([jetImagesDF['QCD_Z'], QCD['QCD_Flat_'+str(i)+'_Z_images'][()]])
   jetBESvarsDF['QCD'] = numpy.concatenate([jetBESvarsDF['QCD'], QCD['QCD_Flat_'+str(i)+'_BES_vars'][()]])
   jetLabETDF['QCD'] = numpy.concatenate([jetLabETDF['QCD'], QCD['QCD_Flat_'+str(i)+'_LabFrameET'][()]])
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
      jetLabETDF[heavy] = numpy.concatenate([jetLabETDF[heavy], imagefile[heavy+heavy+'_'+str(i)+'_LabFrameET'][()]])
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

qcdLabET = jetLabETDF['QCD']
qcdLabET = qcdLabET[:90000]
higgsLabET = jetLabETDF['H']
higgsLabET = higgsLabET[:90000]
topLabET = jetLabETDF['T']
topLabET = topLabET[:90000]
WLabET = jetLabETDF['W']
WLabET = WLabET[:90000]
ZLabET = jetLabETDF['Z']
ZLabET = ZLabET[:90000]
bLabET = jetLabETDF['B']
bLabET = bLabET[:90000]


jetHiggsImages = numpy.concatenate([HiggsQCDImages, HiggsHImages, HiggsTImages, HiggsWImages, HiggsZImages, HiggsBImages ])
jetTopImages = numpy.concatenate([TopQCDImages, TopHImages, TopTImages, TopWImages, TopZImages, TopBImages ])
jetWImages = numpy.concatenate([WQCDImages, WHImages, WTImages, WWImages, WZImages, WBImages ])
jetZImages = numpy.concatenate([ZQCDImages, ZHImages, ZTImages, ZWImages, ZZImages, ZBImages ])

jetLabels = numpy.concatenate([numpy.zeros(len(HiggsQCDImages) ), numpy.ones(len(HiggsHImages) ), numpy.full(len(HiggsTImages), 2), numpy.full(len(HiggsWImages), 3), numpy.full(len(HiggsZImages), 4), numpy.full(len(HiggsBImages), 5)] )

jetBESvars = numpy.concatenate([qcdBESvars, higgsBESvars, topBESvars, WBESvars, ZBESvars, bBESvars])
print jetBESvars
print "Stored data and truth information"

copyqcd = []
copyhiggs = []
copytop = []
copyW = []
copyZ = []
copyb = []

copyqcd = copy.copy(qcdBESvars)
copyhiggs = copy.copy(higgsBESvars)
copytop = copy.copy(topBESvars)
copyW = copy.copy(WBESvars)
copyZ = copy.copy(ZBESvars)
copyb = copy.copy(bBESvars)

besthistqcd = numpy.array(copyqcd).T
besthisthiggs = numpy.array(copyhiggs).T
besthisttop = numpy.array(copytop).T
besthistW = numpy.array(copyW).T
besthistZ = numpy.array(copyZ).T
besthistb = numpy.array(copyb).T

if plotInputs == True:
   plt.figure()
   plt.hist(qcdLabET, bins=100, color='b', label='QCD', histtype='step', normed=True)
   plt.hist(WLabET, bins=100, color='g', label='W', histtype='step', normed=True)
   plt.hist(ZLabET, bins=100, color='y', label='Z', histtype='step', normed=True)
   plt.hist(higgsLabET, bins=100, color='m', label='H', histtype='step', normed=True)
   plt.hist(topLabET, bins=100, color='r', label='T', histtype='step', normed=True)
   plt.hist(bLabET, bins=100, color='c', label='B', histtype='step', normed=True)
   plt.xlabel('Lab Frame E_T')
   plt.legend()
   if savePDF == True:
      plt.savefig("plots/Hist_LabFrameET_.pdf")
   plt.close()

   for index, hist in enumerate(besthistqcd):
      print index
      plt.figure()
      plt.hist(hist, bins=100, color='b', label='QCD', histtype='step', normed=True)
      plt.hist(besthistW[index], bins=100, color='g', label='W', histtype='step', normed=True)
      plt.hist(besthistZ[index], bins=100, color='y', label='Z', histtype='step', normed=True)
      plt.hist(besthisthiggs[index], bins=100, color='m', label='H', histtype='step', normed=True)
      plt.hist(besthisttop[index], bins=100, color='r', label='T', histtype='step', normed=True)
      plt.hist(besthistb[index], bins=100, color='c', label='B', histtype='step', normed=True)
      plt.xlabel(index)
      plt.legend()
      if savePDF == True:
         plt.savefig("plots/Hist_"+str(index)+"_.pdf")
      plt.close()
      pass
   pass

