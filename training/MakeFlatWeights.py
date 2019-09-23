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
import ROOT as r
# get stuff from modules
from sklearn import svm, metrics, preprocessing, neural_network, tree
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

# set up keras


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
jetLabETDF = {}
QCD = h5py.File("images/QCD_Flat_1phiCosThetaBoostedJetImagesX10.h5","r")
jetLabETDF['QCD'] = QCD['QCD_Flat_1_LabFrameET'][()]
QCD.close()
H = h5py.File("images/HH_1phiCosThetaBoostedJetImagesX10.h5","r")
jetLabETDF['H'] = H['HH_1_LabFrameET'][()]
H.close()
T = h5py.File("images/TT_1phiCosThetaBoostedJetImagesX10.h5","r")
jetLabETDF['T'] = T['TT_1_LabFrameET'][()]
T.close()
W = h5py.File("images/WW_1phiCosThetaBoostedJetImagesX10.h5","r")
jetLabETDF['W'] = W['WW_1_LabFrameET'][()]
W.close()
Z = h5py.File("images/ZZ_1phiCosThetaBoostedJetImagesX10.h5","r")
jetLabETDF['Z'] = Z['ZZ_1_LabFrameET'][()]
Z.close()
B = h5py.File("images/BB_1phiCosThetaBoostedJetImagesX10.h5","r")
jetLabETDF['B'] = B['BB_1_LabFrameET'][()]
B.close()


for i in range(2, 6):
   QCD = h5py.File("images/QCD_Flat_"+str(i)+"phiCosThetaBoostedJetImagesX10.h5","r")
   jetLabETDF['QCD'] = numpy.concatenate([jetLabETDF['QCD'], QCD['QCD_Flat_'+str(i)+'_LabFrameET'][()]])
   QCD.close()

for i in range(2, 6):
   for heavy in ('H','T','W','Z','B'):
      if (heavy is 'W' or heavy is 'B') and i == 5: continue
      imagefile = h5py.File("images/"+heavy+heavy+"_"+str(i)+"phiCosThetaBoostedJetImagesX10.h5","r")
      jetLabETDF[heavy] = numpy.concatenate([jetLabETDF[heavy], imagefile[heavy+heavy+'_'+str(i)+'_LabFrameET'][()]])
      imagefile.close()

print "Accessed Jet Images and BES variables"

#h5f.close()

print "Made image dataframes"

#==================================================================================
# Train the Neural Network ////////////////////////////////////////////////////////
#==================================================================================

# Store data and truth

qcdLabET = jetLabETDF['QCD']
higgsLabET = jetLabETDF['H']
topLabET = jetLabETDF['T']
WLabET = jetLabETDF['W']
ZLabET = jetLabETDF['Z']
bLabET = jetLabETDF['B']


w_dict = {}



for label in ['QCD', 'H', 'T', 'W', 'Z', 'B']:
   print label+':', len(jetLabETDF[label])
   w_dict[label] = numpy.zeros((len(jetLabETDF[label]), 1))
   it_hist = r.TH1F(label+'_source', label+'_source', 50, 0, 2500)
   for entry, hist in enumerate(jetLabETDF[label]):
      it_hist.Fill(hist[0]) #Literally just turning the numpy array in the h5 back into a root hist.
      pass
   it_flat = r.TH1F(label+'_flat', label+'_flat', 50, 0, 2500)
   for entry, hist in enumerate(jetLabETDF[label]):
      rand_seed = numpy.random.uniform(0, 1)
      pt = hist[0]
      keep_chance = 100 / float(it_hist.GetBinContent(it_hist.FindBin(pt)))
      w_dict[label][entry] = keep_chance
      if keep_chance > rand_seed:
         it_flat.Fill(pt)
         pass
      pass
   canv = r.TCanvas('c1', 'c1')
   canv.cd()
   it_flat.Draw()
   it_flat.SetMinimum(0.0)
   canv.SaveAs('Flat'+label+'_100bin.pdf')
   it_hist.Draw()
   it_hist.SetMinimum(0.0)
   canv.SaveAs('Normal'+label+'_100bin.pdf')
   h5f = h5py.File(label+'EventWeights.h5', 'w')
   h5f.create_dataset(label, data=w_dict[label], compression='lzf')
   print label+' Number of Events:', it_flat.Integral()
   pass

