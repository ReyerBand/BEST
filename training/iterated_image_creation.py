#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# boost_jetImageCreator.py ////////////////////////////////////////////////////////
#==================================================================================
# This program makes boosted frame cosTheta phi jet images ////////////////////////
#==================================================================================

# modules
import ROOT as root
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
import timeit
import sys

# set up keras
from os import environ
environ["KERAS_BACKEND"] = "tensorflow" #must set backend before importing keras

# user modules
#import functions as tools
#import imageOperations as img
import tools.functions as tools
import tools.imageOperations as img

# enter batch mode in root (so python can access displays)
root.gROOT.SetBatch(True)

# set options 
plotJetImages = True
boostAxis = False
savePDF = True
savePNG = False

filename = sys.argv[1]
print filename
print filename[:-5]
#==================================================================================
# Load Monte Carlo ////////////////////////////////////////////////////////////////
#==================================================================================

# access the TFiles
#Storing data in /eos/, accessing through url 
fileCandidates = root.TFile.Open('root://cmsxrootd.fnal.gov//store/user/rband/BESTSamples/'+filename)

# access the trees
treeCandidates = fileCandidates.Get("run/jetTree")
print treeCandidates, type(treeCandidates)

#Declare the file to be written to
h5f = h5py.File("images/"+filename[:-5]+"JetImages.h5","w")
bestVars = tools.getBestBranchNames(treeCandidates)

# Loop over tree, making a numpy array for each image and the BES variables
print "Number of jets:", treeCandidates.GetEntries()
num_pass = 0
for index, jet in enumerate(treeCandidates):
   #Selection criteria here
   if index%1000 == 1: print "Imaging jet", index
   if (jet.et > 500 and jet.tau32 < 9999 and jet.SDmass > 10):
      if num_pass == 0:
         H_image = img.prepareBoostedImages(jet, 'H')
         T_image = img.prepareBoostedImages(jet, 'T')
         W_image = img.prepareBoostedImages(jet, 'W')
         Z_image = img.prepareBoostedImages(jet, 'Z')
         BES_vars = tools.GetBESVars(jet, bestVars)
         num_pass += 1
      else:
         H_image = numpy.append(H_image, img.prepareBoostedImages(jet, 'H'))
         T_image = numpy.append(T_image, img.prepareBoostedImages(jet, 'T'))
         W_image = numpy.append(W_image, img.prepareBoostedImages(jet, 'W'))
         Z_image = numpy.append(Z_image, img.prepareBoostedImages(jet, 'Z'))
         BES_vars = numpy.append(BES_vars, tools.GetBESVars(jet, bestVars))
         num_pass += 1
         
      if num_pass%1000 == 1: print "Jet,", num_pass


h5f.create_dataset(filename[:-5]+'_H_image', data=H_image, compression='lzf')
h5f.create_dataset(filename[:-5]+'_T_image', data=T_image, compression='lzf')
h5f.create_dataset(filename[:-5]+'_W_image', data=W_image, compression='lzf')
h5f.create_dataset(filename[:-5]+'_Z_image', data=Z_image, compression='lzf')
h5f.create_dataset(filename[:-5]+'_BES_vars', data=BES_vars, compression='lzf')

# plot with python
# if plotJetImages == True:
#    print "Plotting Average Boosted jet images"
#    img.plotAverageBoostedJetImage(jetImagesDF[filename[:-5]+'_Higgs_images'], filename[:-5]+'boost_Higgs', savePNG, savePDF)
#    img.plotThreeBoostedJetImages(jetImagesDF[filename[:-5]+'_Higgs_images'], filename[:-5]+'boost_Higgs', savePNG, savePDF)
#    img.plotAverageBoostedJetImage(jetImagesDF[filename[:-5]+'_Top_images'], filename[:-5]+'boost_Top', savePNG, savePDF)
#    img.plotThreeBoostedJetImages(jetImagesDF[filename[:-5]+'_Top_images'], filename[:-5]+'boost_Top', savePNG, savePDF)
#    img.plotAverageBoostedJetImage(jetImagesDF[filename[:-5]+'_W_images'], filename[:-5]+'boost_W', savePNG, savePDF)
#    img.plotThreeBoostedJetImages(jetImagesDF[filename[:-5]+'_W_images'], filename[:-5]+'boost_W', savePNG, savePDF)
#    img.plotAverageBoostedJetImage(jetImagesDF[filename[:-5]+'_Z_images'], filename[:-5]+'boost_Z', savePNG, savePDF)
#    img.plotThreeBoostedJetImages(jetImagesDF[filename[:-5]+'_Z_images'], filename[:-5]+'boost_Z', savePNG, savePDF)

print "Program was a great success!!!"

