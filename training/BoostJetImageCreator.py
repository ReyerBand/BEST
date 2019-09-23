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

# get stuff from modules
from root_numpy import tree2array

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
fileCandidates = root.TFile(filename, "READ")

# access the trees
treeCandidates = fileCandidates.Get("run/jetTree")

print "Accessed the trees"

# get input variable names from branches
#varnames = img.getBoostCandBranchNames(treeCandidates)

Hnames = img.getBoostCandBranchNamesHiggs(treeCandidates)
Tnames = img.getBoostCandBranchNamesTop(treeCandidates)
Wnames = img.getBoostCandBranchNamesW(treeCandidates)
Znames = img.getBoostCandBranchNamesZ(treeCandidates)


print "Variables for Higgs jet image creation: ", Hnames

# create selection criteria
#sel = ""
#sel = "jetAK8_pt > 500 && jetAK8_mass > 50"
sel = "tau32 < 9999. && et > 500. && SDmass > 10"

# make arrays from the trees
LabArrayJets = tree2array(treeCandidates, ['et'], sel)

HarrayCandidates = tree2array(treeCandidates, Hnames, sel)
TarrayCandidates = tree2array(treeCandidates, Tnames, sel)
WarrayCandidates = tree2array(treeCandidates, Wnames, sel)
ZarrayCandidates = tree2array(treeCandidates, Znames, sel)

HarrayCandidates = tools.appendTreeArray(HarrayCandidates)
TarrayCandidates = tools.appendTreeArray(TarrayCandidates)
WarrayCandidates = tools.appendTreeArray(WarrayCandidates)
ZarrayCandidates = tools.appendTreeArray(ZarrayCandidates)
LabArrayJets = tools.appendTreeArray(LabArrayJets)
print "Number of Jets that will be imaged: ", len(LabArrayJets)
print "Number of constituents that will be imaged: ", len(HarrayCandidates), len(TarrayCandidates), len(WarrayCandidates), len(ZarrayCandidates)

HimgArrayCandidates = img.makeBoostCandFourVector(HarrayCandidates)
TimgArrayCandidates = img.makeBoostCandFourVector(TarrayCandidates)
WimgArrayCandidates = img.makeBoostCandFourVector(WarrayCandidates)
ZimgArrayCandidates = img.makeBoostCandFourVector(ZarrayCandidates)

print "Made candidate 4 vector arrays from the datasets"

#==================================================================================
# Store BEST Variables ////////////////////////////////////////////////////////////
#==================================================================================

# get BEST variable names from branches
bestVars = tools.getBestBranchNames(treeCandidates)
print "Boosted Event Shape Variables: ", bestVars

# make arrays from the trees
#start, stop, step = 0, 167262, 1
bestArrayCandidates = tree2array(treeCandidates, bestVars, sel)#, None, start, stop, step)
bestArrayCandidates = tools.appendTreeArray(bestArrayCandidates)

print "Made array with the Boosted Event Shape Variables"

#==================================================================================
# Make Jet Images /////////////////////////////////////////////////////////////////
#==================================================================================

jetImagesDF = {}
print "Creating boosted Jet Images for ", filename
jetImagesDF[filename[:-5]+'_Higgs_images'] = img.prepareBoostedImages(HimgArrayCandidates, LabArrayJets, 31, False)
jetImagesDF[filename[:-5]+'_Top_images'] = img.prepareBoostedImages(TimgArrayCandidates, LabArrayJets, 31, False)
jetImagesDF[filename[:-5]+'_W_images'] = img.prepareBoostedImages(WimgArrayCandidates, LabArrayJets, 31, False)
jetImagesDF[filename[:-5]+'_Z_images'] = img.prepareBoostedImages(ZimgArrayCandidates, LabArrayJets, 31, False)

print "Made jet image data frames"

#==================================================================================
# Store BEST Variables in DataFrame ///////////////////////////////////////////////
#==================================================================================

jetImagesDF[filename[:-5]+'_BES_vars'] = bestArrayCandidates
print "Stored BES variables"

#==================================================================================
# Store Data in h5 file ///////////////////////////////////////////////////////////
#==================================================================================

h5f = h5py.File("images/"+filename[:-5]+"phiCosThetaBoostedJetImagesX10.h5","w")
h5f.create_dataset(filename[:-5]+'_Higgs_images', data=jetImagesDF[filename[:-5]+'_Higgs_images'], compression='lzf')
h5f.create_dataset(filename[:-5]+'_Top_images', data=jetImagesDF[filename[:-5]+'_Top_images'], compression='lzf')
h5f.create_dataset(filename[:-5]+'_W_images', data=jetImagesDF[filename[:-5]+'_W_images'], compression='lzf')
h5f.create_dataset(filename[:-5]+'_Z_images', data=jetImagesDF[filename[:-5]+'_Z_images'], compression='lzf')
h5f.create_dataset(filename[:-5]+'_BES_vars', data=jetImagesDF[filename[:-5]+'_BES_vars'], compression='lzf')

print "Saved Candidates Boosted Jet Images"

#==================================================================================
# Plot Jet Images /////////////////////////////////////////////////////////////////
#==================================================================================

# plot with python
if plotJetImages == True:
   print "Plotting Average Boosted jet images"
   img.plotAverageBoostedJetImage(jetImagesDF[filename[:-5]+'_Higgs_images'], filename[:-5]+'boost_Higgs', savePNG, savePDF)
   img.plotThreeBoostedJetImages(jetImagesDF[filename[:-5]+'_Higgs_images'], filename[:-5]+'boost_Higgs', savePNG, savePDF)
   img.plotAverageBoostedJetImage(jetImagesDF[filename[:-5]+'_Top_images'], filename[:-5]+'boost_Top', savePNG, savePDF)
   img.plotThreeBoostedJetImages(jetImagesDF[filename[:-5]+'_Top_images'], filename[:-5]+'boost_Top', savePNG, savePDF)
   img.plotAverageBoostedJetImage(jetImagesDF[filename[:-5]+'_W_images'], filename[:-5]+'boost_W', savePNG, savePDF)
   img.plotThreeBoostedJetImages(jetImagesDF[filename[:-5]+'_W_images'], filename[:-5]+'boost_W', savePNG, savePDF)
   img.plotAverageBoostedJetImage(jetImagesDF[filename[:-5]+'_Z_images'], filename[:-5]+'boost_Z', savePNG, savePDF)
   img.plotThreeBoostedJetImages(jetImagesDF[filename[:-5]+'_Z_images'], filename[:-5]+'boost_Z', savePNG, savePDF)

print "Program was a great success!!!"

