#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# imageOperations.py --------------------------------------------------------------
#==================================================================================
# This module contains functions to make Jet Images -------------------------------
#==================================================================================

# modules
import ROOT as root
import numpy
import pandas as pd
import matplotlib as mpl
mpl.use('Agg') #prevents opening displays, must use before pyplot
import matplotlib.pyplot as plt
import copy
import random
import itertools
import types
import tempfile
import timeit
import sys
# grab some keras stuff
from os import environ
environ["KERAS_BACKEND"] = "tensorflow" #must set backend before importing keras
import keras.backend as K

#==================================================================================
# Get PF Candidate Branches -------------------------------------------------------
#----------------------------------------------------------------------------------
# tree is TTree -------------------------------------------------------------------
#----------------------------------------------------------------------------------

def getPFcandBranchNames(tree ):

   # empty array to store names
   treeVars = []

   # loop over branches
   for branch in tree.GetListOfBranches():
      name = branch.GetName()
      # Only get PF branches
      if 'PF' in name:
         treeVars.append(name)
      if 'jetAK8_pt' in name:
         treeVars.append(name)

   return treeVars

#==================================================================================
# Get Higgs Frame Candidate Branches ----------------------------------------------
#----------------------------------------------------------------------------------
# tree is TTree -------------------------------------------------------------------
#----------------------------------------------------------------------------------
def getBoostCandBranchNamesHiggs(tree ):
   treeVars = []
   for branch in tree.GetListOfBranches():
      name = branch.GetName()
      if 'HiggsFrame' in name:
         if '_px' in name:
            treeVars.append(name)
         if '_py' in name:
            treeVars.append(name)
         if '_pz' in name:
            treeVars.append(name)
         if '_e' in name:
            treeVars.append(name)

   return treeVars
def getBoostCandBranchNamesTop(tree ):
   treeVars = []
   for branch in tree.GetListOfBranches():
      name = branch.GetName()
      if 'TopFrame' in name:
         if '_px' in name:
            treeVars.append(name)
         if '_py' in name:
            treeVars.append(name)
         if '_pz' in name:
            treeVars.append(name)
         if '_e' in name:
            treeVars.append(name)

   return treeVars
def getBoostCandBranchNamesW(tree ):
   treeVars = []
   for branch in tree.GetListOfBranches():
      name = branch.GetName()
      if 'WFrame' in name:
         if '_px' in name:
            treeVars.append(name)
         if '_py' in name:
            treeVars.append(name)
         if '_pz' in name:
            treeVars.append(name)
         if '_e' in name:
            treeVars.append(name)

   return treeVars
def getBoostCandBranchNamesZ(tree ):
   treeVars = []
   for branch in tree.GetListOfBranches():
      name = branch.GetName()
      if 'ZFrame' in name:
         if '_px' in name:
            treeVars.append(name)
         if '_py' in name:
            treeVars.append(name)
         if '_pz' in name:
            treeVars.append(name)
         if '_e' in name:
            treeVars.append(name)

   return treeVars

def getBoostCandBranchNames(tree ):

   # empty array to store names
   treeVars = []

   # loop over branches
   for branch in tree.GetListOfBranches():
      name = branch.GetName()
      # Only get PF branches
      if '_px' in name:
         treeVars.append(name)
      if '_py' in name:
         treeVars.append(name)
      if '_pz' in name:
         treeVars.append(name)
      if '_e' in name:
         treeVars.append(name)

      # if 'Frame_PF' in name:
      #    treeVars.append(name)
      # if 'jetAK8_pt' in name:
      #    treeVars.append(name)
      # if 'jetAK8_phi' in name:
      #    treeVars.append(name)
      # if 'jetAK8_eta' in name:
      #    treeVars.append(name)
      # if 'jetAK8_mass' in name:
      #    treeVars.append(name)

   return treeVars

#==================================================================================
# Make array with PF candidate information ----------------------------------------
#----------------------------------------------------------------------------------
# This function converts the array made from the jetTree to the correct form to ---
#   use with the jet image functions ----------------------------------------------
# array is a numpy array made from a TTree ----------------------------------------
#----------------------------------------------------------------------------------

def makePFcandArray(array):

   tmpArray = []  #use lists not numpy arrays (wayyyyy faster)
   jetCount = 1
   n = 0
   # loop over jets
   while n < len(array) :
      # loop over pf candidates
      for i in range( len(array[n][1][:]) ) :
         jetPt = array[n][0]
         pt = array[n][1][i]
         phi = array[n][2][i]
         eta = array[n][3][i]
         tmpArray.append([jetCount, jetPt, pt, eta, phi])
      jetCount +=1
      n += 1

   newArray = copy.copy(tmpArray)
   return newArray

#==================================================================================
# Make array with Boosted PF candidate information --------------------------------
#----------------------------------------------------------------------------------
# This function converts the array made from the jetTree to the correct form to ---
#   use with the jet image functions ----------------------------------------------
# array is a numpy array made from a TTree ----------------------------------------
#----------------------------------------------------------------------------------

def makeBoostCandArray(array):

   tmpArray = []  #use lists not numpy arrays (wayyyyy faster)
   jetCount = 1
   n = 0
   # loop over jets
   while n < len(array) :
      # loop over pf candidates
      for i in range( len(array[n][4][:]) ) :
         jetPt = array[n][0]
         px = array[n][4][i]
         py = array[n][5][i]
         pz = array[n][6][i]
         e = array[n][7][i]
         candLV = root.TLorentzVector(px, py, pz, e)
         tmpArray.append([jetCount, jetPt, candLV.Pt(), candLV.Eta(), candLV.Phi()])
      jetCount +=1
      n += 1

   newArray = copy.copy(tmpArray)
   return newArray

#==================================================================================
# Make array with Boosted PF candidate 4 vectors ----------------------------------
#----------------------------------------------------------------------------------
# This function converts the array made from the jetTree to the correct form to ---
#   use with the boosted jet image functions --------------------------------------
# array is a numpy array made from a TTree ----------------------------------------
#----------------------------------------------------------------------------------

def makeBoostCandFourVector(array):

   tmpArray = []  #use lists not numpy arrays (wayyyyy faster)
   jetCount = 1
   entryNum = 0
   n = 0
   # loop over jets
   while n < len(array) :
      if n % 10000 == 0: print "Making 4 vector array for jet number: ", jetCount, sys.getsizeof(tmpArray),"bytes"
      # loop over pf candidates
      #changed index arguement to match changes to variable list
      for i in range( len(array[n][0][:]) ) :
         px = array[n][0][i]
         py = array[n][1][i]
         pz = array[n][2][i]
         e = array[n][3][i]
         candLV = root.TLorentzVector()
         candLV.SetPxPyPzE(px, py, pz, e)
         # List the most energetic candidate first
         if i == 0:
            tmpArray.append([jetCount, candLV])
         elif i > 0 and candLV.E() > tmpArray[entryNum - i][1].E() :
            tmpArray.append([jetCount, tmpArray[entryNum - i][1] ])
            tmpArray[entryNum - i] = [jetCount, candLV]
         else:
            tmpArray.append([jetCount, candLV]) 
         entryNum += 1
      jetCount +=1
      n += 1

   newArray = copy.copy(tmpArray)
   
   return newArray

#==================================================================================
# Boosted Candidate Rotations -----------------------------------------------------
#----------------------------------------------------------------------------------
# candArray is an array of four vectors for one jet -------------------------------
#----------------------------------------------------------------------------------

def boostedRotationsLeadZ(candArray):
   phiPrime = []
   thetaPrime = []

   # define the rotation angles for first two rotations
   rotPhi = candArray[0].Phi()
   rotTheta = candArray[0].Theta()
   subPhi = 0

   # Perform the first two rotations
   subleadE = -1
   leadE = candArray[0].E()
   leadLV = candArray[0]
   for icand in candArray :

      # Waring for incorrect energy sorting
      if icand.E() > leadE : print "WARNING: Energy sorting was done incorrectly!"

      icand.RotateZ(-rotPhi)
      #set small py values to 0
      if abs(icand.Py() ) < 0.01 : icand.SetPy(0) 

      icand.RotateY(-rotTheta)
      # make sure leading candidate vector has been rotated
      if icand.E() == leadE : leadLV = icand
      #set small px values to 0
      if abs(icand.Px() ) < 0.01 : icand.SetPx(0)

      # Find subleading candidate
      if icand.E() > subleadE and icand.E() < leadE :
         if abs( (icand.CosTheta() - leadLV.CosTheta() ) ) > 0.3 :
            subleadE = icand.E()
            # store its phi for a third rotation
            subPhi = icand.Phi()

   # Perform the third rotation
   for icand in candArray :

      # warning for subleading identification
      if icand.E() > subleadE and icand.E() < leadE : 
         if abs( (icand.CosTheta() - leadLV.CosTheta() ) ) > 0.3 : print "WARNING: Subleading candidate was improperly identified!"  

      icand.RotateZ(-subPhi)
      #set small py values to 0
      if abs(icand.Py() ) < 0.01 : icand.SetPy(0)

      # store image info
      phiPrime.append(icand.Phi() )
      thetaPrime.append( icand.CosTheta() )

   return numpy.array(phiPrime), numpy.array(thetaPrime)

#==================================================================================
# Boosted Candidate Rotations -----------------------------------------------------
#----------------------------------------------------------------------------------
# candArray is an array of four vectors for one jet -------------------------------
#----------------------------------------------------------------------------------

def boostedRotations(candArray):
   phiPrime = []
   thetaPrime = []
   energyNorm = []
   # define the rotation angles for first two rotations
   rotPhi = candArray[0].Phi()
   rotTheta = candArray[0].Theta()
   subPsi = 0

   # Perform the first two rotations
   subleadE = -1
   leadE = candArray[0].E()
   leadLV = candArray[0]
   it_counter = 0
   for icand in candArray :
      energyNorm.append(icand.E() / leadE)
      it_counter+=1
      # Warning for incorrect energy sorting
      if icand.E() > leadE : print "WARNING: Energy sorting was done incorrectly!"

      icand.RotateZ(-rotPhi)
      #set small py values to 0
      if icand.E() == leadE:
         if abs(icand.Py() ) < 0.01 : icand.SetPy(0) 

      icand.RotateY(numpy.pi/2 - rotTheta)

      # make sure leading candidate vector has been rotated
      if icand.E() == leadE : 
         leadLV = icand
      #set small pZ values to 0
         if abs(icand.Pz() ) < 0.01 : icand.SetPz(0)

      # Find subleading candidate
      if icand.E() > subleadE and icand.E() < leadE :
         if abs( (icand.Phi() - leadLV.Phi() ) ) > 1.0 :
            subleadE = icand.E()
            # store its y z projection angle to Z axis (psi) for a third rotation
            # arctan2 is very important
            subPsi = numpy.arctan2(icand.Py(), icand.Pz() )

   # Perform the third rotation
   it_counter = 0
   for icand in candArray :
      it_counter+=1
      # warning for subleading identification
      if icand.E() > subleadE and icand.E() < leadE : 
         if abs( (icand.Phi() - leadLV.Phi() ) ) > 1.0 : print "WARNING: Subleading candidate was improperly identified!"  

      # rotatate about x with psi to get to x-y plane      
      icand.RotateX(subPsi - numpy.pi/2)

      #set small pz values to 0
      #Only do this for the leading and sub-leading candidates, which by definition should have zero pz
      if icand.E() == subleadE or icand.E() == leadE:
         if abs(icand.Pz() ) < 0.01 : icand.SetPz(0)

      # store image info
      #phiPrime.append(icand.Phi() )
      #thetaPrime.append( icand.CosTheta() )

   # Reflect if bottomSum > topSum and/or leftSum > rightSum
   leftSum, rightSum = 0, 0
   topSum, bottomSum = 0, 0
   it_counter = 0
   for icand in candArray :
      it_counter+=1
      if icand.CosTheta() > 0 :
         topSum += icand.E()
      if icand.CosTheta() < 0 :
         bottomSum += icand.E()

      if icand.Phi() > 0 :
         rightSum += icand.E()
      if icand.Phi() < 0 :
         leftSum += icand.E()

   # store image info
   if topSum == bottomSum :
      print "Both sides are the same #EnlightenedCentrism#DumpCandidateInfo"
   for icand in candArray :
      if bottomSum > topSum :
         thetaPrime.append( -icand.CosTheta() )
      if topSum > bottomSum :
         thetaPrime.append( icand.CosTheta() )
      if topSum == bottomSum :
         print "Cos theta:", icand.CosTheta(), "Energy, Px, Py, Pz", icand.E(), icand.Px(), icand.Py(), icand.Pz()

      if leftSum > rightSum :
         phiPrime.append( -icand.Phi() )
      if leftSum < rightSum :
         phiPrime.append(icand.Phi() )

   return numpy.array(phiPrime), numpy.array(thetaPrime), numpy.array(energyNorm)

#==================================================================================
# Boosted Candidate Rotations relative to Boost Axis ------------------------------
#----------------------------------------------------------------------------------
# candArray is an array of four vectors for one jet -------------------------------
#----------------------------------------------------------------------------------

def boostedRotationsRelBoostAxis(candArray, jetLV):
   phiPrime = []
   thetaPrime = []

   # define the rotation angles for first two rotations
   rotPhi = jetLV.Phi()
   rotTheta = jetLV.Theta()
   subPhi = 0

   # Perform the first two rotations
   subleadE = -1
   leadE = candArray[0].E()
   for icand in candArray :

      if icand.E() > leadE : print "WARNING: Energy sorting was done incorrectly!"

      icand.RotateZ(-rotPhi)
      #set small py values to 0
      if abs(icand.Py() ) < 0.01 : icand.SetPy(0) 

      icand.RotateY(-rotTheta)
      #set small px values to 0
      if abs(icand.Px() ) < 0.01 : icand.SetPx(0)

      # store leading phi for a third rotation
      if icand.E() == leadE : 
         leadPhi = icand.Phi()

      # Find subleading candidate
      if icand.DeltaR(candArray[0]) > 0.35 and icand.E() > subleadE :
         subleadE = icand.E()

   # Perform the third rotation
   for icand in candArray :
      icand.RotateZ(-leadPhi)
      #set small py values to 0
      if abs(icand.Py() ) < 0.01 : icand.SetPy(0)

      # store image info
      phiPrime.append(icand.Phi() )
      thetaPrime.append( icand.CosTheta() )

   return numpy.array(phiPrime), numpy.array(thetaPrime)

#==================================================================================
# Make boosted frame Jet Images ---------------------------------------------------
#----------------------------------------------------------------------------------
# make jet image histograms using the candidate data frame and the original -------
#    jet array --------------------------------------------------------------------
# refFrame is the reference frame for the images to be created in -----------------
#----------------------------------------------------------------------------------

def prepareBoostedImages(jet, which_frame, nbins=31):
#removed references here to the lab jet axis, since we aren't considering rotations around jet axis
    nx = nbins #30 # number of image bins in phi
    ny = nbins #30 # number of image bins in theta
    # set limits on relative phi and theta for the histogram
    xbins = numpy.linspace(-numpy.pi,numpy.pi,nx+1)
    ybins = numpy.linspace(-1,1,ny+1)
    if K.image_dim_ordering()=='tf':
        # 4D tensor (tensorflow backend)
        # 1st dim is eta bin
        # 2nd dim is phi bin
        # 3rd dim is pt value (or rgb layer, etc.)
       jet_image = numpy.zeros((nx, ny, 1))
    else:        
       jet_image = numpy.zeros((1, nx, ny))

    #Make list of candidates as LorentzVector
    candLV = []
    sumE = 0
    for i in xrange(0,len(jet.HiggsFrame_PF_candidate_px)):
        candidate = root.TLorentzVector()
        if which_frame == 'H': candidate.SetPxPyPzE(jet.HiggsFrame_PF_candidate_px[i], jet.HiggsFrame_PF_candidate_py[i], jet.HiggsFrame_PF_candidate_pz[i], jet.HiggsFrame_PF_candidate_e[i]) 
        if which_frame == 'T': candidate.SetPxPyPzE(jet.TopFrame_PF_candidate_px[i], jet.TopFrame_PF_candidate_py[i], jet.TopFrame_PF_candidate_pz[i], jet.TopFrame_PF_candidate_e[i])
        if which_frame == 'W': candidate.SetPxPyPzE(jet.WFrame_PF_candidate_px[i], jet.WFrame_PF_candidate_py[i], jet.WFrame_PF_candidate_pz[i], jet.WFrame_PF_candidate_e[i])
        if which_frame == 'Z': candidate.SetPxPyPzE(jet.ZFrame_PF_candidate_px[i], jet.ZFrame_PF_candidate_py[i], jet.ZFrame_PF_candidate_pz[i], jet.ZFrame_PF_candidate_e[i])
        sumE+=candidate.E()
        candLV.append(candidate)

    # perform boosted frame rotations
    # Needs to be sorted first!
    candLV.sort(key = lambda x: x.E(), reverse=True)
    phiPrime,thetaPrime,energyNorm = boostedRotations(candLV)

    if sumE < 1.0:
       print "No energy? Number of candidates", len(jet.HiggsFrame_PF_candidate_px)

       # make a 2D numpy hist for the image
    if len(phiPrime) == 0 or len(thetaPrime) == 0 or len(energyNorm) == 0:
       print "Length of phi, theta, weights", len(phiPrime), len(thetaPrime), len(weights)
    
    hist, xedges, yedges = numpy.histogram2d(phiPrime, thetaPrime, weights = energyNorm, bins=(xbins,ybins))
    for ix in range(0,nx):
       for iy in range(0,ny):
          if K.image_dim_ordering()=='tf':
             jet_image[ix,iy,0] = hist[ix,iy]
          else:
             jet_image[0,ix,iy] = hist[ix,iy]
    return jet_image

#==================================================================================
# Plot Averaged Boosted Jet Images ------------------------------------------------
#----------------------------------------------------------------------------------
# Average over the jet images and plot the result as a 2D histogram ---------------
# title has limited options, see if statements ------------------------------------
#----------------------------------------------------------------------------------

def plotAverageBoostedJetImage(jetImageDF, title, plotPNG, plotPDF):

   # sum and average jet images
   summed = numpy.sum(jetImageDF, axis=0)
   avg = numpy.apply_along_axis(lambda x: x/len(jetImageDF), axis=1, arr=summed)

   # plot the images
   plt.figure('N') 
   plt.imshow(avg[:,:,0].T, norm=mpl.colors.LogNorm(), origin='lower', interpolation='none', extent=[-numpy.pi, numpy.pi, -1, 1], aspect = "auto")
   cbar = plt.colorbar()
   cbar.set_label(r'Energy [GeV]')
   if title == 'boost_QCD' :
      plt.title('QCD Boosted Jet Image', fontsize = 22)
   if title == 'boost_HH4W' :
      plt.title(r'$H\rightarrow WW$ Boosted Jet Image', fontsize = 22)
   if title == 'boost_HH4B' :
      plt.title(r'$H\rightarrow bb$ Boosted Jet Image', fontsize = 22)
   plt.xlabel(r'$\phi$', fontsize = 18)
   plt.ylabel(r'cos($\theta$)', fontsize = 20)
   if plotPNG == True :
      plt.savefig('plots/'+title+'_jetImage.png')
   if plotPDF == True :
      plt.savefig('plots/'+title+'_jetImage.pdf')
   plt.close()

#==================================================================================
# Plot 3 Boosted Jet Images -------------------------------------------------------
#----------------------------------------------------------------------------------
# Average over the jet images and plot the result as a 2D histogram ---------------
# title has limited options, see if statements ------------------------------------
#----------------------------------------------------------------------------------

def plotThreeBoostedJetImages(jetImageDF, title, plotPNG, plotPDF):

   for i in range (0, 3) :
      # plot the images
      plt.figure('N') 
      plt.imshow(jetImageDF[i,:,:,0].T, norm=mpl.colors.LogNorm(), origin='lower', interpolation='none', extent=[-numpy.pi, numpy.pi, -1, 1], aspect = "auto")
      cbar = plt.colorbar()
      cbar.set_label(r'Energy [GeV]')
      if title == 'boost_QCD' :
         plt.title('QCD Boosted Jet Image #'+str(i), fontsize = 22)
      if title == 'boost_HH4W' :
         plt.title(r'$H\rightarrow WW$ Boosted Jet Image #'+str(i), fontsize = 22)
      if title == 'boost_HH4B' :
         plt.title(r'$H\rightarrow bb$ Boosted Jet Image #'+str(i), fontsize = 22)
      plt.xlabel(r'$\phi$', fontsize = 18)
      plt.ylabel(r'cos($\theta$)', fontsize = 20)
      if plotPNG == True :
         plt.savefig('plots/'+title+'_jetImageNum'+str(i)+'.png')
      if plotPDF == True :
         plt.savefig('plots/'+title+'_jetImageNum'+str(i)+'.pdf')
      plt.close()

#==================================================================================
# Plot Molleweide Boosted Jet Images ----------------------------------------------
#----------------------------------------------------------------------------------
# Average over the jet images and plot the result as a 2D histogram ---------------
# title has limited options, see if statements ------------------------------------
#----------------------------------------------------------------------------------

def plotMolleweideBoostedJetImage(jetImageDF, title, plotPNG, plotPDF):

   # sum and average jet images
   summed = numpy.sum(jetImageDF, axis=0)
   avg = numpy.apply_along_axis(lambda x: x/len(jetImageDF), axis=1, arr=summed)

   # plot the images
   fig = plt.figure()
   ax = fig.add_subplot(111, projection = 'mollweide')

   lon = numpy.linspace(-numpy.pi, numpy.pi, 42) 
   lat = numpy.linspace(-numpy.pi/2, numpy.pi/2, 42) 
   Lon, Lat = numpy.meshgrid(lon, lat)

   im = ax.pcolormesh(Lon, Lat, avg[:,:,0].T, norm=mpl.colors.LogNorm() )
   cbar = fig.colorbar(im, orientation='horizontal')
   cbar.set_label(r'Energy [GeV]')
   if title == 'boost_QCD' :
      plt.title('QCD Boosted Jet Image')
   if title == 'boost_HH4W' :
      plt.title(r'$H\rightarrow WW$ Boosted Jet Image')
   if title == 'boost_HH4B' :
      plt.title(r'$H\rightarrow bb$ Boosted Jet Image')
   plt.xlabel(r'$\phi_i$')
   plt.ylabel(r'$\theta_i$')
#   plt.xticks(numpy.arange(-4, 4, step=1.0) )
#   plt.yticks(numpy.arange(0, 4, step=1.0) )
   if plotPNG == True :
      plt.savefig('plots/'+title+'_jetImage.png')
   if plotPDF == True :
      plt.savefig('plots/'+title+'_jetImage.pdf')
   plt.close()

#==================================================================================
# Make lab frame Jet Images -------------------------------------------------------
#----------------------------------------------------------------------------------
# make jet image histograms using the candidate data frame and the original -------
#    jet array --------------------------------------------------------------------
# refFrame is the reference frame for the images to be created in -----------------
#----------------------------------------------------------------------------------

def prepareImages(candDF, jetArray, refFrame):

    nx = 30 # size of image in eta
    ny = 30 # size of image in phi
    # set limits on relative phi and eta for the histogram
    if refFrame == 'lab' :
       xbins = numpy.linspace(-1.4,1.4,nx+1)
       ybins = numpy.linspace(-1.4,1.4,ny+1)
    if refFrame == 'boost' :
       xbins = numpy.linspace(-7,7,nx+1)
       ybins = numpy.linspace(-7,7,ny+1)

    list_x = []
    list_y = []
    list_w = []
    if K.image_dim_ordering()=='tf':
        # 4D tensor (tensorflow backend)
        # 1st dim is jet index
        # 2nd dim is eta bin
        # 3rd dim is phi bin
        # 4th dim is pt value (or rgb layer, etc.)
        jet_images = numpy.zeros((len(jetArray), nx, ny, 1))
    else:        
        jet_images = numpy.zeros((len(jetArray), 1, nx, ny))
    
    for i in range(0,len(jetArray)):
        if i % 1000 == 0: print "Imaging jet number: ", i+1
        # get the ith jet
        mask = candDF['njet'].values == i + 1 # boolean mask ... the fastest way from stack
        df_unsorted_cand_i = candDF[mask] # new data frame with only ith jet candidates
        df_dict_cand_i = df_unsorted_cand_i.sort_values(by=['cand_pt'], ascending=False) # sort candidates by pt

        # relative eta
        x = df_dict_cand_i['cand_eta']-df_dict_cand_i['cand_eta'].iloc[0]
        # relative phi
        y = df_dict_cand_i['cand_phi']-df_dict_cand_i['cand_phi'].iloc[0]
        weights = df_dict_cand_i['cand_pt'] # pt of candidate is the weight

        # rotate rel eta and rel phi
        x,y = rotate_and_reflect(x,y,weights)
        list_x.append(x)
        list_y.append(y)
        list_w.append(weights)


        hist, xedges, yedges = numpy.histogram2d(x, y, weights=weights, bins=(xbins,ybins))
        for ix in range(0,nx):
            for iy in range(0,ny):
                if K.image_dim_ordering()=='tf':
                    jet_images[i,ix,iy,0] = hist[ix,iy]
                else:
                    jet_images[i,0,ix,iy] = hist[ix,iy]
    return jet_images

#==================================================================================
# Rotate and Reflect Jet Images ---------------------------------------------------
#----------------------------------------------------------------------------------
# x is relative eta, i.e. the difference between jet eta and the eta of each ------
#   daughter ----------------------------------------------------------------------
# y is relative phi, i.e. the difference between jet phi and the phi of each ------
#   daughter ----------------------------------------------------------------------
# w is the weight. Typically jet pT -----------------------------------------------
#----------------------------------------------------------------------------------

def rotate_and_reflect(x,y,w):
    rot_x = []
    rot_y = []
    theta = 0
    maxPt = -1
    for ix, iy, iw in zip(x, y, w):
        dv = numpy.matrix([[ix],[iy]])-numpy.matrix([[x.iloc[0]],[y.iloc[0]]])
        dR = numpy.linalg.norm(dv)
        thisPt = iw

        # Find the second highest Pt cand that is not in leading subjet
        if dR > 0.35 and thisPt > maxPt:
            maxPt = thisPt

            # rotation in eta-phi plane c.f  https://arxiv.org/abs/1407.5675 and https://arxiv.org/abs/1511.05190:
            theta = -numpy.arctan2(iy,ix)-numpy.radians(90)

            # rotation by lorentz transformation c.f. https://arxiv.org/abs/1704.02124:
            #px = iw * numpy.cos(iy)
            #py = iw * numpy.sin(iy)
            #pz = iw * numpy.sinh(ix)
            # calculate polar angle
            #theta = numpy.arctan2(py,pz)+numpy.radians(90)
    # make the rotation matrix        
    c, s = numpy.cos(theta), numpy.sin(theta)
    R = numpy.matrix('{} {}; {} {}'.format(c, -s, s, c))

    # rotate all candidates using theta
    for ix, iy, iw in zip(x, y, w):

        # rotation in eta-phi plane:
        rot = R*numpy.matrix([[ix],[iy]])
        rix, riy = rot[0,0], rot[1,0]

        # rotation by lorentz transformation
        #px = iw * numpy.cos(iy)
        #py = iw * numpy.sin(iy)
        #pz = iw * numpy.sinh(ix)
        #rot = R*numpy.matrix([[py],[pz]])
        #px1 = px
        #py1 = rot[0,0]
        #pz1 = rot[1,0]
        #iw1 = numpy.sqrt(px1*px1+py1*py1)
        #rix, riy = numpy.arcsinh(pz1/iw1), numpy.arcsin(py1/iw1) #range of arcsine is -pi to pi
        rot_x.append(rix)
        rot_y.append(riy)
        
    # now reflect if leftSum > rightSum
    leftSum = 0
    rightSum = 0
    for ix, iy, iw in zip(x, y, w):
        if ix > 0: 
            rightSum += iw
        elif ix < 0:
            leftSum += iw
    if leftSum > rightSum:
        ref_x = [-1.*rix for rix in rot_x]
        ref_y = rot_y
    else:
        ref_x = rot_x
        ref_y = rot_y
    
    return numpy.array(ref_x), numpy.array(ref_y)

#==================================================================================
# Plot Averaged Jet Images --------------------------------------------------------
#----------------------------------------------------------------------------------
# Average over the jet images and plot the result as a 2D histogram ---------------
# title has limited options, see if statements ------------------------------------
#----------------------------------------------------------------------------------

def plotAverageJetImage(jetImageDF, title, plotPNG, plotPDF):

   summed = numpy.sum(jetImageDF, axis=0)
   avg = numpy.apply_along_axis(lambda x: x/len(jetImageDF), axis=1, arr=summed)
   plt.figure('N') 
   plt.imshow(avg[:,:,0].T, norm=mpl.colors.LogNorm(), origin='lower', interpolation='none', extent=[-1.4,1.4, -1.4, 1.4])
   cbar = plt.colorbar()
   cbar.set_label(r'$p_T$ [GeV]')
   if title == 'lab_QCD' :
      plt.title('QCD Lab Jet Image', fontsize = 22)
   if title == 'lab_HH4W' :
      plt.title(r'$H\rightarrow WW$ Lab Jet Image', fontsize = 22)
   if title == 'lab_HH4B' :
      plt.title(r'$H\rightarrow bb$ Lab Jet Image', fontsize = 22)
   plt.xlabel(r'$\eta_i$', fontsize = 18)
   plt.ylabel(r'$\phi_i$', fontsize = 20)
   if plotPNG == True :
      plt.savefig('plots/'+title+'_jetImage.png')
   if plotPDF == True :
      plt.savefig('plots/'+title+'_jetImage.pdf')
   plt.close()
