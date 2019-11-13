#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# functions.py ////////////////////////////////////////////////////////////////////
#==================================================================================
# This module contains functions to be used with HHESTIA //////////////////////////
#==================================================================================

# modules
import numpy
import matplotlib
matplotlib.use('Agg') #prevents opening displays, must use before pyplot
import matplotlib.pyplot as plt
import copy
import random
import itertools
import types
import tempfile
import keras.models

# grab some keras stuff
from os import environ
environ["KERAS_BACKEND"] = "tensorflow" #must set backend before importing keras
import keras.backend as K

# functions from modules
from sklearn import svm, metrics, preprocessing, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

#==================================================================================
# Plot Confusion Matrix ///////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# cm is the comfusion matrix //////////////////////////////////////////////////////
# classes are the names of the classes that the classifier distributes among //////
#----------------------------------------------------------------------------------

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
   """
   This function prints and plots the confusion matrix.
   Normalization can be applied by setting `normalize=True`.
   """
   if normalize:
       cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
       print("Normalized confusion matrix")
   else:
       print('Confusion matrix, without normalization')

   print(cm)

   plt.imshow(cm, interpolation='nearest', cmap=cmap)
   plt.title(title)
   plt.colorbar()
   tick_marks = numpy.arange(len(classes))
   plt.xticks(tick_marks, classes, rotation=45)
   plt.yticks(tick_marks, classes)

   fmt = '.2f' if normalize else 'd'
   thresh = cm.max() / 2.
   for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
       plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

   plt.ylabel('True label')
   plt.xlabel('Predicted label')
   plt.tight_layout() #make all the axis labels not get cutoff

#==================================================================================
# Get Branch Names ////////////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# tree is a TTree /////////////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------

def getBranchNames(tree ):

   # empty array to store names
   treeVars = []

   # loop over branches
   for branch in tree.GetListOfBranches():
      name = branch.GetName()
      if 'nJets' in name:
         continue
      if 'SoftDropMass' in name:
         continue
      if 'mass' in name:
         continue
      if 'gen' in name:
         continue
      if 'pt' in name:
         continue
      treeVars.append(name)

   return treeVars

#==================================================================================
# Get BEST Branch Names ///////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# tree is a TTree /////////////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------

def getBestBranchNames(tree ):

   # empty array to store names
   treeVars = []

   # loop over branches
   for branch in tree.GetListOfBranches():
      name = branch.GetName()

      # if 'nJets' in name:
      #    continue
      # if 'SoftDropMass' in name:
      #    continue
      # if 'mass' in name:
      #    continue
      # if 'gen' in name:
      #    continue
      # if 'pt' in name:
      #    continue
      # if 'candidate' in name:
      #    continue
      # if 'subjet' in name:
      #    continue
      # if 'PuppiWeight' in name:
      #    continue
      # if 'NNout' in name:
      #    continue
      # if 'is' in name:
      #    continue

      if 'Njets' in name:
         continue
      if 'target' in name:
         continue
      if 'NNout' in name:
         continue
      if 'gen' in name:
         continue
      if 'flatten' in name:
         continue
      if 'dist' in name:
         continue
      if 'npv' in name:
         continue
      if 'sorting' in name:
         continue
      if name == 'mass':
         continue
      if name == 'minDist':
         continue
      if 'et' in name:
         continue
      if 'Frame' in name:
         continue
      if 'Njets' in name:
         continue
      if 'target' in name:
         continue
      if 'NNout' in name:
         continue
      if 'gen' in name:
         continue
      if 'flatten' in name:
         continue
      if 'dist' in name:
         continue
      if 'npv' in name:
         continue
      if 'sorting' in name:
         continue
      if name == 'mass':
         continue
      if name == 'minDist':
         continue
      if 'et' in name:
         continue
      if name == 'isB':
         continue
      if name == 'isT':
         continue
      if name == 'isW':
         continue
      if name == 'isZ':
         continue
      if name == 'isH':
         continue
      if 'over' in name:
         continue
      if 'CSV' in name:
         continue
      if 'PF' in name:
         continue
      if 'm12_' in name and not 'm12_H' in name:
         continue
      if 'm13_' in name and not 'm13_H' in name:
         continue
      if 'm23_' in name and not 'm23_H' in name:
         continue
      if 'm1234_' in name and not 'm1234_H' in name:
         continue
      if 'isotropy' in name and not 'isotropy_H' in name:
         continue
      if 'sumP' in name:
         continue
      if 'PuppiWeight' in name:
         continue

      treeVars.append(name)
   print len(treeVars), "variables used in BES will be:", treeVars
   return treeVars

#==================================================================================                                                                                                                          
# Save BES Vars to a numpy array ////////////////////////////////////////////////////////                                                                                                                     
def GetBESVars(jet, treeVars):
   bes_vars = []
   for var_name in treeVars:
      bes_vars.append(getattr(jet,var_name))
   return numpy.array(bes_vars)

#==================================================================================
# Append Arrays from trees ////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# array is a numpy array made from a TTree ////////////////////////////////////////
#----------------------------------------------------------------------------------

def appendTreeArray(array):

   tmpArray = []
   for entry in array[:] :
      a = list(entry)
      tmpArray.append(a)
   newArray = copy.copy(tmpArray)
   return newArray

#==================================================================================
# Randomize Data //////////////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# array is an array of TTree arrays ( [ tree1array, tree2array, ...] ) ////////////
#----------------------------------------------------------------------------------

def randomizeData(array):

   trainData = []
   targetData = []
   nEvents = 0
   for iArray in range(len(array) ) :
      nEvents = nEvents + len(array[iArray])
   while nEvents > 0:
      rng = random.randint(0,len(array)-1 )
      if (len(array[rng]) > 0):
         trainData.append(array[rng].pop() )
         targetData.append(rng)
         nEvents = nEvents - 1
   return trainData, targetData

#==================================================================================
# Plot Performance ////////////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# loss is an array of loss and loss_val from the training /////////////////////////
# acc is an array of acc and acc_val from the training ////////////////////////////
# train_test is the train data that has not been trained on ///////////////////////
# target_test is the target data that has not been trained on /////////////////////
# target_predict is the models prediction of data that has not been trained on ////
#----------------------------------------------------------------------------------

def plotPerformance(loss, acc, adToTitle): #, train_test, target_test, target_predict):
   
   # plot loss vs epoch
   plt.figure()
   plt.plot(loss[0], label='loss')
   plt.plot(loss[1], label='val_loss')
   plt.legend(loc="upper right")
   plt.xlabel('epoch')
   plt.ylabel('loss')
   plt.savefig("plots/"+adToTitle+"_loss.pdf")
   plt.savefig("plots/"+adToTitle+"_loss.png")
   plt.close()

   # plot accuracy vs epoch
   plt.figure()
   plt.plot(acc[0], label='acc')
   plt.plot(acc[1], label='val_acc')
   plt.legend(loc="upper left")
   plt.xlabel('epoch')
   plt.ylabel('acc')
   plt.savefig("plots/"+adToTitle+"_acc.pdf")
   plt.savefig("plots/"+adToTitle+"_acc.png")
   plt.close()

   # Plot ROC
#   fpr, tpr, thresholds = roc_curve(target_test, target_predict)
#   roc_auc = auc(fpr, tpr)
#   ax = plt.subplot(2, 2, 3)
#   ax.plot(fpr, tpr, lw=2, color='cyan', label='auc = %.3f' % (roc_auc))
#   ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='random chance')
#   ax.set_xlim([0, 1.0])
#   ax.set_ylim([0, 1.0])
#   ax.set_xlabel('false positive rate')
#   ax.set_ylabel('true positive rate')
#   ax.set_title('receiver operating curve')
#   ax.legend(loc="lower right")

#==================================================================================
# Plot Probabilities //////////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# probs is an array of probabilites, labels, and colors ///////////////////////////
#    [ [probArray, label, color], .. ] ////////////////////////////////////////////
#----------------------------------------------------------------------------------

def plotProbabilities(probs):

   for iProb in range(len(probs) ) :
      for jProb in range(len(probs) ) :
         plt.figure()
         plt.xlabel("Probability for " + probs[iProb][1] + " Classification")
         plt.hist(probs[jProb][0].T[iProb], bins=20, range=(0,1), label=probs[jProb][1], color=probs[jProb][2], histtype='step', 
                  normed=True, log = True)
         plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=6, mode="expand", borderaxespad=0.)
         plt.savefig("prob_" + probs[iProb][1] + ".pdf")
         plt.close()

#==================================================================================
# Generate Filter Visualizations //////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# This function allows us to see what patterns each convolutional filter is -------
#    trained to extract -----------------------------------------------------------
#----------------------------------------------------------------------------------

def generate_pattern(model,layer_name, filter_index, size):
   
   # build a loss function that maximizes the activation of the nth filter of the layer
   layer_output = model.get_layer(layer_name).output
   loss = K.mean(layer_output[:, :, :, filter_index])

   # Compute the gradient of the input picture using this loss
   grads = K.gradients(loss, model.input)[0]

   # Normalize the gradient (nice trick)
   grads /= (K.sqrt(K.mean(K.square(grads) ) ) + 1e-5)

   # Get loss and grads for an input picture
   iterate = K.function([model.input], [loss, grads] )

   # Start with a grey image with some noise
   input_img_data = numpy.random.random((1,size, size, 1)) * 20 + 128

   # Run gradient ascent for 40 steps
   step = 1
   for i in range(40):
      loss_value, grads_value = iterate([input_img_data])
      input_img_data += grads_value * step

   img = input_img_data[0]

   return deprocess_image(img)

#==================================================================================
# Convert Tensor into an Image ////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# Utility function to convert a tensor into a valid image -------------------------
#----------------------------------------------------------------------------------

def deprocess_image(x):
   
   # Normalize the tensor
   x -= x.mean()
   x /= (x.std() + 1e-5)
   x *= 0.1

   # clip to [0,1]
   x += 0.5
   #x = numpy.clip(x, 0, 1)

   # convert to RGB array
   x *= 25.5
   x = numpy.clip(x, 0, 255).astype('uint8')

   return x

#==================================================================================
# Make Keras Picklable  ///////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# A patch to make Keras give results in pickle format /////////////////////////////
#----------------------------------------------------------------------------------

def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__


    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__






   
