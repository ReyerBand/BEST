import h5py
import numpy
import random
import keras
import matplotlib
matplotlib.use('Agg') #prevents opening displays, must use before pyplot                                                                                                                          
import matplotlib.pyplot as plt
from sklearn import svm, metrics, preprocessing, neural_network, tree
from sklearn.model_selection import train_test_split
from collections import OrderedDict
class GenerateBatch(object):
   def __init__(self, batch_size = 1200, validation_frac = 0.1, debug_info = False, debug_plots = False):
      #File list should be a dict mapping the class to the file
      filelist = {'QCD' : 'images/FullQCD.h5', 'H' : 'images/FullH.h5', 'T' : 'images/FullT.h5', 'W' : 'images/FullW.h5', 'Z' : 'images/FullZ.h5', 'B' : 'images/FullB.h5'}
      weightfilelist = {'QCD' : 'QCDEventWeights.h5', 'H' : 'HEventWeights.h5', 'T' : 'TEventWeights.h5', 'W' : 'WEventWeights.h5', 'Z' : 'ZEventWeights.h5', 'B' : 'BEventWeights.h5'}
      self.filelist = filelist
      self.weightfilelist = weightfilelist
      self.batch_size = batch_size
#      self.inputs = ['H', 'T', 'W', 'Z', 'BES']
      self.inputs = ['H', 'T', 'BES']
      self.classes = ['QCD', 'H', 'T', 'W', 'Z', 'B']
      self.num_classes = 6
      self.validation_frac = validation_frac
      self.debug_info = debug_info
      self.debug_plots = debug_plots
      #Load all the data files into memory
      self.weights = self.OpenWeightFiles()
      self.data = self.OpenDataFiles()
      self.last_train_keep = {'QCD' : [], 'H' : [], 'T' : [], 'W' : [], 'Z' : [], 'B' : []}
      self.last_valid_keep = {'QCD' : [], 'H' : [], 'T' : [], 'W' : [], 'Z' : [], 'B' : []}
      self.train_indices, self.valid_indices =  self.split_train_valid()
      print 'Initialized Generator'

   def OpenWeightFiles(self):
      weight_dict = {}
      for flavor in self.classes:
         temp_file = h5py.File(self.weightfilelist[flavor], "r")
         weight_dict[flavor] = temp_file[flavor][()]
#         print type(temp_file), type(weight_dict), type(weight_dict[flavor])
         temp_file.close()
      print 'Loaded All Weights Into Memory'
      return weight_dict

   def OpenDataFiles(self):
      data_dict = {}
      for flavor in self.classes:
         temp_file = h5py.File(self.filelist[flavor], "r")
         for label in self.inputs:
            data_dict[flavor+'_'+label] = temp_file[flavor+'_'+label][()]
         temp_file.close()
      print 'Loaded All Data Into Memory'
      if self.debug_plots:
         print len(data_dict['QCD_BES']), len(data_dict['H_BES']), len(data_dict['T_BES']), len(data_dict['W_BES']), len(data_dict['Z_BES']), len(data_dict['B_BES'])
         for i in range(0,len(data_dict['QCD_BES'][0])):
            plt.figure()
            plt.hist(data_dict['QCD_BES'][i], bins=20, color='b', label='QCD', histtype='step', normed=True)
            plt.hist(data_dict['H_BES'][i], bins=20, color='m', label='H', histtype='step', normed=True)
            plt.hist(data_dict['T_BES'][i], bins=20, color='r', label='T', histtype='step', normed=True)
            plt.hist(data_dict['W_BES'][i], bins=20, color='g', label='W', histtype='step', normed=True)
            plt.hist(data_dict['Z_BES'][i], bins=20, color='y', label='Z', histtype='step', normed=True)
            plt.hist(data_dict['B_BES'][i], bins=20, color='c', label='B', histtype='step', normed=True)
            plt.xlabel(i)
            plt.legend()
            plt.savefig("plots/Hist_"+str(i)+"_.pdf")
            plt.close()
         print 'Plotted All BES Inputs'
      return data_dict

   def split_train_valid(self):
      keep_train_list = {}
      keep_valid_list = {}
      for flavor in self.classes:
         keep_train_list[flavor] = []
         keep_valid_list[flavor] = []
         pass
      #Each entry in this dict should be a list of indices, where the key is which class
      #Initialize a list with entries being indices of dataset, shuffle it to randomize order weights are viewed in.      
      #Loop over weights instead of dataset, much smaller amount to load                                                                                                                                           
      #This should give number per pT bin in weights                                                                                                                                                                 
      for flavor in self.classes:   
#         weights = h5py.File(self.weightfilelist[flavor], "r")
         temp_list = []
         for i in range(0, len(self.weights[flavor])):
            temp_list.append(i)
            pass
         numpy.random.shuffle(temp_list)
         train_stop_index = int((1-self.validation_frac) * len(self.weights[flavor]))
         for index in temp_list[0:train_stop_index]:
            keep_train_list[flavor].append(index)
            pass
         for index in temp_list[train_stop_index:len(self.weights[flavor])]:
            keep_valid_list[flavor].append(index)
            pass
#         weights.close()
         pass
      if self.debug_info:
         for flavor in self.classes:
            with open('TrainIndices_'+flavor+'.txt', 'w') as indexfile:
               for weight_val in keep_train_list[flavor]:
                  indexfile.write('%s \n' %weight_val)
               indexfile.close()
            with open('ValidIndices_'+flavor+'.txt', 'w') as indexfile:
               for weight_val in keep_valid_list[flavor]:
                  indexfile.write('%s \n' %weight_val)
               indexfile.close()
            set_train = set(keep_train_list[flavor])
            set_valid = set(keep_valid_list[flavor])
            print 'Are training/validation disjoint?', set_valid.isdisjoint(set_train)
      return keep_train_list, keep_valid_list
   
   #Method for iterating over a file, to be used with a file of 1 class.  Returns batch_size / 6 because it will be called six times.                                                                                
   def generate_train_batch(self, file_name, weight_file):
 #     dataset = h5py.File(file_name, "r")
 #     weights = h5py.File(weight_file, "r")
      particle = weight_file[:-15]
      validation_frac = self.validation_frac
      keep_train_list = []
      
      for index in self.train_indices[particle]:
         rand_num = numpy.random.uniform(0, 1)
         weight = self.weights[particle][index]
         if weight > rand_num:
            keep_train_list.append(index)
            pass
         pass

      numpy.random.shuffle(keep_train_list)

      #Slice these to the correct length, should be batch_size/num_classes
      if len(keep_train_list) > (self.batch_size/self.num_classes):
         keep_train_list = keep_train_list[0:int(self.batch_size/self.num_classes)]

      if self.debug_info:
         print "Number of duplicated training events between batches: ", len(set(keep_train_list).intersection(set(self.last_train_keep[particle])))
         print set(keep_train_list).intersection(set(self.last_train_keep[particle]))
         self.last_train_keep[particle] = keep_train_list
      return_train_batch = [] 
      temp_image_train_list = []
      best_vars_train_list = []

      if 'QCD' in particle: particle_index = 1
      if 'H' in particle: particle_index = 2
      if 'T' in particle: particle_index = 3
      if 'W' in particle: particle_index = 4
      if 'Z' in particle: particle_index = 5
      if 'B' in particle: particle_index = 6

      for key in self.inputs:
         if 'BES' not in key:
            temp_image_train_list.append(numpy.zeros((len(keep_train_list), 31, 31, 1)))
#            temp_image_train_list.append(numpy.full((len(keep_train_list), 31, 31, 1), particle_index))
            pass
         pass 

      for i, key in enumerate(self.inputs):
         if 'BES' not in key:
            for n, index in enumerate(keep_train_list):
               temp_image_train_list[i][n] = self.data[particle+'_'+key][index]
#               temp_image_train_list[i].append(self.data[particle+'_'+key][index])
               if self.debug_info and n is 0:
                  print key, type(self.data[particle+'_'+key][index]), len(self.data[particle+'_'+key][index])
               pass
            pass
         if 'BES' in key:
            for n, index in enumerate(keep_train_list):
               best_vars_train_list.append(self.data[particle+'_'+key][index])
#               best_vars_train_list.append(numpy.zeros(44))
               if self.debug_info and n is 0:
                  print key, type(self.data[particle+'_'+key][index]), len(self.data[particle+'_'+key][index])
#               return_train_batch.append(dataset[particle+'_'+key][index])
               pass
      return_train_batch = [temp_image_train_list[0], temp_image_train_list[1], best_vars_train_list]
#      dataset.close()
#      weights.close()
      return return_train_batch

   def generate_valid_batch(self, file_name, weight_file):
#      dataset = h5py.File(file_name, "r")
#      weights = h5py.File(weight_file, "r")
      particle = weight_file[:-15]
      validation_frac = self.validation_frac
      
      keep_valid_list = []

      for index in self.valid_indices[particle]:
         rand_num = numpy.random.uniform(0, 1)
         weight = self.weights[particle][index]
         if weight > rand_num:
            keep_valid_list.append(index)
            pass
         pass

      numpy.random.shuffle(keep_valid_list)

      #Slice these to the correct length, should be batch_size/num_classes                                                                                                                                           
      if len(keep_valid_list) > (self.validation_frac * self.batch_size/self.num_classes):
#         keep_valid_list = keep_valid_list[0:int(self.validation_frac * (self.batch_size/self.num_classes))]
         keep_valid_list = keep_valid_list[0:int((self.batch_size/self.num_classes))]                  
         pass
      if self.debug_info:
         print "Number of duplicated validation events between batches: ", len(set(keep_valid_list).intersection(set(self.last_valid_keep[particle])))
         print set(keep_valid_list).intersection(set(self.last_valid_keep[particle]))
         self.last_valid_keep[particle] = keep_valid_list

      return_valid_batch = []
      temp_image_valid_list = []
      best_vars_valid_list = []

      if 'QCD' in particle: particle_index = 1
      if 'H' in particle: particle_index = 2
      if 'T' in particle: particle_index = 3
      if 'W' in particle: particle_index = 4
      if 'Z' in particle: particle_index = 5
      if 'B' in particle: particle_index = 6


      for key in self.inputs:
         if 'BES' not in key:
            temp_image_valid_list.append(numpy.zeros((len(keep_valid_list), 31, 31, 1)))
#            temp_image_valid_list.append(numpy.full((len(keep_valid_list), 31, 31, 1), particle_index))
            pass
         pass #Should result in 4 entries in the list                                                                                                                                       

      for i, key in enumerate(self.inputs):
         if 'BES' not in key:
            for n, index in enumerate(keep_valid_list):
               temp_image_valid_list[i][n] = (self.data[particle+'_'+key])[index]               
               pass
            pass
         if 'BES' in key:
            for index in keep_valid_list:
               best_vars_valid_list.append(self.data[particle+'_'+key][index])
#               best_vars_valid_list.append(numpy.zeros(44))
               pass
      return_valid_batch = [temp_image_valid_list[0], temp_image_valid_list[1], best_vars_valid_list]
#      dataset.close()
#      weights.close()
      return return_valid_batch


   def train_looping(self):
      #This uses generate_batch on each input file, returns the training tuples (data, truth)
      for index, key in enumerate(self.filelist):
         f = self.filelist[key]
         f_w = self.weightfilelist[key]
#         if self.debug_info: print 'Data class:', f, 'Weight class:', f_w, 'Index:', index
         train_temp = self.generate_train_batch(f, f_w)
         if index == 0:
            big_train_batch = train_temp
            big_train_truth = numpy.full(len(train_temp[0]),index)
         else:
            big_train_truth = numpy.concatenate([big_train_truth, numpy.full(len(train_temp[0]),index)])
            for i in range(0,len(train_temp)):
               big_train_batch[i] = numpy.concatenate([big_train_batch[i], train_temp[i]])


      empty_train_batch = [None for i in range(0, len(big_train_batch))]
      #Needs to be set by hand to correct length? seems a bit wrong 
      big_train_batch[0], empty_train_batch[0], big_train_batch[1], empty_train_batch[1], big_train_batch[2], empty_train_batch[2], big_train_truth, empty_train_truth = train_test_split(big_train_batch[0], big_train_batch[1], big_train_batch[2], big_train_truth, test_size = 0.0)
      return big_train_batch, keras.utils.to_categorical(big_train_truth, num_classes = self.num_classes)

   def valid_looping(self):
      for index, key in enumerate(self.filelist):
         f = self.filelist[key]
         f_w = self.weightfilelist[key]
         valid_temp = self.generate_valid_batch(f, f_w)
         if index == 0:
            big_valid_batch = valid_temp
            big_valid_truth = numpy.full(len(valid_temp[0]),index)
            pass
         else:
            big_valid_truth = numpy.concatenate([big_valid_truth, numpy.full(len(valid_temp[0]),index)])
            for i in range(0,len(valid_temp)):
               big_valid_batch[i] = numpy.concatenate([big_valid_batch[i], valid_temp[i]])


#      shuffle_tuple = list(zip(big_valid_batch, big_valid_truth))
#      numpy.random.shuffle(shuffle_tuple)
#      big_valid_batch, big_valid_truth = (list(t) for t in (zip(*shuffle_tuple)))
      empty_valid_batch = [None for i in range(0, len(big_valid_batch))]
      big_valid_batch[0], empty_valid_batch[0], big_valid_batch[1], empty_valid_batch[1], big_valid_batch[2], empty_valid_batch[2], big_valid_truth, empty_valid_truth = train_test_split(big_valid_batch[0], big_valid_batch[1], big_valid_batch[2], big_valid_truth, test_size = 0.0)


      return big_valid_batch, keras.utils.to_categorical(big_valid_truth, num_classes = self.num_classes)
#      return shuffle_tuple
#      return big_valid_batch, big_valid_truth

   def generator_train(self):
      while True:
         training = self.train_looping()
         yield training
   def generator_valid(self):
      while True:
         valid = self.valid_looping()
         yield valid


