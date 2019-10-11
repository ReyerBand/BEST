import h5py
import numpy
import random
import keras

class GenerateBatch(object):
   def __init__(self, batch_size = 1200, validation_frac = 0.1):
      #File list should be a dict mapping the class to the file
      filelist = {'QCD' : 'images/FullQCD.h5', 'H' : 'images/FullH.h5', 'T' : 'images/FullT.h5', 'W' : 'images/FullW.h5', 'Z' : 'images/FullZ.h5', 'B' : 'images/FullB.h5'}
      weightfilelist = {'QCD' : 'QCDEventWeights.h5', 'H' : 'HEventWeights.h5', 'T' : 'TEventWeights.h5', 'W' : 'WEventWeights.h5', 'Z' : 'ZEventWeights.h5', 'B' : 'BEventWeights.h5'}
      self.filelist = filelist
      self.weightfilelist = weightfilelist
      self.batch_size = batch_size
#      self.inputs = ['H', 'T', 'W', 'Z', 'BES']
      self.inputs = ['H', 'BES']
      self.classes = ['QCD', 'H', 'T', 'W', 'Z', 'B']
      self.num_classes = 6
      self.validation_frac = validation_frac
      self.train_indices, self.valid_indices =  self.split_train_valid()
      print 'Initialized Generator'

   def split_train_valid(self):
#      keep_train_list = {'QCD' : [], 'H' : [], 'T' : [], 'W' : [], 'Z' : [], 'b' : []}
#      keep_valid_list = {'QCD' : [], 'H' : [], 'T' : [], 'W' : [], 'Z' : [], 'b' : []}
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
#         dataset = h5py.File(self.filelist[flavor])
         weights = h5py.File(self.weightfilelist[flavor])
         temp_list = []
#         print 'Key :', flavor, 'Weight length: ', len(weights[flavor])
         for i in range(0, len(weights[flavor])):
            temp_list.append(i)
            pass
         numpy.random.shuffle(temp_list)
         train_stop_index = int((1-self.validation_frac) * len(weights[flavor]))
#         print 'Going up to entry ', train_stop_index
         for index in temp_list[0:train_stop_index]:
            keep_train_list[flavor].append(index)
            pass
         for index in temp_list[train_stop_index:len(weights[flavor])]:
            keep_valid_list[flavor].append(index)
            pass
#         dataset.close()
         weights.close()
         pass
      return keep_train_list, keep_valid_list
   
   #Method for iterating over a file, to be used with a file of 1 class.  Returns batch_size / 6 because it will be called six times.                                                                                
   def generate_train_batch(self, file_name, weight_file):
      dataset = h5py.File(file_name)
      weights = h5py.File(weight_file)
      particle = weight_file[:-15]
      validation_frac = self.validation_frac
      keep_train_list = []
      
      #Loop over weights instead of dataset, much smaller amount to load                                                                                                                                             
      #This should give number per pT bin in weights                                                                                                                                                                 
      for index in self.train_indices[particle]:
         rand_num = numpy.random.uniform(0, 1)
         weight = weights[particle][index]
         if weight > rand_num:
            keep_train_list.append(index)
            pass
         pass

      numpy.random.shuffle(keep_train_list)

      #Slice these to the correct length, should be batch_size/num_classes                                                                                                                                           
      if len(keep_train_list) > (self.batch_size/self.num_classes):
         keep_train_list = keep_train_list[0:(self.batch_size/self.num_classes)]
         pass

      return_train_batch = [] 
      temp_image_train_tuple = []
      best_vars_train_tuple = []
      for key in self.inputs:
         if 'BES' not in key:
            temp_image_train_tuple.append(numpy.zeros((len(keep_train_list), 31, 31, 1)))
            pass
         pass #Should result in 4 entries in the tuple

      for i, key in enumerate(self.inputs):
         if 'BES' not in key:
            for n, index in enumerate(keep_train_list):
               temp_image_train_tuple[i][n] = (dataset[particle+'_'+key])[index]
               pass
            pass
         if 'BES' in key:
            for index in keep_train_list:
               best_vars_train_tuple.append(dataset[particle+'_'+key][index])
               pass
      return_train_batch = [temp_image_train_tuple, best_vars_train_tuple]

      dataset.close()
      weights.close()
      return return_train_batch
   def generate_valid_batch(self, file_name, weight_file):
      dataset = h5py.File(file_name)
      weights = h5py.File(weight_file)
      particle = weight_file[:-15]
      validation_frac = self.validation_frac
      
      keep_valid_list = []

      for index in self.valid_indices[particle]:
         rand_num = numpy.random.uniform(0, 1)
         weight = weights[particle][index]
         if weight > rand_num:
            keep_valid_list.append(index)
            pass
         pass

      numpy.random.shuffle(keep_valid_list)

      #Slice these to the correct length, should be batch_size/num_classes                                                                                                                                           
      if len(keep_valid_list) > (self.validation_fraction * self.batch_size/self.num_classes):
	      keep_valid_list = keep_valid_list[0:(self.validation_fraction * (self.batch_size/self.num_classes))]
              pass

      return_valid_batch = []
#      temp_image_valid_tuple = numpy.zeros((len(keep_valid_list), 31, 31, 1))
      temp_image_valid_tuple = []
      best_vars_valid_tuple = []

      for key in self.inputs:
         if 'BES' not in key:
            temp_image_valid_tuple.append(numpy.zeros((len(keep_valid_list), 31, 31, 1)))
            pass
         pass #Should result in 4 entries in the tuple                                                                                                                                       
      for i, key in enumerate(self.inputs):
         if 'BES' not in key:
            for n, index in enumerate(keep_valid_list):
               temp_image_valid_tuple[i][n] = (dataset[particle+'_'+key])[index]
               pass
            pass
         if 'BES' in key:
            for index in keep_valid_list:
               best_vars_valid_tuple.append(dataset[particle+'_'+key][index])
               pass
      return_valid_batch = [temp_image_valid_tuple, best_vars_valid_tuple]

      dataset.close()
      weights.close()
      return return_valid_batch


   def train_looping(self):
      #This uses generate_batch on each input file, returns the training and validation tuples 
      big_train_batch = []
      big_train_truth = []
      for index, key in enumerate(self.filelist):
         f = self.filelist[key]
         f_w = self.weightfilelist[key]
#         print f, f_w
#         print index
         train_temp = self.generate_train_batch(f, f_w)
         big_train_batch.append(train_temp)
         big_train_truth.append(index)
#         big_train_truth = numpy.full(len(big_train_batch), truth_number)
#         big_valid_truth = numpy.full(len(big_valid_batch), truth_number)
         pass
      shuffle_tuple = list(zip(big_train_batch, big_train_truth))
      numpy.random.shuffle(shuffle_tuple)
      big_train_batch, big_train_truth = zip(*shuffle_tuple)
      return big_train_batch, keras.utils.to_categorical(big_train_truth, num_classes = self.num_classes)

   def valid_looping(self):
      big_valid_batch = []
      big_valid_truth = []

      for index, key in enumerate(self.filelist):
         f = self.filelist[key]
         f_w = self.weightfilelist[key]
         valid_temp = self.generate_valid_batch(f, f_w)
         big_valid_batch.append(valid_temp)
         big_valid_truth.append(index)
         pass
      shuffle_tuple = list(zip(big_valid_batch, big_valid_truth))
      numpy.random.shuffle(shuffle_tuple)
      big_valid_batch, big_valid_truth = zip(*shuffle_tuple)
      return big_valid_batch, keras.utils.to_categorical(big_valid_truth, num_classes = self.num_classes)


   def generator_train(self):
#      while True:
      training = self.train_looping()
      yield training
   def generator_valid(self):
#      while True:
      valid = self.valid_looping()
      yield valid
