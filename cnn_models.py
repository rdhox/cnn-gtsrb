#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 11:25:49 2021

@author: renaud
"""

import numpy as np
from numpy import mean
from numpy import std
from sklearn.utils import shuffle 
from matplotlib import pyplot
import sys
import h5py
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import KFold

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Dropout
from keras.layers import Flatten

sys.path.append('..')

def read_dataset(enhanced_dir, dataset_name):
   '''Reads h5 dataset
   Args:
       filename     : datasets filename
       dataset_name : dataset name, without .h5
   Returns:    x_train,y_train, x_test,y_test data, x_meta,y_meta'''
   # ---- Read dataset
   filename = f'{enhanced_dir}/{dataset_name}.h5'
   with  h5py.File(filename,'r') as f:
       x_train = f['x_train'][:]
       y_train = f['y_train'][:]
       x_test  = f['x_test'][:]
       y_test  = f['y_test'][:]
       x_meta  = f['x_meta'][:]
       y_meta  = f['y_meta'][:]
   print(x_train.shape, y_train.shape)
   # ---- Shuffle
   x_train,y_train=shuffle(x_train,y_train)

   # ---- done
   return x_train, y_train, x_test,y_test, x_meta,y_meta


# ***** Declaring models *****
def get_model_v1(lx, ly, lz):
    model = Sequential()
    
    model.add(Conv2D(96, (3, 3), activation='relu', input_shape=(lx, ly, lz)))
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(192, (3, 3), activation='relu'))
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(1500, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(43, activation='softmax'))
    
    plot_model(model, to_file='model_V1_plot.png', show_shapes=True, show_layer_names=True)
    
    return model

def get_model_v2(lx,ly,lz):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(lx,ly,lz), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3, 3), padding='same',activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(43, activation='softmax'))
    return model

def get_model_v3(lx,ly,lz):
    model = Sequential()
    
    model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(lx,ly,lz)))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten()) 
    model.add(Dense(1152, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(43, activation='softmax'))
    return model

# plot diagnostic learning curves
def summarize_diagnostics(histories):
	for i in range(len(histories)):
		# plot loss
		pyplot.subplot(2, 1, 1)
		pyplot.title('Cross Entropy Loss')
		pyplot.plot(histories[i].history['loss'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
		# plot accuracy
		pyplot.subplot(2, 1, 2)
		pyplot.title('Classification Accuracy')
		pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')
	pyplot.show()
    
# summarize model performance
def summarize_performance(scores):
	# print summary
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
	# box and whisker plots of results
	pyplot.boxplot(scores)
	pyplot.show()


# parameters
enhanced_dir = './data/enhanced'
final_dir = './data/final'

dataset_name = 'set-24x24-RGB'
batch_size = 64
epochs = 5
scale = 1

# ***** Evalution of the Model using Kfold *****
def evaluate_model_kfold(dataX, dataY, fn_model, n_folds=5):
    scores, histories = list(), list()
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    (n, lx, ly, lz) = dataX.shape
    
    for train_ix, test_ix in kfold.split(dataX):
        model = fn_model(lx, ly, lz)
        # Select rows for train and test
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        # compile model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # fit model
        history = model.fit(trainX, trainY,
                            batch_size = batch_size,
                            epochs = epochs,
                            validation_data=(testX, testY),
                            verbose=1)
        # evaluate model
        _, accuracy = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (accuracy*100))
        # keep track
        scores.append(accuracy)
        histories.append(history)
    summarize_diagnostics(histories)
    summarize_performance(scores)
    

def evaluate_model(dataX, dataY, testX, testY, fn_model):
    (n, lx, ly, lz) = dataX.shape
    model = fn_model(lx,ly,lz)
    model.summary()
    model.compile(optimizer = 'adam',
                  loss      = 'sparse_categorical_crossentropy',
                  metrics   = ['accuracy'])
    history = model.fit(dataX, dataY,
                      batch_size      = batch_size,
                      epochs          = epochs,
                      verbose         = 1,
                      validation_data = (testX, testY))
    score = model.evaluate(testX, testY, verbose=0)   
    summarize_diagnostics([history])
    summarize_performance([score])

def run_model():
    # import the dataset
    x_train, y_train, x_test,y_test, x_meta, y_meta = read_dataset(final_dir, dataset_name)
    # show example of data
    # pyplot.imshow(x_train[1], interpolation='nearest')
    # pyplot.show()
    
    # Evaluate with KFold
    evaluate_model_kfold(x_train, y_train, get_model_v1)
    
    # Evaluate directly
    # evaluate_model(x_train, y_train, x_test, y_test, get_model_v1)
    


run_model()




