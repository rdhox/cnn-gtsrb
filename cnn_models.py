#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 11:25:49 2021

@author: renaud
"""
import os, random, time
import numpy as np
from numpy import mean, std
from sklearn.utils import shuffle 
from matplotlib import pyplot
import sys
import h5py
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import KFold

import json

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from keras.callbacks import ModelCheckpoint, TensorBoard

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
def get_model_v0(lx, ly, lz):
    model = Sequential()
    
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(lx, ly, lz)))
    model.add(MaxPool2D(2, 2))
    
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(43, activation='softmax'))
    
    plot_model(model, to_file='model_V0_plot.png', show_shapes=True, show_layer_names=True)
    
    return model
    
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
    
    plot_model(model, to_file='model_V2_plot.png', show_shapes=True, show_layer_names=True)
    
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
    
    plot_model(model, to_file='model_V3_plot.png', show_shapes=True, show_layer_names=True)
    
    return model

# parameters
enhanced_dir = './data/enhanced'
final_dir = './data/final'
data_result = './results'
tag_id = '{:06}'.format(random.randint(0,99999))

dataset_name = 'set-24x24-'
batch_size = 32
epochs = 5
scale = 1


# variables
quality_label = ['RGB', 'RGB-HE', 'L', 'L-LHE']
fn_models = [get_model_v0, get_model_v1, get_model_v2, get_model_v3]
results = [
    { 'accuracy': list(), 'std': list(), 'time': list() },
    { 'accuracy': list(), 'std': list(), 'time': list() },
    { 'accuracy': list(), 'std': list(), 'time': list() },
    { 'accuracy': list(), 'std': list(), 'time': list() }
]

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


# ***** Evalution of the Model using Kfold *****
def evaluate_model_kfold(dataX, dataY, fn_model, d_name, m_name, n_folds=3):
    scores, histories, times = list(), list(), list()
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    (n, lx, ly, lz) = dataX.shape
    
    for train_ix, test_ix in kfold.split(dataX):
        model = fn_model(lx, ly, lz)
        # Select rows for train and test
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        # compile model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # Callbacks
        log_dir = f'{data_result}/logs_{tag_id}/tb_{d_name}_{m_name}'
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        save_dir = f'{data_result}/models_{tag_id}/model_{d_name}_{m_name}.h5'
        bestmodel_callback = ModelCheckpoint(filepath=save_dir, verbose=0, monitor='accuracy', save_best_only=True)
        # Start the timer
        start_time = time.time()
        # fit model
        history = model.fit(trainX, trainY,
                            batch_size = batch_size,
                            epochs = epochs,
                            validation_data=(testX, testY),
                            verbose=0,
                            callbacks=[tensorboard_callback, bestmodel_callback])
        # End the timer
        end_time = time.time()
        # evaluate model
        _, accuracy = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (accuracy*100))
        # keep track
        times.append(end_time - start_time)
        scores.append(accuracy)
        # histories.append(history)
    return scores, times

def run_model():
    # ---- Logs and models dir
    os.makedirs(f'{data_result}/logs_{tag_id}',   mode=0o750, exist_ok=True)
    os.makedirs(f'{data_result}/models_{tag_id}', mode=0o750, exist_ok=True)
    
    # Evaluate with KFold
    for k in range(4):
        for quality in quality_label:
            x_train, y_train, x_test,y_test, x_meta, y_meta = read_dataset(enhanced_dir, f'{dataset_name}{quality}')
            scores, times = evaluate_model_kfold(x_train, y_train, fn_models[k], quality, f'v_{k}')
            results[k]['accuracy'].append(mean(scores)*100)
            results[k]['std'].append(std(scores)*100)
            results[k]['time'].append(mean(times))
            print('Model v%d: accuracy=%.3f , std=%.3f, time=%.3f' % (k, mean(scores)*100, std(scores)*100, mean(times)))
            with open('result.txt', 'w') as outfile:
                json.dump(results, outfile)
    

    # We save the result
    # Evaluate directly
    # evaluate_model(x_train, y_train, x_test, y_test, get_model_v0)

run_model()

