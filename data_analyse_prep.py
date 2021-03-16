#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 14:09:24 2021

@author: renaud
"""
import os
import h5py
from sklearn.utils import shuffle

# imported functions from other files
from fn_dataset_statistics import draw_stats_datasets
from fn_read_csv import read_csv_dataset
from fn_images_enhancement import images_enhancement

# parameters for short tests
# scale = 0.1
# output_dir = './data/enhanced'
# parameters for total dataset generation
scale = 1
output_dir = './data/final'

# Read the dataset
(x_train, y_train) = read_csv_dataset('./data/GTSRB/origine/Train.csv')
(x_test, y_test) = read_csv_dataset('./data/GTSRB/origine/Test.csv')
(x_meta, y_meta) = read_csv_dataset('./data/GTSRB/origine/Meta.csv')

# Shuffle the train data
x_train, y_train = shuffle(x_train, y_train)

# Sort the Meta
combined = list(zip(x_meta, y_meta))
combined.sort(key=lambda x:x[1])
x_meta, y_meta = zip(*combined)

# draw the statistics of the dataset
# draw_stats_datasets(x_train, y_train, x_test, y_test)

# Generate enhanced datasets

def save_h5_dataset(x_train, y_train, x_test, y_test, x_meta,y_meta, filename):
        
    # ---- Create h5 file
    with h5py.File(filename, "w") as f:
        f.create_dataset("x_train", data=x_train)
        f.create_dataset("y_train", data=y_train)
        f.create_dataset("x_test",  data=x_test)
        f.create_dataset("y_test",  data=y_test)
        f.create_dataset("x_meta",  data=x_meta)
        f.create_dataset("y_meta",  data=y_meta)
        
    # ---- done
    size=os.path.getsize(filename)/(1024*1024)
    print('Dataset : {:24s}  shape : {:22s} size : {:6.1f} Mo   (saved)'.format(filename, str(x_train.shape),size))


n_train = int(len(x_train)*scale)
n_test = int(len(x_test)*scale)

print(f'Scale is : {scale}')
print(f'x_train length is : {n_train}')
print(f'x_test  length is : {n_test}')
print(f'output dir is     : {output_dir}\n')


for s in [24, 48]:
    for m in ['RGB', 'RGB-HE', 'L', 'L-LHE']:
        filename = f'{output_dir}/set-{s}x{s}-{m}.h5'
        # ---- Enhancement
        #      Note : x_train is a numpy array of python objects (images with <> sizes)
        #             but images_enhancement() return a real array of float64 numpy (images with same size)
        #             so, we can save it in nice h5 files
        #
        x_train_new = images_enhancement(x_train[:n_train], width=s, height=s, mode=m)
        x_test_new = images_enhancement(x_test[:n_test], width=s, height=s, mode=m)
        x_meta_new = images_enhancement(x_meta, width=s, height=s, mode='RGB')
        #save
        save_h5_dataset(x_train_new, y_train[:n_train], x_test_new, y_test[:n_test], x_meta_new, y_meta, filename)
        












    



