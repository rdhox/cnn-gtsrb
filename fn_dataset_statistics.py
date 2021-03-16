#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 15:37:58 2021

@author: renaud
"""

import matplotlib.pyplot as plt


def draw_stats_datasets(x_train, y_train, x_test, y_test):
    # Statistics about the dataset
    train_size  = []
    train_ratio = []
    train_lx    = []
    train_ly    = []
    
    test_size   = []
    test_ratio  = []
    test_lx     = []
    test_ly     = []
    
    for image in x_train:
        (lx,ly,lz) = image.shape
        train_size.append(lx*ly/1024)
        train_ratio.append(lx/ly)
        train_lx.append(lx)
        train_ly.append(ly)
    
    for image in x_test:
        (lx,ly,lz) = image.shape
        test_size.append(lx*ly/1024)
        test_ratio.append(lx/ly)
        test_lx.append(lx)
        test_ly.append(ly)
    
    # ------ Global stuff
    print("x_train shape : ",x_train.shape)
    print("y_train shape : ",y_train.shape)
    print("x_test  shape : ",x_test.shape)
    print("y_test  shape : ",y_test.shape)
    
    # ------ Statistics / sizes
    plt.figure(figsize=(16,6))
    plt.hist([train_size,test_size], bins=100)
    plt.gca().set(title='Sizes in Kpixels - Train=[{:5.2f}, {:5.2f}]'.format(min(train_size),max(train_size)), 
                  ylabel='Population', xlim=[0,30])
    plt.legend(['Train','Test'])
    plt.show()
    
    # ------ Statistics / ratio lx/ly
    plt.figure(figsize=(16,6))
    plt.hist([train_ratio,test_ratio], bins=100)
    plt.gca().set(title='Ratio lx/ly - Train=[{:5.2f}, {:5.2f}]'.format(min(train_ratio),max(train_ratio)), 
                  ylabel='Population', xlim=[0.8,1.2])
    plt.legend(['Train','Test'])
    plt.show()
    
    # ------ Statistics / lx
    plt.figure(figsize=(16,6))
    plt.hist([train_lx,test_lx], bins=100)
    plt.gca().set(title='Images lx - Train=[{:5.2f}, {:5.2f}]'.format(min(train_lx),max(train_lx)), 
                  ylabel='Population', xlim=[20,150])
    plt.legend(['Train','Test'])
    plt.show()
    
    # ------ Statistics / ly
    plt.figure(figsize=(16,6))
    plt.hist([train_ly,test_ly], bins=100)
    plt.gca().set(title='Images ly - Train=[{:5.2f}, {:5.2f}]'.format(min(train_ly),max(train_ly)), 
                  ylabel='Population', xlim=[20,150])
    plt.legend(['Train','Test'])
    plt.show()
    
    # ------ Statistics / classId
    plt.figure(figsize=(16,6))
    plt.hist([y_train,y_test], bins=43)
    plt.gca().set(title='ClassesId', ylabel='Population', xlim=[0,43])
    plt.legend(['Train','Test'])
    plt.show()
