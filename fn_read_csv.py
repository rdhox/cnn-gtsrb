#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 15:44:54 2021

@author: renaud
"""
import os
import numpy as np
import pandas as pd
from skimage import io


def read_csv_dataset(file_name):
    path = os.path.dirname(file_name)
    name = os.path.basename(file_name)
    
    # Read csv file
    df = pd.read_csv(file_name, header=0)
    # get filename and class Ids
    filenames = df['Path'].to_list()
    y = df['ClassId'].to_list()
    x = []
    # Read images
    for filename in filenames:
        image = io.imread(f'{path}/{filename}')
        x.append(image)
    return np.array(x, dtype=object), np.array(y)