#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 19:51:00 2021

@author: renaud
"""

from matplotlib import pyplot
import json

x_label = ['RGB', 'RGB-HE', 'L', 'L-LHE']
colors = ['blue', 'red', 'green', "pink"]
positions = [1, -1.5, 1, -1.5]

# params
batch_size = 32
epochs = 5
nfold = 3

with open('result.txt') as json_file:
    results = json.load(json_file)

fig = pyplot.figure()
ax = fig.add_subplot(111)
pyplot.title('Result 24x24 img dataset GTSRB, small quantity')
for k in range(4):
    pyplot.plot(results[k]['accuracy'], color=colors[k], label=f'v{k}', marker='o')
    for i, v in enumerate(results[k]['accuracy']):
        ax.annotate("%.2f" % v, xy=(i,v), xytext=(-7,positions[k]*10), color=colors[k], textcoords='offset points')
pyplot.xlabel("Dataset image treatment")
pyplot.ylabel("Model Accuracy")
pyplot.xticks([0,1,2,3],x_label)
pyplot.legend(loc="center right")
pyplot.show()