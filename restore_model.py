#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:22:09 2018

@author: dmitriy
"""

from __future__ import print_function
import os.path as osp
import numpy as np
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint

np.random.seed(1337)

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda

from keras.optimizers import RMSprop


from helper_functions import *

model = build_two_branches()
model.load_weights('my_model_total_stage_two_2.h5')

#model = load_model('my_model_total_stage_two_2.h5',  custom_objects={'contrastive_loss_2': contrastive_loss_2})
images_opt, images_sar, targets, discr_dist, discr_vectors, samples_weight =  read_data_test()
result = model.predict([images_opt, images_sar])


pos = []
neg = []

for i in range(0, len(result)):
    if(targets[i] == 0):
        neg.append(result[i])
    if(targets[i] == 1):
        pos.append(result[i])


pos = np.array(pos)
neg = np.array(neg)

np.save("pos", pos)
np.save("neg", neg)
