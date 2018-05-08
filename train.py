#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:18:02 2018

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

train_net()
