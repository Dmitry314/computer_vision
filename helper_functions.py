#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 09:43:32 2018

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


img_rows, img_cols = 64, 64
input_shape = (img_rows, img_cols, 1)
dim_of_feature_vector = 128

def numpy_read(filename):
    filedir, basename = osp.split(filename)
    basename, _ext = osp.splitext(basename)

    parts = basename.split('_')
    shape = parts[-1]
    shape = tuple(map(int, shape.split('x')))
    dtype = np.dtype('float32')
    return np.memmap(
        filename,
        mode='readonly',
        shape=shape,
        dtype=dtype)

def read_data(number, test_interface = False):
    if(number != 2):
        images_opt = numpy_read('../ws/batches/first_' + str(number) +'_65536x64x64x1.raw')
        images_sar = numpy_read('../ws/batches/second_' + str(number) +'_65536x64x64x1.raw')
        targets = np.memmap(
            '../ws/batches/targets_' + str(number) + '.raw', 
            mode='readonly', 
            shape=(len(images_opt),), 
            dtype='float32')
    else:
        images_opt = numpy_read('../ws/batches/first_' + str(number) +'_33761x64x64x1.raw')
        images_sar = numpy_read('../ws/batches/second_' + str(number) +'_33761x64x64x1.raw')
        targets = np.memmap(
            '../ws/batches/targets_' + str(number) + '.raw', 
            mode='readonly', 
            shape=(len(images_opt),), 
            dtype='float32')
        
    
    discr_dist = []
    discr_vectors = []
    class_weights = [0, 0]
    for i in targets:
        if(i == 0):
            discr_dist.append(0)
            discr_vectors.append([1, 0]) #first class mean that photos is different
            class_weights[0] += 1
        if(i == 1):
            class_weights[1] += 1
            discr_dist.append(1)
            discr_vectors.append([0, 1])
    
    class_weights[0] = float(class_weights[0]) / len(targets)
    class_weights[1] = float(class_weights[1]) / len(targets)
    
    samples_weight = []
    for i in range(0, len(targets)):
        if(targets[i] == 0):
            samples_weight.append(1)
        else:
            samples_weight.append(float(class_weights[0]) / class_weights[1])
    
    samples_weight = np.array(samples_weight)
    discr_dist = np.array(discr_dist)
    discr_vectors = np.array(discr_vectors)
    assert(discr_dist.shape[0] == targets.shape[0])
 
    if(test_interface):
        return images_opt[:5000], images_sar[:5000], targets[:5000], discr_dist[:5000], discr_vectors[:5000], samples_weight[:5000]
    else:
        return images_opt, images_sar, targets, discr_dist, discr_vectors, samples_weight



def read_data_test(test_interface = False):

    images_opt = numpy_read('../ws/validation_batches/first_0_8822x64x64x1.raw')
    images_sar = numpy_read('../ws/validation_batches/second_0_8822x64x64x1.raw')
    targets = np.memmap(
        '../ws/validation_batches/targets_0' +  '.raw', 
        mode='readonly', 
        shape=(len(images_opt),), 
        dtype='float32')
  
    
    discr_dist = []
    discr_vectors = []
    class_weights = [0, 0]
    for i in targets:
        if(i == 0):
            discr_dist.append(0)
            discr_vectors.append([1, 0]) #first class mean that photos is different
            class_weights[0] += 1
        if(i == 1):
            class_weights[1] += 1
            discr_dist.append(1)
            discr_vectors.append([0, 1])
    
    class_weights[0] = float(class_weights[0]) / len(targets)
    class_weights[1] = float(class_weights[1]) / len(targets)
    
    samples_weight = []
    for i in range(0, len(targets)):
        if(targets[i] == 0):
            samples_weight.append(1)
        else:
            samples_weight.append(float(class_weights[0]) / class_weights[1])
    
    samples_weight = np.array(samples_weight)
    discr_dist = np.array(discr_dist)
    discr_vectors = np.array(discr_vectors)
    assert(discr_dist.shape[0] == targets.shape[0])
 
    if(test_interface):
        return images_opt[:5000], images_sar[:5000], targets[:5000], discr_dist[:5000], discr_vectors[:5000], samples_weight[:5000]
    else:
        return images_opt, images_sar, targets, discr_dist, discr_vectors, samples_weight


def my_train_test_split():
    pass




def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred))


def contrastive_loss_2(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def build_one_branch_nn():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='tanh',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (3, 3), activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(256, (3, 3), activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Conv2D(128, (3,3), activation = 'relu'))
    #model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='tanh'))
    #model.add(Dropout(0.2))
    model.add(Dense(dim_of_feature_vector, activation='sigmoid')) #relu???
    
    return model
    

def build_simple_nn():
    model = Sequential()
    model.add(Conv2D(4, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    
    model.add(Dense(dim_of_feature_vector, activation='softmax')) #relu???
    
    return model
    


def build_two_branches():
    print("Qudratic loss!!")
    base_network = build_one_branch_nn()
    
    
    input_a = Input(shape=(img_rows, img_cols, 1))
    input_b = Input(shape=(img_rows, img_cols, 1))    
    
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    #distance = keras.layers.concatenate([processed_a, processed_b])
    #distance = keras.layers.Dense(1)(distance)
    
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    model = Model(input=[input_a, input_b], output=distance)
        
    return model



def build_two_branches_smooth():
    base_network = build_simple_nn()
    
    
    input_a = Input(shape=(img_rows, img_cols, 1))
    input_b = Input(shape=(img_rows, img_cols, 1))    
    
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    distance = keras.layers.subtract([processed_a, processed_b])
    distance = keras.layers.Dense(128, activation = 'relu')(distance)
    #distance = keras.layers.Dense(64)(distance)
    #distance = keras.layers.Dense(32)(distance)
    distance = keras.layers.Dense(2)(distance)
    #distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    model = Model(input=[input_a, input_b], output=distance)
        
    return model

    
from keras.utils import plot_model

def train_net():
    
    file_path="weights_base_best3.hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        
    early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
    callbacks_list = [checkpoint, early] #early
 
    rms = RMSprop(lr=0.001, decay = 0.01)
    sgd = keras.optimizers.SGD(lr=0.05, momentum=0.0, decay=0.0, nesterov=False)
    model = build_two_branches()
    
    
    
    
 
    model.compile(loss = contrastive_loss_2, optimizer='sgd')
    #plot_model(model, to_file='model.png')
    images_opt, images_sar, targets, discr_dist, discr_vectors, samples_weight =  read_data(0)
    model.fit([images_opt, images_sar], discr_dist, batch_size = 10, sample_weight = samples_weight, epochs = 5, validation_split=0.1, callbacks=callbacks_list)
    print("ended training on dataset 0ne")
    model.save('my_model_total_stage_one_1.h5')
    
    images_opt, images_sar, targets, discr_dist, discr_vectors, samples_weight =  read_data(1)
    
    model.fit([images_opt, images_sar], discr_dist, batch_size = 10, sample_weight = samples_weight, epochs = 5, validation_split=0.1, callbacks=callbacks_list)
    print("ended training on dataset two")
    
    images_opt, images_sar, targets, discr_dist, discr_vectors, samples_weight =  read_data(2)
    
    model.fit([images_opt, images_sar], discr_dist, batch_size = 10, sample_weight = samples_weight, epochs = 5, validation_split=0.1, callbacks=callbacks_list)
    print("ended training on dataset three")
    
    
    
    
    model.save('my_model_total_stage_two_2.h5')


def FPR_TPR(targets, result_num):
    num_of_plus = 0
    num_of_minus = 0
    
    true_class = 0
    false_class = 0
    for i in range(0, len(targets)):
        if(targets[i] == 1):
            num_of_plus += 1
        else:
            num_of_minus += 1
    
    for i in range(0, len(result_num)):
        if(targets[i] == 1 and result_num[i] == 1):
            true_class += 1
        
        if(targets[i] == 0 and result_num[i] == 1):
            false_class += 1
            
            
    return float(true_class)/num_of_plus, float(false_class)/num_of_minus
        
    
    
 
    

def get_auc():

    model = build_two_branches()
    model.load_weights('my_model.h5')
    images_opt, images_sar, targets, discr_dist, discr_vectors, samples_weight =  read_data(1)
    
    result = model.predict([images_opt, images_sar])
    
    minim = np.min(result)
    maxim = np.max(result)
    
    step = float(maxim - minim) / 20
    auc = []
    
    z = minim
    curr = 0
    res = 0
    while(z < maxim):
        result_num = []
        for i in result:
            if(i < z ):
                result_num.append(1)
            else:
                result_num.append(0)
            
        fpr_tpr = FPR_TPR(targets, result_num)
        auc.append([])
        auc[curr].append((z, fpr_tpr[0], fpr_tpr[1]))
        
        res +=  fpr_tpr[0] * fpr_tpr[1] * step
        
        
        z+= step
        curr += 1
    
    thefile = open('auc.txt', 'w')
    print(auc)
    
    
    print("**********************")
    print(res)
    
from keras.models import load_model    
def save_res():
    

    model = load_model('my_model_total_stage_two.h5')
    images_opt, images_sar, targets, discr_dist, discr_vectors, samples_weight =  read_data(2)
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
    
'''
def train_the_same():
    print("training on dataset where the size of positive examples is equal the size of negative examples")
    rms = RMSprop(lr=0.001, decay = 0.)
    sgd = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=True)
    model = build_two_branches()
    model.compile(loss=contrastive_loss, optimizer='sgd')
    #plot_model(model, to_file='model.png')
    images_opt, images_sar, targets, discr_dist, discr_vectors =  read_data(0)
    images_opt_1 = []
    images_sar_1 = []
    discr_dist_1 = []
    discr_vectors_1 = []
    
    
    for i in range(0, len(discr_dist)):
        if(discr_dist[i] == 0):
            images_opt_1.append(images_opt[i])
            images_sar_1.append(images_sar[i])
            discr_dist_1.append(discr_dist[i])
            discr_vectors_1.append(discr_vectors[i])
            images_opt_1.append(images_opt[i + 1])
            images_sar_1.append(images_sar[i + 1])
            discr_dist_1.append(discr_dist[i + 1])
            discr_vectors_1.append(discr_vectors[i + 1])
            
    images_opt_1 = np.array(images_opt_1)
    images_sar_1 = np.array(images_sar_1)
    discr_dist_1 = np.array(discr_dist_1)
    discr_vectors_1 = np.array(discr_vectors_1)
 
    model.fit([images_opt_1, images_sar_1], discr_dist_1, batch_size = 10, epochs = 10)

    model.save('my_model_quadratic_dist.hd5')
    
#train_the_same()


    res = model.predict([images_opt, images_sar])
    
    acc = 0
    for i in range(0, len(res)):
        if(res[i][0] > res[i][1] and discr_vectors[i][0] >discr_vectors[i][1]):
            acc += 1
        
        if(res[i][1] > res[i][0] and discr_vectors[i][1] >discr_vectors[i][0]):
            acc += 1
    print(float(acc) / len(res))


def test_number_two(model):
    images_opt, images_sar, targets, discr_dist, discr_vectors =  read_data()
    images_opt_1 = []
    images_sar_1 = []
    discr_dist_1 = []
    discr_vectors_1 = []
    
    
    for i in range(0, len(discr_dist)):
        if(discr_dist[i] == 0):
            images_opt_1.append(images_opt[i])
            images_sar_1.append(images_sar[i])
            discr_dist_1.append(discr_dist[i])
            discr_vectors_1.append(discr_vectors[i])
    
    images_opt_1 = np.array(images_opt_1)
    images_sar_1 = np.array(images_sar_1)
    discr_dist_1 = np.array(discr_dist_1)
    
    res = model.predict([images_opt_1, images_sar_1])
    
    acc = 0
    for i in range(0, len(res)):
        if(res[i][0] > res[i][1] and discr_vectors_1[i][0] >discr_vectors_1[i][1]):
            acc += 1
        
        if(res[i][1] > res[i][0] and discr_vectors_1[i][1] >discr_vectors_1[i][0]):
            acc += 1
    print(float(acc) / len(res))
    
    
def get_diff_metrics():
    pass


def tune_network():
    pass
    

def see_at_output(model):
    for i in range(1000):
        
        print(model.predict([images_opt[i:i+1], images_sar[i:i+1]]), discr_dist[i])


    z = model.predict([images_opt_1, images_sar_1])
    ones = []
    zeros = []
    for i in range( len(z)):
        if(discr_dist[i] == 1):
            ones.append(z[i])
        else:
            zeros.append(z[i])
            
    ones = pd.DataFrame(ones)
    zeros = pd.DataFrame(zeros)
    
    
    ones.plot.hist()
    zeros.plot.hist()            
'''
