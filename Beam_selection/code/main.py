#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 15:23:32 2020

@author: diego, ailton, marcus
"""

'''Trains a deep NN for choosing top-K beams
Adapted by AK: Aug 7, 2018
See
https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/
and
https://stackoverflow.com/questions/45642077/do-i-need-to-use-one-hot-encoding-if-my-output-variable-is-binary
See for explanation about convnet and filters:
https://datascience.stackexchange.com/questions/16463/what-is-are-the-default-filters-used-by-keras-convolution2d
and
http://cs231n.github.io/convolutional-networks/
'''
import csv
import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras.models import model_from_json,Model
from tensorflow.keras.layers import Dense,concatenate
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adadelta,Adam
from sklearn.model_selection import train_test_split
from ModelHandler import ModelHandler
import numpy as np
import argparse
from enum import Enum


###############################################################################
# Support functions
###############################################################################

#For description about top-k, including the explanation on how they treat ties (which can be misleading
#if your classifier is outputting a lot of ties (e.g. all 0's will lead to high top-k)
#https://www.tensorflow.org/api_docs/python/tf/nn/in_top_k
def top_10_accuracy(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=10)

def top_50_accuracy(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=50)

def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind

def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return (rows, cols)

def beamsLogScale(y,thresholdBelowMax):
        y_shape = y.shape
        
        for i in range(0,y_shape[0]):            
            thisOutputs = y[i,:]
            logOut = 20*np.log10(thisOutputs + 1e-30)
            minValue = np.amax(logOut) - thresholdBelowMax
            zeroedValueIndices = logOut < minValue
            thisOutputs[zeroedValueIndices]=0
            thisOutputs = thisOutputs / sum(thisOutputs)
            y[i,:] = thisOutputs
        
        return y

def getBeamOutput(output_file):
    
    thresholdBelowMax = 6
    
    print("Reading dataset...", output_file)
    output_cache_file = np.load(output_file)
    yMatrix = output_cache_file['output_classification']
    
    yMatrix = np.abs(yMatrix)
    yMatrix /= np.max(yMatrix)
    yMatrixShape = yMatrix.shape
    num_classes = yMatrix.shape[1] * yMatrix.shape[2]
    
    y = yMatrix.reshape(yMatrix.shape[0],num_classes)
    y = beamsLogScale(y,thresholdBelowMax)
    
    return y,num_classes


parser = argparse.ArgumentParser(description='Configure the files before training the net.')
parser.add_argument('data_folder', help='Location of the data directory', type=str)
#TODO: limit the number of input to 3
parser.add_argument('--input', nargs='*', default=['coord'], 
choices = ['img', 'coord', 'lidar'],
help='Which data to use as input. Select from: img, lidar or coord.') 
args = parser.parse_args()

###############################################################################
# Data configuration
###############################################################################
tf.device('/device:GPU:0')
data_dir = args.data_folder+'/'
tgtRec = 3

###############################################################################
# Coordinate configuration
coord_input_file = data_dir+'coord/coord_'+str(tgtRec)+'.npz'
coord_cache_file = np.load(coord_input_file)
X_coord = coord_cache_file['coordinates']

coord_input_shape = X_coord.shape

###############################################################################
# Image configuration
resizeFac = 20 # Resize Factor
nCh = 1 # The number of channels of the image
imgDim = (360,640) # Image dimensions
method = 1

img_input_file = data_dir+'image/img_input_train_tst_'+str(method)+'_'+str(resizeFac)+'.npz'
print("Reading dataset... ",img_input_file)
img_cache_file = np.load(img_input_file)
X_img = img_cache_file['inputs']

img_input_shape = X_img.shape

###############################################################################
# LIDAR configuration
lidar_input_file = data_dir+'lidar/lidar_'+str(tgtRec)+'.npz'
print("Reading dataset... ",lidar_input_file)
lidar_cache_file = np.load(lidar_input_file)
X_lidar = lidar_cache_file['inputs']

lidar_input_shape = X_lidar.shape

###############################################################################
# Output configuration
output_file = data_dir+'beam_output/beams_output_user'+str(tgtRec)+'.npz'
y,num_classes = getBeamOutput(output_file)

##############################################################################
# Model configuration
##############################################################################

#multimodal
multimodal = False if len(args.input) == 1 else len(args.input)

num_epochs = 400
batch_size = 32
validationFraction = 0.2 #from 0 to 1
modelHand = ModelHandler()
opt = Adam()

if 'coord' in args.input:
    coord_model = modelHand.createArchitecture('coord_mlp',num_classes,coord_input_shape[1],'complete')
if 'img' in args.input:
    if nCh==1:   
        img_model = modelHand.createArchitecture('light_image',num_classes,[img_input_shape[1],img_input_shape[2],1],'complete')
    else:
        img_model = modelHand.createArchitecture('light_image',num_classes,[img_input_shape[1],img_input_shape[2],img_input_shape[3]],'complete')
if 'lidar' in args.input:
    lidar_model = modelHand.createArchitecture('lidar_marcus',num_classes,[lidar_input_shape[1],lidar_input_shape[2],lidar_input_shape[3]],'complete')

if multimodal:
    if 'coord' and 'lidar' in args.input:
        combined_model = concatenate([coord_model.output,lidar_model.output])
        z = Dense(num_classes,activation="relu")(combined_model)
        model = Model(inputs=[coord_model.input,lidar_model.input],outputs=z)
        model.compile(loss=categorical_crossentropy,
                    optimizer=opt,
                    metrics=[metrics.categorical_accuracy,
                            metrics.top_k_categorical_accuracy,
                            top_50_accuracy])
        model.summary()
        model.fit([X_coord,X_lidar],y,epochs=num_epochs,batch_size=batch_size)

    elif 'coord' and 'img' in args.input:
        combined_model = concatenate([coord_model.output,img_model.output])
        z = Dense(num_classes,activation="relu")(combined_model)
        model = Model(inputs=[coord_model.input,img_model.input],outputs=z)
        model.compile(loss=categorical_crossentropy,
                    optimizer=opt,
                    metrics=[metrics.categorical_accuracy,
                            metrics.top_k_categorical_accuracy,
                            top_50_accuracy])
        model.summary()
        model.fit([X_coord,X_img],y,epochs=num_epochs,batch_size=batch_size)
    
    else:
        combined_model = concatenate([lidar_model.output,img_model.output])
        z = Dense(num_classes,activation="relu")(combined_model)
        model = Model(inputs=[lidar_model.input,img_model.input],outputs=z)
        model.compile(loss=categorical_crossentropy,
                    optimizer=opt,
                    metrics=[metrics.categorical_accuracy,
                            metrics.top_k_categorical_accuracy,
                            top_50_accuracy])
        model.summary()
        model.fit([X_lidar,X_img],y,epochs=num_epochs,batch_size=batch_size)

else:
    if 'coord' in args.input:
        input_model = modelHand.createArchitecture('coord_mlp',num_classes,coord_input_shape[1],'complete')
        z = Dense(num_classes,activation="relu")(input_model.output)
        model = Model(inputs=input_model.input, outputs=z)
        model.compile(loss=categorical_crossentropy,
                            optimizer=opt,
                            metrics=[metrics.categorical_accuracy,
                                    metrics.top_k_categorical_accuracy,
                                    top_50_accuracy])
        model.summary()
        model.fit(X_coord,y,epochs=num_epochs,batch_size=batch_size)

    elif 'img' in args.input:
        input_model = img_model  
        input_model.compile(loss=categorical_crossentropy,
                    optimizer=opt,
                    metrics=[metrics.categorical_accuracy,
                            metrics.top_k_categorical_accuracy,
                            top_50_accuracy])
        input_model.summary()
        input_model.fit(X_img,y,epochs=num_epochs,batch_size=batch_size)

    else:
        input_model = modelHand.createArchitecture('lidar_marcus',num_classes,[lidar_input_shape[1],lidar_input_shape[2],lidar_input_shape[3]],'complete')
        z = Dense(num_classes,activation="relu")(input_model.output)
        model = Model(inputs=input_model.input, outputs=z)
        model.compile(loss=categorical_crossentropy,
                    optimizer=opt,
                    metrics=[metrics.categorical_accuracy,
                            metrics.top_k_categorical_accuracy,
                            top_50_accuracy])
        model.summary()
        model.fit(X_lidar,y,epochs=num_epochs,batch_size=batch_size)
    