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

# Copyright (c) 2020, Marcus Dias[^1], Ailton Oliveira[^1], Diego gomes[^1], Aldebaro Klautau[^1]
#
# [^1]: Lasse, Universidade Federal do Pará,
#  Belém, Brazil. marcus.dias@itec.ufpa.br, ailton.pinto@itec.ufpa.br, diagomes@unifesspa.edu.br, aldebaro@ufpa.br
#  Support Email: Raymobtime@gmail.com

#ITU Challenge version

from __future__ import print_function

import keras
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adagrad
import numpy as np
from sklearn.preprocessing import minmax_scale
import keras.backend as K
import copy
from datetime import datetime
from keras.utils import plot_model
#import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import h5py

#For description about top-k, including the explanation on how they treat ties (which can be misleading
#if your classifier is outputting a lot of ties (e.g. all 0's will lead to high top-k)
#https://www.tensorflow.org/api_docs/python/tf/nn/in_top_k
def top_2_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)

def top_3_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_4_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=4)

#default top_k seems to be k=5

def top_10_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=10)

def top_20_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=20)

def top_30_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=30)

def top_40_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=40)

def top_50_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=50)

def top_100_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=100)

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

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
batch_size = 32
epochs = 400 #20000
thresholdBelowMax = 6

#fileNameIdentifier = 'obstacles_c2N_removedx25y25_nofloor_hist_inputs_sG3sL0.1'
fileNameIdentifier = 'obstacles'
f = open(fileNameIdentifier + '.txt','w')
#fileNameIdentifier = 'obstacles_c2N_removedx25y25_nofloor_hist_inputs_sG0sL0'
#inputsFileName = 'd:/gits/lasse/software/5gm-pcd-nn/matlab/' + fileNameIdentifier + '.mat'
#print("Reading dataset...", inputsFileName)
#arrayName = 'obstacleInputs'
#X = read_matlab_array_from_mat(inputsFileName, arrayName)
#before I was converting from mat to npz and reading an npz:
#input_cache_file = np.load(inputsFileName)
#X = input_cache_file['obstacleInputs'] #inputs

inputsFileName_lidar = 'lidar_input_user3.npz'
print("Reading dataset...", inputsFileName_lidar)
input_cache_file = np.load(inputsFileName_lidar)
X_lidar = input_cache_file['input_classification'] #inputs

inputsFileName_camera1 = 'image_matrix_camera1.npz'
print("Reading dataset...", inputsFileName_camera1)
input_cache_file_camera1 = np.load(inputsFileName_camera1)
X_image1 = input_cache_file_camera1['image_matrix'] #inputs

inputsFileName_camera2 = 'image_matrix_camera3.npz'
print("Reading dataset...", inputsFileName_camera2)
input_cache_file_camera2 = np.load(inputsFileName_camera2)
X_image2 = input_cache_file_camera2['image_matrix'] #inputs

inputsFileName_camera3 = 'image_matrix_camera3.npz'
print("Reading dataset...", inputsFileName_camera3)
input_cache_file_camera3 = np.load(inputsFileName_camera3)
X_image3 = input_cache_file_camera3['image_matrix'] #inputs

lidar_shape = X_lidar.shape

X = np.zeros([lidar_shape[0],lidar_shape[1],lidar_shape[2],lidar_shape[3]+3])

X[:,:,:,0:10] = X_lidar
X[:,:,:,10] = X_image1
X[:,:,:,11] = X_image2
X[:,:,:,12] = X_image3

#outputsFileName = 'sumoAndInsiteInfoValids.npz'
outputsFileName = 'beams_output_user3.npz'
print("Reading dataset...", outputsFileName)
output_cache_file = np.load(outputsFileName)
yMatrix = output_cache_file ['output_classification'] #inputs

yMatrix = np.abs(yMatrix)
yMatrix /= np.max(yMatrix)
yMatrixShape = yMatrix.shape
numClasses = yMatrix.shape[1] * yMatrix.shape[2]

y = yMatrix.reshape(yMatrix.shape[0], numClasses)

print('num Examples = ', X.shape[0])
print('Input share: Each example is a matrix = ', X.shape[1], X.shape[2], X.shape[3])
print('numClasses = ', numClasses)
#fraction to be used for training set
validationFraction = 0.2 #from 0 to 1

# split for train and for test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validationFraction, random_state=seed, shuffle=True)
print('num Examples train = ', X_train.shape[0])
print('num Examples test = ', X_test.shape[0])

#Need to do for each one, train and test, to be able to save test before zero'ing to use in Matlab
for i in range(len(y_train)):
    thisOutputs = y_train[i,:]
    logOut = 20*np.log10(thisOutputs + 1e-30)
    #indLog = np.argsort(logOut) #sort in ascending order
    minValue = np.amax(logOut) - thresholdBelowMax
    zeroedValueIndices = logOut < minValue
    thisOutputs[zeroedValueIndices]=0
    thisOutputs = thisOutputs / sum(thisOutputs)
    y_train[i] = thisOutputs
    #print(sum(y[i]))

#before zero'ing, save it
y_test_original = y_test

for i in range(len(y_test)):
    thisOutputs = y_test[i,:]
    logOut = 20*np.log10(thisOutputs + 1e-30)
    #indLog = np.argsort(logOut) #sort in ascending order
    minValue = np.amax(logOut) - thresholdBelowMax
    zeroedValueIndices = logOut < minValue
    thisOutputs[zeroedValueIndices]=0
    thisOutputs = thisOutputs / sum(thisOutputs)
    y_test[i] = thisOutputs

#Keras is requiring an extra dimension: I will add it with reshape
#X = X.reshape(X_train.shape[0], nrows, ncolumns, X_train.shape[3])
input_shape = [X.shape[1], X.shape[2], X.shape[3]]

print("Finished reading datasets")

# convert class vectors to binary class matrices. This is equivalent to using OneHotEncoder
#from sklearn.preprocessing import OneHotEncoder
#encoder = OneHotEncoder()
#y_train = encoder.fit_transform(y_train.reshape(-1, 1))
#y_train = keras.utils.to_categorical(y_train, 2)
#original_y_test = copy.deepcopy(y_test).astype(int)
#y_test = keras.utils.to_categorical(y_test, 2)

# declare model Convnet with two conv1D layers following by MaxPooling layer, and two dense layers
# Dropout layer consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting.

NN = 50
print('Running with NN = ', NN)
model = Sequential()

#model.add(Conv2D(24, kernel_size=(2, 2), activation='relu', padding='same', input_shape=(135,2)))
#model.add(MaxPooling1D(3))
#model.add(Conv2D(16, kernel_size=(2, 2), activation='relu', padding='same'))
#model.add(MaxPooling1D(3))
#model.add(Dropout(0.5))
#model.add(Dense(8, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Flatten())
#model.add(Dense(numClasses, activation='softmax'))

#model.summary()

#model.add(Dense(1,input_shape=input_shape, activation='relu'))
if True:
    dropProb=0.3
    model.add(Conv2D(NN, kernel_size=(13,13),
                        activation='relu',
                        #				 strides=[1,1],
                        padding="SAME",
                        input_shape=input_shape))
    model.add(Conv2D(NN, (11, 11), padding="SAME", activation='relu'))
    model.add(Conv2D(NN, (9, 9), padding="SAME", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Dropout(dropProb))
    model.add(Conv2D(2*NN, (7, 7), padding="SAME", activation='relu'))
    #model.add(Dropout(dropProb))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Conv2D(4*NN, (5, 5), padding="SAME", activation='relu'))
    model.add(Dropout(dropProb))
    model.add(Conv2D(NN, (3, 3), padding="SAME", activation='relu'))
    #model.add(Dropout(dropProb))
    model.add(Conv2D(1, (1, 1), padding="SAME", activation='relu'))
    #model.add(Dropout(dropProb))
    model.add(Flatten())
    model.add(Dense(numClasses, activation='softmax'))
else:
    dropProb=0.01
    model.add(Dense(NN,
                        activation='relu',
                        #				 strides=[1,1],
                        #				 padding="SAME",
                        input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(NN, activation='relu'))
    model.add(Dropout(dropProb))
    #model.add(Dense(NN, activation='relu'))
    #model.add(Dropout(dropProb))
    model.add(Dense(NN, activation='relu'))
    model.add(Dropout(dropProb))
    model.add(Dense(numClasses, activation='softmax'))

#model.add(Conv2D(20, kernel_size=(16, 16),
#                 activation='relu',
#				 strides=[1,1],
#				 padding="SAME",
#                 input_shape=input_shape))
#model.add(Conv2D(4, (6, 4), padding="SAME", activation='relu'))
#model.add(Conv2D(16, (10, 2), padding="SAME", activation='relu'))
#model.add(MaxPooling2D(pool_size=(4, 2)))
#model.add(Dropout(0.5))
#model.add(Flatten())
#model.add(Dense(2, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(numClasses, activation='softmax'))

model.summary()
try: #to catch CTRL+C #AK-TODO NOT WORKING!
    if False:
        model.compile(loss=keras.losses.categorical_crossentropy,
                        optimizer=keras.optimizers.Adadelta(),
                        metrics=[metrics.categorical_accuracy])
    else:
        model.compile(loss=keras.losses.categorical_crossentropy,
                        optimizer=keras.optimizers.Adadelta(),
                        metrics=[metrics.categorical_accuracy,
                                #top_2_accuracy,
                                #top_3_accuracy,
                                #top_4_accuracy,
                                metrics.top_k_categorical_accuracy,
                                top_10_accuracy,
                                #top_20_accuracy,
                                top_30_accuracy,
                                #top_40_accuracy,
                                top_50_accuracy,
                                top_100_accuracy
                                ])

    # compile model.
    #model.compile(loss='mean_squared_error',
    #              optimizer=Adagrad(),
    #              metrics=['accuracy','mae'])
    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        shuffle=True,
                        #validation_split=validationFraction)
                        validation_data=(X_test, y_test))
    #validation_data=(X_test, y_test))
except KeyboardInterrupt:
    print('WARNING: interrupted by user! But it is safe.') #AK-TODO not working

# print results
print(model.metrics_names)
#print('Test loss rmse:', np.sqrt(score[0]))
#print('Test accuracy:', score[1])
print(history.history)
f.write(str(history.history))
# val_acc = history.history['val_acc']
# acc = history.history['acc']
# f = open('classification_output.txt','w')
# f.write('validation_acc\n')
# f.write(str(val_acc))
# f.write('\ntrain_acc\n')
# f.write(str(acc))
# f.close()

#score = model.evaluate(X_test, y_test, verbose=1)
#y_pred = model.predict(X_test) #get the prediction for test set, with 1 for correct 'class' and 0 for others
#a = model.get_layer('dense_1').output
#model.outputs = a
#y_pred = model.predict_proba(X_test) #get the prediction for test set
y_pred = model.predict(X_test) #get the prediction for test set
#print(y_pred.shape)

outputFileName =  fileNameIdentifier + '_N' + str(NN) +'.hdf5'
fh5py = h5py.File(outputFileName, 'w')
fh5py['y_pred'] = y_pred
fh5py['y_test_original'] = y_test_original
fh5py.close()
print('==> Wrote file ' + outputFileName)

f.close()
