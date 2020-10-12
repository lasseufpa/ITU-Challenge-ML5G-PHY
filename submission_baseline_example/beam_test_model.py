"""
Created on 2020-10-11

Description: This file evalutes the model and pre-trained weights. It loads the
model and weights, loads the data, and generates predictions with NN. The 
predictions are saved in a CSV file to be evaluated.

@author: Ilan Correa <ilan@ufpa.br>
"""

import argparse
import numpy as np
import sys
import os

from tensorflow.keras.models import model_from_json
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import metrics

parser = argparse.ArgumentParser(description='Configure the files before testing the net.')
parser.add_argument('data_folder', help='Location of the data directory', type=str)
parser.add_argument('--input', nargs='*', default=['coord'], 
choices = ['img', 'coord', 'lidar'],
help='Which data to use as input. Select from: img, lidar or coord.')
parser.add_argument('-p','--plots', 
help='Use this parametter if you want to see the accuracy and loss plots',
action='store_true')
args = parser.parse_args()

# This example uses data in baseline_data folder
args.data_folder = os.path.join(args.data_folder, 'baseline_data')

def top_10_accuracy(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=10)

def top_50_accuracy(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=50)

###############################################################################
# Load trained model
###############################################################################
opt = Adam()

with open('my_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)

model.compile(loss=categorical_crossentropy,
                    optimizer=opt,
                    metrics=[metrics.categorical_accuracy,
                            metrics.top_k_categorical_accuracy])

model.load_weights('my_model_weights.h5')

###############################################################################
# Load Data
###############################################################################
data_dir = args.data_folder+'/'
if 'coord' in args.input: 
    ###############################################################################
    # Coordinate configuration
    coord_input_file = data_dir+'coord_input/coord_test.npz'
    coord_cache_file = np.load(coord_input_file)
    X_coord = coord_cache_file['coordinates']

    coord_input_shape = X_coord.shape

if 'img' in args.input:
    ###############################################################################
    # Image configuration
    resizeFac = 20 # Resize Factor
    
    img_input_file = data_dir+'image_input/img_input_test_'+str(resizeFac)+'.npz'
    print("Reading dataset... ",img_input_file)
    img_cache_file = np.load(img_input_file)
    X_img = img_cache_file['inputs']

    img_input_shape = X_img.shape

if 'lidar' in args.input:
    ###############################################################################
    # LIDAR configuration
    #train
    lidar_input_file = data_dir+'lidar_input/lidar_test.npz'
    print("Reading dataset... ",lidar_input_file)
    lidar_cache_file = np.load(lidar_input_file)
    X_lidar = lidar_cache_file['input']

    lidar_input_shape = X_lidar.shape

###############################################################################
# Perform the evaluation
###############################################################################
multimodal = False if len(args.input) == 1 else len(args.input)

if multimodal == 2:
    if 'coord' in args.input and 'lidar' in args.input:
        Y_test = model.predict([X_coord,X_lidar])

    elif 'coord' in args.input and 'img' in args.input:
        Y_test = model.predict([X_coord,X_img])
    else:
        Y_test = model.predict([X_lidar,X_img])
elif multimodal == 3:
    Y_test = model.predict([X_lidar,X_img,X_coord])
else:
    if 'coord' in args.input:
        Y_test = model.predict(X_coord)
    elif 'img' in args.input:
        Y_test = model.predict(X_img)
    else:
        Y_test = model.predict(X_lidar)
        
        
###############################################################################
# Save the result for evaluation
###############################################################################
np.savetxt('beam_test_pred.csv', Y_test, delimiter=',')