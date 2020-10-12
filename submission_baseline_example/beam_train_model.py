"""
Created on 2020-10-11

Description: This file uses the example in main.py, which creates the NN model
and trains it. Then, the model and weights are saved.

@author: Ilan Correa <ilan@ufpa.br>
"""

import argparse
import sys
import common

parser = argparse.ArgumentParser(description='Configure the files before training the net.')
parser.add_argument('data_folder', help='Location of the data directory', type=str)
parser.add_argument('--input', nargs='*', default=['coord'], 
choices = ['img', 'coord', 'lidar'],
help='Which data to use as input. Select from: img, lidar or coord.')
parser.add_argument('-p','--plots', 
help='Use this parametter if you want to see the accuracy and loss plots',
action='store_true')
args = parser.parse_args()

common.parser = parser

# This example uses another example previously provided by the organization team
# of the challenge. The module main creates and trains the model
sys.path.append('../Beam_selection/code/')
import main as my_ml_impl

# After training the model, its parameters are saved
my_ml_impl.model.save_weights('my_model_weights.h5', save_format='h5') 
model_json = my_ml_impl.model.to_json()
with open('my_model.json', "w") as json_file:
    json_file.write(model_json)


