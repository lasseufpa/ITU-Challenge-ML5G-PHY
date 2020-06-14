#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 18:46:16 2020

@author: diego
"""
import csv
import numpy as np
from CSVHandler import CSVHandler
import os
from numpy import load

def processCoordinates(data_folder, dataset, episodesForTrain):
    print('Generating Beams ...')
    csvHand = CSVHandler()
    
    inputDataDir = data_folder+'/coord_input/'
    coordFileName = 'CoordVehiclesRxPerScene_s008'
    coordURL = dataset +'/'+coordFileName + '.csv'
    
    coordinates_train, coordinates_test = csvHand.getCoord(coordURL, episodesForTrain)
    
    train_channels = len(coordinates_train)

    #train
    np.savez(inputDataDir+'coord_train'+'.npz',coordinates=coordinates_train)
    #test
    np.savez(inputDataDir+'coord_validation'+'.npz',coordinates=coordinates_test)
    
    print ('Coord npz files saved!')

    return train_channels