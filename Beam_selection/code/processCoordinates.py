#Script context use	: This script uses Raymotime data (https://www.lasse.ufpa.br/raymobtime/) in the context of the UFPA - ITU Artificial Intelligence/Machine Learning in 5G Challenge (http://ai5gchallenge.ufpa.br/).
#Author       		: Ailton Oliveira, Aldebaro Klautau, Arthur Nascimento, Diego Gomes, Jamelly Ferreira, Walter Frazão
#Email          	: ml5gphy@gmail.com                                          
#License		: This script is distributed under "Public Domain" license.
###################################################################

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import numpy as np
from CSVHandler import CSVHandler
import os
from numpy import load

def processCoordinates(data_folder, dataset):
    print('Generating Beams ...')
    csvHand = CSVHandler()
    
    inputDataDir = data_folder+'/coord_input/'
    coordFileName = 'CoordVehiclesRxPerScene_s008'
    coordURL = dataset +'/'+coordFileName + '.csv'
    
    coordinates_train, coordinates_test = csvHand.getCoord(coordURL, 1564)
    
    train_channels = len(coordinates_train)

    #train
    np.savez(inputDataDir+'coord_train'+'.npz',coordinates=coordinates_train)
    #test
    np.savez(inputDataDir+'coord_validation'+'.npz',coordinates=coordinates_test)
    
    print ('Coord npz files saved!')

    return train_channels