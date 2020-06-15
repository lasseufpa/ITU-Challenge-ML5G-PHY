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

def processLidarData(data_folder, dataset, limit):
    csvHand = CSVHandler()
    print('Generating LIDAR ...')
    lidarDataDir = dataset+'/'+ 'lidar_data_s008/'
    inputDataDir = data_folder+'/lidar_input/'
    coordFileName = 'CoordVehiclesRxPerScene_s008'
    coordURL = dataset +'/'+coordFileName + '.csv'

    if not(os.path.exists(inputDataDir)):
        os.mkdir(inputDataDir)
        print("Directory '% s' created" % inputDataDir)

    nSamples, lastEpisode, epi_scen  = csvHand.getEpScenValbyRec(coordURL)
    obstacles_matrix_array_lidar = np.ones((nSamples,20,200,10), np.int8)
    lidar_inputs_train = []
    lidar_inputs_test = []
    with open(coordURL) as csvfile:
        reader = csv.DictReader(csvfile)
        id_count = 0
        alreadyInMemoryEpisode = -1
        for row in reader:
            episodeNum = int(row['EpisodeID'])
            #if (episodeNum < numEpisodeStart) | (episodeNum > numEpisodeEnd):
            #    continue #skip episodes out of the interval
            isValid = row['Val'] #V or I are the first element of the list thisLine
            if isValid == 'I':
                continue #skip invalid entries
            if episodeNum != alreadyInMemoryEpisode: #just read if a new episode
                print('Reading Episode '+str(episodeNum)+' ...')
                currentEpisodesInputs = np.load(os.path.join(lidarDataDir,'obstacles_e_'+str(episodeNum)+'.npz'))
                obstacles_matrix_array = currentEpisodesInputs['obstacles_matrix_array']
                alreadyInMemoryEpisode = episodeNum #update for other iterations
            r = int(row['VehicleArrayID']) #get receiver number
            obstacles_matrix_array_lidar[id_count] = obstacles_matrix_array[r]
            id_count = id_count + 1

    lidar_inputs_test = obstacles_matrix_array_lidar[limit:]
    lidar_inputs_train = obstacles_matrix_array_lidar[:limit]

    #train
    np.savez(inputDataDir+'lidar_train.npz',input=lidar_inputs_train)
    #test
    np.savez(inputDataDir+'lidar_validation.npz',input=lidar_inputs_test)