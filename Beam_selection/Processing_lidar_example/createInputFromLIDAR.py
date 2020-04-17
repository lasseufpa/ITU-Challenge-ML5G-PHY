# Copyright (c) 2020, Ailton Oliveira[^1], Diego gomes[^1], Aldebaro Klautau[^1]
#
# [^1]: Lasse, Universidade Federal do Pará,
#  Belém, Brazil. ailton.pinto@itec.ufpa.br, diagomes@unifesspa.edu.br, aldebaro@ufpa.br
#  Support Email: Raymobtime@gmail.com

import os
import csv
import numpy as np
import sys

def getInfo(filename,inputPath):
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        numExamples = 0
        for row in reader:
            isValid = row['Val'] #V or I are the first element of the list thisLine
            valid_user = int(row['VehicleArrayID'])
            if isValid == 'V' and valid_user == user:
                numExamples = numExamples + 1
                #continue #skip invalid entries 
            
    run = 0
    while(1):
        npz_name = os.path.join(inputPath,'obstacles_e_'+str(run)+'.npz')
        if os.path.exists(npz_name):
            currentEpisodesInputs = np.load(npz_name)
            obstacles_matrix_array = currentEpisodesInputs['obstacles_matrix_array']
            return numExamples,obstacles_matrix_array.shape
        elif run > 200:
            print('You may indicated a wrong folder. Check your input path')
            break
        else:
            run += 1


#config variables
filename = 'CoordVehiclesRxPerScene_s008.csv' #information file 
argv = sys.argv
print('Usage: python createInputFromLIDAR.py inputPath npz_name')
inputPath = argv[1] #input (PCD) path/name
npz_name = argv[2] #output path/name
user = 3 #desired user
numExamples,input_shape = getInfo(filename,inputPath) #extract useful information
with open(filename) as csvfile:
    reader = csv.DictReader(csvfile)   
    allInputs = np.zeros((numExamples,input_shape[1],input_shape[2],input_shape[3]), np.int8)
    id_count = 0
    alreadyInMemoryEpisode = -1
    
    if os.path.exists(npz_name):
        print( npz_name, 'already exists')
        exit(1)
    for row in reader:
        episodeNum = int(row['EpisodeID'])
        isValid = row['Val'] #V or I are the first element of the list thisLine
        if isValid == 'I':
            continue #skip invalid entries
        if episodeNum != alreadyInMemoryEpisode: #just read if a new episode
            print('Reading Episode '+str(episodeNum)+' ...')
            currentEpisodesInputs = np.load(os.path.join(inputPath,'obstacles_e_'+str(episodeNum)+'.npz'))
            obstacles_matrix_array = currentEpisodesInputs['obstacles_matrix_array']
            alreadyInMemoryEpisode = episodeNum #update for other iterations
        
        if (int(row['VehicleArrayID']) == user):
            s = int(row['SceneID']) - 1 #get scene number
            r = int(row['VehicleArrayID']) #get receiver number
            allInputs[id_count] = obstacles_matrix_array[r]
            id_count = id_count + 1

    np.savez_compressed(npz_name, input_classification=allInputs)
    print('Saved file ', npz_name)
