# Copyright (c) 2020, Ailton Oliveira[^1], Diego gomes[^1], Aldebaro Klautau[^1]
#
# [^1]: Lasse, Universidade Federal do Pará,
#  Belém, Brazil. ailton.pinto@itec.ufpa.br, diagomes@unifesspa.edu.br, aldebaro@ufpa.br
#  Support Email: Raymobtime@gmail.com

import os
import csv
import numpy as np
import sys

def getInfo(filename, user):
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        numExamples = 0
        episode_i = []
        scene_i = []
        line_i = []
        valid_line = 0
        for row in reader:
            isValid = row['Val'] #V or I are the first element of the list thisLine
            valid_user = int(row['VehicleArrayID'])
            if isValid == 'V': #filter the valid channels 
                if valid_user == user: #filter the desired user
                    numExamples = numExamples + 1
                    episode_i.append(row['EpisodeID'])
                    scene_i.append(row['SceneID'])
                    line_i.append(valid_line)
                    #continue #skip invalid entries
                valid_line += 1 
    return numExamples, episode_i, scene_i, line_i

#config variables
argv = sys.argv
filename = argv[1] #information file 
user = int(argv[2]) #user you want to track
npz_name = 'beams_output_user'+ str(user) + '.npz' #npz output name

numExamples,episode_list, scene_list, line_list = getInfo(filename, user) #extract useful information


BeamFileName = './beams_output.npz' 
print("Reading dataset...", BeamFileName)
beam_cache_file = np.load(BeamFileName)
X_Beam = beam_cache_file['output_classification']
shape = X_Beam.shape
print(shape)

filtered_beams = np.zeros((numExamples,shape[1],shape[2]), np.complex128)
filtered_id = 0
for id in line_list:
    filtered_beams[filtered_id,:,:] = X_Beam[id,:,:]
    filtered_id += 1

np.savez(npz_name, output_classification=filtered_beams) #save best beams for desired user
print('Saved file ',npz_name)