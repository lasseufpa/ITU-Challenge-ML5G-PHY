#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 15:07:51 2020

Description: This code reads sumoOutputFiles from the runs of a
Raymobtime simulation to load receivers positions that are used
as input of the NN.

SUMO output files can be found at:
    datasetHome/
    |____s008_Rosslyn_10MobileRx_60GHz_2086episodes_1scenes_Ts1s_InSite3.2/ (extracted zip file)
         |____runX/ (e.g., run000000, run00100)
              |____sumoOutputInfoFileName.txt

It also reads rays information from the HDF5 files provided with
the dataset to calculate the MIMO channel, the equivalent MIMO
chanell with each possible precoding/combining, then, finally,
the best Tx-Rx beam pairs 

HDF5 files can be found at:
    datasetHome/
    |____raw_data/
         |____ray_tracing_data_s008_carrier60GHz/ (extracted zip file)
              |____rosslyn_mobile_60GHz_ts0.1s_V_Lidar_e*.hdf5

@author: Ilan Correa <ilan@ufpa.br>
"""

## Imports
import os
import h5py
import numpy as np
import csv
import sys

sys.path.append('../Beam_selection/code/')
import mimo_channels

############################################################
################Function definitions
############################################################

def getUsersPositions(sumoFile, numEpisodes, numScenes):
    '''This funtion returns the position and angle of each user in a set of
    Raybmotime runs. It return as a data structure with dimensions
    numEpisodes x numScenes x numRx x numParameters. numRx is fixed
    as 10, as, currently, it is the maximum number of receivers in the
    runs. numParameters is 3, and are: xinsite, yinsite, angle
    This function uses the third column ('veh') of the
    sumoOuptutFile as ID for the receiver.'''
    
    def getUsersIds(sumoFile):
        '''Helper function. Everytime the loop below reaches a new episode,
        we need to update the receivers' IDs adopted in that episode.'''
        
        ids = ['','','','','','','','','','']
        
        with open(sumoFile, 'r') as f:
            for i in range(10):
                f.seek(0)
                currentLine = f.readline() # Throw away header
                currentLine = f.readline()
                while currentLine != '':
                    fields = currentLine.split(',')
                    if fields[2] == str(i):
                        ids[i] = fields[3]
                        break
                    currentLine = f.readline()
        return ids
    
    print('## Reading Receivers Positions')
    
    positions = np.nan * np.ones((numEpisodes, numScenes, 10, 3))
    runPtr = 0
    currentEpisode = -1
    while True: # Go over all files/runs
        currentFile = sumoFile.replace('*', '%05d'%runPtr)
        runPtr += 1
        
        if not os.path.isfile(currentFile):
            break # Finish if gone through all files
            
        with open(currentFile, 'r') as f:
            currentLine = f.readline() # Throw away header
            currentLine = f.readline()
            fields = currentLine.split(',')
            
            # Check if started a new episode
            if currentEpisode != int(fields[0]):
                currentEpisode = int(fields[0])
                ids = getUsersIds(currentFile)
            
            currentScene = int(fields[1])
            
            for i in range(10):
                f.seek(0)
                currentLine = f.readline() # Throw away header
                currentLine = f.readline()
                
                while currentLine != '':
                    fields = currentLine.split(',')
                    if fields[3] == ids[i]:
                        positions[currentEpisode, currentScene, i, 0] = fields[6] # xinsite
                        positions[currentEpisode, currentScene, i, 1] = fields[7] # yinsite
                        positions[currentEpisode, currentScene, i, 2] = fields[12] # angle
                        break
                    currentLine = f.readline()
    print('\tDone')
    return positions

def checkNumEpisodes(hdf5file):
    num = 1
    while True:
        if not os.path.isfile(hdf5File.replace('*', str(num-1))):
            num-=1
            break
        num+=1
    return num

############################################################
################ Script configurations
############################################################
number_Tx_antennas = 32 #ULA
number_Rx_antennas = 8  #ULA

## TODO: Please update paths below to reflect the location of
#        these files in your system
hdf5File = '/home/ilan/Raymobtime_dataset/s008/raw_data/ray_tracing_data_s008_carrier60GHz/'+\
    'rosslyn_mobile_60GHz_ts0.1s_V_Lidar_e*.hdf5'
sumoOutputFile = '/home/ilan/Raymobtime_dataset/s008/auxiliary_simulation_files/'+\
    's008_Rosslyn_10MobileRx_60GHz_2086episodes_1scenes_Ts1s_InSite3.2/run*/sumoOutputInfoFileName.txt'
outputFile = 's008'
    
'''Read one the the episodes to get information of number of scenes per
episode (numScenes), number of receivers (numReceivers) and number of
rays saved from the ray-tracing simulations (numRays). Note that, the
number of saved rays is configured  as up to numRays, but in a given
scene, the number of rays actually saved may be different.'''
b = h5py.File(hdf5File.replace('*', str(0)), 'r')
allEpisodeData = b.get('allEpisodeData')
numScenes = allEpisodeData.shape[0]
numReceivers = allEpisodeData.shape[1]
numRays = allEpisodeData.shape[2]

# Get total number of episodes
numEpisodes = checkNumEpisodes(hdf5File)

# Get users positions corresponding to iEpisode x iScene 
positions_array = getUsersPositions(sumoOutputFile, numEpisodes, numScenes)

# Pre-allocate memory
best_ray_array = -1 * np.ones((numEpisodes, numScenes, numReceivers,2))
dataset_rays = np.zeros((numEpisodes*numScenes*numRays, 2))
dataset_positions = np.zeros((numEpisodes*numScenes*numRays, 3))

#  Loop auxiliary variables
dataset_ptr = 0
totalOfUsers = 0
totalOfValidUsers = 0
totalOfInvalidUsers = 0
for iEpisode in range(numEpisodes):
    currentFile = hdf5File.replace('*', str(iEpisode))
    
    print("Episode # ", iEpisode)
    
    b = h5py.File(currentFile, 'r')
    allEpisodeData = b.get('allEpisodeData')
    
    for iScene in range(numScenes):
        
        for iRx  in range(numReceivers):
            totalOfUsers += 1
            
            insiteData = allEpisodeData[iScene, iRx, :, :]
            
            for iRay in range(numRays):
                if np.isnan(insiteData[iRay, 0]):
                    break
            
            if iRay == 0:
                totalOfInvalidUsers += 1
                continue
            
            totalOfValidUsers += 1
            
            gain_in_dB = insiteData[0:iRay, 0]
            timeOfArrival = insiteData[0:iRay, 1]
            AoD_el = insiteData[0:iRay, 2]
            AoD_az = insiteData[0:iRay, 3]
            AoA_el = insiteData[0:iRay, 4]
            AoA_az = insiteData[0:iRay, 5]
            isLOSperRay = insiteData[0:iRay, 6]
            pathPhases = insiteData[0:iRay, 7]
            if insiteData.shape[1] == 8:
                RxAngle = positions_array[iEpisode, iScene, iRx, 2]
            else:
                RxAngle = insiteData[0:iRay, 8][0]
            RxAngle = RxAngle + 90.0
            if RxAngle > 360.0:
                RxAngle = RxAngle - 360.0
            #Correct ULA with Rx orientation
            AoA_az = - RxAngle + AoA_az #angle_new = - delta_axis + angle_wi;
        
            mimoChannel = mimo_channels.getNarrowBandULAMIMOChannel(\
                AoD_az, AoA_az, gain_in_dB, number_Tx_antennas, number_Rx_antennas)
                
            equivalentChannel = \
                mimo_channels.getDFTOperatedChannel(mimoChannel, number_Tx_antennas, number_Rx_antennas)
            equivalentChannelMagnitude = np.abs(equivalentChannel)
            
            best_ray = np.argwhere( equivalentChannelMagnitude == np.max(equivalentChannelMagnitude) )
            best_ray_array[iEpisode, iScene, iRx, :] = best_ray
            
            dataset_rays[dataset_ptr, :] = best_ray
            dataset_positions[dataset_ptr, :] = positions_array[iEpisode, iScene, iRx, :]
            dataset_ptr += 1

dataset_positions = dataset_positions[range(dataset_ptr), :]
dataset_rays = dataset_rays[range(dataset_ptr), :]

print('Saving dataset')
np.savez(outputFile + '_nTx%d'%number_Tx_antennas + '_nRx%d'%number_Rx_antennas + '_beams_output.npz', output_classification=dataset_rays)
np.savez(outputFile + '_nTx%d'%number_Tx_antennas + '_nRx%d'%number_Rx_antennas + '_coordinates_input.npz', input_classification=dataset_positions)