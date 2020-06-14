#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 20:45:28 2020

@author: Ailton Oliveira[^1], Diego gomes[^1], Aldebaro Klautau[^1]
"""
import csv

class CSVHandler:
    
    def getEpScenValbyRec(self, filename):
    
        with open(filename) as csvfile:
            reader = csv.DictReader(csvfile)
            numExamples = 0
            epi_scen = []
                        
            for row in reader:
                isValid = row['Val'] #V or I are the first element of the list thisLine
                #valid_user = int(row['VehicleArrayID'])
                if isValid == 'V': #check if the channel is valid
                    numExamples = numExamples + 1
                    epi_scen.append([int(row['EpisodeID']),int(row['SceneID'])])
                lastEpisode = int(row['EpisodeID'])
                                                                
        return numExamples, lastEpisode, epi_scen
    
    def getCoord(self,filename, limitEp):

        with open(filename) as csvfile:
            reader = csv.DictReader(csvfile)
            coordinates_train = []
            coordinates_test = []      
            
            for row in reader:
                isValid = row['Val'] #V or I are the first element of the list thisLine
                if isValid == 'V': #check if the channel is valid
                    if int(row['EpisodeID']) <= limitEp:
                        coordinates_train.append([float(row['x']),float(row['y'])])
                    if int(row['EpisodeID']) > limitEp:
                        coordinates_test.append([float(row['x']),float(row['y'])])

        return coordinates_train, coordinates_test
    
    