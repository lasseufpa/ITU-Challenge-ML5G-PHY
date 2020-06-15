#Script context use	: This script uses Raymotime data (https://www.lasse.ufpa.br/raymobtime/) in the context of the UFPA - ITU Artificial Intelligence/Machine Learning in 5G Challenge (http://ai5gchallenge.ufpa.br/).
#Author       		: Ailton Oliveira, Aldebaro Klautau, Arthur Nascimento, Diego Gomes, Jamelly Ferreira, Walter Frazão
#Email          	: ml5gphy@gmail.com                                          
#License		: This script is distributed under "Public Domain" license.
###################################################################

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
from ImgFeatureExtractor import ImgFeatureExtractor
import os

def processImageData(data_folder, dataset, limit):
    learnSet = 'train' #'train' or 'val'
    resizeFac = 20
    color_space = 'Gray' #(BGR, HSV, Gray, LAB, YCrCb) LAB and YCrCb not implemented
    imgDim = [960,540] #TO-DO - Not leave this in "HARD CODE"
    dimResize = (int(imgDim[0]/resizeFac),int(imgDim[1]/resizeFac))
    dataDir = dataset+'/'
    imgDataDir = 'image_data_s008/'
    inputDataDir = data_folder+'/image_input/'
    coordFileName = 'CoordVehiclesRxPerScene_s008'
    imgFeatExt = ImgFeatureExtractor(imgDim,color_space,resizeFac,dataDir,imgDataDir,inputDataDir,learnSet)
    __ = imgFeatExt.resizeAndConcatenate(coordFileName, limit, use_high_pass_filter=True)
