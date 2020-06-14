#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 19:28:10 2020

@author: diego
"""

import csv
from ImgFeatureExtractor import ImgFeatureExtractor
import os

def processImageData(data_folder, dataset, limit):
    learnSet = 'train' #'train' or 'val'
    resizeFac = 20
    color_space = 'Gray' #(BGR, HSV, Gray, LAB, YCrCb) LAB and YCrCb not implemented
    imgDim = [360,640] #TO-DO - Not leave this in "HARD CODE"
    dimResize = (int(imgDim[0]/resizeFac),int(imgDim[1]/resizeFac))
    dataDir = dataset+'/'
    imgDataDir = 'image_data_s008/'
    inputDataDir = data_folder+'/image_input/'
    coordFileName = 'CoordVehiclesRxPerScene_s008'
    imgFeatExt = ImgFeatureExtractor(imgDim,color_space,resizeFac,dataDir,imgDataDir,inputDataDir,learnSet)
    __ = imgFeatExt.resizeAndConcatenate(coordFileName, limit, use_high_pass_filter=False)
