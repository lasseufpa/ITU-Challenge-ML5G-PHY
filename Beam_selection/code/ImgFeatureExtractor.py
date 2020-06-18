#Script context use	: This script uses Raymotime data (https://www.lasse.ufpa.br/raymobtime/) in the context of the UFPA - ITU Artificial Intelligence/Machine Learning in 5G Challenge (http://ai5gchallenge.ufpa.br/).
#Author       		: Ailton Oliveira, Aldebaro Klautau, Arthur Nascimento, Diego Gomes, Jamelly Ferreira, Walter Frazão
#Email          	: ml5gphy@gmail.com                                          
#License		: This script is distributed under "Public Domain" license.
###################################################################

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from scipy.signal import convolve2d
import numpy as np
from CSVHandler import CSVHandler

class ImgFeatureExtractor:
    
    def __init__(self,imgDim,color_space,resizeFac,dataDir,imgDataDir,inputDataDir,learnSet):
        
        self.color_space = color_space
        if self.color_space =='Gray':
            self.nCh = 1
        elif self.color_space in ['HSV', 'BGR']:
            self.nCh = 3
        self.resizeFac = resizeFac
        self.dimResize = (int(imgDim[0]/self.resizeFac),int(imgDim[1]/self.resizeFac))
        self.dataDir = dataDir
        self.imgDataDir = imgDataDir
        self.inputDataDir = inputDataDir
        self.learnSet = learnSet

    def color_space_cvt(self, imgURL):
        if self.color_space == 'Gray':
            imgTmp = cv2.imread(imgURL, cv2.IMREAD_GRAYSCALE) #Grayscale
        elif self.color_space == 'BGR':
            imgTmp = cv2.imread(imgURL, cv2.IMREAD_COLOR) #BGR
        elif self.color_space == 'HSV':
            imgTmp = cv2.imread(imgURL, cv2.IMREAD_COLOR)
            imgTmp = cv2.cvtColor(imgTmp, cv2.COLOR_BGR2HSV) #right way to convert to hsv
        else:
            print('Please enter a valid color space')
            #TO-DO - Implement this metods
            #imgTmp = cv2.imread(imgURL, cv2.COLOR_BGR2LAB) #LAB
            #imgTmp = cv2.imread(imgURL, cv2.COLOR_BGR2YCrCb)#YCrCb or lumem
        return imgTmp
        
    def resizeAndConcatenate(self,coordFileName, limit, use_high_pass_filter='False'):
        
        csvHand = CSVHandler()
        coordURL = self.dataDir + coordFileName + '.csv'
        nSamples, lastEpisode, epi_scen_list  = csvHand.getEpScenValbyRec(coordURL)                
        dimResize = self.dimResize
        if self.nCh > 1:
            inputs = np.zeros([nSamples,dimResize[0],dimResize[1]*3,self.nCh],dtype=np.uint8)
        else:
            inputs = np.zeros([nSamples,dimResize[0],dimResize[1]*3,1],dtype=np.uint8)        
        

        for samp in range(0,nSamples):
            for cam in range(1,4):
                epi_scen = epi_scen_list[samp]
                imgURL = self.dataDir+self.imgDataDir+'camera'+str(cam)+'/'+'{:0>1}'.format(epi_scen[0])+'.png'
                imgTmp = self.color_space_cvt(imgURL) #convert for the right color space
                #cv2.imshow("Resized image", imgTmp) #plot debug
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                imgRes = cv2.resize(imgTmp,(dimResize[1],dimResize[0]),interpolation = cv2.INTER_AREA)
                
                if self.nCh == 1:
                    if use_high_pass_filter == True:
                        highPassKernel = np.array([[0,-1/4,0], [-1/4,2,-1/4], [0,-1/4,0]])
                        imgResFilt = convolve2d(imgRes, highPassKernel, mode='same')
                        inputs[samp,:,dimResize[1]*(cam-1):dimResize[1]*cam,0] = imgResFilt
                    else:
                        inputs[samp,:,dimResize[1]*(cam-1):dimResize[1]*cam,0] = imgRes
                elif self.nCh == 3:
                    inputs[samp,:,dimResize[1]*(cam-1):dimResize[1]*cam,:] = imgRes                    
                        
            if(np.mod(samp+1,21)==0):
                print("Generated samples: "+str(samp))
        input_validation = inputs[limit:]
        input_train =  inputs[:limit]

        #np.savez(self.inputDataDir+'img_input_'+self.learnSet+'_tst'+'_1_'+str(self.resizeFac)+'.npz',inputs=inputs)
        np.savez(self.inputDataDir+'img_input_train_'+str(self.resizeFac)+'.npz',inputs=input_train)          
        np.savez(self.inputDataDir+'img_input_validation_'+str(self.resizeFac)+'.npz',inputs=input_validation)  
        return inputs
