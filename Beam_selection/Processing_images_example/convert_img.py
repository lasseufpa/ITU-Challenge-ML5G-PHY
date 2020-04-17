# Copyright (c) 2020, Ailton Oliveira[^1], Diego gomes[^1], Aldebaro Klautau[^1]
#
# [^1]: Lasse, Universidade Federal do Pará,
#  Belém, Brazil. ailton.pinto@itec.ufpa.br, diagomes@unifesspa.edu.br, aldebaro@ufpa.br
#  Support Email: Raymobtime@gmail.com

import cv2
import numpy as np
import os
import csv


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
            if isValid == 'V':
                if valid_user == user:
                    numExamples = numExamples + 1
                    episode_i.append(int(row['EpisodeID']))
                    scene_i.append(int(row['SceneID']))
                    line_i.append(valid_line)
                    #continue #skip invalid entries
                valid_line += 1
    return numExamples, episode_i, scene_i, line_i

working_directory = os.path.dirname(os.path.realpath(__file__)) 


image_path = os.path.join(working_directory, 'images')
number_of_cameras = 3
user = 3 #user you want to track
filename = 'CoordVehiclesRxPerScene_s008.csv'


numExamples,episode_list, scene_list, line_list = getInfo(filename, user)
for camera in range(1,number_of_cameras+1, 1):
    npz_name = 'image_matrix_camera' + str(camera) + '.npz'
    storage = 0
    camera_path = os.path.join(image_path,'camera{}'.format(camera))
    images_list = os.listdir(camera_path)
    print(len(images_list))
    image_matrix_storage = np.zeros((numExamples,20, 200)) #expect image size is 90x160
    images_list.sort()

    for image in images_list:    
        image_id = int(image.strip('.png'))
        if image_id in episode_list:

            img = cv2.imread(os.path.join(camera_path, image), cv2.IMREAD_GRAYSCALE) #Grayscale
            #img = cv2.imread('./0000.png', cv2.IMREAD_COLOR) #RGB
            #img = cv2.imread('./0000.png', cv2.IMREAD_UNCHANGED) 
            
            print('Scene Frame Dimensions : ',img.shape)
            
            scale_percent = 25 # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (200, 20)
            # resize image
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            
            print('Resized Dimensions : ',resized.shape)
            '''cv2.imshow("Resized image", resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()'''
            image_matrix_storage[storage, :,:] = resized
            storage =+ 1
    print(image_matrix_storage.shape)
    np.savez(npz_name, image_matrix = image_matrix_storage)
    print('Saved file ', npz_name)

        

