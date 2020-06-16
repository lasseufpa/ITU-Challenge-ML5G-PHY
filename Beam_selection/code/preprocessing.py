###################################################################
#Script context use	: This script uses Raymotime data (https://www.lasse.ufpa.br/raymobtime/) in the context of the UFPA - ITU Artificial Intelligence/Machine Learning in 5G Challenge (http://ai5gchallenge.ufpa.br/).
#Author       		: Ailton Oliveira, Aldebaro Klautau, Arthur Nascimento, Diego Gomes, Jamelly Ferreira, Walter Frazão
#Email          	: ml5gphy@gmail.com                                          
#License		: This script is distributed under "Public Domain" license.
###################################################################

import argparse
from processImageData import processImageData
from processCoordinates import processCoordinates
from processLidarData import processLidarData
from processBeamsOutput import processBeamsOutput
from pathlib import Path

        
def main():
    parser = argparse.ArgumentParser(description='Configure the files before training the net.')
    parser.add_argument('dataset', help='Path to the Raymobtime dataset directory')
    parser.add_argument('data_folder', help='Where to place the processed data')
    args = parser.parse_args()
    
    checkDataDirectory(args.data_folder)

    limit = processCoordinates(args.data_folder, args.dataset)
    processImageData(args.data_folder, args.dataset, limit)
    processLidarData(args.data_folder, args.dataset, limit)
    processBeamsOutput(args.data_folder, args.dataset, limit)


def checkDataDirectory(path):
    #passing from string to libpath's Path obj
    path = Path(path)
    if path.exists():
        print('Folder already exists')
    else:
        print('Creating the folder structure...\n')
        path.mkdir()
    if (path.joinpath('coord_input')).exists():
        print('Folder already exists')
    else:
        print('Creating the folder structure...\n')
        path.joinpath('coord_input').mkdir()
    if (path.joinpath('image_input')).exists():
        print('Folder already exists')
    else:
        print('Creating the folder structure...\n')
        path.joinpath('image_input').mkdir()
    if (path.joinpath('lidar_input')).exists():
        print('Folder already exists')
    else:
        print('Creating the folder structure...\n')
        path.joinpath('lidar_input').mkdir()
    if (path.joinpath('beam_output')).exists():
        print('Folder already exists')
    else:
        print('Creating the folder structure...\n')
        path.joinpath('beam_output').mkdir()

if __name__ == "__main__":
    main()
