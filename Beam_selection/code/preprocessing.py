import argparse
from processImageData import processImageData
from processCoordinates import processCoordinates
from processLidarData import processLidarData
from processBeamsOutput import processBeamsOutput
from pathlib import Path

coordFileName = '../../Datasets/Raymobtime_s008/CoordVehiclesRxPerScene_s008.csv'
episodesForTrain = 1564

def checkDataDirectory(path):
    #passing from string to libpath's Path obj
    path = Path(path)
    if path.exists():
        print('Folder already exists')
    else:
        print('Creating the folder structure...\n')
        path.mkdir()
        path.joinpath('coord_input').mkdir()
        path.joinpath('image_input').mkdir()
        path.joinpath('lidar_input').mkdir()
        path.joinpath('beam_output').mkdir()
        
def main():
    parser = argparse.ArgumentParser(description='Configure the files before training the net.')
    parser.add_argument('dataset', help='Path to the Raymobtime dataset directory')
    parser.add_argument('data_folder', help='Where to place the processed data')
    args = parser.parse_args()
    
    checkDataDirectory(args.data_folder)

    limit = processCoordinates(args.data_folder, args.dataset, episodesForTrain)
    processImageData(args.data_folder, args.dataset, limit)
    processLidarData(args.data_folder, args.dataset, limit)
    processBeamsOutput(args.data_folder, args.dataset, limit)

if __name__ == "__main__":
    main()
