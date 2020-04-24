import pandas as pd
import numpy as np
import h5py
import os

from mimo_channels_data_generator import RandomChannelMimoDataGenerator

# fix random seed for reproducibility
seed = 1
np.random.seed(seed)

#num_samples = [10, 20, 50, 60, 100, 150, 200, 220, 240, 256]
num_samples = [256]
files = ["rosslyn60Ghz.mat"]

short_files = ["60GHz"]

SNRs = [0]

# common parameters
Nt = 8  # num of Rx antennas, will be larger than Nt for uplink massive MIMO
Nr = 8  # num of Tx antennas
min_randomized_snr_db = -1
max_randomized_snr_db = 1
batch_size = 5  # numExamplesWithFixedChannel
method = "manual"
num_training = 1600
num_testing = 800
numExamplesWithFixedChannel = 1

index = 0

print("Extracting dataset...")
for file_name in files:
    
    for samples in num_samples:
        numSamplesPerExample = samples
        numSamplesPerFixedChannel = (
            numExamplesWithFixedChannel * numSamplesPerExample
        )  # 

        training_generator = RandomChannelMimoDataGenerator(
            batch_size=batch_size,
            Nr=Nr,
            Nt=Nt,
            # num_clusters=num_clusters,
            numSamplesPerFixedChannel=numSamplesPerFixedChannel,
            # numSamplesPerExample=numSamplesPerExample, SNRdB=SNRdB,
            numSamplesPerExample=numSamplesPerExample,
            # method='random')
            method=method,
            file=file_name
        )

        training_generator.randomize_SNR = True
        training_generator.min_randomized_snr_db = min_randomized_snr_db
        training_generator.max_randomized_snr_db = max_randomized_snr_db

        inputs, outputs = training_generator.get_examples(num_training)

        hf = h5py.File('train.hdf5', 'w')
        hf.create_dataset('inputs', data=inputs)
        hf.create_dataset('outputs', data=outputs)
        hf.close()
        

        for SNRdb in SNRs:
        
            training_generator.randomize_SNR = False
            training_generator.SNRdB = SNRdb
            #training_generator.min_randomized_snr_db = min_randomized_snr_db
            #training_generator.max_randomized_snr_db = max_randomized_snr_db

            inputs, outputs = training_generator.get_examples(num_testing)

            hf = h5py.File('test.hdf5', 'w')
            hf.create_dataset('inputs', data=inputs)
            hf.create_dataset('outputs', data=outputs)
            hf.close()

    index += 1
    
print("Extracting done...")



