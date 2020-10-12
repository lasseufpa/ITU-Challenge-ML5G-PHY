'''
This script receives a dataset as input and generates two csv files:
beam_train_pred.csv and beam_train_label.csv. The former presents the equivalent
channels for each combination of precoder and combiner and it also illustrates
the assumed ordering when comparing the predictions in beam_test_pred.csv with
the correct index in beam_test_label.txt. The latter saves the top-1 beam pairs.
'''
import numpy as np
import sys
import csv
import os
np.set_printoptions(threshold=sys.maxsize)

# Change here by the correct location in your system
datasetPath = '/home/ilan/Documents/Raymobtime_dataset/s009'
 
NpzFileName_train = \
    datasetPath+'/baseline_data/beam_output/beams_output_test.npz'

("Reading dataset...", NpzFileName_train)
beam_cache = np.load(NpzFileName_train)
beam = beam_cache['output_classification']
   
equivalentChannel_CsvFileName = \
    datasetPath+'/baseline_data/beam_output/beam_test_pred.csv'
trueBeamIndex_CsvFileName = \
    datasetPath+'/baseline_data/beam_output/beam_test_label.csv'

limit = len(beam) #total number of examples
with open(equivalentChannel_CsvFileName, "w", newline='') as f,\
    open(trueBeamIndex_CsvFileName, 'w') as g:
    writer = csv.writer(f, delimiter=',')            
    for i in range(0,limit,1): #go over all examples
        codebook = np.absolute(beam[i, :]) #read matrix
        Rx_size = codebook.shape[0] # 8 antenna elements
        Tx_size = codebook.shape[1] # 32 antenna elements
        new_line = np.zeros( (Rx_size*Tx_size,), np.float ) #initialize
        for tx in range(0,Tx_size,1):
            for rx in range(0,Rx_size,1): #inner loop goes over receiver
                new_line[tx * Rx_size + rx] = codebook[rx,tx] #impose ordering
        writer.writerow(list(new_line))
        g.write(str(np.argwhere( new_line == np.max(new_line))[0][0]) + '\n')
