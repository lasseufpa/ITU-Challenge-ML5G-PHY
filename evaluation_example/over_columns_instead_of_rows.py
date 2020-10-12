'''
Code to illustrate how to switch the ordering scheme when 
converting a matrix into a vector.
Note that reshape in Matlab is column-wise by default,
while in Python Numpy it is row-wise by default.

The context is that we have Tx=32 vectors at the transmitter codebook
and Rx=8 vectors at the receiver codebook. Let us say the ML model
output its prediction as a matrix X of dimension 8 x 32, with rows
and columns indicating the indices of the Rx and Tx codebooks, respectively.

When indicating the best pair of indices (i, j) as a single index k,
the convention we are adopting is such that in the example above,
X is parsed column-wise. This code assumes the vectors with 256 values
were obtained by using a row-wise reshape of X, and converts it to
the values that would be obtained with a column-wise reshape of X.

This can be used in case the ML model generated beam_test_pred.csv
using a row-wise reshape, while the correct index to be compared with
beam_test_label.txt is obtained with a column-wise reshape.
'''
import numpy as np
import sys
import csv

np.set_printoptions(threshold=sys.maxsize)

def reorder(data, num_rows, num_columns):
    '''
    Reorder a vector obtained from a matrix: read row-wise and write column-wise.
    '''
    original_vector  = np.asarray(data, dtype = np.float)
    #read row-wise
    original_matrix = np.reshape(original_vector, (num_rows, num_columns))
    #write column-wise
    new_vector = np.reshape(original_matrix, num_rows*num_columns, 'F')
    return new_vector

#Important: inform the correct number of rows and columns of your
#matrix before it was converted into a vector
num_rows = 8
num_columns = 32

#file names:
input_file_name = 'row.csv'
output_file_name = 'column.csv'

with open(output_file_name, "w", newline='') as output_f:
    writer = csv.writer(output_f, delimiter=',')            
    with open(input_file_name,'r') as input_f:
        data_iter = csv.reader(input_f,
                            delimiter = ',')
        for data in data_iter:
            if len(data) != (num_rows * num_columns):
                raise Exception('Number of elements in this row is not the product num_rows * num_columns')
            new_vector = reorder(data, num_rows, num_columns)
            writer.writerow(list(new_vector))
