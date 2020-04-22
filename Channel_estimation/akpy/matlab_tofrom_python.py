import numpy as np
import scipy.io as spio
import h5py
'''
Code to interface Python and Matlab. It allows to read arrays (one
needs to know the name of the array):
Matlab to Python: read .mat (version 6) files into Python 
Python to Matlab: write HDF5 files to be read by Matlab.
Example:
    The array to be read by Python was saved in a Matlab .mat file with commands as:
    save -v6 'test.mat' arrayName1 arrayName2 arrayName3
    or
    save(thisFileName,'myArray','-v6')
'''

def convert_npz_to_hdf5(npzFileName):
    '''
    Convert npz into hdf5. Note hdf5 does not support compression and files can be much larger
    than a compressed npz. But the hdf5 can be conveniently read by Matlab.
    See: http://christopherlovell.co.uk/blog/2016/04/27/h5py-intro.html
    %To read hdf5 file in Matlab as follows:    
    %https://stackoverflow.com/questions/21624653/python-created-hdf5-dataset-transposed-in-matlab
    %do not forget the / as prefix onf the second argument below:
    allEpisodeData=h5read(fileName,'/allEpisodeData');
    %need to permute dimensions of 4-d tensor
    allEpisodeData = permute(allEpisodeData,ndims(allEpisodeData):-1:1);
    :param npzFileName:
    :return:
    '''
    npz_cache_file = np.load(npzFileName)
    allKeys = npz_cache_file.keys()
    numKeys = len(allKeys)
    outputHDF5FilaName = npzFileName.replace('.npz','.hdf5')
    h5pyFile = h5py.File(outputHDF5FilaName, 'w')
    for thisKey in allKeys:
        npArray = npz_cache_file[thisKey] #get array with name thisKey
        #Matched ok with result of test_convert_npz_to_hdf5.m
        # squeeze(obstacles_matrix_array(10,4,196,570,1:15))'
        #that was: 0   1   1   1   1   1   1   1   1   1   0   0   0   0   0
        #print(npArray[9,3,195,569,0:15])
        #output the same result
        h5pyFile[thisKey] = npArray #store in hdf5
    h5pyFile.close()
    print('==> Wrote file ', outputHDF5FilaName, ' with keys: ', allKeys)

def read_matrix_from_file(fileName, numRows, numColumns):
    '''
    Read in Python a real-valued matrix written in Matlab as binary file with doubles (64 bits).
    We deal here with transposing the matrix given that Matlab organizes the numbers
    ordering along columns while Python, C, etc. go along the rows.

    It assumes the file was written in Matlab with function writeFloatMatrixToBinaryFile.m below:

    function writeFloatMatrixToBinaryFile(matrix, fileName)

    fileID = fopen(fileName,'wb');
    if fileID == -1
        error(['Could not open file ' fileName])
    end
    [N,M]=size(matrix);
    fwrite(fileID, N, 'double');
    fwrite(fileID, M, 'double');
    fwrite(fileID, matrix, 'double');
    fclose(fileID);

    :param fileName: file name
    :param numRows: number of rows
    :param numColumns: number of columns
    :return: matrix of dimension numRows x numColumns
    '''
    matrix = np.fromfile(fileName,dtype=np.double)
    #there is a small header: first two values are the matrix dimension
    if matrix[0] != numRows: #check consistency
        print(matrix[0], ' != ', numRows)
        exit(-1)
    if matrix[1] != numColumns: #check consistency
        print(matrix[1], ' != ', numColumns)
        exit(-1)
    matrix = matrix[2:] #skip the header (two first values)
    #recall that it was written along columns in Matlab, so will need to transpose
    #in order to adjust to the way Python reads it in
    matrix = np.reshape(matrix,(numColumns, numRows)) #read with "transposed" shape
    matrix = matrix.T #now effectively transpose elements
    return matrix

def read_matlab_array_from_mat(fileName, arrayName):
    '''
    Read an array saved in Matlab with something like:
    save -v6 'test.mat' arrayName1 arrayName2 arrayName3
    or
    save(thisFileName,'myArray','-v6')
    Note that this gives support to multidimensional complex-valued arrays.
    :param fileName: file name (test.mat in this example)
    :param arrayName: name of array of interest
    :return: numpy array
    '''
    arrays=spio.loadmat(fileName)
    x=arrays[arrayName]
    #return np.transpose(x) #scipy already transposes. No need to:
    #https://stackoverflow.com/questions/21624653/python-created-hdf5-dataset-transposed-in-matlab
    return x

def test_read_matlab_array_from_mat():

    if False:
        #some example of complex-valued 3-d array called H_all in a mat file:
        fileName = 'D:/ak/Works/2018-massive-mimo/GampMatlab/trunk/code/examples/ProbitSE/test3.mat'
        arrayName = 'H_all'
        x=read_matlab_array_from_mat(fileName, arrayName)
        print(x[50,60,70])
    else:
        if True:
            #this array is complex and has shape (140, 7, 4, 2, 18, 50)
            fileName = 'D:/gits/lasse/software/mimo-matlab/clusteredChannels/802_16_outdoor/mainFolder/allChannelsEpisode1.mat'
            arrayName = 'allHs'
            x=read_matlab_array_from_mat(fileName, arrayName)
            print(x.shape)
            print(x[50,6,3,1,3,40]) #and compare with allHs(50+1,6+1,3+1,1+1,3+1,40+1) in Matlab
        else:
            #another example saved with
            #save(fileName,'ak_receivedSignal_y', 'ak_transmittedSignal_x', 'ak_channel', 'ak_noise','-v6');
            fileName = 'gamp_simulation.mat'
            arrayName = 'ak_receivedSignal_y'
            x=read_matlab_array_from_mat(fileName, arrayName)
            #Matlab: ak_receivedSignal_y(10,5,20)
            print(x[9,4,19])

def test_convert_npz_to_hdf5():
    fileName = 'D:/gits/lasse/software/5gm-pcd-nn/3dobstacles/aug11norotation/obstacles_e_1.npz'
    print("Reading dataset...", fileName)
    convert_npz_to_hdf5(fileName)

if __name__ == '__main__':
    #test_read_matlab_array_from_mat()
    test_convert_npz_to_hdf5()
