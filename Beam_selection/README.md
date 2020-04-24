# Beam Selection Challenge
The ML5G-PHY beam-selection challenge assumes a mmWave MIMO system. Both transmitter (Tx) and receiver (Rx) have sets of pre-defined beams organized as “codebooks” (sets of vectors, one for the Tx and another for the Rx). The beam-selection task is to select the best pair (i,j) of beams for communication, where i is the i-th vector of the Tx codebook and j is the j-th vector of the Rx codebook. The ML model to be developed (e.g., a neural network or decision tree) takes input features extracted from a multimodal dataset, which include, for instance, LIDAR point clouds and positions of vehicles. The ML model then outputs a pair of indices (i,j), which should be used for achieving the strongest magnitude of the effective channel created by the pair (i,j) of adopted beams.

Details about the challenge and the datasets are avaiable at http://ai5gchallenge.ufpa.br/ and https://www.lasse.ufpa.br/raymobtime/, respectively.

To generate neural network files, see the documentation in the processing directories.
After generate the neural network files, move the files for this folder.

```bash
# Install all used packages
pip install numpy Keras h5py scikit-learn DateTime
``` 
If **just** looking for the machine learning data, you can download directly [here](https://nextcloud.lasseufpa.org/s/zN5Yj956steYNHz)

To run the neural network after already had the files, run the following command:

```bash
python classifierTopKBeams.py
```

It is assumed that **Python 3** is used

