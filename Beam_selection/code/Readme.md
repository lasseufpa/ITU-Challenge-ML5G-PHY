# BeamSelection Framework

This repository contains Python code for preprocessing data and train/test a
NN (Neural Network) for a beam-selection task based on Raymobtime datasets.

## How to use it?

### Preprocessing
First, download the desired dataset [here](https://nextcloud.lasseufpa.org/s/7FrX2883E4yorbB).
After downloading the dataset, run the following command:

```
python code/preprocessing.py data_folder Raymobtime_root
```
*data_folder* is a directory to put the processed files and *Raymobtime_root* is
the root of the Raymobtime dataset.

### Training and testing
To train and test the model, you should run the following command:

```shell
python code/main.py data_folder --input type_of_input
```

*data_folder* is the same one generated on the preprocessing step and the
possible values for *--input* are: img, lidar or coord. You can use a single
type of data or as many as necessary(don't repeat the same parametters or pass more than 3 different)
