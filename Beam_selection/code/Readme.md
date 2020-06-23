## Beam Selection baseline

This repository contains Python codes for preprocessing data and train/validate
a Neural Network for the beam selection task using the Raymobtime dataset.
To download the files for preprocessing or/and the Neural network [Access here](https://github.com/lasseufpa/ITU-Challenge-ML5G-PHY/tree/master/Beam_selection/data)
Or download directly [Here](https://nextcloud.lasseufpa.org/s/FQgjXx7r52c7Ww9)

### Python dependencies
If you want to use the already available preprocessed data, to train and test this baseline
model the only dependencies are:  
* [TensorFlow](https://www.tensorflow.org/install)
* [Scikit-learn](https://scikit-learn.org/stable/install.html)
* [Numpy](https://numpy.org/install/)

You may install these packages using pip or similar software. For example, with pip:

pip install tensorflow

In the other hand, if you want to preprocess data or plot model's accuracy and 
loss, you will also need the following ones:
  * [OpenCV](https://pypi.org/project/opencv-python/) For image processing, e.g: image resampling
  * [Matplotlib](https://matplotlib.org/users/installing.html) For plotting

<!-- ### Getting data
Before  -->

### How to use it?

#### Preprocessing
Before training your model, preprocess the data using:

```bash
python preprocessing.py raymobtime_root data_folder
```
* Parameters
  
  * (**Mandatory**) *raymobtime_root* is the directory where you placed the files related to the Raymobtime dataset, downloaded using one of the scripts available [here](https://github.com/lasseufpa/ITU-Challenge-ML5G-PHY/tree/master/Beam_selection/data).

  * (**Mandatory**) *data_folder* is the directory where you want to place the processed files.

> If *data_folder* doesn't exist, it will be created for you

#### Training and validation
After processing the data, you can train and validate your model using the
following command:

```bash
python main.py data_folder --input type_of_input
```

* Parameters 

  * (**Mandatory**) *data_folder* is the same one directory generated on the preprocessing step.

  * (**Optional**) *--input* is a list of the types of data that you want to feed into your model. You can pass up to 3 different types, the possible ones are : *img, coord and lidar*. In the absence of the *--input* parameter, the coord data will be used as a default
  * (**Optional**) *--plots* plot the accuracy and validation accuracy of your model.

##### Usage example
To train a model that uses *images, lidar and coordinates* use the command:
```bash
python main.py data --input img lidar coord
```
