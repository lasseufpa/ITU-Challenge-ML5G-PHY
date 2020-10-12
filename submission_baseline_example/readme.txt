Description: The files in this folder (submission_baseline_example) present an
example of submission that uses the baseline data without a customized frontend.
This file is an example of a mandatory readme.txt.

#### Building the environment

1 - Dependencies

1.1 System dependencies
- Hardware: CUDA Capable GPU
- Libraries: CUDA 10.1 and cuDNN 7.6 (see https://www.tensorflow.org/install/source)
- Python 3: it is in general installed installed with Linux distributions.
- Others, install using the following commands in Ubuntu 18
    pip: "sudo apt install python-pip"
    virtualvenv: "sudo apt install virtualvenv"

1.2 - Python Dependencies
- Python packages: tensorflow-gpu 2.3.1, sklearn, matplotlib (and dependencies of
these packages)
- Please run the script build_env.sh to build the Python environment required
to run our code.
    "bash build_env.sh"


2 - Running the code

After building the environment, you can train the model as shown in 2.1 and 
test it as shown in 2.2. The step in 2.1 saves the model and its trained weights
(my_model.json and my_model_weights.h5 files).

2.1 - Training the model

python beam_train_model.py --input coord img lidar -p ~/Documents/Raymobtime_dataset/s008

usage: beam_train_model.py [-h]
                           [--input [{img,coord,lidar} [{img,coord,lidar} ...]]]
                           [-p]
                           data_folder

Configure the files before training the net.

positional arguments:
  data_folder           Location of the data directory

optional arguments:
  -h, --help            show this help message and exit
  --input [{img,coord,lidar} [{img,coord,lidar} ...]]
                        Which data to use as input. Select from: img, lidar or
                        coord.
  -p, --plots           Use this parametter if you want to see the accuracy
                        and loss plots

2.2 - Testing the model

python beam_test_model.py --input coord img lidar -p  ~/Documents/Raymobtime_dataset/s009

usage: beam_test_model.py [-h]
                          [--input [{img,coord,lidar} [{img,coord,lidar} ...]]]
                          [-p]
                          data_folder

Configure the files before testing the net.

positional arguments:
  data_folder           Location of the data directory

optional arguments:
  -h, --help            show this help message and exit
  --input [{img,coord,lidar} [{img,coord,lidar} ...]]
                        Which data to use as input. Select from: img, lidar or
                        coord.
  -p, --plots           Use this parametter if you want to see the accuracy
                        and loss plots

3 - Pre-trained model and weights

As described in the rules of the challenge (see Google Docs link below), the
participants must provide with the submission the trained model. Therefore, in
this example, we also include the files my_model.json and my_model_weights.h5.
These files are the results of running the training described in 2.1 with the
following arguments:
"python beam_train_model.py --input coord -p ~/Documents/Raymobtime_dataset/s008".

Note that the trained model included was trained only with coordinates data,
which allows generating my_model_weights.h5 with a small size to be provided as
an example in this repository. To achieve better performance, the participants
should use all the available data (coordinates, images and lidar), and upload
their model in the submission.

https://docs.google.com/document/d/1_TEpxu40E1sL5jmJVG1Qng7Giiq0qYDCSwU0Q15DPVs/edit#
