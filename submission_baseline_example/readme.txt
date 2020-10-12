Description: The files in this folder (submission_baseline_example) present an
example submission that uses the baseline data without customized frontend.
This file is an example of a mandatory readme.txt required in the submission.

#### Building the environment

1 - Dependencies: 

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

python beam_train_model.py --input coord img lidar -p ~/Documents/Raymobtime_dataset/s008/baseline_data

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

python beam_test_model.py --input coord img lidar -p  ~/Documents/Raymobtime_dataset/s009/baseline_data

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
