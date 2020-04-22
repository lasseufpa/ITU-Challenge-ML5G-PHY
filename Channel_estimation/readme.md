# Channel Estimation Challenge
The channel estimation challenge assumes a multiple-input multiple-output (MIMO) communication system using millimeter wave (mmWave) frequencies. In this challenge, the inputs are preamble (or pilot) signals obtained at the receiver, after propagation through the channel. The channels are obtained with ray-tracing simulations. Based on the received pilots, the task is to estimate the MIMO channel.

## Requirements
Requirements to run the baseline model are:
(work in progress!)

## Directory structure
(work in progress!)

## How to run the code
Follow steps below to run the baseline models
### 1. download dateset
Inside dataset folder, run the following command in terminal.

`>>>python download_dataset.py`
### 2. preprocess dataset
Inside preprocessing folder, and after dataset has been downloaded, run the following command in terminal.

`>>>python get_channels.py`
### 3. run main.py
You can run main.py scritp passing arguments in the terminal to tell what you want to do. You can train the model, or you can train and test or you can only test:

a) to train and test (default option): `>>>python main.py`

b) just to train: `>>>python main.py --mode train`    

c) just to test: `>>>python main.py --mode test`

Before you run, please create a folder named `models` in the same directory of main.py file to save your models.