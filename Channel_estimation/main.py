"""
Beseline channel estimation.
"""
import os
import shutil
import sys
from datetime import datetime
import argparse
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import (
    Reshape,
    Dense,
    Lambda,
    Flatten,
)
from mimo_channels_data_generator import RandomChannelMimoDataGenerator
from keras import backend as K

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode", choices=["train", "test", "trte"], default="trte")
args = parser.parse_args()

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_channel_unstructured3"
logdir = "{}/run-{}/".format(root_logdir, now)

logdir = "./mimo64x64"
# save this script
os.makedirs(logdir, exist_ok=True)
ini = sys.argv[0]
shutil.copyfile(sys.argv[0], os.path.join(
    logdir, os.path.basename(sys.argv[0])))
print("Copied:", sys.argv[0], "to", os.path.join(
    logdir, os.path.basename(sys.argv[0])))

# fix random seed for reproducibility
seed = 1
np.random.seed(seed)


# training parameters
epochs = 100

# Parameters
global Nt
global Nr
Nt = 64  # num of Rx antennas, will be larger than Nt for uplink massive MIMO
Nr = 64  # num of Tx antennas
# the sample is a measurement of Y values, and their collection composes an example. The channel estimation
min_randomized_snr_db = -1
max_randomized_snr_db = 1

# must be done per example, each one having a matrix of Nr x numSamplesPerExample of complex numbers
num_samples_per_example = 256  # number of channel uses, input and output pairs
# if wants a gradient calculated many times with same channel
num_examples_with_fixed_channel = 1
num_samples_per_fixed_channel = (
    num_examples_with_fixed_channel * num_samples_per_example
)  # coherence time
# obs: it may make sense to have the batch size equals the coherence time
batch_size = 5  # numExamplesWithFixedChannel
num_test_examples = 400  # for evaluating in the end, after training
# get small number to avoid slowing down the simulation, test in the end
num_validation_examples = 450
num_training_examples = 1600
file_name = "dataset/rosslyn60Ghz.mat"
method = "manual"
# Generator
training_generator = RandomChannelMimoDataGenerator(
    batch_size=batch_size,
    Nr=Nr,
    Nt=Nt,
    numSamplesPerFixedChannel=num_samples_per_fixed_channel,
    numSamplesPerExample=num_samples_per_example,
    method=method,
    file=file_name
)

training_generator.randomize_SNR = True
training_generator.min_randomized_snr_db = min_randomized_snr_db
training_generator.max_randomized_snr_db = max_randomized_snr_db

input_train, output_train = training_generator.get_examples(
    num_training_examples)
input_val, output_val = training_generator.get_examples(
    num_validation_examples)

global H_normalization_factor
H_normalization_factor = np.sqrt(Nr * Nt)

global K_H_normalization_factor
K_H_normalization_factor = K.variable(H_normalization_factor, dtype=np.float32)

# real / compl as twice number of rows
input_shape = (num_samples_per_example, 2 * (Nr))
output_dim = (2 * Nr, Nt)

num_inputs = np.prod(input_shape)
num_outputs = np.prod(output_dim)
print(num_inputs, " ", num_outputs)


def baseline_model():
    global Nt
    global Nr
    H_normalization_factor = np.sqrt(Nr * Nt)
    # K_H_normalization_factor = K.variable(H_normalization_factor,dtype=np.float32)

    N = 150
    model = Sequential()

    model.add(Dense(N, activation="tanh", input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(N, activation="tanh"))
    model.add(Dense(num_outputs, activation="linear"))
    model.add(Lambda(lambda x: H_normalization_factor *
                     K.l2_normalize(x, axis=-1)))
    model.add(Reshape(output_dim))
    return model


model_filepath = "models/model-rossslyn.h5"
model = baseline_model()
# Compile model
model.compile(loss="mse", optimizer="adam")
print(model.summary())

# Train model on dataset
if args.mode == "train" or args.mode == "trte":
    history = model.fit(input_train, output_train,
                        validation_data=(input_val, output_val),
                        epochs=epochs,
                        verbose=1,
                        callbacks=[
                            keras.callbacks.EarlyStopping(
                                monitor="val_loss",
                                min_delta=1e-7,
                                patience=5,
                            ),
                            keras.callbacks.ModelCheckpoint(
                                filepath=model_filepath,
                                monitor="val_loss",
                                verbose=1,
                                save_best_only=True,
                            ),
                            keras.callbacks.ReduceLROnPlateau(
                                factor=0.5,
                                min_delta=1e-7,
                                patience=2,
                                cooldown=5,
                                verbose=1,
                                min_lr=1e-6,
                            ),
                        ],
                        )

    print("Num of used channels over train: ",
          training_generator.num_channel_used)

if args.mode == "test" or args.mode == "trte":
    model = keras.models.load_model(model_filepath)
    # test with disjoint test set
    SNRdB = 0
    training_generator.randomize_SNR = False
    training_generator.SNRdB = SNRdB
    training_generator.method = "manual"
    # get rid of the last example in the training_generator's memory (flush it)
    test_input, test_output = training_generator.get_examples(1)
    # now get the actual examples:
    test_input, test_output = training_generator.get_examples(
        num_test_examples)
    predicted_output = model.predict(test_input)
    error = test_output - predicted_output

    mseTest = np.mean(error[:] ** 2)
    print("overall MSE = ", mseTest)
    mean_nmse = mseTest / (Nr * Nt)
    print("overall NMSE = ", mean_nmse)
    nmses = np.zeros((num_test_examples,))
    for i in range(num_test_examples):
        this_H = test_input[i]
        this_error = error[i]
        nmses[i] = np.mean(this_error[:] ** 2) / np.mean(this_H[:] ** 2)

    print("NMSE: mean", np.mean(nmses), "min",
          np.min(nmses), "max", np.max(nmses))
    nmses_db = 10 * np.log10(nmses)
    print(
        "NMSE dB: mean",
        np.mean(nmses_db),
        "min",
        np.min(nmses_db),
        "max",
        np.max(nmses_db),
    )