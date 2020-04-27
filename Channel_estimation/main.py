"""
Beseline channel estimation.
"""
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
from keras import backend as K
from sklearn.model_selection import train_test_split
import h5py

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train", "test", "trte"], default="trte")
args = parser.parse_args()

# fix random seed for reproducibility
seed = 1
np.random.seed(seed)


# training parameters
epochs = 10
validation_fraction = 0.2

# load dataset for training
train_dataset = h5py.File('dataset/train.hdf5', 'r')
train_input, validation_input, train_output, validation_output = \
    train_test_split(np.array(train_dataset['inputs']), np.array(train_dataset['outputs']), \
    test_size=validation_fraction, random_state=seed)
train_dataset.close()

num_samples_per_example = 256
global Nt
global Nr
Nt = 8  # num of Rx antennas
Nr = 8  # num of Tx antennas
# real / compl as twice number of rows
input_shape = (num_samples_per_example, 2 * (Nr))
output_dim = (2 * Nr, Nt)
global H_normalization_factor
H_normalization_factor = np.sqrt(Nr * Nt)

# sanity check
num_inputs = np.prod(input_shape)
num_outputs = np.prod(output_dim)
print(num_inputs, " ", num_outputs)


def baseline_model():
    global Nt
    global Nr
    H_normalization_factor = np.sqrt(Nr * Nt)

    N = 150
    model = Sequential()

    model.add(Dense(N, activation="tanh", input_shape=input_shape))
    model.add(Flatten())
    #model.add(Dense(N, activation="tanh"))
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
    history = model.fit(train_input, train_output,
                        validation_data=(validation_input, validation_output),
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
                        ],
                        )

if args.mode == "test" or args.mode == "trte":
    model = keras.models.load_model(model_filepath)
    test_input, test_output = validation_input, validation_output
    # now get the actual examples:
    predicted_output = model.predict(test_input)
    error = test_output - predicted_output

    mseTest = np.mean(error[:] ** 2)
    print("overall MSE = ", mseTest)
    mean_nmse = mseTest / (Nr * Nt)
    print("overall NMSE = ", mean_nmse)
    nmses = np.zeros((len(test_input),))
    for i in range(len(test_input)):
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