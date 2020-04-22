import numpy as np
from scipy.io import loadmat  # read .mat files
import keras
import numpy.linalg as la
from akpy.mimo_channels import (
    ak_generate_sparse_channels,
    initialize_matrices_for_channelFromAngularToArrayDomain,
    channelFromAngularToArrayDomain,
    initialize_matrices_for_channelFromArrayToAngularDomain,
    channelFromArrayToAngularDomain,
)

#from hysteresis_quantizer import hysteresis_quantize

from akpy.quantizer import ak_quantizer

from sklearn.preprocessing import MinMaxScaler

"""
# In a loop, generate:
# 1) random complex-valued channel matrix H with dimension Nr x Nt
# 2) input symbols as a complex-valued matrix with dimension Tt x Nt training
Here the training sequence is fixed
# 3) pass input symbols through the channel and get Nr x Tt matrix
# 4) quantize channel outputs with 1-bit using sign

# Use a generator:
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
Real and Imag are represented as real matrix with duplicated number of rows (not as new dimension as before). 
"""


class MimoChannel:
    "Simulate MIMO channel"

    """Constructor"""

    def __init__(
        self,
        Nr=3,
        Nt=2,
        num_clusters=2,
        numSamples=32,
        method="random",
        manual_input=(None, None),
    ):
        if method not in ["random", "sparse", "randomized_sparse", "manual"]:
            print("Method", method, "not support!")
            exit(1)
        # Initialization
        self.Nr = Nr
        self.Nt = Nt
        self.method = method
        self.num_clusters = num_clusters

        if method == "random":
            # just make explicit the Gaussian parameters we are using:
            mu = 0
            sigma = 1
            # get initial channel, which stays fixed until changed
            self.H = np.random.normal(
                mu, sigma, (self.Nr, self.Nt)
            ) + 1j * np.random.normal(mu, sigma, (self.Nr, self.Nt))
            # normalize: make the squared Frobenius norm of H = Nt*Nr
            self.H *= np.sqrt(self.Nr * self.Nt) / la.norm(self.H, "fro")
            UdirectFFT_Rx, UinverseFFT_Tx = initialize_matrices_for_channelFromArrayToAngularDomain(
                self.Nr, self.Nt
            )
            self.Hv = channelFromArrayToAngularDomain(
                self.H, UdirectFFT_Rx, UinverseFFT_Tx
            )

        if method == "randomized_sparse":
            num_clusters = np.random.randint(1, Nr * Nt + 1)  # second is exclusive
            self.Hv = ak_generate_sparse_channels(
                num_clusters, Nr, Nt, tau_sigma=1e-9, mu=0.2
            )
            # normalize: make the squared Frobenius norm of Hv = Nt*Nr
            self.Hv *= np.sqrt(self.Nr * self.Nt) / la.norm(self.Hv, "fro")
            UinverseFFT_Rx, UdirectFFT_Tx = initialize_matrices_for_channelFromAngularToArrayDomain(
                Nr, Nt
            )
            self.H = channelFromAngularToArrayDomain(
                self.Hv, UinverseFFT_Rx, UdirectFFT_Tx
            )

        if method == "sparse":
            self.Hv = ak_generate_sparse_channels(
                num_clusters, Nr, Nt, tau_sigma=1e-9, mu=0.2
            )
            # normalize: make the squared Frobenius norm of Hv = Nt*Nr
            self.Hv *= np.sqrt(self.Nr * self.Nt) / la.norm(self.Hv, "fro")
            UinverseFFT_Rx, UdirectFFT_Tx = initialize_matrices_for_channelFromAngularToArrayDomain(
                Nr, Nt
            )
            self.H = channelFromAngularToArrayDomain(
                self.Hv, UinverseFFT_Rx, UdirectFFT_Tx
            )
        if method == "manual":
            self.H, self.Hv = manual_input
            if self.H is None or self.Hv is None:
                print("Must provide manual_input")
                exit(1)
            self.H *= np.sqrt(self.Nr * self.Nt) / la.norm(self.H, "fro")
            self.Hv *= np.sqrt(self.Nr * self.Nt) / la.norm(self.Hv, "fro")

        # print('New channel!') #

    def get_mimo_signals(
        self,
        X,
        randomize_SNR=True,
        SNRdB=0,
        min_randomized_snr_db=-10,
        max_randomized_snr_db=10,
        hysteresis_range=0.5,
    ):
        # noiseless transmission
        Y = np.matmul(self.H, X)  # multiplication with training sequence
        power_y = la.norm(Y, "fro") ** 2  # square norm to get power

        if randomize_SNR == True:
            snrdb = np.random.uniform(min_randomized_snr_db, max_randomized_snr_db)
            SNRlinear = 10 ** (0.1 * snrdb)
        else:
            # create noise
            # just make explicit the Gaussian parameters we are using:
            SNRlinear = 10 ** (0.1 * SNRdB)
        power_noise = power_y / SNRlinear

        numSamples = X.shape[1]  # X is fixed because it is a training sequence
        mu = 0
        sigma = 1
        noise = np.random.normal(
            mu, sigma, (self.Nr, numSamples)
        ) + 1j * np.random.normal(mu, sigma, (self.Nr, numSamples))
        # print(SNRlinear) power_noise
        # power_original_noise = la.norm(noise, 'fro')**2  #accurate calculation
        # estimate to be faster, factor 2 to account for for real / imag
        power_original_noise = 2 * self.Nr * numSamples
        # normalize noise to obtain desired SNR
        noise *= np.sqrt(power_noise / power_original_noise)

        Y = Y + noise  # add noise

        # Quantize (it will change the norm)

        shouldUseHysteresisQuantizer = False
        if shouldUseHysteresisQuantizer:
            std = np.random.rand() * hysteresis_range
            #Y = hysteresis_quantize(Y, std=std)
            Y = ak_quantizer(np.real(Y), 5, -1, 1, 1)[0] + 1j * ak_quantizer(np.imag(Y), 5, -1, 1, 1)[0]
        else:
            Y = np.sign(np.real(Y)) + 1j * np.sign(np.imag(Y))

        # print('New signals!') #


        '''
        scaler = MinMaxScaler(feature_range=(-1,1))
        if False:
            #Y = ak_quantizer(np.real(Y), 2, -1, 1, 1)[0] + 1j * ak_quantizer(np.imag(Y), 2, -1, 1, 1)[0]
            Y = scaler.fit_transform(np.real(Y)) + 1j * scaler.fit_transform(np.imag(Y))
        '''
        return Y, noise


# See # https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
"""
Recall that the channel is kept constant by a duration numSamplesPerFixedChannel
(given by coherence time).
To give more context to a network, the pairs of inputs X and outputs Y can be
organized as a simple "example" with numSamplesPerExample "samples" each, where
a sample is the array of symbols X, that generates an array Y.
For simplicity we will require numSamplesPerFixedChannel to be a multiple of
numSamplesPerExample, such that 
 numExamplesPerFixedChannel = numSamplesPerFixedChannel / numSamplesPerExample
 is an integer.
 
 We have the following strategy for the mini-batch. The mini-batch accounts for
 the number of "examples", where each example counts as one train/test
pair with numSamplesPerExample each. A "samples" is a vector.
 If the mini-batch is larger than the
 numExamplesPerFixedChannel, more than one channels will be used in this batch.
 Otherwise, only one channel will be used and a variable will track that.
 
We will require that mini-batch is a multiple of numExamplesPerFixedChannel
such that 
numDifferentChannelsPerBatch = mini-batch / numExamplesPerFixedChannel 
For example, if numDifferentChannelsPerBatch = 1, all examples in that mini-batch 
will be using the same channel.
"""


class RandomChannelMimoDataGenerator(keras.utils.Sequence):
    "Generates data for Keras"

    # during numSamplesPerChannel the channel is the same
    def __init__(
        self,
        batch_size=24,
        Nr=5,
        Nt=3,
        num_clusters=1,
        numSamplesPerFixedChannel=12,
        numSamplesPerExample=4,
        SNRdB=0,
        method="random",
        #data_type = "train",
        file = "",
    ):
        self.numExamplesPerFixedChannel = (
            numSamplesPerFixedChannel / numSamplesPerExample
        )
        if (self.numExamplesPerFixedChannel < 1) | (
            int(self.numExamplesPerFixedChannel) != self.numExamplesPerFixedChannel
        ):
            print(
                "numSamplesPerFixedChannel is ",
                numSamplesPerFixedChannel,
                " but it needs to be a multiple of ",
                "numSamplesPerExample = ",
                numSamplesPerExample,
            )
            exit(1)
        # 'Initialization'
        self.num_channel_used = 0  # count number of different channel used
        self.numExamplesPerFixedChannel = int(self.numExamplesPerFixedChannel)
        self.batch_size = int(batch_size)
        self.Nr = Nr
        self.Nt = Nt
        self.num_clusters = num_clusters
        self.numSamplesPerFixedChannel = numSamplesPerFixedChannel
        self.numSamplesPerExample = int(numSamplesPerExample)
        # self.channelMatrixDimension = (Nr, Nt)
        # self.channel = RandomChannelMimo(Nr, Nt, sparsityRatio)
        self.SNRdB = SNRdB
        self.method = method
        self.randomize_SNR = False
        self.min_randomized_snr_db = -10
        self.max_randomized_snr_db = 10
        #self.data_type = data_type
        self.file = file

        if self.method == "manual":
            mat = loadmat(self.file)
            self.Harray, self.Hvirtual = [mat[key] for key in ["Harray", "Hvirtual"]]

            self.num_channels = self.Harray.shape[0]
            channelIndex = np.random.randint(0, self.num_channels//3)
            manual_input = (*[H[channelIndex] for H in [self.Harray, self.Hvirtual]],)
            self.mimo_channel = MimoChannel(
                Nr=self.Nr,
                Nt=self.Nt,
                num_clusters=self.num_clusters,
                numSamples=self.numSamplesPerExample,
                method=self.method,
                manual_input=manual_input,
            )
        else:
            # create first channel
            self.mimo_channel = MimoChannel(
                Nr=self.Nr,
                Nt=self.Nt,
                num_clusters=self.num_clusters,
                numSamples=self.numSamplesPerExample,
                method=self.method,
            )

        self.numOfSamplesAlreadyGeneratedByCurrentChannel = 0  # initialize counter

        # define the input, which is fixed because it is a training sequence
        # RandomChannelMimo.X = np.random.normal(mu, sigma, (self.Nt, numSamples)) + \
        #         1j * np.random.normal(mu, sigma, (self.Nt, numSamples))
        # RandomChannelMimo.X /= la.norm(RandomChannelMimo.X, 'fro')  # force to have unitary Frobenious norm
        # RandomChannelMimo.X *= np.sqrt(numSamples)  # now the square of that norm is numSamples
        # X *= np.sqrt(SNRlinear/self.Nt)
        # just make explicit the Gaussian parameters we are using:

        if True:  # QPSK
            # From https://stackoverflow.com/questions/46820182/randomly-generate-1-or-1-positive-or-negative-integer
            self.X = (
                2 * np.random.randint(0, 2, size=(self.Nt, numSamplesPerExample)) - 1
            ) + 1j * (
                2 * np.random.randint(0, 2, size=(self.Nt, numSamplesPerExample)) - 1
            )
        else:  # Gaussian
            mu = 0
            sigma = 1
            self.X = np.random.normal(
                mu, sigma, (self.Nt, numSamplesPerExample)
            ) + 1j * np.random.normal(mu, sigma, (self.Nt, numSamplesPerExample))
            # force to have unitary Frobenious norm
            self.X /= la.norm(self.X, "fro")
            # now the square of that norm is numSamplesPerExample
            self.X *= np.sqrt(numSamplesPerExample)

    def get_examples(self, num_batches=1):
        if num_batches == 1:
            return self.__getitem__()  # simply return a batch
        # compose with data from several batches
        if False:  # re / im as extra dimension
            all_inputs = np.NaN * np.ones(
                (self.batch_size * num_batches, self.Nr, self.numSamplesPerExample, 2),
                dtype=float,
            )
            all_outputs = np.NaN * np.ones(
                (self.batch_size * num_batches, self.Nr, self.Nt, 2), dtype=float
            )
        else:  # re / im as twice number of rows
            all_inputs = np.NaN * np.ones(
                (
                    self.batch_size * num_batches,
                    self.numSamplesPerExample,
                    2 * (self.Nr),
                ),
                dtype=float,
            )
            all_outputs = np.NaN * np.ones(
                (self.batch_size * num_batches, 2 * self.Nr, self.Nt), dtype=float
            )
        for i in range(num_batches):
            inputTensor, outputTensor = self.__getitem__()  # get a batch
            start_index = i * self.batch_size
            end_index = start_index + self.batch_size
            all_inputs[start_index:end_index] = inputTensor
            all_outputs[start_index:end_index] = outputTensor
        return all_inputs, all_outputs

    # private method (indicated with __ prefix)
    def __getitem__(self, index=0):
        # X : (n_samples, *dim, n_channels)
        "Generates data containing batch_size samples"
        # Initialization
        if False:  # re / im as extra dimension
            inputTensor = np.NaN * np.ones(
                (self.batch_size, self.Nr, self.numSamplesPerExample, 2), dtype=float
            )
            outputTensor = np.NaN * np.ones(
                (self.batch_size, self.Nr, self.Nt, 2), dtype=float
            )
        else:  # re / im as twice number of rows
            inputTensor = np.NaN * np.ones(
                (self.batch_size, self.numSamplesPerExample, 2 * (self.Nr)), dtype=float
            )
            outputTensor = np.NaN * np.ones(
                (self.batch_size, 2 * self.Nr, self.Nt), dtype=float
            )

        numExamplesAlreadyCreated = 0
        while numExamplesAlreadyCreated < self.batch_size:
            # try to generate all samples with current channel
            numRemainingExamplesWithThisChannel = self.numExamplesPerFixedChannel - (
                self.numOfSamplesAlreadyGeneratedByCurrentChannel
                / self.numSamplesPerExample
            )
            # initialize with all values remaining for current channel
            numExamplesToBeCreatedWithThisChannel = numRemainingExamplesWithThisChannel
            # but now check if that exceeds what is still needed to complete the batch
            if (
                numRemainingExamplesWithThisChannel
                > self.batch_size - numExamplesAlreadyCreated
            ):
                numExamplesToBeCreatedWithThisChannel = (
                    self.batch_size - numExamplesAlreadyCreated
                )
            # get the samples with the same channel
            numExamplesToBeCreatedWithThisChannel = int(
                numExamplesToBeCreatedWithThisChannel
            )
            # incorporate the data to the input and output tensors
            # two things need to be done: the arrays need to be split into "examples", and the examples
            # properly stacked
            for j in range(numExamplesToBeCreatedWithThisChannel):
                # numExamplesToBeCreatedWithThisChannel*self.numSamplesPerExample
                Yreal_valued, Hreal_valued = self.__data_generation_for_single_channel()
                # startSample = 0 #j * self.numSamplesPerExample
                # lastSample = startSample + self.numSamplesPerExample
                # organize the samples in different examples
                # all output will have the same channel H within this inner loop
                outputTensor[numExamplesAlreadyCreated + j] = Hreal_valued
                # thisX = Xreal_valued[:, startSample:lastSample, :]
                # thisY = Yreal_valued[:, startSample:lastSample, :]
                # first the channel outputs
                inputTensor[numExamplesAlreadyCreated + j] = Yreal_valued
                # inputTensor[numExamplesAlreadyCreated + j, self.Nr:, :] = thisX
                # inputTensor[numExamplesAlreadyCreated + j, self.Nr:, :] = np.divide(thisY,thisX+1e-10)
            # update counters within the class and local
            self.numOfSamplesAlreadyGeneratedByCurrentChannel += (
                numExamplesToBeCreatedWithThisChannel * self.numSamplesPerExample
            )
            numExamplesAlreadyCreated += numExamplesToBeCreatedWithThisChannel
            numRemainingExamplesWithThisChannel -= numExamplesToBeCreatedWithThisChannel
            if numRemainingExamplesWithThisChannel < 1:
                channelIdx = 0
                if self.method == "manual":
                    channelIdx = np.random.randint(0, self.num_channels)
                    manual_input = (
                        *[H[channelIdx] for H in [self.Harray, self.Hvirtual]],
                    )
                    self.mimo_channel = MimoChannel(
                        Nr=self.Nr,
                        Nt=self.Nt,
                        num_clusters=self.num_clusters,
                        numSamples=self.numSamplesPerExample,
                        method=self.method,
                        manual_input=manual_input,
                    )
                else:
                    # need to create new channel and reset its counter
                    self.mimo_channel = MimoChannel(
                        Nr=self.Nr,
                        Nt=self.Nt,
                        num_clusters=self.num_clusters,
                        numSamples=self.numSamplesPerExample,
                        method=self.method,
                    )
                self.num_channel_used += 1  # increment counter of different channels
                self.numOfSamplesAlreadyGeneratedByCurrentChannel = (
                    0
                )  # initialize counter

        # print('\nmin=', np.min(inputTensor[:]), ' max=', np.max(inputTensor[:]), ' zeros=', np.argwhere(inputTensor[:]==0))
        # print('New tensors!') #
        return inputTensor, outputTensor

    def __len__(self):
        "Denotes the number of batches per epoch"
        # return int(np.floor(len(self.list_IDs) / self.batch_size))
        return (
            1000
        )  # AK: this was chosen arbitrarily given that the dataset is infinite and there is no "episode" end

    def __splitRealAndImaginary(self, complexMatrix):
        num_rows = complexMatrix.shape[0]
        realMatrix = np.zeros((2 * num_rows, complexMatrix.shape[1]), dtype=float)
        realMatrix[0:num_rows] = np.real(complexMatrix)
        realMatrix[num_rows:] = np.imag(complexMatrix)
        return realMatrix

    def __splitMagnitudeAndPhase(self, complexMatrix):
        num_rows = complexMatrix.shape[0]
        realMatrix = np.zeros((2 * num_rows, complexMatrix.shape[1]), dtype=float)
        # AK TODO I am playing here
        realMatrix[0:num_rows] = np.abs(complexMatrix)
        # np.zeros(complexMatrix.shape,dtype=np.float32)
        realMatrix[num_rows:] = np.angle(complexMatrix) / np.pi
        return realMatrix

    def __splitRealAndImaginary_add_dimension(self, complexMatrix):
        realMatrix = np.zeros(
            (complexMatrix.shape[0], complexMatrix.shape[1], 2), dtype=float
        )
        realMatrix[:, :, 0] = np.real(complexMatrix)
        realMatrix[:, :, 1] = np.imag(complexMatrix)
        return realMatrix

    def __splitMagnitudeAndPhase_add_dimension(self, complexMatrix):
        realMatrix = np.zeros(
            (complexMatrix.shape[0], complexMatrix.shape[1], 2), dtype=float
        )
        realMatrix[:, :, 0] = np.abs(complexMatrix)
        # AK TODO I am playing here
        # np.zeros(complexMatrix.shape,dtype=np.float32)
        realMatrix[:, :, 1] = np.angle(complexMatrix) / np.pi
        return realMatrix

    def __data_generation_for_single_channel(self):
        "Generate data for a single channel"
        # Y, noise = self.mimo_channel.get_mimo_signals(self.X, randomize_SNR=True, self.SNRdB)
        Y, noise = self.mimo_channel.get_mimo_signals(
            self.X,
            randomize_SNR=self.randomize_SNR,
            SNRdB=self.SNRdB,
            min_randomized_snr_db=self.min_randomized_snr_db,
            max_randomized_snr_db=self.max_randomized_snr_db,
        )
        # create additional dimension on tensors to store real and imag
        # choose the H that will be used as input to machine learning
        H = self.mimo_channel.Hv  # virtual domain always
        if False:
            Yreal_valued = self.__splitMagnitudeAndPhase(Y)
            Hreal_valued = self.__splitMagnitudeAndPhase(H)
            # Xreal_valued = self.__splitMagnitudeAndPhase(self.X)
        else:
            Yreal_valued = self.__splitRealAndImaginary(Y)
            Hreal_valued = self.__splitRealAndImaginary(H)
            Xreal_valued = self.__splitRealAndImaginary(self.X)
        # Y is the received signals and H is the channel, both as real matrices

        return np.transpose(Yreal_valued), Hreal_valued
        # return np.transpose(np.vstack((Yreal_valued, Xreal_valued))), Hreal_valued


if __name__ == "__main__":
    # Parameters
    Nr = 8  # should be larger than Nt: uplink massive MIMO
    Nt = 8
    numSamplesPerExample = 24
    numExamplesWithFixedChannel = 1
    numSamplesPerFixedChannel = numExamplesWithFixedChannel * numSamplesPerExample  # L
    batch_size = 3
    SNRdB = -10

    generator = RandomChannelMimoDataGenerator(
        batch_size=batch_size,
        Nr=Nr,
        Nt=Nt,
        numSamplesPerFixedChannel=numSamplesPerFixedChannel,
        numSamplesPerExample=numSamplesPerExample,
        SNRdB=SNRdB,
        method="manual",
    )
    for i in range(10):
        inputTensor, outputTensor = generator.__getitem__()
    print(inputTensor.shape)
    print(outputTensor.shape)

    if False:
        mychannel = MimoChannel(Nr=8, Nt=32, sparsityRatio=1, numSamples=128)
        # num of examples
        Y, noise = mychannel.get_mimo_signals(generator.X, SNRdB=10)
        X = mychannel.X
        print("Squared Frobenius norm of X = ", la.norm(X, "fro") ** 2)
        print("Squared Frobenius norm of H = ", la.norm(H, "fro") ** 2)
        print("Squared Frobenius norm of Y = ", la.norm(Y, "fro") ** 2)
        print("Squared Frobenius norm of Y - noise= ", la.norm(Y - noise, "fro") ** 2)
        print("Squared Frobenius norm of noise = ", la.norm(noise, "fro") ** 2)
        print("Dimensions:")
        print(X.shape)
        print(H.shape)
        print(Y.shape)
        if 0:
            print("X=", X)
            print("Y=", Y)
            print("H=", H)

        mychannel = MimoChannel(Nr=2, Nt=3, sparsityRatio=1, numSamples=128)
        Y, noise = mychannel.get_mimo_signals(SNRdB=10)
        X = mychannel.X
        # print('Error=', np.sum(np.abs(H - H2)[:]))

