'''
DSP functions
'''

import numpy as np

def ak_fftmtx(N, option=1):
    '''
    # function [Ah, A] = ak_fftmtx(N, option)
    #FFT direct Ah and inverse A matrices with 3 options for
    #the normalization factors:
    #1 (default) ->orthonormal matrices
    #2->conventional discrete-time (DT) Fourier series
    #3->used in Matlab/Octave, corresponds to samples of DTFT
    #Example that gives the same result as Matlab/Octave:
    # Ah=ak_fftmtx(4,3); x=[1:4]'; Ah*x, fft(x)
    '''
    W = np.exp(- 1j * 2 * np.pi / N) #twiddle factor W_N
    Ah=np.zeros((N,N), dtype=np.complex) #pre-allocate space
    for n in range(N): #create the matrix with twiddle factors
        for k in range(N):
            Ah[k,n] = W ** (n*k)

    #choose among three different normalizations
    if option == 1: #orthonormal (also called unitary)
        Ah = Ah/np.sqrt(N)
        A = np.conj(Ah)
    if option == 2: #read X(k) in Volts in case x(n) is in Volts
        A = np.conj(Ah)
        Ah = Ah/N
    if option == 3: #as in Matlab/Octave: Ah = Ah
        A = np.conj(Ah)/N;
    if option < 1 or option > 3:
        print('Invalid option value: ', option)
        exit(-1)
    return Ah, A

def test_ak_fftmtx():
    Ah, A = ak_fftmtx(3,1)
    print('Ah = ', Ah)
    print('A = ', A)
    I=np.matmul(Ah,A)
    print('I=', I)

if __name__ == '__main__':
    test_ak_fftmtx()
