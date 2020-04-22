'''
MIMO processing
'''
import numpy as np
from scipy.stats import expon

from akpy.signal_processing import ak_fftmtx

def initialize_matrices_for_channelFromAngularToArrayDomain(Nr, Nt):
    #generate square orthonormal FFT matrices with dimensions Nr and Nt
    option = 1
    [UdirectFFT_Rx, UinverseFFT_Rx] = ak_fftmtx(Nr, option)
    [UdirectFFT_Tx, UinverseFFT_Tx] = ak_fftmtx(Nt, option)
    return UinverseFFT_Rx, UdirectFFT_Tx

def initialize_matrices_for_channelFromArrayToAngularDomain(Nr, Nt):
    #generate square orthonormal FFT matrices with dimensions Nr and Nt
    #Hv = UdirectFFT_Rx*Hk*UinverseFFT_Tx;
    option = 1
    [UdirectFFT_Rx, UinverseFFT_Rx] = ak_fftmtx(Nr, option)
    [UdirectFFT_Tx, UinverseFFT_Tx] = ak_fftmtx(Nt, option)
    return UdirectFFT_Rx, UinverseFFT_Tx

def channelFromAngularToArrayDomain(Hv,UinverseFFT_Rx,UdirectFFT_Tx):
    '''
    #function Hk=channelFromAngularToArrayDomain(Hv,UinverseFFT_Rx, ...
                                                 #    UdirectFFT_Tx)
    #Output Hk is the 3-d array with the wideband channel represented in the
    #physical (also called array) domain. UinverseFFT_Rx and UdirectFFT_Tx
    #are unitary DFT matrices as described in Eq. (7.70) in [1]. The input Hv
    #is the channel in the angular or virtual domain. Because the matrices are
    #unitary (orthonormal) the norm of Hk is the same as Hv.
    #References:
    # [1] Fundamentals of Wireless Communications. David Tse.
    # [2] Akbar M. Sayeed, "Deconstructing Multiantenna Fading Channels"
    # IEEE TRANSACTIONS ON SIGNAL PROCESSING, VOL. 50, NO. 10, OCTOBER 2002,
    # pag. 2563.
    #See also channelFromArrayToAngularDomain.m
    '''
    if len(Hv.shape)==2: #case of narrowband channel, no subcarriers
        Hk = np.matmul(UinverseFFT_Rx, Hv) #temporary result
        Hk = np.matmul(Hk, UdirectFFT_Tx)
    else:
        [Nr,Nt,Nfft]=Hv.shape #num Rx antennas, num Tx and num of FFT points
        Hk = np.zeros(Nr,Nt,Nfft,dtype=complex)
        for k in range(Nfft):
            #Eq. (12) in [1]
            Hk[:,:,k] = np.matmul(UinverseFFT_Rx*Hv[:,:,k], UdirectFFT_Tx)
    return Hk

def channelFromArrayToAngularDomain(H,UdirectFFT_Rx, UinverseFFT_Tx):
    #Hv = UdirectFFT_Rx*Hk*UinverseFFT_Tx;
    if len(H.shape)==2: #case of narrowband channel, no subcarriers
        Hv = np.matmul(UdirectFFT_Rx, H) #temporary result
        Hv = np.matmul(Hv, UinverseFFT_Tx)
    else:
        [Nr,Nt,Nfft]=H.shape #num Rx antennas, num Tx and num of FFT points
        Hv = np.zeros(Nr,Nt,Nfft,dtype=complex)
        for k in range(Nfft):
            #Eq. (12) in [1]
            Hv[:,:,k] = np.matmul(UdirectFFT_Rx*H[:,:,k], UinverseFFT_Tx)
    return Hv


def ak_generate_sparse_channels(num_clusters, Nr, Nt, tau_sigma=1e-9, mu=0.2):
    '''
    ########################################################################################
    # Modified from
    # From: Author: Anum Ali
    #
    # If you use this code or any (modified) part of it in any publication, please cite
    # the paper: Anum Ali, Nuria González-Prelcic and Robert W. Heath Jr.,
    # "Millimeter Wave Beam-Selection Using Out-of-Band Spatial Information",
    # IEEE Transactions on Wireless Communications.
    #
    # Contact person email: anumali@utexas.edu
    ########################################################################################
    # Input Arguments:
    # tau_sigma: The RMS delay spread of the channel
    # mu: The exponential PDF parameter
    # num_clusters: The number of clusters
    # Nr and Nt: num of antennas at Rx and Tx
    tau_sigma=3e-9; #seconds
    mu=0.2;
    num_clusters=4;
    Nr=4;
    Nt=4;
    ########################################################################################
    '''
    #find different delays
    taus = tau_sigma*np.log(np.random.rand(num_clusters,1)) #beautiful transformation
    taus = np.sort(taus-np.min(taus))[::-1] #synchronize Rx with first impulse
    PDP = expon.pdf(taus/tau_sigma, scale=mu) #Exponential PDP. Matlab: PDP=exppdf(taus/tau_sigma,mu)
    PDP=PDP/np.sum(PDP) # Normalizing to unit power PDP
    gains=np.sqrt(PDP/2/num_clusters)*(np.random.randn(num_clusters,1)+1j*np.random.randn(num_clusters,1))
    Hv=np.zeros( (Nr,Nt), dtype=complex)
    num_H_elements = Nr*Nt
    #choose without replacement
    chosen_indices = np.random.choice(num_H_elements, num_clusters, replace=False) #Matlab's randsample
    for i in range(num_clusters):
        chosen_index_uraveled = np.unravel_index(chosen_indices[i], Hv.shape)
        Hv[chosen_index_uraveled]=gains[i]
    return Hv

def test_channelFromAngularToArrayDomain():
    Nr = 3
    Nt = 2
    UinverseFFT_Rx, UdirectFFT_Tx = initialize_matrices_for_channelFromAngularToArrayDomain(Nr, Nt)
    Hv = np.zeros((Nr,Nt),dtype=complex)
    Hv[1,1]=1+5j
    Hv[0,0]=2-3j
    Hk = channelFromAngularToArrayDomain(Hv,UinverseFFT_Rx,UdirectFFT_Tx)
    print('Hk=',Hk)

    UdirectFFT_Rx, UinverseFFT_Tx = initialize_matrices_for_channelFromArrayToAngularDomain(Nr, Nt)
    Hv2 = channelFromArrayToAngularDomain(Hk,UdirectFFT_Rx, UinverseFFT_Tx)
    print('Hv2=',Hv2)
    print('max error real=',np.max(np.real(Hv2-Hv)))
    print('max error imag=',np.max(np.imag(Hv2-Hv)))

def test_ak_generate_sparse_channels():
    Nr = 3
    Nt = 2
    num_clusters = 3
    Hv=ak_generate_sparse_channels(num_clusters, Nr, Nt, tau_sigma=1e-9, mu=0.2)
    UinverseFFT_Rx, UdirectFFT_Tx = initialize_matrices_for_channelFromAngularToArrayDomain(Nr, Nt)
    Hk = channelFromAngularToArrayDomain(Hv,UinverseFFT_Rx,UdirectFFT_Tx)
    print('Hv =', Hv)
    print('Hk =', Hk)

if __name__ == '__main__':
    test_channelFromAngularToArrayDomain()
    #test_ak_generate_sparse_channels()