'''
Based on:
%Emil Bjornson, Jakob Hoydis and Luca Sanguinetti (2017),
%"Massive MIMO Networks: Spectral, Energy, and Hardware Efficiency",
%Foundations and Trends in Signal Processing: Vol. 11, No. 3-4,
%pp. 154-655. DOI: 10.1561/2000000093.
%https://www.massivemimobook.com
'''
#in all reshapes I need to use 'F': h_temp = np.reshape(h_temp,(self.K,self.L),'F')

#https://stackoverflow.com/questions/41816973/modulenotfounderror-what-does-it-mean-main-is-not-a-package
from matlab_tofrom_python import read_matlab_array_from_mat
#from .matlab_tofrom_python import read_matlab_array_from_mat
import numpy as np
from scipy.linalg import sqrtm

class ChannelEstimator:
    def __init__(self, discount=0.9):
        #Number of BSs
        self.L = 16
        #Number of UEs per BS
        self.K = 10
        #Number of BS antennas
        self.M = 64
        #Define the pilot reuse factor
        self.f = 2
        #Select the number of setups with random UE locations
        nbrOfSetups = 100
        #Select the number of channel realizations per setup
        nbrOfRealizations = 100
        ## Propagation parameters
        #Communication bandwidth
        B = 20e6

        #Total uplink transmit power per UE (mW)
        self.p = 100

        #Total downlink transmit power per UE (mW)
        rho = 100

        #Maximum downlink transmit power per BS (mW)
        self.Pmax = self.K*rho

        #Compute downlink power per UE in case of equal power allocation
        self.rhoEqual = (self.Pmax/self.K)*np.ones((self.K,self.L))

        #Define noise figure at BS (in dB)
        noiseFigure = 7

        #Compute noise power
        noiseVariancedBm = -174 + 10*np.log10(B) + noiseFigure

        #Select length of coherence block
        self.tau_c = 200

        #Use the approximation of the Gaussian local scattering model
        accuracy = 2

        #Angular standard deviation in the local scattering model (in degrees)
        ASDdeg = 10
        fileName = 'D:/gits/lasse/software/mimo-toolbox/third_party/emil_massivemimobook/Code/R_channelGaindB.mat'
        channelGaindB=read_matlab_array_from_mat(fileName, 'channelGaindB')
        self.R=read_matlab_array_from_mat(fileName, 'R') #normalized to have norm = M

        #print('R', self.R.shape)

        #Compute the normalized average channel gain, where the normalization
        #is based on the noise power
        channelGainOverNoise = channelGaindB - noiseVariancedBm;

        self.channel_gain_over_noise_linear = 10**(channelGainOverNoise/10)

        #from functionChannelEstimates.m
        #Length of pilot sequences
        self.tau_p = self.f*self.K

        #Generate pilot pattern
        if self.f == 1:
            self.pilotPattern = np.ones((self.L,1))
        elif self.f == 2: #Only works in the running example with its 16 BSs
            self.pilotPattern = np.kron([[1,1]],[1, 2, 1, 2, 2, 1, 2, 1])
        elif self.f == 4: #Only works in the running example with its 16 BSs
            self.pilotPattern = np.kron([[1,1]],[1, 2, 1, 2, 3, 4, 3, 4])
        elif self.f == 16: #Only works in the running example with its 16 BSs
            self.pilotPattern = np.arange(1,self.L)

        print('Pilot groups=', self.pilotPattern)
        #Store identity matrix of size M x M
        self.eyeM = np.eye(self.M)


    '''
    From: section7_figure2.m
    '''
    def estimate_channels(self):
        #Compute the normalized average channel gain, where the normalization
        #is based on the noise power
        x=3

    '''
    Get channel realizations
    '''
    def channel_realizations(self,num_realizations):
        #Go through all channels and apply the channel gains to the spatial
        H = np.random.randn(self.M,num_realizations,self.K,self.L,self.L) + \
                1j * np.random.randn(self.M,num_realizations,self.K,self.L,self.L)
        for j in range(self.L):
            for l in range(self.L):
                for k in range(self.K):
                    Rtemp = self.channel_gain_over_noise_linear[k,j,l] * self.R[:,:,k,j,l]
                    #print(Rtemp.shape)
                    #Rtemp=np.matrix([[8,1+3j,6],[3-100j,5,7],[4,9,2]])
                    Rsqrt= sqrtm(Rtemp)
                    #print(Rsqrt)
                    #exit(-1)
                    Htemp = H[:,:,k,j,l]
                    #Apply correlation to the uncorrelated channel realizations
                    H[:,:,k,j,l] = np.sqrt(0.5) * np.matmul(Rsqrt,Htemp)
        return H

    '''
    Linear MMSE channel estimator 
    '''
    def mmse_estimation(self,H):
        fileName = 'D:/gits/lasse/software/mimo-toolbox/third_party/emil_massivemimobook/Code/H_Np_R.mat'
        self.R=read_matlab_array_from_mat(fileName, 'R')
        H=read_matlab_array_from_mat(fileName, 'H')
        num_realizations = H.shape[1]
        if True:
            Np=read_matlab_array_from_mat(fileName, 'Np')
        else:
            #Generate realizations of normalized noise
            Np = np.sqrt(0.5)*(np.random.randn(self.M,num_realizations,self.K,self.L,self.f) + \
                1j * np.random.randn(self.M,num_realizations,self.K,self.L,self.f) )
        #Prepare to store MMSE channel estimates
        Hhat_MMSE = np.zeros((self.M,num_realizations,self.K,self.L,self.L),dtype=np.complex64)
        #Prepare to store estimation error correlation matrices
        C_MMSE = np.zeros((self.M,self.M,self.K,self.L,self.L),dtype=np.complex64)

        # Go through all cells
        for j in range(self.L):
            #Go through all f pilot groups
            for g in range(self.f):
                #Extract the cells that belong to pilot group g
                groupMembers = np.where(self.pilotPattern == g+1) #add 1 because first pilot group is 1
                groupMembers = groupMembers[1] #AK-TODO buggy, fix it, now need to discard 1st dimension
                #if not groupMembers.any():
                #    raise RuntimeError('No cell is assigned to pilot group = ', g+1)
                #Compute processed pilot signal for all UEs that use these pilots, according to (3.5)

                Htemp = H[:,:,:,groupMembers,j]
                yp = np.sqrt(self.p)*self.tau_p*np.sum(Htemp,3) + \
                    np.sqrt(self.tau_p)*Np[:,:,:,j,g]
                #Go through all UEs
                for k in range(self.K):
                    #Compute the matrix that is inverted in the MMSE estimator
                    Rtemp = self.R[:,:,k,groupMembers,j]
                    PsiInv = (self.p*self.tau_p*np.sum(Rtemp,2) + self.eyeM)
                    for l in groupMembers:
                        #Compute MMSE estimate of channel between BS l and UE k in
                        #cell j using (3.9) in Theorem 3.1
                        #x = B/A is the solution to the equation xA = B. mrdivide in Matlab
                        RPsi = np.matmul(self.R[:,:,k,l,j], np.linalg.inv(PsiInv))
                        Hhat_MMSE[:,:,k,l,j] = np.sqrt(self.p)*np.matmul(RPsi,yp[:,:,k])
                        #Compute corresponding estimation error correlation matrix, using (3.11)
                        C_MMSE[:,:,k,l,j] = self.R[:,:,k,l,j] - self.p*self.tau_p*np.matmul(RPsi,self.R[:,:,k,l,j])
            return Hhat_MMSE, C_MMSE

    '''
    from functionComputeSINR_DL.m
    '''
    def compute_SINR_DL(self):
        fileName = 'D:/gits/lasse/software/mimo-toolbox/third_party/emil_massivemimobook/Code/H_Hhat_C.mat'
        H=read_matlab_array_from_mat(fileName, 'H')
        num_realizations = H.shape[1]
        Hhat=read_matlab_array_from_mat(fileName, 'Hhat')
        C=read_matlab_array_from_mat(fileName, 'C')

        #Store identity matrices of different sizes
        #eyeK = np.eye(self.K)
        eyeM = np.eye(self.M)
        
        #Compute sum of all estimation error correlation matrices at every BS
        C_totM = np.reshape(self.p*np.sum(np.sum(C,2),2),[self.M, self.M, self.L],'F')
        
        #Compute the prelog factor assuming only downlink transmission
        #prelogFactor = (self.tau_c-self.tau_p)/(self.tau_c)
        
        #Prepare to store simulation results for signal gains
        signal_MMMSE = np.zeros((self.K,self.L),dtype=np.complex64)
        
        #Prepare to store simulation results for interference powers
        interf_MMMSE = np.zeros((self.K,self.L,self.K,self.L),dtype=np.complex64)
        
        for n in range(num_realizations): #Go through all channel realizations
            #Go through all cells
            for j in range(self.L):
                #Extract channel realizations from all UEs to BS j
                Hallj = np.reshape(H[:,n,:,:,j],(self.M,self.K*self.L),'F')
        
                #Extract channel realizations from all UEs to BS j
                Hhatallj = np.reshape(Hhat[:,n,:,:,j],(self.M,self.K*self.L),'F')
        
                #Compute MR combining in (4.11)
                V_MR = Hhatallj[:,self.K*j:self.K*(j+1)]
                #print('AK', V_MR)
        
                #Compute M-MMSE combining in (4.7)
                #Backslash or matrix left division. If A is a square matrix, A\B is roughly the same as inv(A)*B, except it is computed in a different way
                tempM = self.p*(self.p*(np.matmul(Hhatallj,np.conj(np.transpose(Hhatallj)))) + C_totM[:,:,j]+eyeM)
                V_MMMSE = np.matmul(np.linalg.inv(tempM), V_MR)
                #Go through all UEs in cell j
                for k in range(self.K):
                    if np.linalg.norm(V_MR[:,k])>0:
                        #M-MMSE precoding
                        w = V_MMMSE[:,k]/np.linalg.norm(V_MMMSE[:,k]) #Extract precoding vector
                        w = np.reshape(w,(1,self.M)).conj() #Hermitian: make it a row vector and conjugate

                        #Compute realizations of the terms inside the expectations
                        #of the signal and interference terms of (7.2) and (7.3)
                        h_temp = H[:,n,k,j,j]
                        signal_MMMSE[k,j] = signal_MMMSE[k,j] + (np.inner(w,h_temp))/num_realizations
                        h_temp = np.matmul(w,Hallj)
                        h_temp = np.abs(np.array(h_temp))**2
                        h_temp = np.reshape(h_temp,(self.K,self.L),'F')

                        #print('AK', np.max(h_temp[:]))

                        interf_MMMSE[k,j,:,:] = interf_MMMSE[k,j,:,:] + h_temp/num_realizations
        #Compute the terms in (7.2)
        signal_MMMSE = np.abs(signal_MMMSE)**2
        #print('AK2', signal_MMMSE)
        #Compute the terms in (7.3)
        for j in range(self.L):
            for k in range(self.K):
                interf_MMMSE[k,j,k,j] = interf_MMMSE[k,j,k,j] - signal_MMMSE[k,j]
        return signal_MMMSE, interf_MMMSE

    '''
    %INPUT:
    %rho          = K x L matrix where element (k,j) is the downlink transmit
    %               power allocated to UE k in cell j
    %signal       = K x L matrix where element (k,j,n) is a_jk in (7.2)
    %interference = K x L x K x L matrix where (l,i,jk,n) is b_lijk in (7.3)
    %prelogFactor = Prelog factor
    %
    %OUTPUT:
    %SE = K x L matrix where element (k,j) is the downlink SE of UE k in cell j
    %     using the power allocation given as input
    '''
    def computeSE_DL_poweralloc(self, rho, signal, interference):
        #Compute the prelog factor assuming only downlink transmission
        prelogFactor = (self.tau_c-self.tau_p)/(self.tau_c)
        #Prepare to save results
        SE = np.zeros((self.K,self.L))
        # Go through all cells
        for j in range(self.L):
            #Go through all UEs in cell j
            for k in range(self.K):
                #Compute the SE in Theorem 4.6 using the formulation in (7.1)
                SE[k,j] = prelogFactor*np.log2(1+(rho[k,j]*signal[k,j]) / (sum(sum(rho*interference[:,:,k,j])) + 1))
        return SE


'''
Look at  
D:\gits\lasse\software\mimo-toolbox\third_party\emil_massivemimobook\Code\ak_functionExampleSetup.m
'''
def test_estimation():
    #saved with save -v6 R_channelGaindB R channelGaindB
    fileName = 'D:/gits/lasse/software/mimo-toolbox/third_party/emil_massivemimobook/Code/R_channelGaindB.mat'
    arrayName = 'channelGaindB'
    x=read_matlab_array_from_mat(fileName, arrayName)
    R=read_matlab_array_from_mat(fileName, 'R')
    #Matlab: ak_receivedSignal_y(10,5,20)
    print(x[9,4,9])
    print(R[3,4,3,0,1])

if __name__ == '__main__':
    #test_estimation()
    ce = ChannelEstimator()
    if False:
        num_realizations = 3
        H = ce.channel_realizations(num_realizations)
        Hhat_MMSE, C_MMSE = ce.mmse_estimation(None)
        print(Hhat_MMSE[0,0,0,0,0])
        print(C_MMSE[0,1,0,0,0])
    signal_MMMSE, interf_MMMSE = ce.compute_SINR_DL()
    print('aa', np.max(interf_MMMSE[:]))
    print('aa', np.min(interf_MMMSE[:]))

    SE = ce.computeSE_DL_poweralloc(ce.rhoEqual, signal_MMMSE, interf_MMMSE)
    print('SE=', SE)