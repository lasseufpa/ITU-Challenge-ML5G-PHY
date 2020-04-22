'''
Preprocessing channels
'''

from scipy.io import loadmat, savemat
import numpy as np
import h5py

file_name = '../dataset/H_matrix_s008_10_users.hdf5'
open_file = h5py.File(file_name, 'r')

img_part = open_file['H_matrix_img_part']
real_part = open_file['H_matrix_real_part']
Ht = real_part[...] + img_part[...] * 1j



# permute dimensions before reshape: scenes before episodes
# found out np.moveaxis as alternative to permute in matlab
Ht = Ht[~np.isnan(Ht).any(axis=3)]
Ht = Ht.reshape((-1,1,10,64,64))

n_episodes, n_scenes, n_receivers, n_r, n_t = Ht.shape

Ht = np.moveaxis(Ht, 1, 0)

reshape_dim = n_episodes * n_scenes * n_receivers

Harray = np.reshape(Ht, (reshape_dim, n_r, n_t))
Hvirtual = np.zeros(Harray.shape, dtype='complex128')
scaling_factor = 1 / np.sqrt(n_r * n_t)

for i in range(reshape_dim):
    m = np.squeeze(Harray[i,:,:])
    Hvirtual[i,:,:] = scaling_factor * np.fft.fft2(m)

savemat('../datasets/rosslyn60Ghz.mat', {'Harray': Harray, 'Hvirtual':Hvirtual})
