'''
Download dataset for channel estimation challenge
'''
import wget

URL = 'https://nextcloud.lasseufpa.org/remote.php/webdav/5GM/Channel_estimation/train.hdf5'
TRAIN_FILE_NAME = wget.download(URL)

print(f'{TRAIN_FILE_NAME} was downloaded successfully!')

URL = 'https://nextcloud.lasseufpa.org/remote.php/webdav/5GM/Channel_estimation/test.hdf5'
TEST_FILE_NAME = wget.download(URL)

print(f'{TEST_FILE_NAME} was downloaded successfully!')
