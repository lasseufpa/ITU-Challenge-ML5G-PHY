'''
Download dataset for channel estimation challenge
'''
import wget

URL = 'https://nextcloud.lasseufpa.org/s/EMZP7t97L8CWNJM'
TRAIN_FILE_NAME = wget.download(URL)

print(f'{TRAIN_FILE_NAME} was downloaded successfully!')

#URL = ''
#TEST_FILE_NAME = wget.download(URL)

#print(f'{TEST_FILE_NAME} was downloaded successfully!')
