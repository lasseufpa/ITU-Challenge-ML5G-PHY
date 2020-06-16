'''
Download dataset for channel estimation challenge
'''
import wget

URL = 'https://nextcloud.lasseufpa.org/s/zqeeyDtbbPt7nKz/download'
TRAIN_FILE_NAME = wget.download(URL)

print('\n File was downloaded successfully!')

#URL = ''
#TEST_FILE_NAME = wget.download(URL)

#print(f'{TEST_FILE_NAME} was downloaded successfully!')
