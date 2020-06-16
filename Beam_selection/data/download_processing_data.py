'''
Download dataset for channel estimation challenge
'''
import wget

print('LIDAR files \n')
URL_LIDAR = 'https://nextcloud.lasseufpa.org/s/X4AyXLYKrdfp7np/download'
wget.download(URL_LIDAR)
print('\nCoords files \n')
URL_COORD = 'https://nextcloud.lasseufpa.org/s/Q85T6JqEm4GRMJt/download'
wget.download(URL_COORD)
print('\nImages files \n')
URL_IMAGE = 'https://nextcloud.lasseufpa.org/s/WkwQ3CqNKeqmNTq/download'
wget.download(URL_IMAGE)
print('\nChannels files \n')
URL_CHANNEL = 'https://nextcloud.lasseufpa.org/s/a7wNfzQ9rs27zMb/download'
wget.download(URL_CHANNEL)


print('\nFiles was downloaded successfully!')

#URL = ''
#TEST_FILE_NAME = wget.download(URL)

#print(f'{TEST_FILE_NAME} was downloaded successfully!')
