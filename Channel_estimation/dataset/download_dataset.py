'''
Download dataset for channel estimation challenge
'''
import wget

URL = 'https://nextcloud.lasseufpa.org/s/7FrX2883E4yorbB/download?path=Raymobtime_s008/matrix_channel_data_s008&files=H_matrix_s008_10_users.hdf5'
FILE_NAME = wget.download(URL)

print(f'{FILE_NAME} was downloaded successfully!')
