
import os
# from tensorflow.keras.utils import get_file
# from torchvision.datasets.utils import download_and_extract_archive
import requests

# specify two useful folders
DATA_PATH = './data'
RESULT_PATH = './result'

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

# download the data first
'''
filename = get_file(fname=os.path.abspath(os.path.join(DATA_PATH,'ultrasound_50frames.h5')),
                    origin='https://weisslab.cs.ucl.ac.uk/WEISSTeaching/datasets/raw/fetal/ultrasound_50frames.h5')
'''

print('Downloading and extracting data...')
url = 'https://weisslab.cs.ucl.ac.uk/WEISSTeaching/datasets/raw/fetal/ultrasound_50frames.h5' 
r = requests.get(url,allow_redirects=True)
filename = os.path.join(DATA_PATH,'ultrasound_50frames.h5')
_ = open(filename,'wb').write(r.content)
print('Done.')

if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)

print('Image and label data downloaded: <%s>.' % filename)
print('Result directory created: <%s>.' % os.path.abspath(RESULT_PATH))
