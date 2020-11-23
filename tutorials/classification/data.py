
import os
from tensorflow.keras.utils import get_file


# specify two useful folders
DATA_PATH = './data'
RESULT_PATH = './result'


# download the data first
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
filename = get_file(fname=os.path.abspath(os.path.join(DATA_PATH,'ultrasound_50frames.h5')),
                    origin='https://weisslab.cs.ucl.ac.uk/WEISSTeaching/datasets/raw/fetal/ultrasound_50frames.h5')


if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)

print('Image and label data downloaded: <%s>.' % filename)
print('Result directory created: <%s>.' % os.path.abspath(RESULT_PATH))
