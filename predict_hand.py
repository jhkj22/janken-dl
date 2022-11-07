from keras.preprocessing import image
from keras.models import load_model
import numpy as np
from keras.applications.densenet import preprocess_input, decode_predictions

base_file = '../../hand4/'
f_model = base_file + 'model/'
model = load_model(f_model + 'model.hdf5')

"""
for i, layer in enumerate(model.layers):
   print(i, layer.name)

from keras.utils import plot_model
plot_model(model, to_file='model.png')
"""

classes = ['rock', 'scissors', 'paper', 'others']

s = 100

import glob

gl = glob.glob(base_file + 'val/scissors/*.jpg')

x = []

for file in gl:
    img = image.load_img(file, target_size=(s, s))
    img = image.img_to_array(img)
    x.append(img)
x = np.array(x)
x = preprocess_input(x)
preds = model.predict(x)

for file, pred in zip(gl, preds):
    print(','.join(['{:2d}'.format(int(o)) for o in pred * 100]), file)



