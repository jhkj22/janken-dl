import numpy as np
import matplotlib.pyplot as plt
from keras.applications.mobilenetv2 import MobileNetV2
from keras.preprocessing import image
from keras.applications.mobilenetv2 import preprocess_input

model = MobileNetV2(weights='imagenet', input_shape=(96, 96, 3))

name = 'block_7_depthwise'

layer_i = 0

for i, layer in enumerate(model.layers):
    if layer.name == name:
        break

w = model.layers[i].get_weights()[0]


#from keras.utils import plot_model
#plot_model(model, to_file='MNet_96.png')





