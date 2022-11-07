import numpy as np
import matplotlib.pyplot as plt

from keras.applications.mobilenetv2 import MobileNetV2
from keras.preprocessing import image
from keras.applications.mobilenetv2 import preprocess_input, decode_predictions

model = MobileNetV2(weights='imagenet', input_shape=(96, 96, 3))
img_path = '../res/penguin.jpg'
img = image.load_img(img_path, target_size=(96, 96))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print(decode_predictions(preds, top=3)[0], flush=True)


"""
from keras.applications.densenet import DenseNet121
from keras.preprocessing import image
from keras.applications.densenet import preprocess_input, decode_predictions

model = DenseNet121(weights='imagenet', include_top=False)

for i, layer in enumerate(model.layers):
   print(i, layer.name)
#313~
#from keras.utils import plot_model
#plot_model(model, to_file='dense121.png')
"""

"""
from keras import models

layer_outputs = [layer.output for layer in model.layers[426:]]
model = models.Model(inputs=model.input, outputs=layer_outputs)


img_path = '../res/penguin.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
"""


