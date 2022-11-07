from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.layers import MaxPooling2D, Convolution2D, GlobalAveragePooling2D
from keras.preprocessing import image
from keras import models
import numpy as np
import matplotlib.pyplot as plt

model = MobileNetV2(weights='imagenet', input_shape=(96, 96, 3), include_top=True)

name = 'block_3_project'

layers = []

for layer in model.layers:
    if layer.name == name:
        layers.append(layer)
        break

layer_outputs = [layer.output for layer in layers]
x = layer_outputs[0]
#x = GlobalAveragePooling2D()(x)


activation_model = models.Model(inputs=model.input, outputs=[x])


img = image.load_img('../res/hand_scissors_palm.png', target_size=(96, 96))
input_img = img
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

out = activation_model.predict(img)

out = np.transpose(out[0], (2, 0, 1))
print(out.shape)


plt.subplot(4, 7, 1)
plt.imshow(input_img)
plt.axis('off')

for i in range(1, 4 * 7):
    plt.subplot(4, 7, i + 1)
    plt.imshow(out[i + 4 * 7 * 0], cmap='gray')
    plt.axis('off')


plt.show()

