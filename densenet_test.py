from keras.applications.densenet import DenseNet121, preprocess_input
from keras.layers import MaxPooling2D, Convolution2D, GlobalAveragePooling2D
from keras.preprocessing import image
from keras import models
import numpy as np
import matplotlib.pyplot as plt

model = DenseNet121(weights='imagenet', include_top=False)

name = 'conv2_block2_0_relu'

layers = []

for layer in model.layers:
    if layer.name == name:
        layers.append(layer)
        break

layer_outputs = [layer.output for layer in layers]
x = layer_outputs[0]


activation_model = models.Model(inputs=model.input, outputs=[x])


img = image.load_img('../res/hand_rock_palm.png', target_size=(100, 100))
input_img = img
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

out = activation_model.predict(img)

out = np.transpose(out[0], (2, 0, 1))
print(out.shape)


plt.axis('off')
for i in range(8 * 12):
    plt.subplot(8, 12, i + 1)
    plt.imshow(out[i], cmap='gray')
    plt.axis('off')


plt.show()

