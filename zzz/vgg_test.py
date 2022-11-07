from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import MaxPooling2D, Convolution2D
from keras.preprocessing import image
from keras import models
import numpy as np
import matplotlib.pyplot as plt

model = VGG16(weights='imagenet')

layers = model.layers[1:19]
layer_outputs = [layer.output for layer in layers]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)



img = image.load_img('../res/hand_paper.png', target_size=(224, 224))
input_img = img
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

activations = activation_model.predict(img)

conv_activations = []
for layer, activation in zip(layers, activations):
    if isinstance(layer, Convolution2D):
        conv_activations.append([layer.name, activation])

layer = conv_activations[3]
name, out = layer
out = np.transpose(layer[1][0], (2, 0, 1))
print(name, out.shape)

plt.subplot(4, 7, 1)
plt.imshow(input_img)
plt.axis('off')
for i in range(1, 4 * 7):
    plt.subplot(4, 7, i + 1)
    plt.imshow(out[i + 0], cmap='gray')
    plt.axis('off')


plt.show()

