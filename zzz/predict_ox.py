from keras.preprocessing import image
from keras.models import load_model
from keras.models import Model
import numpy as np
from keras.applications.inception_v3 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt

f_model = '../../ox/model'
model = load_model(f_model + '/model.hdf5')

"""
w = model.get_weights()[0]
print(w.shape)
w = np.transpose(w, (2, 3, 0, 1))[0]

for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(w[i], cmap='gray')
    plt.xticks([]); plt.yticks([])
plt.show()
"""


#for layer in model.layers:
#    print(layer.name)

layer_name = 'max_pooling2d_1'
model = Model(
    inputs=model.input,
    outputs=model.get_layer(layer_name).output
)


classes = ['circle', 'plus']

img_path = '../../ox/val/plus/4.jpg'
#img_path = '../../ox/val/circle/4.jpg'

img = image.load_img(img_path, color_mode='grayscale')
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
pred = model.predict(x)[0]

print(pred.shape)

for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(pred[:,:,i], cmap='gray')
    plt.xticks([]); plt.yticks([])
#plt.savefig('plus.png')
plt.show()

#print([int(o) for o in pred * 100])




