from keras.preprocessing import image
from keras.models import load_model
from keras.models import Model
import numpy as np
from keras.applications.inception_v3 import preprocess_input, decode_predictions

f_model = '../../food5/model'
model = load_model(f_model + '/model02-vloss0.42-vacc0.86.hdf5')

#input_tensor = Input(shape=(200, 200, 3))
#model.input_shape = input_tensor

img_path = '../res/french-toast.jpg'

s = 299

import time
t1 = time.time()

for i in range(100):
    img = image.load_img(img_path, target_size=(s, s))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    pred = model.predict(x)[0]
    print([int(o) for o in pred * 100])

t2 = time.time()

print(t2 - t1)



