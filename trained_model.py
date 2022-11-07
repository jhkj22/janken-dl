from keras.preprocessing import image
from keras.models import load_model
from keras.models import Model
import numpy as np
from keras.applications.densenet import preprocess_input
#from keras.applications.mobilenet_v2 import preprocess_input
import cv2

class trained_model:
    def __init__(self):
        #f_model = '/home/hayato/etc/hand4/model'
        f_model = '../../hand4/model'
        self.model = load_model(f_model + '/model.hdf5')
        self.model._make_predict_function()
    def predict(self, img):
        s = 100
        img = cv2.resize(img, dsize=(s, s))
        x = np.expand_dims(img, axis=0)
        x = preprocess_input(x)
        pred = self.model.predict(x)[0]
        i = np.argmax(pred)
        return [i, pred[i]]




