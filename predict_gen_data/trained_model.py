from keras.preprocessing import image
from keras.models import load_model
from keras.models import Model
import numpy as np
from keras.applications.densenet import preprocess_input, decode_predictions

class trained_model:
    def __init__(self):
        f_model = '../../../hand4/model'
        #f_model = '/home/hayato/etc/hand4/model'
        self.model = load_model(f_model + '/model.hdf5')
        self.model._make_predict_function()
    def predict(self, img):
        classes = ['rock', 'scissors', 'paper', 'others']
        size = np.ones(2) * 150
        if np.any(img.shape[:2] != size):
            return 'unsupported img size.'
        x = np.expand_dims(img, axis=0)
        x = preprocess_input(x)
        pred = self.model.predict(x)[0]
        pred = np.array([int(o) for o in pred * 100])
        i = np.argmax(pred)
        return '{:}: {:}'.format(classes[i], pred[i])




