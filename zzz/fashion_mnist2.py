import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D

import numpy as np

fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

img_rows, img_cols = 28, 28

x_train_r = x_train[:,:,::-1]

x_train_dup = np.concatenate((x_train, x_train_r))
y_train_dup = np.concatenate((y_train, np.copy(y_train)))

x_train_dup = x_train_dup / 255.0
x_test = x_test / 255.0

if keras.backend.image_data_format() == 'channels_first':
    x_train_dup = x_train_dup.reshape(x_train_dup.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train_dup = x_train_dup.reshape(x_train_dup.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

model = Sequential()
model.add(ZeroPadding2D(padding=(2,2), input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(class_names), activation='softmax'))

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_dup, y_train_dup, epochs=5)

test_loss, test_acc = model.evaluate(x_test, y_test)
print("test loss %s, test accuracy %s" % (test_loss, test_acc))


