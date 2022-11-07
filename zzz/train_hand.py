from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import numpy as np
import keras.callbacks
import os.path

classes = ['rock', 'scissors', 'paper', 'others']
nb_classes = len(classes)

train_data_dir = '../../hand4/train'
f_model = '../../hand4/model'

nb_train_samples = 4000
img_width, img_height = 150, 150

train_datagen = ImageDataGenerator(
    zoom_range=0.2,
    rotation_range=20,
    height_shift_range=0.2,
    width_shift_range=0.2,
    shear_range=5,
    channel_shift_range=5.,
    brightness_range=[0.3, 1.0],
    horizontal_flip=True,
    rescale=1.0 / 255
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    color_mode='rgb',
    classes=classes,
    class_mode='categorical',
    batch_size=16)



base_model = InceptionV3(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(nb_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy')

cp = keras.callbacks.ModelCheckpoint(
    filepath = os.path.join(f_model,
        'model{epoch:02d}-loss{loss:.2f}.hdf5'),
    monitor='loss', verbose=1, save_best_only=True, mode='auto')

history = model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    nb_epoch=2,
    callbacks=[cp])





