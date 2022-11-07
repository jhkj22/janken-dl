from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers.core import Dense,Activation,Dropout,Flatten
import keras.callbacks
from keras.utils import np_utils
import os

def make_model():
    model=Sequential()

    model.add(Conv2D(8, (3, 3), padding='same', input_shape=(32, 32, 1)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(8))
    model.add(Activation('relu'))
    model.add(Dense(2,activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

model = make_model()



classes = ['circle', 'plus']
nb_classes = len(classes)
train_data_dir = '../../ox/train'
validation_data_dir = '../../ox/val'
f_model = '../../ox/model'
nb_train_samples = 4000
nb_validation_samples = 1000
img_width, img_height = 32, 32

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)
validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    color_mode='grayscale',
    classes=classes,
    class_mode='categorical',
    batch_size=16
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    color_mode='grayscale',
    classes=classes,
    class_mode='categorical',
    batch_size=16
)


cp = keras.callbacks.ModelCheckpoint(
    filepath = os.path.join(f_model,
        'model{epoch:02d}-vloss{val_loss:.2f}-vacc{val_acc:.2f}.hdf5'),
    monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

history = model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    nb_epoch=2,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples,
    callbacks=[cp])





