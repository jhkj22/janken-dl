from keras.applications.densenet import DenseNet121
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
import keras.callbacks
import os.path
import dat
import numpy as np

classes = ['rock', 'scissors', 'paper', 'others']
nb_classes = len(classes)
base_dir = '..'
train_data_dir = base_dir + '/train'
validation_data_dir = base_dir + '/val'
f_model = base_dir + '/model'

if not os.path.exists(base_dir + '/model'):
    os.mkdir(base_dir + '/model')

img_width, img_height = 100, 100

train_datagen = ImageDataGenerator(
    zoom_range=0.2,
    rotation_range=20,
    height_shift_range=0.2,
    width_shift_range=0.2,
    shear_range=30,
    channel_shift_range=100,
    horizontal_flip=True,
    rescale=1.0 / 255
)
validation_datagen = ImageDataGenerator(
    zoom_range=0.2,
    rotation_range=20,
    height_shift_range=0.2,
    width_shift_range=0.2,
    shear_range=30,
    channel_shift_range=100,
    horizontal_flip=True,
    rescale=1.0 / 255
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    color_mode='rgb',
    classes=classes,
    class_mode='categorical',
    batch_size=10)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    color_mode='rgb',
    classes=classes,
    class_mode='categorical',
    batch_size=10)



base_model = DenseNet121(weights='imagenet', include_top=False)

name = 'pool4_relu'

base_output = None

for layer in base_model.layers:
    if layer.name == name:
        base_output = layer.output
        break

x = base_output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(nb_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)



for layer in model.layers[:295]:
   layer.trainable = False
for layer in model.layers[295:]:
   layer.trainable = True

model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cp = keras.callbacks.ModelCheckpoint(
    filepath = os.path.join(f_model,
        'model-{epoch:02d}-vl{val_loss:.2f}-va{val_acc:.2f}-l{loss:.2f}-a{acc:.2f}.hdf5'),
    monitor='val_loss', verbose=1, save_best_only=True, mode='auto'
)

hist = model.fit_generator(
    train_generator,
    steps_per_epoch=50,
    epochs=80,
    validation_data=validation_generator,
    validation_steps=100,
    callbacks=[cp]
)

h = hist.history
l = np.array([h['acc'], h['val_acc'], h['loss'], h['val_loss']])
dat.write('history', l)





