from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import keras.callbacks
import os.path

classes = ['rock', 'scissors', 'paper', 'others']
nb_classes = len(classes)
base_dir = '../../hand4'
train_data_dir = base_dir + '/train'
validation_data_dir = base_dir + '/val'
f_model = base_dir + '/model'
nb_train_samples = 4000
nb_validation_samples = 1000 
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
validation_datagen = ImageDataGenerator(
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

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
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
    loss='categorical_crossentropy',
    metrics=['accuracy'])

cp = keras.callbacks.ModelCheckpoint(
    filepath = os.path.join(f_model,
        'model{epoch:02d}-vloss{val_loss:.2f}-vacc{val_acc:.2f}.hdf5'),
    monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

history = model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    nb_epoch=5,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples,
    callbacks=[cp])





