from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import keras.callbacks
import os.path

classes = ['apple_pie', 'french_toast', 'hot_dog', 'pizza', 'sashimi']
nb_classes = len(classes)
train_data_dir = '../../food5/train'
validation_data_dir = '../../food5/val'
f_model = '../../food5/model'
nb_train_samples = 4000
nb_validation_samples = 1000 
img_width, img_height = 299, 299

train_datagen = ImageDataGenerator(rescale=1.0 / 255,
                                   zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

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



model = load_model(f_model + '/model_01.hdf5')

cp = keras.callbacks.ModelCheckpoint(
    filepath = os.path.join(f_model,
        'model{epoch:02d}-vloss{val_loss:.2f}-vacc{val_acc:.2f}.hdf5'),
    monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

history = model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    nb_epoch=4,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples,
    callbacks=[cp])





