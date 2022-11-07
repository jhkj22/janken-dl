from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense
from keras import optimizers


classes = ['rock', 'scissors', 'paper', 'others']
nb_classes = len(classes)
train_data_dir = '../../dataset/train'
validation_data_dir = '../../dataset/val'
nb_train_samples = 112
nb_validation_samples = 32
img_width, img_height = 200, 200

train_datagen = ImageDataGenerator(rescale=1.0 / 255, zoom_range=0.2, horizontal_flip=True)
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

input_tensor = Input(shape=(img_width, img_height, 3))
ResNet50 = ResNet50(include_top=False, weights='imagenet',input_tensor=input_tensor)

top_model = Sequential()
top_model.add(Flatten(input_shape=ResNet50.output_shape[1:]))
top_model.add(Dense(nb_classes, activation='softmax'))

model = Model(input=ResNet50.input, output=top_model(ResNet50.output))

model.compile(loss='categorical_crossentropy',
optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
metrics=['accuracy'])

history = model.fit_generator(
train_generator,
samples_per_epoch=nb_train_samples,
nb_epoch=5,
validation_data=validation_generator,
nb_val_samples=nb_validation_samples)


