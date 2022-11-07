from keras.applications.inception_v3 import InceptionV3

model = InceptionV3(weights='imagenet', include_top=False)

#from keras.utils import plot_model
#plot_model(model, to_file='inception_v3.png')

for i, layer in enumerate(model.layers):
    print(i, layer.name)

