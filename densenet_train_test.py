from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D
from keras.models import Model

base_model = DenseNet121(weights='imagenet', include_top=False)

name = 'conv4_block13_0_relu'

base_output = None

for layer in base_model.layers:
    if layer.name == name:
        base_output = layer.output
        break

x = base_output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

#for i, layer in enumerate(model.layers):
#    print(i, layer.name)
w = model.layers[228].get_weights()
print(len(w), w[0].shape)

















