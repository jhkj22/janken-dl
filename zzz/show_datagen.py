from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os.path
import glob
import matplotlib.pyplot as plt
import shutil
import matplotlib.gridspec as gridspec

def draw_images(datagen, x):
    temp_dir = "temp"
    os.mkdir(temp_dir)

    g = datagen.flow(x, batch_size=1, save_to_dir=temp_dir, save_prefix='img', save_format='jpg')
    for i in range(9):
        batch = g.next()

    images = glob.glob(os.path.join(temp_dir, "*.jpg"))
    fig = plt.figure()
    gs = gridspec.GridSpec(3, 3)
    gs.update(wspace=0.1, hspace=0.1)
    for i in range(9):
        img = image.load_img(images[i])
        plt.subplot(gs[i])
        plt.imshow(img, aspect='auto')
        plt.axis("off")

    shutil.rmtree(temp_dir)
    plt.show()

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


if __name__ == '__main__':
    img = image.load_img('../../../hand4/clip/paper/1551362942.4915602.jpg')
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    draw_images(train_datagen, x)




