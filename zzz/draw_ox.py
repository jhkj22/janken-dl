from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

def ellip_rand():
    im = Image.new('RGB', (32, 32), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    s = np.random.rand(4)
    x, y = s[:2]* 10 + 11
    rx, ry = s[2:] * 10 + 3
    draw.ellipse((x - rx, y - ry, x + rx, y + ry), outline=(0xff, 0xff, 0xff))
    return im

def x_rand():
    im = Image.new('RGB', (32, 32), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    s = np.random.rand(2)
    tt1 = s[0] * np.pi
    tt2 = tt1 + np.pi / 6 + s[1] * 2 * np.pi / 3
    s = np.random.rand(4)
    r = s * 10 + 3
    s = np.random.rand(2)
    x, y = s * 16 + 8
    line1 = (x - np.cos(tt1) * r[0], y - np.sin(tt1) * r[0],
        x + np.cos(tt1) * r[1], y + np.sin(tt1) * r[1])
    line2 = (x - np.cos(tt2) * r[2], y - np.sin(tt2) * r[2],
        x + np.cos(tt2) * r[3], y + np.sin(tt2) * r[3])
    draw.line(line1, fill=(0xff, 0xff, 0xff), width=1)
    draw.line(line2, fill=(0xff, 0xff, 0xff), width=1)
    return im

def circle():
    im = Image.new('RGB', (32, 32), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    x, y = 16, 16
    rx, ry = 8, 8
    draw.ellipse((x - rx, y - ry, x + rx, y + ry), outline=(0xff, 0xff, 0xff))
    return im
def plus():
    im = Image.new('RGB', (32, 32), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    draw.line((16, 8, 16, 24), fill=(0xff, 0xff, 0xff), width=1)
    draw.line((8, 16, 24, 16), fill=(0xff, 0xff, 0xff), width=1)
    return im

import os
def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

#base_d = '/home/hayato/etc/ox/'
base_d = '../../ox/'

make_dir(base_d + 'train')
make_dir(base_d + 'train/plus')
make_dir(base_d + 'train/circle')
make_dir(base_d + 'val')
make_dir(base_d + 'val/plus')
make_dir(base_d + 'val/circle')

for i in range(200):
    im = ellip_rand()
    im.save(base_d + 'val/circle/' + str(i) + '.jpg')
    im = x_rand()
    im.save(base_d + 'val/plus/' + str(i) + '.jpg')






