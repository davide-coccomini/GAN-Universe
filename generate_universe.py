import sys
import glob
from PIL import Image
import random
from itertools import islice
from random import randint
import cv2
import numpy as np
import os


def random_chunk(li, min_chunk=15, max_chunk=500):
    it = iter(li)
    while True:
        nxt = list(islice(it, randint(min_chunk, max_chunk)))
        if nxt:
            yield nxt
        else:
            break


def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):

    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)


def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)


def concat_tile_resize(im_list_2d, interpolation=cv2.INTER_CUBIC):
    im_list_v = [hconcat_resize_min(
        im_list_h, interpolation=cv2.INTER_CUBIC) for im_list_h in im_list_2d]
    return vconcat_resize_min(im_list_v, interpolation=cv2.INTER_CUBIC)


def rotate_image(image):
    rotation_random = random.randrange(0, 4)
    if rotation_random == 1:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_random == 2:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif rotation_random == 3:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


folder = sys.argv[1]

filenames = os.listdir(folder)

random_sizes = [420, 410, 400, 390, 380, 370]

images = []
for i in range(0, 2):
    for name in filenames:
        image = cv2.imread(os.path.join(folder, name))
        image = rotate_image(image)
        random_size = random.choice(random_sizes)
        image = cv2.resize(image, (random_size, random_size),
                           interpolation=cv2.INTER_AREA)
        images.append(image)

for i in range(0, 16000):
    empty_tile = cv2.imread(os.path.join(
        "empty_tiles", str(random.randrange(0, 8)))+".png")

    random_size = random.choice(random_sizes)
    image = rotate_image(image)
    image = cv2.resize(empty_tile, (random_size, random_size),
                       interpolation=cv2.INTER_AREA)
    images.append(image)

random.shuffle(images)

chunks = list(random_chunk(images))
im_tile_resize = concat_tile_resize(chunks)

cv2.imwrite('universe.jpg', im_tile_resize)
