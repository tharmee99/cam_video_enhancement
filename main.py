import os
import sys
from PIL import Image
import cv2
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
import pandas as pd
import random

########################################################################################################################

IMG_DIR = os.path.join("res", "small")
TMP_DIR = "temp"

########################################################################################################################

def distort_images(img_list):
    for img in img_list:
        plt.imshow(img)
        plt.show()
        pass


def crop_img(img, size=(100, 100)):
    start_x = random.randrange(0, img.shape[0]-size[0])
    end_x = start_x + size[0]
    start_y = random.randrange(0, img.shape[1]-size[1])
    end_y = start_y + size[1]
    return img[start_x:end_x, start_y:end_y]


def is_greyscale(img):
    w = img.shape[0]
    h = img.shape[1]
    for i in range(w):
        for j in range(h):
            r, g, b = img[i, j]
            if r != g != b:
                return False
    return True


def import_images(directory=IMG_DIR, force_calc=False):
    """
    Import images from a directory and form an array.

    @param directory:   The directory to import the images from
    @param force_calc:  If true, the array is re-built even if a numpy save file exists. Useful if pre-processing is
                        changed
    @return:            (numpy.ndarray) An array of all the images
    """

    img_np_arr = None
    file_name = "img_np_arr.npy"

    # If np array save file exists just read that in, else build array by reading images
    if os.path.exists(os.path.join(TMP_DIR, file_name)) and not force_calc:
        print("Importing images from npy file...")
        img_np_arr = np.load(os.path.join(TMP_DIR, file_name))
        print("...done\n")
    else:
        all_imgs = os.listdir(directory)

        img_arr = []

        for img in tqdm(all_imgs, desc="Importing images", file=sys.stdout):
            img_raw = cv2.imread(os.path.join(directory, img))
            img_cropped = crop_img(img_raw)
            if not is_greyscale(img_cropped):
                img_arr.append(img_cropped)

        img_np_arr = np.asarray(img_arr)

        np.save(os.path.join(TMP_DIR, file_name), img_np_arr)

    return img_np_arr


if __name__ == '__main__':
    imgs = import_images(force_calc=True)

    distort_images(imgs)