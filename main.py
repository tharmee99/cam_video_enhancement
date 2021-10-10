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

########################################################################################################################

IMG_DIR = os.path.join("res", "small")
TMP_DIR = "temp"

########################################################################################################################

def distort_images(img_list):
    for img in img_list:
        plt.imshow(img)
        plt.show()


def crop_images():
    pass


def import_images(directory=IMG_DIR, force_calc=False):
    """
    Import images from a directory and form an array.

    @param directory: The directory to import the images from
    @param force_calc: Re-build the array even if a numpy save file exists
    @return: An array (numpy.ndarray) of all the images
    """

    img_np_arr = None
    file_name = "img_np_arr.npy"

    # If np array save file exists just read that in, else build array by reading images
    if os.path.exists(os.path.join(TMP_DIR, file_name)) and not force_calc:
        img_np_arr = np.load(os.path.join(TMP_DIR, file_name))
    else:
        all_imgs = os.listdir(directory)

        img_arr = []

        # for img in all_imgs:
        for img in tqdm(all_imgs, desc="Importing Images", file=sys.stdout):
            img_arr.append(cv2.imread(os.path.join(directory, img)))

        img_np_arr = np.asarray(img_arr)

        np.save(os.path.join(TMP_DIR, file_name), img_np_arr)

    return img_np_arr


if __name__ == '__main__':
    imgs = import_images()

    print(type(imgs))

    distort_images(imgs)