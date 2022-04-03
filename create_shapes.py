from ast import List
from random import randint
from matplotlib.pyplot import draw
import PIL
from PIL import Image
from PIL import ImageChops
import scipy
import skimage.draw
import numpy as np
import math
import backgrounds
import cv2
from scipy import io

def get_train_data(path: str, classes_to_keep: List):
    silhouettes = io.loadmat(path)
    train_labels = np.ndarray.flatten(np.array(silhouettes['train_labels']))
    train_data = silhouettes['train_data']
    data_to_use = [(np.reshape(np.array(x, dtype=np.float32), (16,16)), classes_to_keep.index(train_labels[i])) for (i, x) in enumerate(train_data) if train_labels[i] in classes_to_keep]
    data = np.array([paste_black(backgrounds.rand_background((20,20)),x[0]) for x in data_to_use])
    labels = np.array([x[1] for x in data_to_use])

    val_labels =  np.ndarray.flatten(np.array(silhouettes['val_labels']))
    val_data = silhouettes['val_data']
    val_to_use = [(np.reshape(np.array(x, dtype=np.float32), (16,16)), classes_to_keep.index(val_labels[i])) for (i, x) in enumerate(val_data) if val_labels[i] in classes_to_keep]
    val_data = np.array([paste_black(backgrounds.rand_background((20,20)),x[0]) for x in val_to_use])
    val_labels = np.array([x[1] for x in val_to_use])

    return data, labels, val_data, val_labels

def paste_black(target, shape):
    shape_image = Image.fromarray(shape.astype('uint8')*255)
    backround = Image.fromarray(target.astype('uint8')*255)

    blank = Image.new("L", backround.size, "white")
    shape_w, shape_h = shape_image.size
    back_w, back_h = backround.size

    top_left = (randint(0, back_w - shape_w), randint(0, back_h - shape_h))
    bottom_right = (top_left[0] + shape_w, top_left[1] + shape_h)
    box = (top_left[0], top_left[1], bottom_right[0], bottom_right[1])

    Image.Image.paste(blank, shape_image, box)
    res = ImageChops.multiply(backround, blank)
    res = np.array(res, dtype=np.float) / 255
    res = np.reshape(res, (res.shape[0], res.shape[1], 1))
    return res
    