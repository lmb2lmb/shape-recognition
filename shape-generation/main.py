from cmath import pi
import math
from pickletools import float8
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pyrsistent import v
import scipy

image_shape = image_height, image_width, image_channels = 1500, 2000, 3
def circle(image, center_y , center_x, radius, x, y, color):
    image = np.copy(image)
    circle = np.sqrt(((x - center_x) ** 2) + ((y - center_y) **2))
    C = circle < radius
    image[C] = color
    return image

def rectangle(image, center_y, center_x, height, width, x, y, color):
    image = np.copy(image)
    Width = np.abs(x - center_x) < (width/2)
    Height = np.abs(y - center_y) < (height/2)

    A = np.logical_and(Width, Height)
    image[A] = color
    return image




image = np.ones(image_shape)


y_coords = np.arange(0, image_height, 1)
x_coords = np.arange(0, image_width, 1)
z_coords = np.arange(0, image_channels, 1)

x, y= np.meshgrid(x_coords, y_coords)


image = np.sin(x*pi*2 / image_width)
cv2.imshow('str(i)', image)
cv2.waitKey(0)


for i in range(100):
   color = np.random.rand(3)
   C = circle(image, np.random.randint(0, image_height), np.random.randint(0, image_width), 100, x, y, 
   color)
   C = rectangle(C, np.random.randint(0, image_height), np.random.randint(0, image_width), 100, 100, x, y,color)
   image = C.astype(np.float)
   kernel = cv2.getGaussianKernel(5, i)
   image = cv2.filter2D(image, -1, kernel)
   cv2.imshow(str(i), C)
   cv2.waitKey(1)
# and finally destroy/close all open windows
cv2.destroyAllWindows()


def circle(image, center_y , center_x, radius):
    for row in range(np.shape(image)[0]):
        for col in range(np.shape(image)[1]):
            distance = math.sqrt(math.pow(row - center_y, 2) + math.pow(col - center_x, 2))
            if distance < radius:
                image[row, col] = 0

