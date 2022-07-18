#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-07-18 15:38:14
# @Author  : kentsin (kentsin@gmail.com)
# @Link    : link
# @Version : 0.0.1

import os

# https://www.youtube.com/watch?v=ADV-AjAXHdc&list=PL2VXyKi-KpYuTAZz__9KVl1jQz74bDG7i&index=5
# https://github.com/wjbmattingly/ocr_python_textbook

import cv2
import numpy as np

import pdf2image
poppler_path = r"E:\Program Files (x86)\poppler-22.04.0\Library\bin"

import matplotlib.pyplot as plt

DPI = 300


def load_images(path, dpi=DPI):
    images = []
    images.extend(list(map(lambda image: cv2.cvtColor(np.asarray(image), code=cv2.COLOR_RGB2BGR),
                  pdf2image.convert_from_path(path, dpi=dpi, poppler_path=poppler_path), )))
    return images

# https://stackoverflow.com/questions/28816046/
# displaying-different-images-with-actual-size-in-matplotlib-subplot
def display(image):
    dpi = 80

    height, width = image.shape[:2]

    # what size does the figurer need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # hide spines, ticks, etc.
    ax.axis('off')

    # display the image
    ax.imshow(image, cmap='gray')

    plt.show()

