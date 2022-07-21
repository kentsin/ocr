#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-07-21 09:30:56
# @Author  : kentsin (kentsin@gmail.com)
# @Link    : link
# @Version : 0.0.1

import os
import ntpath
import glob

import pytesseract
import numpy as np
import cv2

import matplotlib.pyplot as plt

import pdf2image
poppler_path = r"E:\Program Files (x86)\poppler-22.04.0\Library\bin"

DPI = 300


def load_images(path, dpi=DPI):
    images = []
    images.extend(list(map(lambda image: cv2.cvtColor(np.asarray(image), code=cv2.COLOR_RGB2GRAY),
                  pdf2image.convert_from_path(path, dpi=dpi, poppler_path=poppler_path))))
    return images

# https://stackoverflow.com/questions/28816046/
# displaying-different-images-with-actual-size-in-matplotlib-subplot


def display(image):
    dpi = 300

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


def pre_proc(img):
    base_image = img.copy()
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # image load as gray already
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 50))
    dilate = cv2.dilate(thresh, kernal, iterations=1)

    return dilate


if __name__ == "__main__":

    for f in glob.glob("*.pdf"):
        imgs = load_images(f)
        i = 0
        txt = u""
        for img in imgs:
            i += 1
            work = pre_proc(img)
            cv2.imwrite(ntpath.basename(f)[:-4] +
                        "-dilated-"+str(i)+".png", work)
            # h, w = work.shape
            #work = img[MT:h-MB, ML:w-MR]
            #deskewed = deskew(work)
            #cv2.imwrite(ntpath.basename(f)[:-4]+"-"+str(i)+".png", deskewed)
            #txt += pytesseract.image_to_string(deskewed,
            #                                   lang=LANG, #config=TESSERACT_CONFIG)
            # print(txt)
            #cv2.imwrite(ntpath.basename(f)[:-4]+"-"+str(i)+".png", img[MT:h-MB, ML:w-MR])
        #with io.open(ntpath.basename(f)[:-4]+".txt", "w", encoding="utf8") as f:
        #    f.write(txt)
