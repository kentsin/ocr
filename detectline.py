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

MT = 80 
ML = 50
MR = 50
MB = 95


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

def getSkewAngle(image) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = image.copy()
    # gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(newImage, (9, 9), 0)
    thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    # Find all contours
    contours, hierarchy = cv2.findContours(
        dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        cv2.rectangle(newImage, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    # print(len(contours))
    minAreaRect = cv2.minAreaRect(largestContour)
    cv2.imwrite("temp/boxes.jpg", newImage)
    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle

def rotateImage(image, angle: float):
    newImage = image.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(
        newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

def deskew(image):
    angle = getSkewAngle(image)
    if angle>10.0 or angle<-10.0: 
        return image
    else:
        return rotateImage(image, -1.0 * angle)

def cut_margins(img):
    h, w = img.shape
    cv2.rectangle(img, (0,0), (w, MT), (0,0,0), 2)
    cv2.rectangle(img, (0,h-MB), (w, h), (0,0,0), 2)
    cv2.rectangle(img, (0,0), (ML, h), (0,0,0), 2)
    cv2.rectangle(img, (w-MR,0), (w, h), (0,0,0), 2)
    return img


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
