#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-07-18 15:38:14
# @Author  : kentsin (kentsin@gmail.com)
# @Link    : link
# @Version : 0.0.1

import os
import ntpath
import io
import glob
import pickle
from turtle import goto

# https://www.youtube.com/watch?v=ADV-AjAXHdc&list=PL2VXyKi-KpYuTAZz__9KVl1jQz74bDG7i&index=5
# https://github.com/wjbmattingly/ocr_python_textbook

import pytesseract
import cv2
import numpy as np

import pdf2image
poppler_path = r"E:\Program Files (x86)\poppler-22.04.0\Library\bin"


import matplotlib.pyplot as plt

DPI = 150
# Margins 
MT = 100 # 200
ML = 10
MR = 10
MB = 100 # 200

MC = 50

TEMP_DICT_FILE = r"E:\workspace\ocr\template_dict.pki"

template_dict = pickle.load(open(TEMP_DICT_FILE, "rb"))
# template_dict = {'cc': [img1, img2], 'al':[img1, img2]}

TEMPLATE_METHOD = cv2.TM_CCOEFF_NORMED

def load_images(path, dpi=DPI):
    images = []
    images.extend(list(map(lambda image: cv2.cvtColor(np.asarray(image), code=cv2.COLOR_RGB2GRAY),
                  pdf2image.convert_from_path(path, dpi=dpi, poppler_path=poppler_path))))
    return images

# https://stackoverflow.com/questions/28816046/
# displaying-different-images-with-actual-size-in-matplotlib-subplot
def display(image):
    dpi = 150

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

def scale_image(image, s):
    w = int(image.shape[1]/s)
    h = int(image.shape[0]/s)
    dim = (w, h)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def find_contours(image):
    image = deskew(image)
    base_image = image.copy()
    
    for s in [1, 4, 8, 16, 32]:
        workimage = scale_image(image, s)

    #    gray = cv2.cvtColor(workimage, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(workimage, (7, 1), 0) # adjustable (7,7)
        nonoise = cv2.fastNlMeansDenoising(blur, None, 20, 7, 21)
        thresh = cv2.threshold(
            nonoise, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 13)) # adjustable
        dilate = cv2.dilate(thresh, kernal, iterations=1) # iterations adjustable?

        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts)==2 else cnts[1]
        cnts = sorted(cnts, key=lambda y: cv2.boundingRect(y)[1])
        print(s, len(cnts))
        if len(cnts) < MC: break

    #print(s)
    #print(len(cnts))
    cv2.rectangle(base_image, (ML, MT), (MR, MB), (255,0,0), 4)
    i = 0

    for c in cnts:
        i = i+1
        x, y, w, h = cv2.boundingRect(c)
        x = x*s
        y = y*s
        w = w*s
        h = h*s
        #print(x, y)
        #if x<ML: continue
        #if y<MT: continue
        #if x>MR: continue
        #if y>MB: continue
        cv2.rectangle(base_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(base_image, "%d : %d %d, %d %d" % (i, x, y, x+w, y+h), (x+2, y+18), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255,0), 2)
    cv2.drawContours(base_image, cnts, -1, (0, 255, 0), 2)
    return base_image

# https://www.youtube.com/watch?v=T-0lZWYWE9Y

def cut_margins(img):
    h, w = img.shape
    cv2.rectangle(img, (0,0), (w, MT), (0,0,0), 2)
    cv2.rectangle(img, (0,h-MB), (w, h), (0,0,0), 2)
    cv2.rectangle(img, (0,0), (ML, h), (0,0,0), 2)
    cv2.rectangle(img, (w-MR,0), (w, h), (0,0,0), 2)
    return img

if __name__=="__main__":
    
    for f in glob.glob("*.pdf"):
        imgs = load_images(f)
        i = 0
        
        for img in imgs:
            i += 1
            h, w = img.shape

            work = deskew(img[MT:h-MB, ML:w-MR]) 

            work = cv2.copyMakeBorder(work, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255 )
            result = find_contours(work)
            cv2.imwrite(ntpath.basename(f)[:-4]+"-"+str(i)+".png", result)            
            
            #txt += pytesseract.image_to_string(work[MT:h-MB, ML:w-MR], lang="chi_tra+por+eng")
            # print(txt)
            #cv2.imwrite(ntpath.basename(f)[:-4]+"-"+str(i)+".png", img[MT:h-MB, ML:w-MR])
        #with io.open(ntpath.basename(f)[:-4]+".txt", "w", encoding="utf8") as f:
        #    f.write(txt)

