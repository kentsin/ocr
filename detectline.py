#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-07-21 09:30:56
# @Author  : kentsin (kentsin@gmail.com)
# @Link    : link
# @Version : 0.0.1

import os
import io
import ntpath
import glob

import pytesseract
import numpy as np
import cv2

import matplotlib.pyplot as plt

import pdf2image

LANG = r"chi_tra+por+eng"
TESSERACT_CONFIG = r"--psm 6 --oem 3"

poppler_path = r"E:\Program Files (x86)\poppler-22.04.0\Library\bin"

DPI = 150

MT = 80   # 80
ML = 80   # 25
MR = 80   # 25
MB = 60   # 40

TH= 12

def load_images(path, dpi=DPI):
    images = []
    images.extend(list(map(lambda image: cv2.cvtColor(np.asarray(image), code=cv2.COLOR_RGB2GRAY),
                  pdf2image.convert_from_path(path, dpi=dpi, poppler_path=poppler_path))))
    return images

# https://stackoverflow.com/questions/28816046/
# displaying-different-images-with-actual-size-in-matplotlib-subplot


def display(image):
    dpi = DPI

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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 50))
    dilate = cv2.dilate(thresh, kernel, iterations=1)

    # Find all contours
    contours, hierarchy = cv2.findContours(
        dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    i = 0
    for c in contours:
        i += 1
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

    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    dilate = cv2.dilate(thresh, kernal, iterations=1)

    return dilate



if __name__ == "__main__":

    for f in glob.glob("*.pdf"):
        imgs = load_images(f)
        i = 0
        j = 0
        txt = u""
        for img in imgs:

            y_height, x_width = img.shape
            
            i += 1
            work = pre_proc(img)
            cnts = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts)==2 else cnts[1]
            cnts = sorted(cnts, key=lambda cnt1: cv2.boundingRect(cnt1)[1])

            # Draw boxes for debug            
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0,0), 1)

            # skip c when outside of margins
            # combine when vertically close together
            # we just care about x,y, w, h
            # combin contours of same heigh
            
            x, y, w, h = cv2.boundingRect(cnts[0]) # Current Expanding box
            new_cnts = []
            
            for c in cnts:
                
                x1, y1, w1, h1 = cv2.boundingRect(c)

                    #txt += "skip ML : %d %d %d %d\n" % (x1, y1, w1, h1)
                    #x,y,w,h = x1,y1,w1,h1
                    continue
                if MT>=y1 or y1>=y_height-MB :
                    #txt += "skip MT : %d %d %d %d\n"% (x1, y1, w1, h1)
                    #x,y,w,h = x1,y1,w1,h1
                    continue

                if w1*h1 < 999:
                    #txt += "skip 9 : %d %d %d %d\n" % (x1, y1, w1, h1)  
                    #x,y,w,h = x1,y1,w1,h1
                    continue   # 900: continue

                #print( x, y, w, h, "|", x1, y1, w1, h1)
                if y1-(y+h)<TH:  # if it close to last box  
                    x, y, w, h = min(x, x1), min(y, y1), max(x+w, x1+w1)-min(x, x1), max(y+h,y1+h1)-min(y, y1)
                else:

                    new_cnts.append((x, y, w, h))
                    # x, y, w, h = x1, y1, w1, h1  # 
                    x, y, w, h = x1, y1, w1, h1
                     

            if y1-(y+h)<TH:
                new_cnts.append((min(x,x1), min(y,y1), max(x+w, x1+w1)-min(x,x1), max(y+h, y1+h1)-min(y,y1)))
            else:
                new_cnts.append((x,y,w,h))
                if  ML>=x1 or x1>=W-MR or MT>=y1 or y1>=H-MB or w1*h1 < 999:
                    pass
                else:
                    new_cnts.append((x1, y1, w1, h1))
            
            for c in new_cnts:
                j += 1
                x, y, w, h = c
                txt += pytesseract.image_to_string(img[y:y+h, x:x+w], lang=LANG, config = TESSERACT_CONFIG)
                #txt += str(j)+"  |"+ str(x)+", "+str(y)+", "+str(w)+", " +str(h)+", "+str(w*h)+"|  "+str(len(t))+"\n\n"+t+"\n\n"
                cv2.putText(img, str(j), (x+4, y+4), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0), 1)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0,0), 4)

            cv2.imwrite(ntpath.basename(f)[:-4] +
                        "-box-"+str(i)+".jpg", img)

        with io.open(ntpath.basename(f)[:-4]+".txt", "w", encoding="utf8") as f:
            f.write(txt)
