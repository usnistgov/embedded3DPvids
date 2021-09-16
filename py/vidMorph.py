#!/usr/bin/env python
'''Morphological operations'''

# external packages
import cv2 as cv
import imutils
import numpy as np 
import os
import sys
import logging
from typing import List, Dict, Tuple, Union, Any, TextIO
import pandas as pd
import matplotlib.pyplot as plt

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
from imshow import imshow

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)

# info
__author__ = "Leanne Friedrich"
__copyright__ = "This data is publicly available according to the NIST statements of copyright, fair use and licensing; see https://www.nist.gov/director/copyright-fair-use-and-licensing-statements-srd-data-and-software"
__credits__ = ["Leanne Friedrich"]
__version__ = "1.0.0"
__maintainer__ = "Leanne Friedrich"
__email__ = "Leanne.Friedrich@nist.gov"
__status__ = "Development"


#----------------------------------------------


######### MORPHOLOGICAL OPERATIONS

def morph(img:np.array, width:int, func:str, iterations:int=1, shape:bool=cv.MORPH_RECT, aspect:float=1, **kwargs) -> np.array:
    '''erode, dilate, open, or close. func should be erode, dilate, open, or close. aspect is aspect ratio of the kernel, height/width'''
    if not shape in [cv.MORPH_RECT, cv.MORPH_ELLIPSE, cv.MORPH_CROSS]:
        raise NameError('Structuring element must be rect, ellipse, or cross')
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(width, int(width*aspect)))
    if func=='erode':
        return cv.erode(img, kernel, iterations = iterations)
    elif func=='dilate':
        return cv.dilate(img, kernel, iterations = iterations)
    elif func=='open':
        return cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    elif func=='close':
        return cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    else:
        raise NameError('func must be erode, dilate, open, or close')

def erode(img:np.array, size:int, **kwargs) -> np.array:
    '''dilate an image, given a kernel size. https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html'''
    return morph(img, size, 'erode', **kwargs)
    
    
def dilate(img:np.array, size:int, **kwargs) -> np.array:
    '''dilate an image, given a kernel size. https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html'''
    return morph(img, size, 'dilate', **kwargs)

def openMorph(img:np.array, size:int, **kwargs) -> np.array:
    '''open the image (erode then dilate)'''
    return morph(img, size, 'open', **kwargs)

def closeMorph(img:np.array, size:int, **kwargs) -> np.array:
    '''close the image (dilate then erode)'''
    return morph(img, size, 'close', **kwargs)



########### SEGMENTATION

def componentCentroid(img:np.array, label:int) -> List[int]:
    '''identify the centroid of a labeled component. Returns '''
    mask = np.where(img == label)
    x = int(np.mean(mask[0]))
    y = int(np.mean(mask[1]))
    return [label,x,y]

def componentCentroids(img:np.array) -> np.array:
    labels = list(np.unique(img))
    centroids = [componentCentroid(img, l) for l in labels]
    return centroids       

def fillComponents(thresh:np.array)->np.array:
    '''fill the connected components in the thresholded image, removing anything touching the border. https://www.programcreek.com/python/example/89425/cv2.floodFill'''
    thresh2 = thresh.copy()
    # add 1 pixel white border all around
    pad = cv.copyMakeBorder(thresh2, 1,1,1,1, cv.BORDER_CONSTANT, value=255)
    h, w = pad.shape
    # create zeros mask 2 pixels larger in each dimension
    mask = np.zeros([h + 2, w + 2], np.uint8)
    img_floodfill = cv.floodFill(pad, mask, (0,0), 0, (5), (0), flags=8)[1] # floodfill outer white border with black
    thresh2 = img_floodfill[1:h-1, 1:w-1]  # remove border
    
    im_flood_fill = thresh2.copy()
    h, w = thresh.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    im_flood_fill = im_flood_fill.astype("uint8")
    cv.floodFill(im_flood_fill, mask, (0, 0), 255)
    im_flood_fill_inv = cv.bitwise_not(im_flood_fill)
    img_out = thresh2 | im_flood_fill_inv
    return img_out

def onlyBorderComponents(thresh:np.array) -> np.array:
    '''only include the components that are touching the border'''
    return cv.subtract(thresh, fillComponents(thresh))

def removeBorders(im:np.array) -> np.array:
    '''remove borders from color image'''
    
    # create a background image
    average = im.mean(axis=0).mean(axis=0)
    avim = np.ones(shape=im.shape, dtype=np.uint8)*np.uint8(average)
    
    # find the border
    gray = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    thresh = dilate(thresh,10)
    thresh = onlyBorderComponents(thresh)
    thresh = cv.medianBlur(thresh,31)
    t_inv = cv.bitwise_not(thresh)
    
    # remove border from image
    imNoBorder = cv.bitwise_and(im,im,mask = t_inv)
    # create background-colored border
    border = cv.bitwise_and(avim,avim,mask = thresh)
    # combine images
    adjusted = cv.add(imNoBorder, border)
    return adjusted

def closeVerticalTop(thresh:np.array) -> np.array:
    '''if the image is of a vertical line, close the top'''
    if thresh.shape[0]<thresh.shape[1]*2:
        return thresh
    
    imtop = int(thresh.shape[0]*0.03)  
    thresh[0:imtop, :] = np.ones(thresh[0:imtop, :].shape)*0
    
    # vertical line. close top to fix bubbles
    top = np.where(np.array([sum(x) for x in thresh])>0) 
    if len(top[0])==0:
        return thresh

    imtop = top[0][0] # first position in y where black
    marks = np.where(thresh[imtop]==255) 
    if len(marks[0])==0:
        return thresh
    
    first = marks[0][0] # first position in x in y row where black
    last = marks[0][-1]
    if last-first<thresh.shape[1]*0.2:
        thresh[imtop:imtop+3, first:last] = 255*np.ones(thresh[imtop:imtop+3, first:last].shape)
    return thresh

def verticalFilter(gray:np.array) -> np.array:
    '''vertical line filter'''
    sobel_y = np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    filtered_image_y = cv.filter2D(gray, -1, sobel_y)   # x gradient (vertical lines)
    vertthresh = 255-cv.adaptiveThreshold(filtered_image_y,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,21,2) 
        # threshold of vertical lines
    sobel_x = np.array([[-1, -2, -1], [0, 0, 0], [1,2, 1]])
    filtered_image_x = cv.filter2D(gray, -1, sobel_x)   # y gradient (horizontals)
    horizthresh = 255-cv.adaptiveThreshold(filtered_image_x,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,21,2)
        # threshold of horizontal lines
    vertthresh = cv.subtract(vertthresh,horizthresh) # remove horizontals from verticals
    vertthresh = closeMorph(vertthresh,2)          # close holes
    return vertthresh




def threshes(img:np.array, gray:np.array, removeVert, attempt) -> np.array:
    '''threshold the grayscale image'''
    if attempt==0:
        # just threshold on intensity
#         ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)  
        ret, thresh = cv.threshold(gray,127,255,cv.THRESH_BINARY_INV)  
        thresh = closeVerticalTop(thresh)
    elif attempt==1:
        # adaptive threshold, for local contrast points
        thresh = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,11,2)
        filled = fillComponents(thresh)
        thresh = cv.add(255-thresh,filled)
    elif attempt==2:
        # threshold based on difference between red and blue channel
        b = img[:,:,2]
        g = img[:,:,1]
        r = img[:,:,0]
        gray2 = cv.subtract(r,b)
        gray2 = cv.medianBlur(gray2, 5)
        ret, thresh = cv.threshold(gray2,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
        ret, background = cv.threshold(r,0,255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
        background = 255-background
        thresh = cv.subtract(background, thresh)
    elif attempt==3:
        # adaptive threshold, for local contrast points
        thresh = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,21,2)
        filled = fillComponents(thresh)
        thresh2 = cv.add(255-thresh,filled)

        # remove verticals
        if removeVert:
            thresh = cv.subtract(thresh, verticalFilter(gray))
            ret, topbot = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU) 
            thresh = cv.subtract(thresh,topbot)
    elif attempt==4:
        thresh0 = threshes(img, gray, removeVert, 0)
        thresh2 = threshes(img, gray, removeVert, 2)
        thresh = cv.bitwise_or(thresh0, thresh2)
        thresh = cv.medianBlur(thresh,3)
    thresh = closeVerticalTop(thresh)
    return thresh

def segmentInterfaces(img:np.array, acrit:float=2500, attempt0:int=0, diag:bool=False, removeVert:bool=False) -> np.array:
    '''from a color image, segment out the ink, and label each distinct fluid segment'''
    if attempt0>=5:
        return [], [], attempt0
    if len(img.shape)==3:
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    gray = cv.medianBlur(gray, 5)
    attempt = attempt0
    finalAt = attempt
    while attempt<5:
        finalAt = attempt
        thresh = threshes(img, gray, removeVert, attempt)
        filled = fillComponents(thresh)            
        markers = cv.connectedComponentsWithStats(filled, 8, cv.CV_32S)

        if diag:
            imshow(img, gray, thresh, filled)
            plt.title(f'attempt:{attempt}')
        if markers[0]>1:
            boxes = pd.DataFrame(markers[2], columns=['x0', 'y0', 'w', 'h', 'area'])
            if max(boxes.loc[1:,'area'])<acrit:
                # poor segmentation. redo with adaptive thresholding.
                attempt=attempt+1
            else:
                attempt = 5
        else:
            attempt = attempt+1
    return filled, markers, finalAt