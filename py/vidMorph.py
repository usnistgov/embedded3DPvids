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

def segmentInterfaces(img:np.array) -> np.array:
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)
    ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    filled = fillComponents(thresh)
    markers = cv.connectedComponentsWithStats(filled, 8, cv.CV_32S)
    return filled, markers