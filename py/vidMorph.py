#!/usr/bin/env python
'''Morphological operations applied to images'''

# external packages
import cv2 as cv
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
    if width==0:
        return img
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
    '''erode an image, given a kernel size. https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html'''
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
    '''identify the centroid of a labeled component. '''
    mask = np.where(img == label)
    x = int(np.mean(mask[0]))
    y = int(np.mean(mask[1]))
    return [label,x,y]

def componentCentroids(img:np.array) -> np.array:
    '''identify a list of all of the component centroids for a labeled image'''
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

def imAve(im:np.array) -> float:
    '''average value of image'''
    return im.mean(axis=0).mean(axis=0)

def imMax(im:np.array) -> float:
    '''max value of image'''
    return im.max(axis=0).max(axis=0)

def removeBorders(im:np.array, normalizeIm:bool=True) -> np.array:
    '''remove borders from color image'''
    
    # create a background image
    white = imMax(im)
    avim = np.ones(shape=im.shape, dtype=np.uint8)*np.uint8(white)
    
    # find the border
    gray = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
#     thresh[:, -100:] = np.ones(shape=(thresh[:, -100:]).shape, dtype=np.uint8)*255 # cutoff right 100 pixels
    thresh[800:, :] = np.ones(shape=(thresh[800:, :]).shape, dtype=np.uint8)*255 # cutoff bottom 100 pixels
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
    if normalizeIm:
        adjusted = normalize(adjusted)
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




def threshes(img:np.array, gray:np.array, removeVert:bool, attempt:int, botthresh:int=150, topthresh:int=200, whiteval:int=80, diag:int=0, **kwargs) -> np.array:
    '''threshold the grayscale image
    img is the original image
    gray is the grayscale conversion of the image
    removeVert=True to remove vertical lines from the thresholding. useful for horizontal images where stitching leaves edges
    attempt number chooses different strategies for thresholding ink
    botthresh = lower threshold for thresholding
    topthresh is the initial threshold value
    whiteval is the pixel intensity below which everything can be considered white
    increase diag to see more diagnostic messages
    '''
    if attempt==0:
#         ret, thresh = cv.threshold(gray,180,255,cv.THRESH_BINARY_INV)
        # just threshold on intensity
        crit = topthresh
        impx = np.product(gray.shape)
        allwhite = impx*whiteval
        prod = allwhite
        while prod>=allwhite and crit>100: # segmentation included too much
            ret, thresh1 = cv.threshold(gray,crit,255,cv.THRESH_BINARY_INV)
            ret, thresh2 = cv.threshold(gray,crit+10,255,cv.THRESH_BINARY_INV)
            thresh = np.ones(shape=thresh2.shape, dtype=np.uint8)
            thresh[:600,:] = thresh2[:600,:] # use higher threshold for top 2 lines
            thresh[600:,:] = thresh1[600:,:] # use lower threshold for bottom line
            prod = np.sum(np.sum(thresh))
            crit = crit-10
#         ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
        if diag>0:
            logging.info(f'Threshold: {crit+10}, product: {prod/impx}, white:{whiteval}')
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

def segmentInterfaces(img:np.array, acrit:float=2500, diag:int=0, removeVert:bool=False, removeBorder:bool=True, **kwargs) -> np.array:
    '''from a color image, segment out the ink, and label each distinct fluid segment. 
    acrit is the minimum component size for an ink segment
    removeVert=True to remove vertical lines from the thresholded image
    removeBorder=True to remove border components from the thresholded image'''
    if len(img.shape)==3:
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    gray = cv.medianBlur(gray, 5)
    attempt = 0
    finalAt = attempt
    while attempt<1:
        finalAt = attempt
        thresh = threshes(img, gray, removeVert, attempt, diag=diag, **kwargs)
        if removeBorder:
            filled = fillComponents(thresh)    
        else:
            filled = thresh.copy()
        markers = cv.connectedComponentsWithStats(filled, 8, cv.CV_32S)

        if diag>0:
            imshow(img, gray, thresh, filled)
            plt.title(f'attempt:{attempt}')
        if markers[0]>1:
            boxes = pd.DataFrame(markers[2], columns=['x0', 'y0', 'w', 'h', 'area'])
            if max(boxes.loc[1:,'area'])<acrit:
                # poor segmentation. redo with adaptive thresholding.
                attempt=attempt+1
            else:
                attempt = 6
        else:
            attempt = attempt+1
    return filled, markers, finalAt



def markers2df(markers:Tuple) -> pd.DataFrame:
    '''convert the labeled segments to a dataframe'''
    df = pd.DataFrame(markers[2], columns=['x0', 'y0', 'w','h','a'])
    df2 = pd.DataFrame(markers[3], columns=['xc','yc'])
    df = pd.concat([df, df2], axis=1) 
        # combine markers into dataframe w/ label stats
    df = df[df.a<df.a.max()] 
        # remove largest element, which is background
    return df

def normalize(im:np.array) -> np.array:
    '''normalize the image'''
    norm = np.zeros(im.shape)
    im = cv.normalize(im,  norm, 0, 255, cv.NORM_MINMAX) # normalize the image
    return im


def white_balance(img:np.array) -> np.array:
    '''automatically correct white balance'''
    result = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv.cvtColor(result, cv.COLOR_LAB2BGR)
    return result