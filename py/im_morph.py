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
    '''fill the connected components in the thresholded image https://www.programcreek.com/python/example/89425/cv2.floodFill'''
    
    # fill in objects
    im_flood_fill = thresh.copy()
    h, w = thresh.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    im_flood_fill = im_flood_fill.astype("uint8")
    cv.floodFill(im_flood_fill, mask, (0, 0), 255)
    im_flood_fill_inv = cv.bitwise_not(im_flood_fill)
    img_out = thresh | im_flood_fill_inv
    return img_out

def removeBorderAndFill(thresh:np.array) -> np.array:
    '''remove the components touching the border and fill them in'''
    thresh2 = thresh.copy()
    # add 1 pixel white border all around
    pad = cv.copyMakeBorder(thresh2, 1,1,1,1, cv.BORDER_CONSTANT, value=255)
    h, w = pad.shape
    # create zeros mask 2 pixels larger in each dimension
    mask = np.zeros([h + 2, w + 2], np.uint8)
    img_floodfill = cv.floodFill(pad, mask, (0,0), 0, (5), (0), flags=8)[1] # floodfill outer white border with black
    thresh2 = img_floodfill[1:h-1, 1:w-1]  # remove border
    return fillComponents(thresh2)

def onlyBorderComponents(thresh:np.array) -> np.array:
    '''only include the components that are touching the border'''
    return cv.subtract(thresh, fillComponents(thresh))

def imAve(im:np.array) -> float:
    '''average value of image'''
    return im.mean(axis=0).mean(axis=0)

def imMax(im:np.array) -> float:
    '''max value of image'''
    return im.max(axis=0).max(axis=0)

def imMin(im:np.array) -> float:
    '''min value of image'''
    return im.min(axis=0).min(axis=0)

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


def closeHorizLine(thresh:np.array, imtop:int) -> np.array:
    '''draw a black line across the y position imtop between the first and last black point'''
    marks = np.where(thresh[imtop]==255) 
    if len(marks[0])==0:
        return thresh
    
    first = marks[0][0] # first position in x in y row where black
    last = marks[0][-1]
    if last-first<thresh.shape[1]*0.2:
        thresh[imtop:imtop+3, first:last] = 255*np.ones(thresh[imtop:imtop+3, first:last].shape)
    return thresh

def closeVerticalTop(thresh:np.array, cutoffTop:float=0.03, closeBottom:bool=False, **kwargs) -> np.array:
    '''if the image is of a vertical line, close the top'''
    if thresh.shape[0]<thresh.shape[1]*2:
        return thresh
    
    # cut off top 3% of image
    if cutoffTop>0:
        imtop = int(thresh.shape[0]*cutoffTop)  
        thresh[0:imtop, :] = np.ones(thresh[0:imtop, :].shape)*0

    # vertical line. close top to fix bubbles
    top = np.where(np.array([sum(x) for x in thresh])>0) 

    if len(top[0])==0:
        return thresh

    imtop = top[0][0] # first position in y where black
    thresh = closeHorizLine(thresh, imtop)
    if closeBottom:
        imbot = top[0][-1]-3
        thresh = closeHorizLine(thresh, imbot)
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


def removeBlack(im:np.array, threshold:int=70) -> np.array:
    '''remove black portions such as bubbles from the image'''
    if len(im.shape)==3:
        gray = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
    else:
        gray = im.copy()
    _, thresh = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY_INV)
    thresh = dilate(thresh, 5)
    white = imMax(im)
    avim = np.ones(shape=im.shape, dtype=np.uint8)*np.uint8(white)
    cover = cv.bitwise_and(avim, avim, mask=thresh)
    im = cv.add(cover, im)
    return im

def imchannel(im:np.array, channel:int) -> np.array:
    '''return a color image that is just the requested channel'''
    red = np.zeros(shape=im.shape, dtype=np.uint8)
    red[:,:,channel] = im[:,:,channel]
    return red

def removeChannel(im:np.array, channel:int) -> np.array:
    '''remove the requested color channel'''
    red = im
    red[:,:,channel] = np.zeros(shape=(red[:,:,channel]).shape, dtype=np.uint8)
    return red

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
        while prod>=allwhite and crit>50: # segmentation included too much
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
    thresh = closeVerticalTop(thresh, **kwargs)
    return thresh

def segmentInterfaces(img:np.array, acrit:float=2500, diag:int=0, removeVert:bool=False, removeBorder:bool=True, eraseMaskSpill:bool=False, **kwargs) -> np.array:
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
        if 'nozData' in kwargs and 'crops' in kwargs:
            nd = kwargs['nozData']
            thresh = nd.maskNozzle(thresh, ave=False, invert=False, crops=kwargs['crops'])   # add the nozzle back in for filling
            h,w = thresh.shape
            thresh[0, :] = 0   # clear out the top row
            thresh[:int(h/4), 0] = 0  # clear left and right edges at top half
            thresh[:int(h/4),-1] = 0
        
        if removeBorder:
            filled = removeBorderAndFill(thresh)    
        else:
            filled = fillComponents(thresh)
        if 'nozData' in kwargs:
            filled = nd.maskNozzle(filled, ave=False, invert=True, crops=kwargs['crops'])  # remove the nozzle again
            if eraseMaskSpill:
                filled = nd.eraseSpillover(filled, crops=kwargs['crops'])
        labels, markers, ret = filterMarkers(filled, acrit=acrit)
        if ret==0:
            attempt = 6
        else:
            attempt = attempt+1
        if diag>0:
            imshow(img, gray, thresh, labels, maxwidth=13)
            plt.title(f'attempt:{attempt}')
    finalAt = attempt
    return labels, markers, finalAt


def filterMarkers(filled:np.array, acrit:float=2500) -> Tuple[np.array, Tuple]:
    '''get connected components and filter by area, then create a new binary image without small components'''
    markers = cv.connectedComponentsWithStats(filled, 8, cv.CV_32S)
    labels = markers[1]
    if markers[0]==0:
        # no components. redo segmentation
        return labels, markers, 1
    # we collected points
    boxes = pd.DataFrame(markers[2], columns=['x0', 'y0', 'w', 'h', 'area'])

    if len(boxes)==1 or max(boxes.loc[1:,'area'])<acrit:
        # poor segmentation. redo with adaptive thresholding.
        return labels, markers, 1

    boxes = pd.DataFrame(markers[2], columns=['x0', 'y0', 'w', 'h', 'area'])
    for i in list(boxes[boxes.area<acrit].index):
        labels[labels==i] = 0
    markers = markers[0], labels.copy(), markers[2], markers[3]
    labels[labels>0]=255
    labels = labels.astype(np.uint8)
    return labels, markers, 0



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

def reconstructMask(markers:Tuple, df:pd.DataFrame) -> np.array:
    '''construct a binary mask with all components labeled in the dataframe'''
    masks = [(markers[1] == i).astype("uint8") * 255 for i,row in df.iterrows()]
    if len(masks)>0:
        componentMask = masks[0]
        if len(masks)>1:
            for mask in masks[1:]:
                componentMask = cv.add(componentMask, mask)
        return componentMask
    else:
        return np.zeros(markers[1].shape).astype(np.uint8)

def removeDust(im:np.array, acrit:int=1000, diag:int=0) -> np.array:
    '''remove dust from the image'''
    
    # threshold darkest parts
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray,(5,5),0)
    thresh = cv.adaptiveThreshold(blur, 255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 21, 10)
    
    # label thresholded image
    markers = cv.connectedComponentsWithStats(thresh, 8, cv.CV_32S)
    df = markers2df(markers)
    df = df[df.a<acrit]
    if len(df)==0:
        return im
    
    componentMask = reconstructMask(markers, df)
    badParts = cv.bitwise_and(im,im,mask = dilate(componentMask, 3))
    frameMasked = cv.add(im, badParts)
    if diag>0:
        imshow(im, badParts, frameMasked)
    return frameMasked

def blackenRed(im:np.array) -> np.array:
    '''darken red segments'''
    diff = cv.absdiff(im[:,:,2],im[:,:,1])
    blur = cv.GaussianBlur(diff,(5,5),0)
    _,thresh = cv.threshold(blur,30,255,cv.THRESH_BINARY_INV)
    im2 = np.ones(im.shape, dtype=np.uint8)*255
    badParts = cv.bitwise_not(im2,im2,mask = thresh)
    frameMasked = cv.subtract(im, badParts)
    return frameMasked
    