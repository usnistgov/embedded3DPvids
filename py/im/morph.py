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
sys.path.append(os.path.dirname(currentdir))
from imshow import imshow

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)



#----------------------------------------------


######### MORPHOLOGICAL OPERATIONS

def morph(img:np.array, width:int, func:str, iterations:int=1, shape:bool=cv.MORPH_RECT, aspect:float=1, hitBorder:bool=False, **kwargs) -> np.array:
    '''erode, dilate, open, or close. func should be erode, dilate, open, or close. aspect is aspect ratio of the kernel, height/width'''
    if width==0:
        return img
    if not shape in [cv.MORPH_RECT, cv.MORPH_ELLIPSE, cv.MORPH_CROSS]:
        raise NameError('Structuring element must be rect, ellipse, or cross')
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(width, int(width*aspect)))
    if not hitBorder:
        if len(img.shape)==3:
            c = (0,0,0)
        else:
            c = 0
        dy =  int(width*aspect+1)
        dx = int(width+1)
        img = cv.copyMakeBorder(img,dy, dy, dx, dx, cv.BORDER_CONSTANT, None, c)
    if func=='erode':
        img =  cv.erode(img, kernel, iterations = iterations)
    elif func=='dilate':
        img = cv.dilate(img, kernel, iterations = iterations)
    elif func=='open':
        img =  cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    elif func=='close':
        img =  cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    else:
        raise NameError('func must be erode, dilate, open, or close')
    if not hitBorder:
        img = img[dy:-dy, dx:-dx]
    return img
        

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

def emptySpaces(thresh:np.array) -> np.array:
    ff = floodFill(thresh)
    img_out = cv.subtract(ff, thresh)
    return img_out

def floodFill(thresh:np.array) -> np.array:
    '''fill extra components'''
    im_flood_fill = thresh.copy()
    h, w = thresh.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    im_flood_fill = im_flood_fill.astype("uint8")
    cv.floodFill(im_flood_fill, mask, (0, 0), 255)
    im_flood_fill_inv = cv.bitwise_not(im_flood_fill)
    return im_flood_fill_inv

def labelHierarchy(hierarchy:np.array, cnt:np.array) -> pd.DataFrame:
    '''put the hierarchy into a dataframe'''
    hdf = pd.DataFrame(hierarchy[0], columns=['previous', 'next', 'child', 'parent'])
    toplevel = hdf[(hdf.parent<0)]   # no parents
    level1 = hdf[(hdf.parent>=0)]    # has parents
    level2 = level1[level1.parent.isin(level1.index.tolist())]   # has grandparents
    level3 = level2[level2.parent.isin(level2.index.tolist())]   # has great grandparents
    hdf.loc[(toplevel.index), 'level'] = 0
    hdf.loc[(level1.index), 'level'] = 1
    hdf.loc[(level2.index), 'level'] = 2
    hdf.loc[(level3.index), 'level'] = 3
    hdf['area'] = [cv.contourArea(c) for c in cnt]
    return hdf

def displayHierarchy(thresh:np.array, hdf:pd.DataFrame, cnt:np.array) -> np.array:
    '''add the hierarchy contours to the thresholded image'''
    c2 = thresh.copy()
    c2 = cv.cvtColor(c2, cv.COLOR_GRAY2BGR)
    colors = {0:(19, 53, 242), 1: (46, 211, 240), 2:(15, 214, 81), 3:(230, 237, 31)}
    for n in hdf.level.unique():
        color = colors[n]
        for i,row in hdf[hdf.level==n].iterrows():
            if cv.contourArea(cnt[i])>50:
                cv.drawContours(c2, [cnt[i]], -1, color, 2)
    return c2

def fillComponents(thresh:np.array, diag:int=0, leaveHollows:bool=True)->np.array:
    '''fill the connected components in the thresholded image https://www.programcreek.com/python/example/89425/cv.floodFill'''
    
    # fill in objects
    thresh2 = thresh.copy()
    im_flood_fill_inv = floodFill(thresh2)
    img_out = thresh2 | im_flood_fill_inv
    if leaveHollows:
        # remove the inside of any bubbles inside objects
        i2 = thresh2
        i2[-1:, :] = 255
        cnt, hierarchy = cv.findContours(i2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        hdf = labelHierarchy(hierarchy)
        im2 = thresh.copy()*0
        if diag>0:
            thresh2 = displayHierarchy(i2, hdf, cnt)
        for i in (hdf[hdf.level==3]).index:
            if cv.contourArea(cnt[i])>50:
                cv.drawContours(im2, cnt, contourIdx=i, color=(255,255,255),thickness=-1)
        img_out = cv.subtract(img_out, im2)
    if diag>0:
        if leaveHollows:
            imshow(thresh, thresh2, im_flood_fill_inv, img_out, title='fillComponents')
        else:
            imshow(thresh, im_flood_fill_inv, img_out, title='fillComponents')
    return img_out

def isReentrant(cnt:np.array) -> bool:
    '''determine if the contour has its outer surface also inside of its inner surface, i.e. it is a U'''
    if len(cnt)<100:
        return False
    M = cv.moments(cnt)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    dist = cv.pointPolygonTest(cnt, (cx,cy), False)
    return dist<0


def maskContour(im:np.array, cnt:np.array, thickness:int=-1) -> float:
    '''get the mean value inside the contour if thickness=-1, on the contour if thickness>0'''
    if len(im.shape)==3:
        im2 = normalize(cv.cvtColor(im, cv.COLOR_BGR2GRAY))
    else:
        im2 = im
    mask = np.zeros((im.shape[0], im.shape[1]))
    cv.drawContours(mask, [cnt], contourIdx=0, color=(255,255,255),thickness=thickness)  # fill in the contour
    masked = np.ma.masked_where(mask==0, im2).compressed()
    return masked


def meanInsideContour(im:np.array, cnt:np.array, thickness:int=-1) -> float:
    '''get the mean value inside the contour if thickness=-1, on the contour if thickness>0'''
    if len(im)==0:
        return 0
    masked = maskContour(im, cnt, thickness)
    return masked.mean().mean()

def contrastOnContour(im:np.array, cnt:np.array, thickness:int=1) -> float:
    '''get the contrast between average positive and negative values inside the contour if thickness=-1, on the contour if thickness>0'''
    if len(im)==0:
        return 0
    masked = maskContour(im, cnt, thickness)
    posvals = masked[masked>0]
    if len(posvals)>0:
        pos = posvals.mean().mean()
    else:
        pos = 0
    negvals = masked[masked<0]
    if len(negvals)>0:
        neg = negvals.mean().mean()
    else:
        neg = 0
    return pos-neg


def fillByContours(thresh:np.array, im:np.array, amin:int=50, amax:int=5000
                   , diag:int=0, conCrit:int=50, laplacian:np.array = [], **kwargs) -> np.array:
    '''fill the components using the contours, where anything with a size between amin and amax doesn't get filled'''
    i2 = thresh.copy()
    i2[-1, :] = 255
    cnt, hierarchy = cv.findContours(i2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    hdf = labelHierarchy(hierarchy, cnt)
    imout = thresh.copy()
    
    level1pts = hdf[hdf.level==1]

    # fill in tiny contours
    for i in level1pts[(level1pts.area<amin)|(level1pts.area>amax)].index:
        cv.drawContours(imout, cnt, contourIdx=i, color=(255,255,255),thickness=-1)

    for i in level1pts[(level1pts.area>=amin)&(level1pts.area<=amax)].index:
        mo = meanInsideContour(laplacian, cnt[i], thickness=1)
        con = contrastOnContour(laplacian, cnt[i], thickness=2)
        if diag>0:
            a = level1pts.loc[i,'area']
            print(a, mo, con)
        if con<conCrit and mo>-5:
            # fill it if it is chunky, or if it is rough
            cv.drawContours(imout, cnt, contourIdx=i, color=(255,255,255),thickness=-1)

    for i in (hdf[hdf.level==3]).index:
        if hdf.loc[i,'area']>50:
            # empty this bubble
            cv.drawContours(imout, cnt, contourIdx=i, color=(0,0,0),thickness=-1)
    if diag>0:
        im3 = displayHierarchy(i2, hdf, cnt)
        imshow(thresh, imout, im3, titles=['fill: thresh', 'filled', 'contours'])
    return imout

def fillTiny(thresh:np.array, acrit:int=50) -> np.array:
    '''fill the components using the contours, where anything with a size between amin and amax doesn't get filled'''
    i2 = thresh.copy()
    i2[-1, :] = 255
    cnt, hierarchy = cv.findContours(i2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    hdf = labelHierarchy(hierarchy, cnt)
    imout = thresh.copy()
    level1pts = hdf[hdf.level==1]

    # fill in tiny contours
    for i in level1pts[(level1pts.area<acrit)].index:
        cv.drawContours(imout, cnt, contourIdx=i, color=(255,255,255),thickness=-1)
    return imout
    

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

def skeletonize(img:np.array, w:int=3) -> np.array:
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)

    ret,img = cv.threshold(img,127,255,0)
    element = cv.getStructuringElement(cv.MORPH_CROSS,(w,w))
    done = False

    while( not done):
        eroded = cv.erode(img,element)
        temp = cv.dilate(eroded,element)
        temp = cv.subtract(img,temp)
        skel = cv.bitwise_or(skel,temp)
        img = eroded.copy()

        zeros = size - cv.countNonZero(img)
        if zeros==size:
            done = True

    return skel


    
