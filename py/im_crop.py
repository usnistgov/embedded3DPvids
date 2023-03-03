#!/usr/bin/env python
'''Functions for cropping and masking apertures in shear droplet videos'''

# external packages
import cv2 as cv
import numpy as np 
import os
import sys
import time
import logging
from typing import List, Dict, Tuple, Union, Any, TextIO
from matplotlib import pyplot as plt
import traceback

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)


# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)



#----------------------------------------------


####### CROPPING TOOLS

def convertCrop(im:np.array, crops:Union[Dict, int]) -> dict:
    '''convert the given crop dimensions to coordinates that will definitely fit in the '''
    if len(im)>0:
        h,w = im.shape[0:2]
    else:
        return {}

    if type(crops) is int:
        return {'x0':d, 'xf':w-d, 'y0':d, 'yf':h-d}
    elif 'dx' in crops and 'dy' in crops:
        dx = crops['dx']
        dy = crops['dy']
        return {'x0':dx, 'xf':w-dx, 'y0':dy, 'yf':h-dy}
    else:
        out = {'x0':0, 'xf':w, 'y0':0, 'yf':h}
        if 'x0' in crops:
            out['x0'] = max(0, min(w, crops['x0']))
        if 'y0' in crops:
            out['y0'] = max(0, min(h, crops['y0']))
        if 'xf' in crops:
            if crops['xf']<0:
                out['xf'] = min(w, max(out['x0'], w+crops['xf']))
            else:
                out['xf'] = min(w, max(out['x0'], crops['xf']))
        if 'yf' in crops:
            if crops['yf']<0:
                out['yf'] = min(h, max(out['y0'], h+crops['yf']))
            else:
                out['yf'] = min(h, max(out['y0'], crops['yf']))
        return out
    
def imcropDict(im:np.array, crop:dict):
    '''crop an image to the bounds, defined by x0,xf,y0,yf, where y is from top and x is from left'''
    h = im.shape[0]
    return im[int(crop['y0']):int(crop['yf']), int(crop['x0']):int(crop['xf'])]

def imcrop(im:np.array, bounds:Union[Dict, int]) -> np.array:
    '''crop an image to the bounds, defined by single number, dx and dy, or x0,xf,y0,yf, where y is from top and x is from left'''
    crop = convertCrop(im, bounds)
    if len(crop)==0:
        return im
    else:
        return imcropDict(im, crop)


def findCenterXY(x:int, w:int, imw:int, radius:int) -> int:
    '''find the center for x or y'''
    if x<0:
        if w>=imw:
            # maxed out width, unknown center
            xc=int(imw/2)
        else:
            # maxed out width on left side, known right side
            xc=w-radius
    else:
        if x+w>=imw:
            # maxed out width on right side, known left side
            xc=x+radius
        else:
            # x and w both within image
            xc=int(x+w/2)
    return xc

def findCenter(x:int, y:int, w:int, h:int, imh:int, imw:int, bounds:Dict) -> Dict:
    '''find the center of the aperture based on dimensions of bounding box and initial dimensions of image'''
    radius = int(max(w,h)/2) # radius of the aperture
    xc = findCenterXY(x,w,imw,radius)
    yc = findCenterXY(y,h,imh,radius)
    xcc = xc-bounds['x0'] # x aperture center in the cropped image
    ycc = yc-bounds['y0'] # y aperture center in the cropped image
    return {'r':radius, 'xc':xc, 'yc':yc, 'xcc':xcc, 'ycc':ycc}
    
def padBounds(x:int, y:int, w:int, h:int, imh:int, imw:int, pad:int) -> Dict:
    '''pad the bounds of the cropped image'''
    x0 = max(0, x-pad)
    xf = min(imw, x+w+pad)
    y0 = max(0, y-pad)
    yf = min(imh, y+h+pad)
    bounds = {'x0':x0, 'xf':xf, 'y0':y0, 'yf':yf}
    return bounds

            


def getPads(h2:int, h1:int) -> Tuple[int,int]:
    '''get left and right padding given two heights'''
    dh = (h2-h1)/2
    if dh>0:
        # second image is larger: need to crop
        crop=True
    else:
        # second image is smaller: need to pad
        crop=False
        dh = abs(dh)
        
    # fix rounding
    if dh-int(dh)>0:
        dhl = int(dh)
        dhr = int(dh)+1
    else:
        dhl = int(dh)
        dhr = int(dh)
        
    if crop:
        dhl = -dhl
        dhr = -dhr
        
    return dhl,dhr
    
def circleMask(im:np.array, bs:int, background:float, cval:int, bounds:dict, dr:int=0) -> np.array:
    '''get a mask that matches the size of im, has a blur size of bs, a background value of background, a circle value of cval (0-255), and bounds of bounds'''
    mask = np.full((im.shape[0]+2*bs, im.shape[1]+2*bs), background, dtype=np.uint8)  # mask is only 
    cv.circle(mask, (bounds['ycc']+bs, bounds['xcc']+bs), bounds['r']+dr, (cval,cval,cval), -1)
    if bs>0:
        mask = cv.blur(mask,(bs,bs))
        mask = imcrop(mask, bs)
    if len(im.shape)==3:
        mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR ) # convert to color if original image is color
    return mask
    
def removeAperture(img:np.array, bounds:Dict, crop:bool=False, dr:int=75) -> np.array:
    '''given an image, crop it to the aperture (if requested), and then mask everything outside the aperture to white. This is useful if you are trying to detect a dark droplet interface and don't want to detect the outside of the aperture  '''
    if crop:
        im2 = imcrop(img, bounds)
    else:
        im2 = img
    
    bs=201 # blur size
    background = np.median(im2)
    frontmask = circleMask(im2, bs, 0,255, bounds, dr=dr)
    backmask = circleMask(im2, bs, background, 0, bounds, dr=-dr)
    masked = cv.bitwise_and(frontmask,im2)
    removed = cv.add(backmask, masked)
    return removed