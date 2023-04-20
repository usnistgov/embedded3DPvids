#!/usr/bin/env python
'''Functions for cropping images'''

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


# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)

#----------------------------------------------

def relativeCrop(progDims, nozData, tag:str, crops0:dict, **kwargs) -> dict:
    '''get the crop dictionary as y0,yf,x0,xf given a line to target'''
    if tag in crops0:
        crops = crops0[tag]
    else:
        crops = crops0
    out = {}
    if 'relative' in crops and 'w' in crops and 'h' in crops and 'wc' in crops and 'hc' in crops:
        # coordinates relative to the nozzle, centered on the line given by the tag
        # w and h are total width and height, and wc and hc are position of center within that space
        rc = progDims.relativeCoords(tag, **kwargs)   # shift in mm
        c = nozData.absoluteCoords(rc)   # absolute coords of ROI center in px from bottom left
        # print(rc, c)
        out = {}
        for s1 in [['x', 'w'], ['y', 'h']]:
            v = s1[0]
            d = s1[1]
            out[f'{v}0']=c[v]-crops[f'{d}c']
            out[f'{v}f']=c[v]-crops[f'{d}c']+crops[d]
    else:  
        for s1 in [['y','h'], ['x', 'w']]:
            v = s1[0]
            d = s1[1]
            if f'{v}c' in crops and d in crops:
                # centered coordinates
                out[f'{v}0'] = crops[f'{v}c']-crops[d]/2
                out[f'{v}f'] = crops[f'{v}c']+crops[d]/2
            else:
                # start/end coordinates
                for s in [f'{v}0', f'{v}f']:
                    if s in crops:
                        out[s] = crops[s]
    return out

def cropEdge(h:int, w:int, d:int) -> dict:
    '''crop a uniform distance d around the edge'''
    return {'x0':d, 'xf':w-d, 'y0':d, 'yf':h-d}

def cropTBLR(h:int, w:int, crops:dict) -> dict:
    '''crop one length dx from left and right and another length dy from top and bottom'''
    dx = crops['dx']
    dy = crops['dy']
    return {'x0':dx, 'xf':w-dx, 'y0':dy, 'yf':h-dy}

def crop0(w:int, x0:int) -> Tuple[int, int]:
    '''get the padding and position for a 0 position'''
    if x0>w:
        pad = w
        out = w
    elif x0<0:
        pad = -x0
        out = 0
    else:
        out = x0
        pad = 0
    return int(out), int(pad)

def cropf(w:int, xf:int, x0:int) -> Tuple[int, int]:
    '''get the padding and position for a 0 position'''
    if xf>w:
        pad = xf-w
        out = w
    elif xf<x0:
        pad = xf-x0
        out = x0
    else:
        out = xf
        pad = 0
    return int(out), int(pad)

def crop0f(h:int, w:int, crops:dict) -> Tuple[dict, dict]:
    '''crop to given boundaries and find the padding on each side'''
    out = {'x0':0, 'xf':w, 'y0':0, 'yf':h}
    pad = {'x0':0, 'xf':0, 'y0':0, 'yf':0}
    if 'x0' in crops:
        out['x0'], pad['x0'] = crop0(w, crops['x0'])
    if 'y0' in crops:
        out['y0'], pad['y0'] = crop0(h, crops['y0'])
    if 'xf' in crops:
        out['xf'], pad['xf'] = cropf(w, crops['xf'], out['x0'])
    if 'yf' in crops:
        out['yf'], pad['yf'] = cropf(h, crops['yf'], out['y0'])
    return out, pad

def convertCropHW(h:int, w:int, crops:Union[Dict, int])->dict:
    if type(crops) is int:
        return cropEdge(h,w,d)
    elif 'dx' in crops and 'dy' in crops:
        return cropTBLR(h,w,crops)
    else:
        return crop0f(h,w,crops)[0]


def convertCrop(im:np.array, crops:Union[Dict, int]) -> dict:
    '''convert the given crop dimensions to coordinates that will definitely fit in the image'''
    if len(im)>0:
        h,w = im.shape[0:2]
    else:
        return {}
    return convertCropHW(h,w,crops)
    
        
    
def convertCropPad(im:np.array, crops:Union[Dict, int]) -> Tuple[dict, dict]:
    '''convert the given crop dimensions to coordinates that will fit, and get the size of the padding on the side'''
    if len(im)>0:
        h,w = im.shape[0:2]
    else:
        return {}
    pad = {'x0':0, 'xf':0, 'y0':0, 'yf':0}
    if type(crops) is int:
        return cropEdge(h,w,d), pad
    elif 'dx' in crops and 'dy' in crops:
        return cropTBLR(h,w,crops), pad
    else:
        return crop0f(h,w,crops)
    
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
    
def imcropPad(im:np.array, bounds:Union[dict, int]) -> np.array:
    '''crop an image to the bounds and pad it with white if it doesn't fit'''
    crop, pad = convertCropPad(im, bounds)
    if len(crop)==0:
        return im
    im = imcropDict(im, crop)
    dst = cv.copyMakeBorder(im, pad['y0'], pad['yf'], pad['x0'], pad['xf'], cv.BORDER_CONSTANT, None, [255, 255, 255])
    return dst
    
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
