#!/usr/bin/env python
'''Functions for collecting data from stills of single lines'''

# external packages
import os, sys
import traceback
import logging
import pandas as pd
from matplotlib import pyplot as plt
from typing import List, Dict, Tuple, Union, Any, TextIO
import re
import numpy as np
import cv2 as cv
import shutil
import subprocess
import time

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
from im.imshow import imshow
from tools.plainIm import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)
pd.set_option('display.max_rows', 500)


#----------------------------------------------

def openImageInPaint(folder:str, st:str, i:int) -> None:
    '''open the image in paint. this is useful for erasing smudges or debris that are erroneously detected by the algorithm as filaments'''
    file = stitchFile(folder, st, i)
    if not os.path.exists(file):
        return
    subprocess.Popen([cfg.path.paint, file]);



def sem(l:list) -> float:
    l = np.array(l)
    l = l[~np.isnan(l)]
    if len(l)==0:
        return np.nan
    return np.std(l)/np.sqrt(len(l))
    

def getRoughness(componentMask:np.array, diag:int=0) -> float:
    '''measure roughness as perimeter of object / perimeter of convex hull. 
    componentMask is a binarized image of just one segment'''
    contours = cv.findContours(componentMask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    if int(cv.__version__[0])>=4:
        contours = contours[0]
    else:
        contours = contours[1]
    if len(contours)==0:
        return -1
    contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True) # select the largest contour
    cnt = contours[0]
    cnt = cv.approxPolyDP(cnt, 1, True) # smooth the contour by 1 px
    perimeter = cv.arcLength(cnt,True)
    if perimeter==0:
        return {}, {}
    hull = cv.convexHull(cnt)
    hullperimeter = cv.arcLength(hull,True)
    roughness = perimeter/hullperimeter-1  # how much extra perimeter there is compared to the convex hull
    if diag:
        # show annotated image
        cm = componentMask.copy()
        cm = cv.cvtColor(cm,cv.COLOR_GRAY2RGB)
        cv.drawContours(cm, [hull], -1, (110, 245, 209), 6)
        cv.drawContours(cm, cnt, -1, (186, 6, 162), 6)
        
        x,y,w,h = cv.boundingRect(cnt)
        cm = cm[max(y-5,0):min(y+h+5, cm.shape[0]),max(x-5, 0):min(x+w+5, cm.shape[1])]
        imshow(cm)
    return roughness

def widthInRow(row:list) -> int:
    '''distance between first and last 255 value of row'''
    first,last = bounds(row)
    return last-first

def boundsInArray(arr:np.array) -> np.array:
    '''left and right bounds in the array'''
    if arr.sum()==0:
        return []
    a2 = np.stack(np.where(arr)).transpose()
    idx = np.where(np.diff(a2[:,0])!=0)[0]+1
    a3 = np.split(a2,list(idx))
    
    return np.array([[i[0,1],i[-1,1]] for i in a3])
    

def widthsInArray(arr:np.array) -> list:
    '''get the distance between first and last nonzero value of each row'''
    if arr.sum()==0:
        return []
    a2 = np.stack(np.where(arr)).transpose()  # get positions of 255
    idx = np.where(np.diff(a2[:,0])!=0)[0]+1  # find changes in row
    a3 = np.split(a2,list(idx))               # split into rows
    return [i[-1,1]-i[0,1] for i in a3]              # get distance between first and last
    

def bounds(row:list) -> Tuple[int,int]:
    '''get position of first and last 255 value in row'''
    if not type(row) is list:
        row = list(row)
    if not 255 in row:
        return -1, -1
    last = len(row) - row[::-1].index(255) 
    first = row.index(255)
    return first,last



def meanBounds(chunk:np.array, rows:bool=True) -> Tuple[float,float]:
    '''get average bounds across rows or columns'''
    if not rows:
        chunk = chunk.transpose()
    b = boundsInArray(chunk)
    if len(b)==0:
        return -1,-1
    x0 = np.mean(b[:,0])
    xf = np.mean(b[:,1])
    return x0,xf



        
def closestIndex(val:float, l1:list) -> int:
    '''index of closest value in list l1 to value val'''
    l2 = [abs(x-val) for x in l1]
    return l2.index(min(l2))


        
def difference(do:pd.Series, wo:pd.Series, s:str) -> float:
    '''get difference between values'''
    if hasattr(do, s) and hasattr(wo, s) and not pd.isna(do[s]) and not pd.isna(wo[s]):
        return do[s]-wo[s]
    else:
        raise ValueError('No value detected')
        
def convertValue(key:str, val:list, units_in:dict, pxpmm:float, units_out:dict, vals_out:dict) -> Tuple:
    '''convert the values from px to mm'''
    uke = units_in[key]
    if uke=='px':
        c = 1/pxpmm
        u2 = 'mm'
    elif uke=='px^2':
        c = 1/pxpmm**2
        u2 = 'mm^2'
    elif uke=='px^3':
        c = 1/pxpmm**3
        u2 = 'mm^3'
    else:
        c = 1
        u2 = uke
    units_out[key] = u2
    units_out[f'{key}_SE'] = u2
    units_out[f'{key}_N'] = ''
    vals_out[key] = np.mean(val)*c
    vals_out[f'{key}_SE'] = sem(val)*c
    vals_out[f'{key}_N'] = len(val)
    
def calcVest(h:float, r:float) -> float:
    '''estimate the volume of an object, assuming it is a cylinder with spherical end caps'''
    if h>2*r:
        vest = (h - 2*r)*np.pi*(r)**2 + 4/3*np.pi*r**3 # cylinder + hemisphere endcaps
    else:
        vest = 4/3*np.pi*r**2*(h/2) # ellipsoid
    return vest
    
    



