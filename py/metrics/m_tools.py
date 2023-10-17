#!/usr/bin/env python
'''Functions for collecting data from stills of single lines'''

# external packages
import os, sys
import magic
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
import pyautogui

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
from im.imshow import imshow
from tools.plainIm import *
from tools.config import cfg
from m_stats import *
import file_handling as fh

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)
pd.set_option('display.max_rows', 500)

#----------------------------------------------

class paintObject:
    
    def __init__(self, file:str, fullScreen:bool=True, scrollDown:bool=False, scrollRight:bool=True, dropper:bool=False, thickness:int=3, **kwargs):
        self.file = file
        pyautogui.PAUSE = 0.1
        self.w = pyautogui.size().width
        self.h = pyautogui.size().height
        self.openPaint()
        time.sleep(1)
        self.selectPencil()
        self.selectDropper()
        self.selectTopRight()
        self.magnify(5)
        self.selectPencil()
        self.selectThickness(thickness)
        if scrollRight:
            self.scrollRight()
        if scrollDown:
            self.scrollDown()
        if dropper:
            self.selectDropper()
        self.moveFrac(3/4, 1/2)
        
    def openPaint(self):
        subprocess.Popen(["cmd", "/c", "start", "/max", cfg.path.paint, self.file]);   
        
    def selectPencil(self):
        pyautogui.moveTo(347, 115) # Move the mouse to the pencil button
        pyautogui.click() # Click the mouse at its current location.
        
    def selectDropper(self):
        pyautogui.moveTo(386, 159) # Move the mouse to the dropper button
        pyautogui.click() # Click the mouse at its current location.  
        
    def selectTopRight(self):
        pyautogui.moveTo(700, 376) # select from top right
        pyautogui.click() # Click the mouse at its current location.  
        
    def selectMagnifier(self):
        pyautogui.moveTo(426, 159) # Move the mouse to the magnifier button
        pyautogui.click() # Click the mouse at its current location. 
        
    def magnify(self, n:int):
        self.selectMagnifier()
        pyautogui.moveTo(548, 376) # select from top right
        for i in range(n):
            pyautogui.click() # zoom
            
    def selectThickness(self, t:int):  
        pyautogui.moveTo(347, 115) # Move the mouse to the pencil button
        pyautogui.click() # Click the mouse at its current location.
        pyautogui.moveTo(908, 114) # Move the mouse to the line thickness button
        pyautogui.click() # Click the mouse at its current location.
        time.sleep(0.25)
        y = {1:223, 2:276, 3:345, 4:391}[t]
        pyautogui.moveTo(908, y) # Move the mouse to the select 2nd line thickness
        pyautogui.click() # Click the mouse at its current location.
        
    def scrollRight(self):
        pyautogui.moveTo(300, self.h-120) # Move the mouse to the scroll button on bottom
        pyautogui.mouseDown()
        pyautogui.moveTo(self.w-300, self.h-120) # Move the mouse to the scroll button on bottom
        pyautogui.mouseUp()
        
    def scrollDown(self):
        pyautogui.moveTo(3821, 355) # Move the mouse to the scroll button on bottom
        pyautogui.mouseDown()
        pyautogui.moveTo(3821, 1380) # Move the mouse to the scroll button on bottom
        pyautogui.mouseUp()
        
    def moveFrac(self, wfrac:float, hfrac:float):
        pyautogui.moveTo(int(self.w*wfrac), int(self.h*hfrac)) # Move the mouse to the bottom right

    
def openInPaint(file, **kwargs):
    '''open the file in MS paint'''
    paintObject(file, **kwargs)

    
def cursorFinder():
    while True:
        x, y = pyautogui.position()
        positionStr = 'X: ' + str(x).rjust(4) + ' Y: ' + str(y).rjust(4) + '\n'
        print(positionStr, end='')
        print('\b' * len(positionStr), end='', flush=True)
        time.sleep(0.01)
    
    
def openInExcel(file):
    '''open the file in MS excel'''
    subprocess.Popen([cfg.path.excel, file]);    

def ppdist(p1:list, p2:list) -> float:
    '''distance between 2 points'''
    d = 0
    for i in range(len(p1)):
        d = d + (float(p2[i])-float(p1[i]))**2
    d = np.sqrt(d)
    return d


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
        

    
def calcVest(h:float, r:float) -> float:
    '''estimate the volume of an object, assuming it is a cylinder with spherical end caps'''
    if h>2*r:
        vest = (h - 2*r)*np.pi*(r)**2 + 4/3*np.pi*r**3 # cylinder + hemisphere endcaps
    else:
        vest = 4/3*np.pi*r**2*(h/2) # ellipsoid
    return vest
    
def get_image_size(file_path):
    """
    Return (width, height) for a given img file content - no external
    dependencies except the os and struct modules from core
    """
    if not os.path.exists(file_path):
        return 0,0
    t = magic.from_file(file_path)
    v = re.search('(\d+) x (\d+)', t).groups()
    w = int(v[0])
    h = int(v[1])
    return w,h


def whiteoutFile(file:str, val:int=255) -> None:
    '''white out the whole file'''
    if not os.path.exists(file):
        return
    im = cv.imread(file)
    if len(im.shape)==3:
        im[:,:,:] = val
    else:
        im[:,:] = val
    cv.imwrite(file, im)
    ff = file.replace(cfg.path.server, '')
    if val==255:
        logging.info(f'Whited out {ff}')
    elif val==0:
        logging.info(f'Blacked out {ff}')
    else:
        logging.info(f'Covered up {ff}')
    
    
def whiteoutAll(file:str) -> None:
    '''whiteout the file, the cropped file, the ML file, and the Usegment file'''
    if not 'vstill' in file:
        raise ValueError(f'whiteoutAll failed on {file}. Only allowed for vstill files')
    whiteoutFile(file, val=255)
    bn = os.path.basename(file)
    folder = os.path.dirname(file)
    cropfile = os.path.join(folder, 'crop', bn.replace('vstill', 'vcrop'))
    if os.path.exists(cropfile):
        whiteoutFile(cropfile, val=255)
    ufile = os.path.join(folder, 'Usegment', bn.replace('vstill', 'Usegment'))
    mfile = os.path.join(folder, 'MLsegment', bn.replace('vstill', 'MLsegment'))
    mfile2 = os.path.join(folder, 'MLsegment2', bn.replace('vstill', 'MLsegment2'))
    for filei in [ufile, mfile, mfile2]:
        if os.path.exists(filei):
            whiteoutFile(filei, val=0)

def whiteOutFiles(folder:str, canMatch:list=[], mustMatch:list=[]) -> None:
    '''whiteout all files that match the strings'''
    for file in os.listdir(folder):
        if 'vstill' in file and fh.anyIn(canMatch, file) and fh.allIn(mustMatch, file):
            whiteoutAll(os.path.join(folder, file))