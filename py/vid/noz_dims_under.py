#!/usr/bin/env python
'''Functions for storing dimensions of the nozzle'''

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
import csv
import random
import time

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
from noz_dims_tools import *


# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)


#----------------------------------------------

class nozDimsUnder(nozDims):
    '''holds dimensions of the nozzle'''
    
    def __init__(self, pfd:fh.printFileDict, importDims:bool=True):
        self.xC = -1
        self.yC = -1
        self.r = -1
        super().__init__(pfd, importDims)
        if importDims:
            self.importNozzleDims()
        
        
    def adjustForCrop(self, crops:dict) -> None:
        return super().adjustForCrop(crops, ['xC'], ['yC'])
        
    def padNozzle(self, dr:int=0):
        self.r = self.r + dr
        
    def nozDims(self) -> dict:
        '''get the nozzle dimensions
        yB is from top, xL and xR are from left'''
        return {'xC':self.xC, 'yC':self.yC, 'r':self.r}
        
    def setDims(self, d:dict) -> None:
        '''adopt the dimensions in the dictionary'''
        if 'xC' in d:
            self.xC = int(d['xC'])
        if 'yC' in d:
            self.yC = int(d['yC'])
        if 'r' in d:
            self.r = int(d['r'])
        
    def copyDims(self, nd:nozDims) -> None:
        '''copy dimensions from another nozDims object'''
        self.xC = nd.xC
        self.yC = nd.yC
        self.r = nd.r
    
    
    def exportNozzleDims(self, overwrite:bool=False) -> None:
        '''export the nozzle location to file'''
        return super().exportNozzleDims({'yC':self.yC, 'xC':self.xC, 'r':self.r, 'pxpmm':self.pxpmm, 'w':self.w, 'h':self.h}, overwrite=overwrite)
        
        
    def importNozzleDims(self) -> int:
        '''find the target pressure from the calibration file. returns 0 if successful, 1 if not'''
        return super().importNozzleDims(['yC', 'xC', 'xC', 'pxpmm', 'w', 'h'])
        
    def nozCenter(self) -> tuple:
        return self.xC, self.yC

    def nozWidthPx(self):
        return (self.r*2)
    
    def nozCover(self, pad:int=0, val:int=255, y0:int=0, color:bool=False, **kwargs) -> np.array:
        '''get a mask that covers the nozzle'''
        if type(val) is list:
            mask = np.zeros((self.h, self.w, len(val)), dtype="uint8")
            white = (255,255,255)
        else:
            mask = np.zeros((self.h, self.w), dtype="uint8")
            white = 255
        if y0<0:
            y0 = self.yC+y0
        else:
            y0 = self.yC
        cv.circle(mask, (self.xC, y0), (self.r+pad), white,-1, 8, 0)
        if color and len(mask.shape)==2:
            mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        if 'crops' in kwargs:
            mask = vc.imcrop(mask, kwargs['crops'])
        return mask
    
    def nozBounds(self) -> dict:
        return {'x0':self.xC-self.r, 'xf':self.xC+self.r, 'y0':self.yC-self.r, 'yf':self.yC+self.r}

        