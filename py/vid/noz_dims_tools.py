#!/usr/bin/env python
'''Class for storing dimensions of the nozzle'''

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
from im.imshow import imshow
from im.contour import getContours
import im.morph as vm
import im.crop as vc
from tools.config import cfg
from tools.plainIm import *
import file.file_handling as fh
from v_tools import vidData
from noz_plots import *


# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)


#----------------------------------------------

class nozDims:
    '''holds dimensions of the nozzle'''
    
    def __init__(self, pfd:fh.printFileDict, importDims:bool=True):
        self.w = 790
        self.h = 590
        self.pfd = pfd
        self.pxpmm = self.pfd.pxpmm()
        self.nozDetected = False
        
        
    def adjustForCrop(self, crops:dict, xlist:list, ylist:list) -> None:
        '''shift the nozzle dimensions relative to how we cropped the image'''
        for x in xlist:
            setattr(self, x, int(getattr(self, x)+crops['x0']))
        for y in ylist:
            setattr(self, y, int(getattr(self, y)+crops['y0']))
    
    def nozDimsFN(self) -> str:
        '''file name of nozzle dimensions table'''
        # store noz dimensions in the subfolder
        return self.pfd.newFileName('nozDims', 'csv')
    
    def setSize(self, im:np.array) -> None:
        '''set the current dimensions equal to the dimensions of the image'''
        self.h, self.w = im.shape[:2]
    
    
    def exportNozzleDims(self, d:dict, overwrite:bool=False) -> None:
        '''export the nozzle location to file'''
        fn = self.nozDimsFN()  # nozzle dimensions file name
        if os.path.exists(fn) and not overwrite:
            return
        plainExpDict(fn,d)
        
        
    def importNozzleDims(self, tlist:list) -> int:
        '''find the target pressure from the calibration file. returns 0 if successful, 1 if not'''
        fn = self.nozDimsFN()      # nozzle dimensions file name
        if not os.path.exists(fn):
            self.nozDetected = False
            return 1
        d,_ = plainImDict(fn, unitCol=-1, valCol=1)
        for st,val in d.items():
            setattr(self, st, int(val))
        if len(set(tlist)-set(d))==0:
            # we have all values
            self.nozDetected = True
            return 0
        else:
            return 1
        
        
    def absoluteCoords(self, d:dict) -> dict:
        '''convert the relative coordinates in mm to absolute coordinates on the image in px. y is from the bottom, x is from the left'''  
        xc,yc = self.nozCenter()
        out = {'x':xc+d['dx']*self.pxpmm, 'y':yc-d['dy']*self.pxpmm} # convert y to from the bottom
        return out 
    
    def relativeCoords(self, x:float, y:float, reverse:bool=False) -> dict:
        '''convert the absolute coordinates in px to relative coordinates in px, where y is from the top and x is from the left. reverse=True to go from mm to px'''
        nx,ny = self.nozCenter()
        if reverse:
            return x*self.pxpmm+nx, ny-y*self.pxpmm
        else:
            return (x-nx)/self.pxpmm, (ny-y)/self.pxpmm
        
    def nozWidth(self):
        '''nozzle width in mm'''
        return self.nozWidthPx()/self.pxpmm
        
    def dentHull(self, hull:list, crops:dict) -> list:
        '''do not do this. meant for side views'''
        return hull
        