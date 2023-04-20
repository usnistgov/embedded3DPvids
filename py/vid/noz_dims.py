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
from im.imshow import imshow
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
        self.xL = -1
        self.xR = -1
        self.yB = -1
        self.xM = -1
        self.w = 790
        self.h = 590
        self.pfd = pfd
        self.pxpmm = self.pfd.pxpmm()
        self.nozDetected = False
        if importDims:
            self.importNozzleDims()
        
        
    def adjustForCrop(self, crops:dict) -> None:
        self.xL = int(self.xL + crops['x0'])
        self.xR = int(self.xR + crops['x0'])
        self.xM = int(self.xM + crops['x0'])
        self.yB = int(self.yB + crops['y0'])
        
    def padNozzle(self, left:int=0, right:int=0, bottom:int=0):
        self.xL = self.xL-left
        self.xR = self.xR+right
        self.yB = self.yB+bottom
        
    def nozDims(self) -> dict:
        '''get the nozzle dimensions
        yB is from top, xL and xR are from left'''
        return {'xL':self.xL, 'xR':self.xR, 'yB':self.yB}
    
    def nozDimsFN(self) -> str:
        '''file name of nozzle dimensions table'''
        # store noz dimensions in the subfolder
        return self.pfd.newFileName('nozDims', 'csv')
    
    def setSize(self, im:np.array) -> None:
        '''set the current dimensions equal to the dimensions of the image'''
        self.h, self.w = im.shape[:2]
        
    def setDims(self, d:dict) -> None:
        '''adopt the dimensions in the dictionary'''
        if 'xL' in d:
            self.xL = int(d['xL'])
        if 'xR' in d:
            self.xR = int(d['xR'])
        if 'yB' in d:
            self.yB = int(d['yB'])
        self.xM = int((self.xL+self.xR)/2)
        
    def copyDims(self, nd:nozDims) -> None:
        '''copy dimensions from another nozDims object'''
        self.yB = nd.yB
        self.xL = nd.xL
        self.xR = nd.xR
        self.xM = nd.xM
    
    
    def exportNozzleDims(self, overwrite:bool=False) -> None:
        '''export the nozzle location to file'''
        fn = self.nozDimsFN()  # nozzle dimensions file name
        if os.path.exists(fn) and not overwrite:
            return
        plainExpDict(fn, {'yB':self.yB, 'xL':self.xL, 'xR':self.xR, 'pxpmm':self.pxpmm, 'w':self.w, 'h':self.h})
        
        
    def importNozzleDims(self) -> int:
        '''find the target pressure from the calibration file. returns 0 if successful, 1 if not'''
        fn = self.nozDimsFN()      # nozzle dimensions file name
        if not os.path.exists(fn):
            self.nozDetected = False
            return 1
        tlist = ['yB', 'xL', 'xR', 'pxpmm', 'w', 'h']
        d,_ = plainImDict(fn, unitCol=-1, valCol=1)
        for st,val in d.items():
            setattr(self, st, int(val))
        if len(set(tlist)-set(d))==0:
            # we have all values
            self.xM = int((self.xL+self.xR)/2)
            self.nozDetected = True
            return 0
        else:
            return 1
        
        
    def absoluteCoords(self, d:dict) -> dict:
        '''convert the relative coordinates in mm to absolute coordinates on the image in px. y is from the bottom, x is from the left'''
        nc = [self.xM, self.yB]    # convert y to from the bottom
        out = {'x':nc[0]+d['dx']*self.pxpmm, 'y':nc[1]-d['dy']*self.pxpmm}
        return out
    
    def relativeCoords(self, x:float, y:float, reverse:bool=False) -> dict:
        '''convert the absolute coordinates in px to relative coordinates in px, where y is from the top and x is from the left. reverse=True to go from mm to px'''
        nx = self.xM
        ny = self.yB
        if reverse:
            return x*self.pxpmm+nx, ny-y*self.pxpmm
        else:
            return (x-nx)/self.pxpmm, (ny-y)/self.pxpmm
        
    def nozWidth(self):
        '''nozzle width in mm'''
        return (self.xR-self.xL)/self.pxpmm
    
    def nozCover(self, padLeft:int=0, padRight:int=0, padBottom:int=0, val:int=255, y0:int=0, color:bool=False, **kwargs) -> np.array:
        '''get a mask that covers the nozzle'''
        if type(val) is list:
            mask = np.zeros((self.h, self.w), dtype="uint8")
        else:
            mask = np.zeros((self.h, self.w), dtype="uint8")
        if y0<0:
            y0 = self.yB+y0
        yB = int(self.yB + padBottom)
        xL = int(self.xL - padLeft)
        xR = int(self.xR + padRight)
        mask[y0:yB, xL:xR]=val
        if color and len(mask.shape)==2:
            mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        if 'crops' in kwargs:
            mask = vc.imcrop(mask, kwargs['crops'])
        return mask
    
    def dentHull(self, hull:list, crops:dict) -> list:
        '''conform the contour to the nozzle'''
        yB = int(self.yB-crops['y0'])
        xL = int(self.xL-crops['x0'])
        xR = int(self.xR-crops['x0'])
        df = pd.DataFrame(hull[:,0] , columns=['x', 'y'])
        if xL-10>df.x.max() or xL<df.x.min() or yB<df.y.min():
            # all points are left or right of the left edge of the nozzle or below the nozzle
            return hull
        under = df[(df.x>xL-1)&(df.x<xR+1)&(df.y>yB-1)]
        if len(under)==0:
            return hull
        left = df[(df.x<xL+1)&(df.x>xL-10)&(df.y<yB)]
        if len(left)==0:
            i = under.index.min()
        else:
            # points go clockwise, so the under point will be after the left point
            i = left.index.max()+1  # index of the transition point
        xL = max(hull[i-1, 0, 0],xL)
        # if i<len(hull):
        #     yB = hull[i, 0, 1]
        hull = np.vstack([hull[:i, 0, :], np.array([xL, yB]), hull[i:, 0, :]])
        return hull
        