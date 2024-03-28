#!/usr/bin/env python
'''Functions for storing dimensions of the nozzle, for side views where the nozzle is a rectangle'''

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

class nozDimsSide(nozDims):
    '''holds dimensions of the nozzle'''
    
    def __init__(self, pfd:fh.printFileDict, importDims:bool=True):
        self.xL = -1
        self.xR = -1
        self.yB = -1
        self.xM = -1
        super().__init__(pfd, importDims)
        if importDims:
            self.importNozzleDims()
        
        
    def adjustForCrop(self, crops:dict) -> None:
        '''shift the nozzle position relative to how we cropped the image'''
        super().adjustForCrop(crops, ['xL', 'xR', 'xM'], ['yB'])
        
    def padNozzle(self, left:int=0, right:int=0, bottom:int=0):
        '''add white space around the nozzle'''
        self.xL = self.xL-left
        self.xR = self.xR+right
        self.yB = self.yB+bottom
        
    def nozDims(self) -> dict:
        '''get the nozzle dimensions
        yB is from top, xL and xR are from left'''
        return {'xL':self.xL, 'xR':self.xR, 'yB':self.yB}
        
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
        super().exportNozzleDims({'yB':self.yB, 'xL':self.xL, 'xR':self.xR, 'pxpmm':self.pxpmm, 'w':self.w, 'h':self.h}, overwrite=overwrite)
        
        
    def importNozzleDims(self) -> int:
        '''find the target pressure from the calibration file. returns 0 if successful, 1 if not'''
        out = super().importNozzleDims(['yB', 'xL', 'xR', 'pxpmm', 'w', 'h'])
        if out==0:
            self.xM = int((self.xL+self.xR)/2)
        return out
        
    def nozCenter(self) -> tuple:
        '''get position of the center of the nozzle'''
        return self.xM, self.yB
        
    def nozWidthPx(self):
        '''nozzle outer diameter in px'''
        return (self.xR-self.xL)
    
    def nozBounds(self) -> dict:
        '''get the coordinates of the edges of the nozzle'''
        return {'x0':self.xL, 'xf':self.xR, 'y0':0, 'yf':self.yB}
    
    def nozCover(self, padLeft:int=0, padRight:int=0, padBottom:int=0, val:int=255, y0:int=0, color:bool=False, **kwargs) -> np.array:
        '''get a mask that covers the nozzle'''
        if type(val) is list:
            mask = np.zeros((self.h, self.w, len(val)), dtype="uint8")
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
        
        image1 = np.zeros((self.h, self.w), dtype=np.uint8)
        cv.drawContours(image1, [hull], -1, 1, 1)
        hull2 = getContours(image1, cv.CHAIN_APPROX_NONE)[0]
        df = pd.DataFrame(hull2[:,0], columns=['x', 'y'])
        insidepts = df[(df.x>=xL)&(df.x<=xR)&(df.y<=yB)]

        if len(insidepts)==0 or (insidepts.y.min()==yB) or (insidepts.x.min()==xR) or (insidepts.x.max()==xL):
            # no points to dent
            return hull
        
        # find the points that intersect the nozzle
        leftedge = insidepts[insidepts.x==insidepts.x.min()]
        rightedge = insidepts[insidepts.x==insidepts.x.max()]
        li = leftedge[leftedge.y==leftedge.y.max()].iloc[0].name
        ri = rightedge[rightedge.y==rightedge.y.max()].iloc[-1].name     
        lpt = df.loc[li]   # point at left
        rpt = df.loc[ri]   # point at right
        
        # get the nozzle points
        nozpts = []
        if lpt.x==xL:
            # intersect with left edge of nozzle
            nozpts.append([xL, lpt.y])
            nozpts.append([xL, yB])
            if rpt.y==yB:
                # intersect with bottom edge of nozzle
                nozpts.append([rpt.x, yB])
            elif rpt.x==xR:
                # intersect with right edge of nozzle
                nozpts.append([xR, yB])
                nozpts.append([xR, rpt.y])
            elif rpt.x<xR:
                nozpts.append([rpt.x, rpt.y])
            else:
                
                raise ValueError('Unexpected intersection points')
        elif lpt.y==yB:
            nozpts.append([lpt.x, yB])
            if rpt.x==xR:
                # intersect with right edge of nozzle
                nozpts.append([xR, yB])
                nozpts.append([xR, rpt.y])
            else:
                raise ValueError('Unexpected intersection points')
        else:
            raise ValueError('Unexpected intersection points')
            
        # reconstitute the list of points
        if li>ri or ri>li:
            # points go counterclockwise
            nozpts.reverse()
            hull3 = np.vstack([hull2[:ri, 0, :], nozpts, hull2[li:, 0, :]])
        else:
            hull3 = np.vstack([hull2[:li, 0, :], nozpts, hull2[ri:, 0, :]])
            
        # simplify the list of points
        image2 = np.zeros((self.h, self.w), dtype=np.uint8)
        cv.drawContours(image2, [hull3], -1, 1, 1)
        hull4 = getContours(image2, cv.CHAIN_APPROX_SIMPLE)[0]
        return hull4[:,0,:]
        