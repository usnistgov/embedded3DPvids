#!/usr/bin/env python
'''Functions for collecting data from stills of xs'''

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
import copy

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
sys.path.append(os.path.dirname(os.path.dirname(currentdir)))
from file_metric import *


# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)

#----------------------------------------------

class fileXS(fileMetric):
    '''collects data about XS segments'''
    
    def __init__(self, file:str, diag:int=0, acrit:int=2500, **kwargs):
        super().__init__(file, diag=diag, acrit=acrit, **kwargs)
        
        
    def filterXSComponents(self) -> None:
        '''filter out cross-section components that don't make sense'''
        errorRet = [], []
        h,w = self.segmenter.labelsBW.shape[:2]
        xest = w/2 # estimated x
        if h>600:
            yest = h-300
            dycrit = 200
        else:
            yest = h/2
            dycrit = h/2
        if len(self.segmenter.df)>1:
            secondLargest = 2*list(self.segmenter.df.a.nlargest(2))[1]
        seg1 = copy.deepcopy(self.segmenter)    # make a copy of the segmenter in case we need to roll back changes  
        self.segmenter.eraseBorderComponents(10)  # remove anything too close to the border
        goodpts = (abs(self.segmenter.df.xc-xest)<100)&(abs(self.segmenter.df.yc-yest)<dycrit)
        self.segmenter.selectComponents(goodpts)
            # filter by location relative to expectation and area
        if not self.segmenter.success:
            self.segmenter = seg1
            self.segmenter.selectComponents(self.segmenter.df.a>1000)   # just filter by area
        if len(self.segmenter.df)>1 and self.segmenter.df.a.max() < secondLargest:
            # largest object not much larger than 2nd largest
            self.segmenter.success = False
        
    
    def dims(self) -> None:
        '''get the dimensions of the segments'''
        roughness = getRoughness(self.segmenter.labelsBW, diag=max(0,self.diag-1))
        m = self.segmenter.largestObject() # select largest object
        self.x0 = int(m['x0'])
        self.y0 = int(m['y0'])
        self.w = int(m['w'])
        self.h = int(m['h'])
        area = int(m['a'])
        self.xc = m['xc']
        self.yc = m['yc']
        aspect = self.h/self.w # height/width
        boxcx = self.x0+self.w/2 # x center of bounding box
        boxcy = self.y0+self.h/2 # y center of bounding box
        xshift = (self.xc-boxcx)/self.w
        yshift = (self.yc-boxcy)/self.h
        self.units = {'line':'', 'aspect':'h/w', 'xshift':'w', 'yshift':'h', 'area':'px'
                  , 'w':'px', 'h':'px'
                      , 'xc':'px', 'yc':'px', 'roughness':''} # where pixels are in original scale
        self.stats = {'line':self.name, 'aspect':aspect, 'xshift':xshift, 'yshift':yshift, 'area':area*self.scale**2
                  , 'w':self.w*self.scale, 'h':self.h*self.scale
                      , 'xc':self.xc*self.scale, 'yc':self.yc*self.scale, 'roughness':roughness}


    def display(self, title:str='') -> None:
        '''display diagnostics'''
        if self.diag<=0:
            return
        # show the image with annotated dimensions
        self.segmenter.display()
        if hasattr(self.segmenter, 'labelsBW'):
            im2 = cv.cvtColor(self.segmenter.labelsBW,cv.COLOR_GRAY2RGB)
        else:
            im2 = self.im.copy()
        imgi = self.im.copy()
        if hasattr(self, 'x0'):
            cv.rectangle(imgi, (self.x0,self.y0), (self.x0+self.w,self.y0+self.h), (0,0,255), 1)   # bounding box
            cv.circle(imgi, (int(self.xc), int(self.yc)), 2, (0,0,255), 2)     # centroid
        # cv.circle(imgi, (self.x0+int(self.w/2),self.y0+int(self.h/2)), 2, (0,255,255), 2) # center of bounding box
        if hasattr(self, 'idealspx'):
            io = {}
            for s in ['x0', 'xf', 'y0', 'yf']:
                io[s] = int(self.idealspx[s]/self.scale)
            cv.rectangle(imgi, (io['x0'],io['yf']), (io['xf'],io['y0']), (237, 227, 26), 1)   # bounding box of intended
        if hasattr(self, 'nozPx'):
            io = {}
            for s in ['x0', 'xf', 'y0', 'yf']:
                io[s] = int(self.nozPx[s]/self.scale)
            cv.rectangle(imgi, (io['x0'],io['yf']), (io['xf'],io['y0']), (0,0,0), 1)   # bounding box of nozzle
        if hasattr(self, 'hull'):
            # show the roughness
            imshow(imgi, self.roughnessIm(), self.statText(), title='xsFile')
        else:
            imshow(imgi, im2, self.statText(), title='xsFile')
        if hasattr(self, 'title'):
            plt.title(self.title)
        
    
    def singleMeasure(self) -> None:
        '''measure a single cross section'''
        self.filterXSComponents()
        if not self.segmenter.success:
            return 
        self.dims()
        self.display()
        
    def multiMeasure(self) -> None:
        '''measure multiple cross sections'''
        # find contours
        self.componentMask = self.segmenter.labelsBW
        
        out = self.getContour(combine=True)
        if out<0:
            return 
        self.x0,self.y0,self.w,self.h = cv.boundingRect(self.hull)   # x0,y0 is top left

        # measure components
        filledArea = self.segmenter.df.a.sum()
        emptiness = self.getEmptiness()
        roughness = self.getRoughness()

        
        aspect = self.h/self.w

        M = cv.moments(self.cnt)
        if M['m00']==0:
            self.xc=0
            self.yc=0
        else:
            self.xc = int(M['m10']/M['m00'])
            self.yc = int(M['m01']/M['m00'])
        boxcx = self.x0+self.w/2 # x center of bounding box
        boxcy = self.y0+self.h/2 # y center of bounding box
        xshift = (self.xc-boxcx)/self.w
        yshift = (self.yc-boxcy)/self.h
        self.stats['line'] = self.name
        self.units['line'] = ''
        units = {'segments':'', 
                      'aspect':'h/w', 'xshift':'w', 'yshift':'h', 'area':'px'
                      , 'x0':'px', 'y0':'px'
                      , 'xf':'px', 'yf':'px'
                        , 'xc':'px', 'yc':'px'
                         , 'w':'px', 'h':'px'
                      , 'emptiness':'', 'roughness':''} # where pixels are in original scale
        ret = {'segments':len(self.segmenter.df)
                      , 'aspect':aspect, 'xshift':xshift, 'yshift':yshift, 'area':filledArea*self.scale**2
                      , 'x0':self.x0*self.scale, 'y0':self.y0*self.scale
                      , 'xf':self.x0*self.scale+self.w*self.scale, 'yf':self.y0*self.scale+self.h*self.scale
                      , 'xc':self.xc*self.scale, 'yc':self.yc*self.scale
                     , 'w':self.w*self.scale, 'h':self.h*self.scale
                      , 'emptiness':emptiness, 'roughness':roughness}
        self.units = {**self.units, **units}
        self.stats = {**self.stats, **ret}
        