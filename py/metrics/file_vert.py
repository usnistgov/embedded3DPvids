#!/usr/bin/env python
'''Functions for collecting data from stills of single line xs'''

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
from file_metric import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)

#----------------------------------------------

class fileVert(fileMetric):
    '''collects data about vertical segments'''
    
    def __init__(self, file:str, diag:int=0, acrit:int=2500, **kwargs):
        super().__init__(file, diag=diag, acrit=acrit, **kwargs)
        
    def dims(self, numLines:int=1, largest:bool=True, getLDiff:bool=False) -> None:
        '''get the dimensions of the segments'''
        self.initializeTimeCounter('fileVert')
        df2 = self.segmenter.df.copy()
        if len(df2)==0:
            return
        
        if numLines==1:
            # take only components in line with the largest component
            filI = df2.a.idxmax() # index of filament label, largest remaining object
            component = df2.loc[filI]
            inline = df2.copy()
            inline = inline[(inline.x0>component['x0']-50)&(inline.x0<component['x0']+50)] # other objects inline with the biggest object
        else:
            # take all components
            inline = df2

        # measure overall dimensions
        inline['xf'] = inline.x0+inline.w
        inline['yf'] = inline.y0+inline.h
        self.x0 = int(inline.x0.min())  # unscaled
        self.y0 = int(inline.y0.min())  # unscaled
        self.xf = int(inline.xf.max()) # unscaled 
        self.yf = int(inline.yf.max()) # unscaled 
        self.w = self.xf - self.x0    # unscaled 
        if numLines==1:
            self.h = int(inline.h.sum())    # unscaled
        else:
            self.h = int(self.yf - self.y0)
        self.xc = int(sum(inline.a * inline.xc)/sum(inline.a))
        self.yc = int(sum(inline.a * inline.yc)/sum(inline.a))
        co = {'area':int(inline.a.sum())*self.scale**2
                     , 'x0':self.x0*self.scale, 'y0':self.y0*self.scale
                      , 'w':self.w*self.scale, 'h':self.h*self.scale
                      , 'xf':self.xf*self.scale, 'yf':self.yf*self.scale
                     , 'xc':self.xc*self.scale, 'yc':self.yc*self.scale, 'segments':len(inline)} 
        units = {'line':'', 'aspect':'h/w', 'area':'px'
                 ,'x0':'px', 'y0':'px', 'xf':'px', 'yf':'px', 'w':'px', 'h':'px'
                 , 'xc':'px', 'yc':'px', 'segments':''} # where pixels are in original scale
        
        # get combined mask of all objects in line
        if numLines==1:
            self.componentMask = self.segmenter.reconstructMask(inline)
        else:
            self.componentMask = self.segmenter.labelsBW.copy()
        # measure roughness, thickness, etc.
        componentMeasures, cmunits = self.measureComponent(horiz=False, reverse=True, diag=max(0,self.diag-1), combine=not largest, emptiness=True, atot=inline.a.sum())
        if len(componentMeasures)==0:
            return
        
        # component measures and co are pre-scaled
        aspect = co['h']/componentMeasures['meanT'] # height/width
        ret = {**{'line':self.name, 'aspect':aspect}, **co}
        units = {**units}
        
        r = componentMeasures['meanT']/2
        ret['vest'] = calcVest(co['h'], r)
        units['vest'] = 'px^3'

        if getLDiff:
            ret['ldiff'] = self.getLDiff(horiz=False)/self.h
            units['ldiff'] = 'h'

        ret = {**ret, **componentMeasures}
        units = {**units, **cmunits}
        self.stats = {**self.stats, **ret}
        self.units = {**self.units, **units}
        
    def gaps(self, distancemm:float) -> None:
        if not hasattr(self, 'nd') or not hasattr(self, 'crop') or 'o' in self.name:
            return
        # get displacements
        disps = self.displacement('z', distancemm*self.nd.pxpmm)
        dispunits = dict([[ii, 'px'] for ii in disps])
        self.stats = {**self.stats, **disps}
        self.units = {**self.units, **dispunits}
        
   
    def display(self) -> None:
        '''display diagnostics'''
        if self.diag<=0:
            return
        self.segmenter.display()
        if hasattr(self, 'componentMask'):
            im2 = cv.cvtColor(self.componentMask,cv.COLOR_GRAY2RGB)
        else:
            im2 = self.im.copy()
        if hasattr(self, 'im0'):
            imgi = self.im0.copy() 
        else:
            imgi = np.zeros(im2.shape, dtype=np.uint8)
        for im in [im2, imgi]:
            if hasattr(self, 'x0'):
                cv.rectangle(im, (self.x0,self.y0), (self.x0+self.w,self.y0+self.h), (0,0,255), 2)
                cv.circle(im, (self.xc, self.yc), 2, (0,0,255), 2)
            if hasattr(self, 'idealspx'):
                io = {}
                for s in ['x0', 'xf']:
                    io[s] = int(self.idealspx[s]/self.scale)
                cv.rectangle(im, (io['x0'],0), (io['xf'],600), (237, 227, 26), 2)   # bounding box of intended
            if hasattr(self, 'nozPx'):
                io = {}
                for s in ['x0', 'xf', 'y0', 'yf']:
                    io[s] = int(self.nozPx[s]/self.scale)
                cv.rectangle(im, (io['x0'],io['yf']), (io['xf'],io['y0']), (125, 125, 125), 3)   # bounding box of nozzle
        if hasattr(self, 'hull'):
            # show the roughness
            imshow(imgi, im2, self.roughnessIm(), self.statText(), title='vertFile')
        else:
            imshow(imgi, im2, self.statText(), title='vertFile')
        plt.title(os.path.basename(self.file))
        return 