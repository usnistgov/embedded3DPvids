#!/usr/bin/env python
'''Functions for collecting data from stills of horiz lines'''

# external packages
import os, sys
import traceback
import logging
import pandas as pd
from typing import List, Dict, Tuple, Union, Any, TextIO
import re
import numpy as np
import cv2 as cv

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


class fileHoriz(fileMetric):
    '''collects data about horizontal segments in a single horizontal line'''
    
    def __init__(self, file:str, diag:int=0, acrit:int=2500, **kwargs):
        super().__init__(file, diag=diag, acrit=acrit, **kwargs)
    

    def markHorizOnIm(self, row:pd.Series) -> np.array:
        '''mark horizontal element on the image'''
        self.annotated = cv.rectangle(self.annotated, (int(row['x0']),int(row['y0'])), (int(row['x0']+row['w']),int(row['y0']+row['h'])), (0,0,255), 2)
        self.annotated = cv.circle(self.annotated, (int(row['xc']), int(row['yc'])), 3, (0,0,255), 3)
    
    def display(self) -> None:
        '''display diagnostics'''
        if self.diag<=0:
            return
        if hasattr(self, 'segmenter'):
            self.segmenter.display()
        if hasattr(self, 'im0'):
            imgi = self.im0.copy() 
        elif hasattr(self, 'im'):
            imgi = self.im.copy()
        else:
            imgi = np.zeros(self.im.shape, dtype=np.uint8)
        for im in [imgi]:
            if hasattr(self, 'x0'):
                cv.rectangle(im, (self.x0,self.y0), (self.x0+self.w,self.y0+self.h), (0,0,255), 2)
                cv.circle(im, (self.xc, self.yc), 2, (0,0,255), 2)
            if hasattr(self, 'idealspx'):
                io = {}
                for s in ['y0', 'yf']:
                    io[s] = int(self.idealspx[s]/self.scale)
                cv.rectangle(im, (0,io['y0']), (800,io['yf']), (237, 227, 26), 2)   # bounding box of intended
            if hasattr(self, 'nozPx'):
                io = {}
                for s in ['x0', 'xf', 'y0', 'yf']:
                    io[s] = int(self.nozPx[s]/self.scale)
                cv.rectangle(im, (io['x0'],io['yf']), (io['xf'],io['y0']), (125, 125, 125), 3)   # bounding box of nozzle
                
        fn = os.path.basename(self.file).replace('_HO', '\nHO')
        if hasattr(self, 'hull'):
            # show the roughness
            imshow(imgi, self.roughnessIm(), self.statText(cols=2), titles=[fn, 'roughness', ''])
        else:
            imshow(imgi, self.statText(cols=2), titles=[fn,  ''])
        return 

        
    def selectLine(self, df:pd.DataFrame, maxlen:float, j:int) -> None:
        '''narrow down the selection to just the segments in the dataframe'''
        self.segmenter.selectComponents(df.w==df.w)
        self.maxlen = maxlen
        self.name = j
        self.title = f'{os.path.basename(self.file)}: horiz{j}'
        self.dims()
        self.display()

    def dims(self) -> None:
        '''measure one horizontal line. df has been filtered down from the full dataframe to only include one row 
        labeled is an image from connected component labeling
        im2 is the original image
        s is is the scaling of the stitched image compared to the raw images, e.g. 0.33
        j is the line number
        maxPossibleLen is the longest length in mm that should have been extruded
        '''
        df = self.segmenter.df     
        
        maxlen0 = df.w.max()   # length of the longest segment
        self.totlen = df.w.sum()    # total length of all segments
        maxarea = df.a.max()   # area of largest segment
        totarea = df.a.sum()   # total area of all segments
        self.xc = int(sum(df.a * df.xc)/sum(df.a))   # weighted center of mass
        self.yc = int(sum(df.a * df.yc)/sum(df.a))

        co = {'line':self.name, 'segments':len(df)
            , 'maxlen':maxlen0*self.scale, 'totlen':df.w.sum()*self.scale
            , 'maxarea':df.a.max()*self.scale**2, 'area':int(df.a.sum())*self.scale**2
              , 'xc':self.xc*self.scale, 'yc':self.yc*self.scale
            }  
        counits = {'line':'', 'segments':''
            , 'maxlen':'px', 'totlen':'px'
            , 'maxarea':'px^2', 'area':'px^2'
                   , 'xc':'px', 'yc':'px'
            }  
        longest = df[df.w==maxlen0] # measurements of longest component
        self.componentMask = self.segmenter.reconstructMask(longest)   # get the image of just this component
        componentMeasures, cmunits = self.measureComponent(True, reverse=(self.name==1), diag=max(0,self.diag-1), combine=combine)
        # component measures and co are pre-scaled
        if 'totlen' in co and 'meanT' in componentMeasures:
            r = componentMeasures['meanT']/2
            h = co['totlen']
            aspect = h/(2*r) # height/width
            vest = calcVest(h,r)
        else:
            aspect = 0
            vest = 0
            
        units = {'line':'', 'aspect':'h/w'} # where pixels are in original scale
        ret = {**{'line':self.name, 'aspect':aspect}, **co, **{'vest':vest}, **componentMeasures}
        units = {**units, **counits, **{'vest':'px^3'}, **cmunits}
        
        self.stats = {**self.stats, **ret}
        self.units = {**self.units, **units}
        
    def dimsMulti(self, getLDiff:bool=False) -> None:
        '''measure one horizontal line. df has been filtered down from the full dataframe to only include one row 
        labeled is an image from connected component labeling
        im2 is the original image
        s is is the scaling of the stitched image compared to the raw images, e.g. 0.33
        j is the line number
        maxPossibleLen is the longest length in mm that should have been extruded
        '''
        df = self.segmenter.df
        df['xf'] = df.x0+df.w
        df['yf'] = df.y0+df.h
        
        self.x0 = df.x0.min()
        self.xf = df.xf.max()
        self.y0 = df.y0.min()
        self.yf = df.yf.max()
        self.h = self.yf-self.y0
        self.w = self.xf-self.x0

        self.area = df.a.sum()   # total area of all segments
        self.xc = int(sum(df.a * df.xc)/sum(df.a))   # weighted center of mass
        self.yc = int(sum(df.a * df.yc)/sum(df.a))

        co = {'line':self.name, 'segments':len(df), 'area':self.area*self.scale**2
                     , 'y0':self.y0*self.scale
                      , 'w':self.w*self.scale, 'h':self.h*self.scale
                      , 'yf':self.yf*self.scale
                     , 'yc':self.yc*self.scale} 
        counits = {'line':'', 'segments':'', 'area':'px'
                 , 'y0':'px', 'yf':'px', 'w':'px', 'h':'px'
                 , 'yc':'px'} # where pixels are in original scale
        
        self.componentMask = self.segmenter.labelsBW.copy()
        componentMeasures, cmunits = self.measureComponent(True, reverse=(self.name==1), diag=max(0,self.diag-1), combine=True)
        # component measures and co are pre-scaled
        ret = {**co,  **componentMeasures}
        units = {**counits, **cmunits}
        
        if getLDiff:
            ret['ldiff'] = self.getLDiff(horiz=True)/self.w
            units['ldiff'] = 'w'
        
        self.stats = {**self.stats, **ret}
        self.units = {**self.units, **units}
        
    def gaps(self, distancemm:float) -> None:
        '''measure the gaps between the nozzle and the object'''
        if not hasattr(self, 'nd') or not hasattr(self, 'crop') or 'o' in self.name:
            return
        # get displacements
        disps = self.displacement('y', distancemm*self.nd.pxpmm, diag=self.diag-1)
        dispunits = dict([[ii, 'px'] for ii in disps])
        self.stats = {**self.stats, **disps}
        self.units = {**self.units, **dispunits}
        