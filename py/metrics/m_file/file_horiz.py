#!/usr/bin/env python
'''Functions for collecting data from stills horiz lines'''

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
        self.annotated = self.segmenter.labelsBW.copy()
        self.annotated = cv.cvtColor(self.annotated,cv.COLOR_GRAY2RGB)
        if self.diag==0:
            return
        for i,row in self.segmenter.df.iterrows():
            self.markHorizOnIm(row)
        imshow(self.im, self.annotated, self.statText(cols=2))
        
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
        totlen = df.w.sum()    # total length of all segments
        maxarea = df.a.max()   # area of largest segment
        totarea = df.a.sum()   # total area of all segments
        xc = int(sum(df.a * df.xc)/sum(df.a))   # weighted center of mass
        yc = int(sum(df.a * df.yc)/sum(df.a))

        co = {'line':self.name, 'segments':len(df)
            , 'maxlen':maxlen0*self.scale, 'totlen':df.w.sum()*self.scale
            , 'maxarea':df.a.max()*self.scale**2, 'totarea':int(df.a.sum())*self.scale**2
              , 'xc':xc*self.scale, 'yc':yc*self.scale
            }  
        counits = {'line':'', 'segments':''
            , 'maxlen':'px', 'totlen':'px'
            , 'maxarea':'px^2', 'totarea':'px^2'
                   , 'xc':'px', 'yc':'px'
            }  
        longest = df[df.w==maxlen0] # measurements of longest component
        self.componentMask = self.segmenter.reconstructMask(longest)   # get the image of just this component
        componentMeasures, cmunits = self.measureComponent(True, reverse=(self.name==1), diag=max(0,self.diag-1))
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
        
    def gaps(self, distancemm:float) -> None:
        if not hasattr(self, 'nd') or not hasattr(self, 'crop') or 'o' in self.name:
            return
        # get displacements
        disps = self.displacement('y', distancemm*self.nd.pxpmm, diag=self.diag-1)
        dispunits = dict([[ii, 'px'] for ii in disps])
        self.stats = {**self.stats, **disps}
        self.units = {**self.units, **dispunits}