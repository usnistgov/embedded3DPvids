#!/usr/bin/env python
'''filling empty spaces in images'''

# external packages
import cv2 as cv
import numpy as np 
import os
import sys
import logging
from typing import List, Dict, Tuple, Union, Any, TextIO
import pandas as pd
import matplotlib.pyplot as plt

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
from imshow import imshow
import morph as vm

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)

#----------------------------------------------

def getContours(mask:np.array, mode:int=cv.CHAIN_APPROX_SIMPLE) -> np.array:
    '''get all the contours'''
    contours = cv.findContours(mask,cv.RETR_TREE,mode)
    if int(cv.__version__[0])>=4:
        contours = contours[0]
    else:
        contours = contours[1]
    return contours


def contourRoughness(cnt:np.array) -> float:
    '''measure the roughness of the contour'''
    hull = cv.convexHull(cnt)
    return cv.arcLength(cnt,True)/cv.arcLength(hull,True)-1 


class contourH:
    '''for holding information about contours'''
    
    def __init__(self, thresh:np.array, mode:int=cv.CHAIN_APPROX_SIMPLE):
        self.thresh = thresh
        contours = cv.findContours(self.thresh,cv.RETR_TREE,mode)
        if int(cv.__version__[0])>=4:
            self.cnt = contours[0]
            self.hierarchy = contours[1]
        else:
            self.cnt = contours[1]
            
    def fillContour(self, im:np.array, i:int) -> None:
        '''fill the whole area of the contour'''
        cv.drawContours(im, self.cnt, contourIdx=i, color=(255,255,255), thickness=-1)
        
    def emptyContour(self, im:np.array, i:int) -> None:
        '''empty the whole area of the contour'''
        cv.drawContours(im, self.cnt, contourIdx=i, color=(0,0,0), thickness=-1)
            
    def labelHierarchy(self) -> pd.DataFrame:
        '''put the hierarchy into a dataframe'''
        self.hdf = pd.DataFrame(self.hierarchy[0], columns=['previous', 'next', 'child', 'parent'])
        toplevel = self.hdf[(self.hdf.parent<0)]   # no parents
        level1 = self.hdf[(self.hdf.parent>=0)]    # has parents
        level2 = level1[level1.parent.isin(level1.index.tolist())]   # has grandparents
        level3 = level2[level2.parent.isin(level2.index.tolist())]   # has great grandparents
        self.hdf.loc[(toplevel.index), 'level'] = 0
        self.hdf.loc[(level1.index), 'level'] = 1
        self.hdf.loc[(level2.index), 'level'] = 2
        self.hdf.loc[(level3.index), 'level'] = 3
        self.hdf['area'] = [cv.contourArea(c) for c in self.cnt]
        
    def maskContour(self, im:np.array, i:int, thickness:int=-1) -> float:
        '''get the mean value inside the contour if thickness=-1, on the contour if thickness>0'''
        if len(im.shape)==3:
            im2 = vm.normalize(cv.cvtColor(im, cv.COLOR_BGR2GRAY))
        else:
            im2 = im
        mask = np.zeros((im.shape[0], im.shape[1]))
        cv.drawContours(mask, [self.cnt[i]], contourIdx=0, color=(255,255,255),thickness=thickness)  # fill in the contour
        masked = np.ma.masked_where(mask==0, im2).compressed()
        return masked
        
    def contrastOnContour(self, im:np.array, i:int, thickness:int=1) -> float:
        '''get the contrast between average positive and negative values inside the contour if thickness=-1, on the contour if thickness>0'''
        if len(im)==0:
            return 0
        masked = self.maskContour(im, i, thickness)
        posvals = masked[masked>0]
        if len(posvals)>0:
            pos = posvals.mean().mean()
        else:
            pos = 0
        negvals = masked[masked<0]
        if len(negvals)>0:
            neg = negvals.mean().mean()
        else:
            neg = 0
        return pos-neg
    
    def meanInsideContour(self, im:np.array, i:int, thickness:int=-1) -> float:
        '''get the mean value inside the contour if thickness=-1, on the contour if thickness>0'''
        if len(self.thresh)==0:
            return 0
        masked = self.maskContour(thresh, i, thickness)
        return masked.mean().mean()
        
    def display(self) -> np.array:
        '''add the hierarchy contours to the thresholded image'''
        c2 = self.thresh.copy()
        if len(c2.shape)==2:
            c2 = cv.cvtColor(c2, cv.COLOR_GRAY2BGR)
        colors = {0:(19, 53, 242), 1: (46, 211, 240), 2:(15, 214, 81), 3:(230, 237, 31)}
        for n in self.hdf.level.unique():
            color = colors[n]
            for i,row in self.hdf[self.hdf.level==n].iterrows():
                if cv.contourArea(self.cnt[i])>50:
                    cv.drawContours(c2, [self.cnt[i]], -1, color, 2)
        return c2
    

