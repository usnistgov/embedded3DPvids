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
import contour as co

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)

#----------------------------------------------

class fillMode:
    removeBorder = 0
    fillSimple = 1
    fillSimpleWithHoles = 2
    fillByContours = 3
    fillTiny = 4

class filler:
    '''for filling holes in images'''
    
    def __init__(self, thresh:np.array, mode:int=fillMode.fillSimple, diag:int=0, **kwargs) -> None:
        self.thresh0 = thresh  # thresholded image
        self.thresh = thresh.copy()
        self.mode = mode   # how to fill the image
        self.diag = diag
        if 'laplacian' in kwargs:
            self.laplacian = kwargs['laplacian']
        else:
            self.laplacian = np.array([])
            
        # fill the image
        if self.mode == fillMode.removeBorder:
            self.removeBorder(**kwargs)
            self.fillComponents(leaveHollows=False, **kwargs)
        elif self.mode == fillMode.fillSimple:
            self.fillComponents(leaveHollows=False, **kwargs)
        elif self.mode == fillMode.fillSimpleWithHoles:
            self.fillComponents(leaveHollows=True, **kwargs)
        elif self.mode == fillMode.fillByContours:
            self.fillByContours(**kwargs)
        elif self.mode == fillMode.fillTiny:
            self.fillTiny(**kwargs)
        
        
    def gapsToFill(self) -> np.array:
        '''fill extra components'''
        im_flood_fill = self.thresh.copy()
        h, w = self.thresh.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        im_flood_fill = im_flood_fill.astype("uint8")
        cv.floodFill(im_flood_fill, mask, (0, 0), 255)
        self.gaps = cv.bitwise_not(im_flood_fill)
        self.gaps = cv.subtract(self.gaps, self.thresh)   # remove any parts that are also in the original image
        return self.gaps
    
    def initializeContours(self) -> None:
        '''initialize the contour hierarchy'''
        if not hasattr(self, 'ch'):
            i2 = self.thresh.copy()
            i2[-1, :] = 255
            self.ch = co.contourH(i2, mode=cv.CHAIN_APPROX_SIMPLE)
            self.ch.labelHierarchy()

    def removeHollows(self) -> np.array:
        # remove the inside of any bubbles inside objects
        self.initializeContours()
        imremove = self.thresh.copy()*0       # new image with contours filled
        if self.diag>0:
            self.contourLabels = self.ch.display()
        for i in (hdf[hdf.level==3]).index:
            if cv.contourArea(cnt[i])>50:
                cv.drawContours(imremove, cnt, contourIdx=i, color=(255,255,255),thickness=-1)
        self.filled = cv.subtract(self.filled, imremove)
        return self.filled
    
    def fillComponents(self, leaveHollows:bool=True, **kwargs)->np.array:
        '''fill the connected components in the thresholded image https://www.programcreek.com/python/example/89425/cv.floodFill'''
        self.gapsToFill()   # find gaps to fill
        self.filled = cv.add(self.thresh, self.gaps)  # add all gaps to the image
        if leaveHollows:
            self.removeHollows(diag=diag)  # remove hollow parts, e.g. the inside of a hole
        if self.diag>0:
            if hasattr(self, 'hollowEmptied'):
                imshow(self.thresh, self.contourLabels, self.gaps, self.filled, titles=['thresh', 'contourLabels', 'gaps', 'filled'])
            else:
                imshow(self.thresh, self.gaps, self.filled, titles=['thresh', 'gaps', 'filled'])
        return self.filled

    def removeBorder(self, **kwargs) -> np.array:
        '''remove the components touching the border'''
        # add 1 pixel white border all around
        pad = cv.copyMakeBorder(self.thresh, 1,1,1,1, cv.BORDER_CONSTANT, value=255)
        h, w = pad.shape
        # create zeros mask 2 pixels larger in each dimension
        mask = np.zeros([h + 2, w + 2], np.uint8)
        img_floodfill = cv.floodFill(pad, mask, (0,0), 0, (5), (0), flags=8)[1] # floodfill outer white border with black
        self.thresh = img_floodfill[1:h-1, 1:w-1]  # remove border

    def fillByContours(self, amin:int=50, amax:int=5000, conCrit:int=50, moCrit:int=-10, **kwargs) -> np.array:
        '''fill the components using the contours, where anything with a size between amin and amax doesn't get filled'''
        self.initializeContours()
        self.filled = self.thresh.copy()
        hdf = self.ch.hdf
        cnt = self.ch.cnt
        
        level1pts = hdf[hdf.level==1]

        # fill in tiny contours
        for i in level1pts[(level1pts.area<amin)|(level1pts.area>amax)].index:
            self.ch.fillContour(self.filled, i)

        for i in level1pts[(level1pts.area>=amin)&(level1pts.area<=amax)].index:
            mo = self.ch.meanInsideContour(self.laplacian, i, thickness=1)
            con = self.ch.contrastOnContour(self.laplacian, i, thickness=2)
            if self.diag>0:
                a = level1pts.loc[i,'area']
                print(a, mo, con)
            if con<conCrit and mo>moCrit:
                # fill it if it is chunky, or if it is rough
                self.ch.fillContour(self.filled, i)

        for i in (hdf[hdf.level==3]).index:
            if hdf.loc[i,'area']>amin:
                # empty this bubble
                self.ch.emptyContour(self.filled, i)
        if self.diag>0:
            contourLabels = self.ch.display()
            imshow(self.thresh, self.filled, contourLabels, titles=['fill: thresh', 'filled', 'contours'])
    
    def fillTiny(self, acrit:int=50, **kwargs) -> np.array:
        '''fill the components using the contours, where anything with a size between amin and amax doesn't get filled'''
        self.initializeContours()
        self.filled = self.thresh.copy()
        hdf = self.ch.hdf
        level1pts = hdf[hdf.level==1]

        # fill in tiny contours
        for i in level1pts[(level1pts.area<acrit)].index:
            self.ch.fillContour(self.filled, i)
        