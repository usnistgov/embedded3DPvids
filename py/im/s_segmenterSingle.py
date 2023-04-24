#!/usr/bin/env python
'''Morphological operations applied to images'''

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
from s_segmenter import segmenter
from morph import *
from imshow import imshow
import contour as co
import im_fill as fi
from tools.timeCounter import timeObject

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)

#----------------------------------------------

class segmenterSingle(segmenter):
    
    def __init__(self, im:np.array, acrit:float=2500, diag:int=0, removeVert:bool=False, removeBorder:bool=True, **kwargs):
        self.removeVert = removeVert
        self.fillMode = fi.fillMode.removeBorder
        super().__init__(im, acrit=acrit, diag=diag, removeBorder=removeBorder, **kwargs)
 
        
    def threshes(self, attempt:int, topthresh:int=200, whiteval:int=80, **kwargs) -> None:
        '''threshold the grayscale image
        attempt number chooses different strategies for thresholding ink
        topthresh is the initial threshold value
        whiteval is the pixel intensity below which everything can be considered white
        increase diag to see more diagnostic messages
        '''
        if attempt==0:
    #         ret, thresh = cv.threshold(self.gray,180,255,cv.THRESH_BINARY_INV)
            # just threshold on intensity
            crit = topthresh
            impx = np.product(self.gray.shape)
            allwhite = impx*whiteval
            prod = allwhite
            while prod>=allwhite and crit>100: # segmentation included too much
                ret, thresh1 = cv.threshold(self.gray,crit,255,cv.THRESH_BINARY_INV)
                ret, thresh2 = cv.threshold(self.gray,crit+10,255,cv.THRESH_BINARY_INV)
                thresh = np.ones(shape=thresh2.shape, dtype=np.uint8)
                thresh[:600,:] = thresh2[:600,:] # use higher threshold for top 2 lines
                thresh[600:,:] = thresh1[600:,:] # use lower threshold for bottom line
                prod = np.sum(np.sum(thresh))
                crit = crit-10
    #         ret, thresh = cv.threshold(self.gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
            if diag>0:
                logging.info(f'Threshold: {crit+10}, product: {prod/impx}, white:{whiteval}')
        elif attempt==1:
            # adaptive threshold, for local contrast points
            thresh = cv.adaptiveThreshold(self.gray,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,11,2)
            filled = filler(thresh, self.fillMode, diag=self.diag-2).filled
            thresh = cv.add(255-thresh,filled)
        elif attempt==2:
            # threshold based on difference between red and blue channel
            b = self.im[:,:,2]
            g = self.im[:,:,1]
            r = self.im[:,:,0]
            self.gray2 = cv.subtract(r,b)
            self.gray2 = cv.medianBlur(self.gray2, 5)
            ret, thresh = cv.threshold(self.gray2,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
            ret, background = cv.threshold(r,0,255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
            background = 255-background
            thresh = cv.subtract(background, thresh)
        elif attempt==3:
            # adaptive threshold, for local contrast points
            thresh = cv.adaptiveThreshold(self.gray,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,21,2)
            filled = filler(thresh, self.fillMode, diag=self.diag-2).filled
            thresh2 = cv.add(255-thresh,filled)

            # remove verticals
            if self.removeVert:
                # removeVert=True to remove vertical lines from the thresholding. useful for horizontal images where stitching leaves edges
                thresh = cv.subtract(thresh, verticalFilter(self.gray))
                ret, topbot = cv.threshold(self.gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU) 
                thresh = cv.subtract(thresh,topbot)
        elif attempt==4:
            thresh0 = threshes(self.im, self.gray, self.removeVert, 0)
            thresh2 = threshes(self.im, self.gray, self.removeVert, 2)
            thresh = cv.bitwise_or(thresh0, thresh2)
            thresh = cv.medianBlur(thresh,3)
        self.thresh = closeVerticalTop(thresh)
    
    def segmentInterfaces(self) -> np.array:
        '''from a color image, segment out the ink, and label each distinct fluid segment. '''
        self.getGray()
        attempt = 0
        self.finalAt = attempt
        while attempt<1:
            self.finalAt = attempt
            self.threshes(attempt, **self.kwargs)
            if self.removeBorder:
                self.filled = filler(self.thresh, self.fillMode, diag=self.diag-2).filled
            else:
                self.filled = self.thresh.copy()
            self.markers = cv.connectedComponentsWithStats(self.filled, 8, cv.CV_32S)

            if self.self.diag>0:
                imshow(self.im, self.gray, self.thresh, self.filled)
                plt.title(f'attempt:{attempt}')
            if self.markers[0]>1:
                self.df = pd.DataFrame(self.markers[2], columns=['x0', 'y0', 'w', 'h', 'area'])
                if max(self.df.loc[1:,'area'])<self.acrit:
                    # poor segmentation. redo with adaptive thresholding.
                    attempt=attempt+1
                else:
                    attempt = 6
            else:
                attempt = attempt+1
        return self.filled, self.markers, self.finalAt