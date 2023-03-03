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
from imshow import imshow
from im_morph import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)



#----------------------------------------------

class segmenter:
    '''for thresholding and segmenting images'''
    
    def __init__(self, im:np.array, acrit:float=2500, diag:int=0, removeBorder:bool=True, eraseMaskSpill:bool=False, closeTop:bool=True, **kwargs):
        self.im = im
        self.w = self.im.shape[1]
        self.h = self.im.shape[0]
        self.acrit = acrit
        self.diag = diag
        self.removeBorder = removeBorder
        self.eraseMaskSpill = eraseMaskSpill
        self.closeTop = closeTop
        self.kwargs = kwargs
        self.success = False
        if 'nozData' in kwargs:
            self.nd = kwargs['nozData']
        if 'crops' in kwargs:
            self.crops = kwargs['crops']
        self.segmentInterfaces()
        self.getConnectedComponents()
        
    def display(self):
        if self.diag>0:
            if hasattr(self, 'labeledIm'):
                imshow(self.im, self.gray, self.thresh, self.labeledIm, maxwidth=13)
            else:
                imshow(self.im, self.gray, self.thresh, maxwidth=13) 
        
    def getGray(self) -> None:
        '''convert the image to grayscale and store the grayscale image as self.thresh'''
        if len(self.im.shape)==3:
            gray = cv.cvtColor(self.im,cv.COLOR_BGR2GRAY)
        else:
            gray = self.im.copy()
        self.gray = cv.medianBlur(gray, 5)

        
    def threshes(self, topthresh:int=200, whiteval:int=80, **kwargs) -> None:
        '''threshold the grayscale image and store the resulting binary image as self.thresh
        topthresh is the initial threshold value
        whiteval is the pixel intensity below which everything can be considered white
        '''
        crit = topthresh
        impx = np.product(self.gray.shape)
        allwhite = impx*whiteval
        prod = allwhite
        while prod>=allwhite and crit>50: # segmentation included too much
            ret, thresh = cv.threshold(self.gray,crit,255,cv.THRESH_BINARY_INV)
            prod = np.sum(np.sum(thresh))/impx
            crit = crit-10
        if self.diag>0:
            logging.info(f'Threshold: {crit+10}, product: {prod}, white:{whiteval}')
        self.thresh = thresh
        self.thresh = closeVerticalTop(self.thresh, **kwargs)
        
    def addNozzle(self) -> None:
        '''add the nozzle in black back in for filling'''
        if not (hasattr(self, 'nd') and hasattr(self, 'crops')):
            return
        thresh = self.nd.maskNozzle(self.thresh, ave=False, invert=False, crops=self.crops)   
        h,w = thresh.shape
        thresh[0, :] = 0   # clear out the top row
        thresh[:int(h/4), 0] = 0  # clear left and right edges at top half
        thresh[:int(h/4),-1] = 0
        self.thresh = thresh
        
    def fillParts(self) -> None:
        '''fill the components, and remove the border if needed'''
        if self.removeBorder:
            self.filled = removeBorderAndFill(self.thresh)    
        else:
            self.filled = fillComponents(self.thresh)
        
    def removeNozzle(self) -> None:
        '''remove the black nozzle from the image'''
        if not (hasattr(self, 'nd') and hasattr(self, 'crops')):
            return
        self.filled = self.nd.maskNozzle(self.filled, ave=False, invert=True, crops=self.crops)  # remove the nozzle again
        if self.eraseMaskSpill:
            self.filled = self.nd.eraseSpillover(self.filled, crops=self.crops)  # erase any extra nozzle that is in the image

    def getDataFrame(self):
        '''convert the labeled segments to a dataframe'''
        df = pd.DataFrame(self.stats, columns=['x0', 'y0', 'w', 'h','a'])
        df2 = pd.DataFrame(self.centroids, columns=['xc','yc'])
        df = pd.concat([df, df2], axis=1) 
            # combine markers into dataframe w/ label stats
        df = df[df.a<df.a.max()] 
            # remove largest element, which is background
        self.df = df
        
    def resetStats(self):
        '''reset the number of components and the filtered binary image'''
        self.numComponents = len(self.df)
        self.labelsBW = self.labeledIm.copy()
        self.labelsBW[self.labelsBW>0]=255
        self.labelsBW = self.labelsBW.astype(np.uint8)
        if self.numComponents==0:
            self.success = False
        else:
            self.success = True
            
    def selectComponents(self, goodpts:pd.Series) -> None:
        '''erase any components that don't fall under criterion'''
        for i in list(self.df[~goodpts].index):
            self.labeledIm[self.labeledIm==i] = 0
        self.df = self.df[goodpts] 
        self.resetStats()
            
    def eraseSmallComponents(self):
        '''erase small components from the labeled image and create a binary image'''
        if len(self.df)==0:
            return
        goodpts = (self.df.a>=self.acrit)
        self.selectComponents(goodpts)
        
    def eraseSmallestComponents(self, satelliteCrit:float=0.2, **kwargs) -> None:
        '''erase the smallest relative components from the labeled image'''
        if len(self.df)==0:
            return
        goodpts = (self.df.a>=satelliteCrit*self.df.a.max())
        self.selectComponents(goodpts)
          
    def eraseBorderComponents(self, margin:int) -> None:
        '''remove any components that are too close to the edge'''
        if len(self.df)==0:
            return
        goodpts = (self.df.x0>margin)&(self.df.y0>margin)&(self.df.x0+self.df.w<self.w-margin)&(self.df.y0+self.df.h<self.h-margin)
        self.selectComponents(goodpts)
        
    def eraseFullWidthComponents(self) -> None:
        '''remove components that are the full width of the image'''
        if len(self.df)==0:
            return
        goodpts = (self.df.w<self.w)
        self.selectComponents(goodpts)
        
    def eraseLeftRightBorder(self) -> None:
        '''remove components that are touching the left or right border'''
        if len(self.df)==0:
            return
        goodpts = ((self.df.x0>0)&(self.df.x0+self.df.w<(self.w)))
        self.selectComponents(goodpts)
        
    def eraseTopBottomBorder(self) -> None:
        '''remove components that are touching the top or bottom border'''
        if len(self.df)==0:
            return
        goodpts = (self.df.y0>0)&(self.df.y0+self.df.h<self.h)
        self.selectComponents(goodpts)
        
        
    def largestObject(self) -> pd.Series:
        '''the largest object in the dataframe'''
        if not self.success:
            raise ValueError('Segmenter failed. Cannot take largest object')
        return self.df[self.df.a==self.df.a.max()].iloc[0]
            

    def getConnectedComponents(self) -> int:
        '''get connected components and filter by area, then create a new binary image without small components'''
        self.markers = cv.connectedComponentsWithStats(self.filled, 8, cv.CV_32S)
        self.numComponents = self.markers[0]
        if self.numComponents==1:
            # no components detected
            self.display()
            return 1
        
        self.labeledIm = self.markers[1]  # this image uses different numbers to label each component
        self.stats = self.markers[2]
        self.centroids = self.markers[3]
        self.getDataFrame()       # convert stats to dataframe
        if len(self.df)==0 or max(self.df.a)<self.acrit:
            # no components detected
            self.display()
            return 1
        self.eraseSmallComponents()
        return 0  
    
    def segmentInterfaces(self) -> np.array:
        '''from a color image, segment out the ink, and label each distinct fluid segment. 
        acrit is the minimum component size for an ink segment
        removeVert=True to remove vertical lines from the thresholded image
        removeBorder=True to remove border components from the thresholded image'''
        self.getGray()
        self.threshes(**self.kwargs)
        self.addNozzle()    # add the nozzle to the thresholded image
        self.fillParts()    # fill components
        self.removeNozzle() # remove the nozzle again

            
    def reconstructMask(self, df:pd.DataFrame) -> np.array:
        '''construct a binary mask with all components labeled in the dataframe'''
        masks = [(self.labeledIm == i).astype("uint8") * 255 for i,row in df.iterrows()]
        if len(masks)>0:
            componentMask = masks[0]
            if len(masks)>1:
                for mask in masks[1:]:
                    componentMask = cv.add(componentMask, mask)
            return componentMask
        else:
            return np.zeros(self.gray.shape).astype(np.uint8)
        
    
#----------------------------------------------------------
    
class segmenterSingle(segmenter):
    
    def __init__(self, im:np.array, acrit:float=2500, diag:int=0, removeVert:bool=False, removeBorder:bool=True, **kwargs):
        self.removeVert = removeVert
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
            filled = fillComponents(thresh)
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
            filled = fillComponents(thresh)
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
                self.filled = fillComponents(self.thresh)    
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
        
    