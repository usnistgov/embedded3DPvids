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
from imshow import imshow
from morph import *
import contour as co
import im_fill as fi
from tools.timeCounter import timeObject
from s_segmenterDF import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)

#----------------------------------------------

class sMode:
    threshold = 0
    adaptive = 1
    kmeans = 2
    

class segmenter(timeObject):
    '''for thresholding and segmenting images'''
    
    def __init__(self, im:np.array, acrit:float=2500, diag:int=0
                 , fillMode:int=fi.fillMode.removeBorder, eraseMaskSpill:bool=False, closeTop:bool=True
                 , closing:int=0, grayBlur:int=3, removeSharp:bool=False
                 , leaveHollows:bool=True, **kwargs):
        self.im = im
        self.w = self.im.shape[1]
        self.h = self.im.shape[0]
        self.acrit = acrit
        self.diag = diag
        self.fillMode = fillMode
        self.eraseMaskSpill = eraseMaskSpill
        self.closeTop = closeTop
        self.closing = closing
        self.kwargs = kwargs
        self.leaveHollows = leaveHollows
        self.removeSharp = removeSharp
        self.grayBlur = grayBlur
        if 'nozData' in kwargs:
            self.nd = kwargs['nozData']
        if 'crops' in kwargs:
            self.crops = kwargs['crops']
        self.segmentInterfaces(**kwargs)
        self.makeDF()
        
    def __getattr__(self, s):
        if s in ['success', 'df', 'labeledIm', 'numComponents', 'labelsBW']:
            return getattr(self.sdf, s)
            
    def makeDF(self):
        if hasattr(self, 'filled'):
            self.sdf = segmenterDF(self.filled, self.acrit, diag=self.diag)
        
    def display(self):
        if self.diag>0:
            if hasattr(self, 'labeledIm'):
                imshow(self.im, self.gray, self.thresh, self.sdf.labeledIm, maxwidth=13, titles=['seg.im', 'gray', 'thresh', 'labeled'])
            else:
                imshow(self.im, self.gray, self.thresh, maxwidth=13, titles=['seg.im', 'gray', 'thresh']) 
        
    def getGray(self) -> None:
        '''convert the image to grayscale and store the grayscale image as self.thresh'''
        if len(self.im.shape)==3:
            gray = cv.cvtColor(self.im,cv.COLOR_BGR2GRAY)
        else:
            gray = self.im.copy()
        if self.grayBlur>0:
            self.gray = cv.medianBlur(gray, self.grayBlur)
        else:
            self.gray = gray
        
    def adaptiveThresh(self) -> np.array:
        '''adaptive threshold'''
        return cv.adaptiveThreshold(self.gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,11,6)
    
    def threshThresh(self, topthresh, whiteval) -> np.array:
        '''conventional threshold
        topthresh is the initial threshold value
        whiteval is the pixel intensity below which everything can be considered white'''
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
        return thresh
    
    def kmeansThresh(self) -> np.array:
        '''use kmeans clustering on the color image to segment interfaces'''
        twoDimage = self.im.reshape((-1,3))
        twoDimage = np.float32(twoDimage)
        attempts= 2
        epsilon = 0.5
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, attempts, epsilon)
        K = 2
        h,w = self.im.shape[:2]
        
        ret,label,center=cv.kmeans(twoDimage,K,None,criteria,attempts,cv.KMEANS_PP_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        result_image = res.reshape((self.im.shape))
        for i,c in enumerate(center):
            result_image[result_image==c]=int(i*255)

        result_image = cv.cvtColor(result_image, cv.COLOR_BGR2GRAY)
        if result_image.sum(axis=0).sum(axis=0)/255/(h*w)>0.5:
            result_image = cv.bitwise_not(result_image)
        return result_image

        
    def threshes(self, topthresh:int=200, whiteval:int=80, segmentMode:Union[list, int]=0, **kwargs) -> None:
        '''threshold the grayscale image and store the resulting binary image as self.thresh
        topthresh is the initial threshold value
        whiteval is the pixel intensity below which everything can be considered white
        '''
        threshes = []
        if not type(segmentMode) is list:
            segmentModes = [segmentMode]
        else:
            segmentModes = segmentMode
        for a in segmentModes:
            if a==sMode.threshold:
                threshes.append(self.threshThresh(topthresh, whiteval))
            elif a==sMode.adaptive:
                threshes.append(self.adaptiveThresh())
            elif a==sMode.kmeans:
                # use k-means clstering
                threshes.append(self.kmeansThresh())
        thresh = threshes[0]
        for t in threshes[1:]:
            thresh = cv.add(thresh, t)
        self.thresh = thresh
        
        
    def closeHorizLine(self, im:np.array, imtop:int, close:bool) -> np.array:
        '''draw a black line across the y position imtop between the first and last black point'''

        if close:
            marks = np.where(im[imtop]==255) 
            if len(marks[0])==0:
                return
            val = 255
            first = marks[0][0] # first position in x in y row where black
            last = marks[0][-1]
        else:
            val = 255
            first = 0
            last = im.shape[1]
        if last-first<im.shape[1]*0.2:
            im[imtop:imtop+3, first:last] = val*np.ones(im[imtop:imtop+3, first:last].shape)
        return im

    def closeVerticalTop(self, im:np.array, close:bool=True, cutoffTop:float=0.01, closeBottom:bool=False, **kwargs) -> np.array:
        '''if the image is of a vertical line, close the top'''
        if im.shape[0]<im.shape[1]*2:
            return im

        # cut off top 3% of image
        if cutoffTop>0:
            if close:
                val = 255
            else:
                val = 0
            imtop = int(im.shape[0]*cutoffTop)  
            im[1:imtop, 1:-1] = np.ones(im[1:imtop, 1:-1].shape)*val

        # vertical line. close top to fix bubbles
        top = np.where(np.array([sum(x) for x in im])>0) 

        if len(top[0])==0:
            return im
        imtop = top[0][0] # first position in y where black
        im = self.closeHorizLine(im, imtop, close)
        if closeBottom:
            imbot = top[0][-1]-3
            im = self.closeHorizLine(im, imbot, close)
        return im 
    
    def closeFullBorder(self, im:np.array) -> np.array:
        '''put a white border around the whole image'''
        if len(im.shape)>2:
            zero = [0,0,0]
        else:
            zero = 255
        im2 = im.copy()
        im2[0, :] = zero
        im2[-1, :] = zero
        im2[:, 0] = zero
        im2[:,-1] = zero
        return im2
        
    def addNozzle(self, bottomOnly:bool=False) -> None:
        '''add the nozzle in black back in for filling'''
        if not (hasattr(self, 'nd') and hasattr(self, 'crops')):
            return
        thresh = self.nd.maskNozzle(self.thresh, ave=False, invert=False, crops=self.crops, bottomOnly=bottomOnly)   
        h,w = thresh.shape
        thresh[0, :] = 0   # clear out the top row
        thresh[:int(h/4), 0] = 0  # clear left and right edges at top half
        thresh[:int(h/4),-1] = 0
        self.thresh = thresh
        
    def removeLaplacian(self, sharpCrit:int=20, **kwargs) -> None:
        '''remove from thresh the edges with a sharp gradient from white to black'''
        self.laplacian = cv.Laplacian(self.gray,cv.CV_64F)
        # ret, thresh2 = cv.threshold(laplacian,10,255,cv.THRESH_BINARY)   # sharp transition from black to white
        ret, thresh3 = cv.threshold(self.laplacian,-sharpCrit,255,cv.THRESH_BINARY_INV)  # sharp transition from white to black
        thresh3 = erode(normalize(thresh3), 2)  # remove tiny boxes
        thresh3 = thresh3.astype(np.uint8)
        self.thresh = cv.subtract(self.thresh, thresh3)

    def fillParts(self, fillTop:bool=True, **kwargs) -> None:
        '''fill the components, and remove the border if needed'''
        if fillTop:
            self.thresh = self.closeVerticalTop(self.thresh, close=True)
        if self.closing>0:
            self.thresh = closeMorph(self.thresh, self.closing)
        elif self.closing<0:
            self.thresh = openMorph(self.thresh, -self.closing)
        if self.removeSharp:
            self.removeLaplacian(**self.kwargs)
        if hasattr(self, 'laplacian'):
            kwargs = {'laplacian':self.laplacian}
        else:
            kwargs = {}
        self.filler = fi.filler(self.thresh, self.fillMode, diag=self.diag-2, **kwargs)
        self.filled = self.filler.filled
        self.filled = self.closeVerticalTop(self.filled, close=False)
        if self.closing>0:
            self.filled = closeMorph(self.filled, self.closing)
        elif self.closing<0:
            self.filled = openMorph(self.filled, -self.closing)
            
    def emptyVertSpaces(self) -> None:
        '''empty the vertical spaces between printed vertical lines'''
        if not hasattr(self, 'filled'):
            return
        # Apply morphology operations
        thresh2 = self.nd.maskNozzle(self.thresh, dilate=5, crops=self.crops, invert=True)  # generously remove nozzle
        gX = openMorph(thresh2, 1, aspect=15)    # filter out horizontal lines
        tot1 = closeMorph(gX, 5, aspect=1/5)   # close sharp edges
        tot = fi.filler(tot1).gapsToFill()    # fill gaps
        tot = openMorph(tot, 3)    # remove debris
        er = cv.subtract(tot1, gX)              # get extra filled gaps
        tot = cv.add(tot, er)        # remove from image
        tot = openMorph(tot, 2)           # remove debris
        
        filled = cv.subtract(self.sdf.labelsBW, tot)   # remove from image

        if self.diag>1:
            imshow(gX, tot, filled, self.filled, title='emptyVertSpaces')
        self.filled = filled
        self.makeDF()   
        
    def removeNozzle(self, s:str='filled') -> None:
        '''remove the black nozzle from the image'''
        if not (hasattr(self, 'nd') and hasattr(self, 'crops')):
            return
        setattr(self, s, self.nd.maskNozzle(getattr(self, s), ave=False, invert=True, crops=self.crops))  
        # remove the nozzle again
            
            
    def segmentInterfaces(self, addNozzle:bool=True, addNozzleBottom:bool=False, **kwargs) -> np.array:
        '''from a color image, segment out the ink, and label each distinct fluid segment. 
        acrit is the minimum component size for an ink segment
        removeVert=True to remove vertical lines from the thresholded image
        removeBorder=True to remove border components from the thresholded image'''
        self.getGray()
        self.threshes(**kwargs)  # threshold
        if addNozzle:
            self.addNozzle()    # add the nozzle to the thresholded image
        if addNozzleBottom:
            self.addNozzle(bottomOnly=True)
        self.fillParts(**kwargs)    # fill components
        self.removeNozzle() # remove the nozzle again
    
        
    def eraseSmallComponents(self, **kwargs):
        '''erase small components from the labeled image and create a binary image'''
        return self.sdf.eraseSmallComponents(**kwargs)
        
    def eraseSmallestComponents(self, satelliteCrit:float=0.2, **kwargs) -> None:
        '''erase the smallest relative components from the labeled image'''
        return self.sdf.eraseSmallestComponents(satelliteCrit, **kwargs)
          
    def eraseBorderComponents(self, margin:int, **kwargs) -> None:
        '''remove any components that are too close to the edge'''
        return self.sdf.eraseBorderComponents(margin, **kwargs)
        
    def eraseFullWidthComponents(self, **kwargs) -> None:
        '''remove components that are the full width of the image'''
        return self.sdf.eraseFullWidthComponents(**kwargs)
        
    def eraseLeftRightBorder(self, **kwargs) -> None:
        '''remove components that are touching the left or right border'''
        return self.sdf.eraseLeftRightBorder(**kwargs)
        
    def eraseTopBottomBorder(self, **kwargs) -> None:
        '''remove components that are touching the top or bottom border'''
        return self.sdf.eraseTopBottomBorder(**kwargs)
     
    def removeScragglies(self, **kwargs) -> None:
        return self.sdf.removeScragglies(**kwargs)
        
    def largestObject(self) -> pd.Series:
        '''the largest object in the dataframe'''
        return self.sdf.largestObject()
     
    def reconstructMask(self, df:pd.DataFrame) -> np.array:
        '''construct a binary mask with all components labeled in the dataframe'''
        return self.sdf.reconstructMask(df)
    
    def noDF(self) -> bool:
        return self.sdf.noDF(df)