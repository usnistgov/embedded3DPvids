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

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)



#----------------------------------------------

class segmenter:
    '''for thresholding and segmenting images'''
    
    def __init__(self, im:np.array, acrit:float=2500, diag:int=0, removeBorder:bool=True, eraseMaskSpill:bool=False, closeTop:bool=True, dilation:int=0, closing:int=0, **kwargs):
        self.im = im
        self.w = self.im.shape[1]
        self.h = self.im.shape[0]
        self.acrit = acrit
        self.diag = diag
        self.removeBorder = removeBorder
        self.eraseMaskSpill = eraseMaskSpill
        self.closeTop = closeTop
        self.closing = closing
        self.kwargs = kwargs
        self.dilation = dilation
        if 'nozData' in kwargs:
            self.nd = kwargs['nozData']
        if 'crops' in kwargs:
            self.crops = kwargs['crops']
        self.segmentInterfaces(**kwargs)
        self.makeDF()
            
    def makeDF(self):
        if hasattr(self, 'filled'):
            self.sdf = segmenterDF(self.filled, self.acrit, diag=self.diag)
        
        
        
    def display(self):
        if self.diag>0:
            if hasattr(self, 'labeledIm'):
                imshow(self.im, self.gray, self.thresh, self.sdf.labeledIm, maxwidth=13, title='segmenter')
            else:
                imshow(self.im, self.gray, self.thresh, maxwidth=13, title='segmenter') 
        
    def getGray(self) -> None:
        '''convert the image to grayscale and store the grayscale image as self.thresh'''
        if len(self.im.shape)==3:
            gray = cv.cvtColor(self.im,cv.COLOR_BGR2GRAY)
        else:
            gray = self.im.copy()
        self.gray = cv.medianBlur(gray, 5)

        
    def threshes(self, topthresh:int=200, whiteval:int=80, adaptive:bool=False, **kwargs) -> None:
        '''threshold the grayscale image and store the resulting binary image as self.thresh
        topthresh is the initial threshold value
        whiteval is the pixel intensity below which everything can be considered white
        '''
        if adaptive:
            thresh = cv.adaptiveThreshold(self.gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,11,2)
        else:
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
        
        
    def closeHorizLine(self, im:np.array, imtop:int, close:bool) -> np.array:
        '''draw a black line across the y position imtop between the first and last black point'''
        marks = np.where(im[imtop]==255) 
        if len(marks[0])==0:
            return

        first = marks[0][0] # first position in x in y row where black
        last = marks[0][-1]
        if close:
            val = 255
        else:
            val = 255
        if last-first<im.shape[1]*0.2:
            im[imtop:imtop+3, first:last] = val*np.ones(im[imtop:imtop+3, first:last].shape)
        return im

    def closeVerticalTop(self, im:np.array, close:bool=True, cutoffTop:float=0.01, closeBottom:bool=False, **kwargs) -> np.array:
        '''if the image is of a vertical line, close the top'''
        if im.shape[0]<im.shape[1]*2:
            return

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
            return 
        imtop = top[0][0] # first position in y where black
        im = self.closeHorizLine(im, imtop, close)
        if closeBottom:
            imbot = top[0][-1]-3
            im = self.closeHorizLine(im, imbot, close)
        return im 
        
    def addNozzle(self, bottomOnly:bool=False) -> None:
        '''add the nozzle in black back in for filling'''
        if not (hasattr(self, 'nd') and hasattr(self, 'crops')):
            return
        thresh = self.nd.maskNozzle(self.thresh, ave=False, invert=False, crops=self.crops, dilate=self.dilation, bottomOnly=bottomOnly)   
        h,w = thresh.shape
        thresh[0, :] = 0   # clear out the top row
        thresh[:int(h/4), 0] = 0  # clear left and right edges at top half
        thresh[:int(h/4),-1] = 0
        self.thresh = thresh

        
#     def closeVertLine(self, s:str, w:int=2, aspect:int=10) -> None:
#         '''find smoothed contours of each component and add them to the image'''
#         m1 = closeMorph(getattr(self, s), w, aspect=aspect)
#         m2 = fillComponents(m1)
#         m3 = openMorph(m2, w, aspect=1)
#         imshow(getattr(self, s), m1, m2, m3)
#         setattr(self, s, m3)
                
        
    def fillParts(self, fillTop:bool=True, **kwargs) -> None:
        '''fill the components, and remove the border if needed'''
        if fillTop:
            self.thresh = self.closeVerticalTop(self.thresh, close=True)
        if self.closing>0:
            self.thresh = closeMorph(self.thresh, self.closing)
        if self.removeBorder:
            self.filled = removeBorderAndFill(self.thresh, leaveHollows=True)    
        else:
            self.filled = fillComponents(self.thresh, diag=self.diag-2, leaveHollows=True)
        self.filled = self.closeVerticalTop(self.filled, close=False)
        if self.closing>0:
            self.filled = closeMorph(self.filled, self.closing)
            
    def emptyVertSpaces(self) -> None:
        '''empty the vertical spaces between printed vertical lines'''
        if not hasattr(self, 'filled'):
            return
        # Apply morphology operations
        thresh2 = self.nd.maskNozzle(self.thresh, dilate=5, crops=self.crops, invert=True)  # generously remove nozzle
        gX = openMorph(thresh2, 1, aspect=15)    # filter out horizontal lines
        tot1 = closeMorph(gX, 5, aspect=1/5)   # close sharp edges
        tot = emptySpaces(tot1)    # fill gaps
        tot = openMorph(tot, 3)    # remove debris
        er = cv.subtract(tot1, gX)              # get extra filled gaps
        tot = cv.add(tot, er)        # remove from image
        tot = openMorph(tot, 2)           # remove debris
        
        filled = cv.subtract(self.sdf.labelsBW, tot)   # remove from image
        
#         skeleton = dilate(skeletonize(self.thresh, w=5),2)
#         sfilled = fillComponents(skeleton)
#         emptied = erode(sfilled, 7)
#         imshow(skeleton, sfilled, emptied)

        if self.diag>1:
            imshow(gX, tot, filled, self.filled, title='emptyVertSpaces')
        self.filled = filled
        self.makeDF()
        
            
        
    def removeNozzle(self, s:str='filled') -> None:
        '''remove the black nozzle from the image'''
        if not (hasattr(self, 'nd') and hasattr(self, 'crops')):
            return
        setattr(self, s, self.nd.maskNozzle(getattr(self, s), ave=False, invert=True, crops=self.crops, dilate=self.dilation))  # remove the nozzle again
        if self.eraseMaskSpill:
            setattr(self, s, self.nd.eraseSpillover(getattr(self, s), crops=self.crops, diag=self.diag-1))  # erase any extra nozzle that is in the image
            
            
    def segmentInterfaces(self, addNozzle:bool=True, addNozzleBottom:bool=False, **kwargs) -> np.array:
        '''from a color image, segment out the ink, and label each distinct fluid segment. 
        acrit is the minimum component size for an ink segment
        removeVert=True to remove vertical lines from the thresholded image
        removeBorder=True to remove border components from the thresholded image'''
        self.getGray()
        self.threshes(**kwargs)
        # if self.eraseMaskSpill:
        #     self.thresh = self.nd.eraseSpillover(self.thresh, crops=self.crops, diag=self.diag-1)
        #     self.gray = self.nd.maskNozzle(self.gray, ave=True, crops=self.crops, dilate=3)
        if addNozzle:
            self.addNozzle()    # add the nozzle to the thresholded image
        if addNozzleBottom:
            self.addNozzle(bottomOnly=True)
        self.fillParts(**kwargs)    # fill components
        self.removeNozzle() # remove the nozzle again
        
    def __getattr__(self, s):
        if s in ['success', 'df', 'labeledIm']:
            return getattr(self.sdf, s)
        
    def eraseSmallComponents(self):
        '''erase small components from the labeled image and create a binary image'''
        return self.sdf.eraseSmallComponents()
        
    def eraseSmallestComponents(self, satelliteCrit:float=0.2, **kwargs) -> None:
        '''erase the smallest relative components from the labeled image'''
        return self.sdf.eraseSmallestComponents(satelliteCrit, **kwargs)
          
    def eraseBorderComponents(self, margin:int) -> None:
        '''remove any components that are too close to the edge'''
        return self.sdf.eraseBorderComponents(margin)
        
    def eraseFullWidthComponents(self) -> None:
        '''remove components that are the full width of the image'''
        return self.sdf.eraseFullWidthComponents()
        
    def eraseLeftRightBorder(self) -> None:
        '''remove components that are touching the left or right border'''
        return self.sdf.eraseLeftRightBorder()
        
    def eraseTopBottomBorder(self) -> None:
        '''remove components that are touching the top or bottom border'''
        return self.sdf.eraseTopBottomBorder()
        
        
    def largestObject(self) -> pd.Series:
        '''the largest object in the dataframe'''
        return self.sdf.largestObject()

            
    def reconstructMask(self, df:pd.DataFrame) -> np.array:
        '''construct a binary mask with all components labeled in the dataframe'''
        return self.sdf.reconstructMask(df)
    
    def noDF(self) -> bool:
        return self.sdf.noDF(df)

            
class segmenterDF:
    '''holds labeled components for an image'''
    
    def __init__(self, filled:np.array, acrit:float=100, diag:int=0):
        self.acrit = acrit
        self.filled = filled
        self.success = False
        self.w = self.filled.shape[1]
        self.h = self.filled.shape[0]
        self.diag = diag
        self.getConnectedComponents()
    
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
        if self.diag>0 and self.labeledIm.max().max()>6:
            self.resetNumbering()
        if self.numComponents==0:
            self.success = False
        else:
            self.success = True
            
    def resetNumbering(self):
        '''reset the numbering of the components so the labeledIm is easier to read'''
        j = 1
        for i,row in self.df.iterrows():
            self.labeledIm[self.labeledIm == i] = j
            self.df.rename(index={i:j}, inplace=True)
            j = j+1
            
            
    def noDF(self) -> bool:
        return not hasattr(self, 'df') or len(self.df)==0
            
    def selectComponents(self, goodpts:pd.Series) -> None:
        '''erase any components that don't fall under criterion'''
        for i in list(self.df[~goodpts].index):
            self.labeledIm[self.labeledIm==i] = 0
        self.df = self.df[goodpts] 
        self.resetStats()
            
    def eraseSmallComponents(self):
        '''erase small components from the labeled image and create a binary image'''
        if self.noDF():
            return
        goodpts = (self.df.a>=self.acrit)
        self.selectComponents(goodpts)
        
    def eraseSmallestComponents(self, satelliteCrit:float=0.2, **kwargs) -> None:
        '''erase the smallest relative components from the labeled image'''
        if self.noDF():
            return
        goodpts = (self.df.a>=satelliteCrit*self.df.a.max())
        self.selectComponents(goodpts)
          
    def eraseBorderComponents(self, margin:int) -> None:
        '''remove any components that are too close to the edge'''
        if self.noDF():
            return
        goodpts = (self.df.x0>margin)&(self.df.y0>margin)&(self.df.x0+self.df.w<self.w-margin)&(self.df.y0+self.df.h<self.h-margin)
        self.selectComponents(goodpts)
        
    def eraseFullWidthComponents(self) -> None:
        '''remove components that are the full width of the image'''
        if self.noDF():
            return
        goodpts = (self.df.w<self.w)
        self.selectComponents(goodpts)
        
    def eraseLeftRightBorder(self) -> None:
        '''remove components that are touching the left or right border'''
        if self.noDF():
            return
        goodpts = ((self.df.x0>0)&(self.df.x0+self.df.w<(self.w)))
        self.selectComponents(goodpts)
        
    def eraseTopBottomBorder(self) -> None:
        '''remove components that are touching the top or bottom border'''
        if self.noDF():
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
            return 1
        
        self.labeledIm = self.markers[1]  # this image uses different numbers to label each component
        self.stats = self.markers[2]
        self.centroids = self.markers[3]
        self.getDataFrame()       # convert stats to dataframe
        self.eraseSmallComponents()
        self.resetStats()
        return 0  

            
    def reconstructMask(self, df:pd.DataFrame) -> np.array:
        '''construct a binary mask with all components labeled in the dataframe'''
        masks = [(self.labeledIm == i).astype("uint8") * 255 for i,row in df.iterrows()]
        if len(masks)>0:
            componentMask = masks[0]
            if len(masks)>1:
                for mask in masks[1:]:
                    componentMask = cv.add(componentMask, mask)
            
        else:
            return np.zeros(self.gray.shape).astype(np.uint8)
        return componentMask
        
    
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
        
    