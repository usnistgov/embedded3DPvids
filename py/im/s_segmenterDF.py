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

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)

#----------------------------------------------


class segmenterFail:
    
    def __init__(self):
        self.filled = None
        self.success = False
        
    def display(self):
        return

class segmenterDF(timeObject):
    '''holds labeled components for an image'''
    
    def __init__(self, filled:np.array, acrit:float=100, diag:int=0, **kwargs):
        self.acrit = acrit
        self.filled = filled
        self.trustLargest = 0
        self.success = False
        self.w = self.filled.shape[1]
        self.h = self.filled.shape[0]
        self.diag = diag
        if 'im' in kwargs:
            self.im = kwargs['im']
        self.getConnectedComponents()
        
    def display0(self):
        if self.diag>0:
            v = []
            titles = []
            for key,val in {'im':'seg.im', 'gray':'gray', 'thresh':'thresh', 'labeledIm':'labeled'}.items():
                if hasattr(self, key):
                    v.append(getattr(self, key))
                    titles.append(val)
            imshow(*v, maxwidth=13, titles=titles)
        
    def display(self):
        if self.diag<1:
            return
        if not hasattr(self, 'imML'):
            self.display0()
            return
        imdiag = cv.cvtColor(self.filled, cv.COLOR_GRAY2BGR)
        imdiag[(self.exc==255)] = [0,0,255]
        imshow(self.imML, self.imU, self.dif, imdiag, titles=['ML', 'Unsupervised', 'Difference'])
        return
    
    def getDataFrame(self):
        '''convert the labeled segments to a dataframe'''
        df = pd.DataFrame(self.stats, columns=['x0', 'y0', 'w', 'h','a'])
        df2 = pd.DataFrame(self.centroids, columns=['xc','yc'])
        df = pd.concat([df, df2], axis=1) 
            # combine markers into dataframe w/ label stats
        df['xf'] = df.x0+df.w
        df['yf'] = df.y0+df.h
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
        for i,row in self.df[(self.df.w<self.w)&(self.df.h<self.h)].iterrows():
            self.labeledIm[self.labeledIm == i] = j
            self.df = self.df.rename(index={i:j})
            j = j+1
            while j in self.df.index:
                j = j+1
            
            
    def noDF(self) -> bool:
        return not hasattr(self, 'df') or len(self.df)==0
    
    def touchingBorder(self, row:pd.Series, margin:int=5):
        '''determine if the object is touching the border'''
        if row['x0']<margin:
            return True
        if row['x0']+row['w']>self.w-margin:
            return True
        if row['y0']<margin:
            return True
        if row['y0']+row['h']>self.h-margin:
            return True
        
    def borderLength(self, mask:np.array) -> int:
        '''the length of the mask that is touching the border'''
        contours = co.getContours(mask, mode=cv.CHAIN_APPROX_NONE)
        if len(contours)>0:
            contours = contours[0][:,0]
            xmin = len(contours[contours[:,0]==min(contours[:,0])])
            xmax = len(contours[contours[:,0]==max(contours[:,0])])
            ymin = len(contours[contours[:,1]==min(contours[:,1])])
            ymax = len(contours[contours[:,1]==max(contours[:,1])])
            return xmin+xmax+ymin+ymax
        else:
            return -1
        
    def largestObject(self, **kwargs) -> pd.Series:
        '''the largest object in the dataframe'''
        if len(self.df)==0:
            return []
        return self.df[self.df.a==self.df.a.max()].iloc[0]
    
    def mainComponent(self, margin:int=5, pcrit:int=20) -> int:
        '''the index of the largest, most centrally located component'''
        largest = self.largestObject()
        if type(largest) is int:
            return -1
        if self.trustLargest==1:
            return largest.name
        if self.trustLargest==-1:
            return -1
        if self.touchingBorder(largest):
            # this object is close to border
            mask = self.singleMask(largest.name)
            bl = self.borderLength(mask)
            if bl>0:
                if bl>pcrit:
                    return -1
                else:
                    self.trustLargest = 1
                    return largest.name
            else:
                return -1
        else:
            return largest.name
            
    def selectComponents(self, goodpts:pd.Series, checks:bool=True, **kwargs) -> None:
        '''erase any components that don't fall under criterion'''
        if len(goodpts)==0:
            # don't empty the dataframe
            return
        mc = self.mainComponent()
        for i in list(self.df[~goodpts].index):
            if not checks or not i==mc:
                # remove this object
                self.labeledIm[self.labeledIm==i] = 0
            else:
                # add this point back in
                goodpts = goodpts|(self.df.index==i)
        self.df = self.df[goodpts] 
        self.resetStats()
            
    def eraseSmallComponents(self, **kwargs):
        '''erase small components from the labeled image and create a binary image'''
        if self.noDF():
            return
        goodpts = (self.df.a>=self.acrit)
        self.selectComponents(goodpts, checks=False, **kwargs)
        
    def eraseLargeComponents(self, acrit:int, checks:bool=False, **kwargs):
        '''erase large components from the labeled image'''
        if self.noDF():
            return
        goodpts = (self.df.a<=acrit)
        self.selectComponents(goodpts, checks=checks, **kwargs)
        
    def eraseWhiteComponent(self, checks:bool=False, **kwargs) -> None:
        '''erase components that are white on the labeled im'''
        for i in self.df.index:
            if i in self.labeledIm:
                val = max(self.labelsBW[self.labeledIm==i])
                if val==0:
                    self.selectComponents(self.df.index!=i, checks=checks, **kwargs)
                    return
        
    def eraseSmallestComponents(self, satelliteCrit:float=0.2, **kwargs) -> None:
        '''erase the smallest relative components from the labeled image'''
        if self.noDF():
            return
        goodpts = (self.df.a>=satelliteCrit*self.df.a.max())
        self.selectComponents(goodpts, **kwargs)
          
    def eraseBorderComponents(self, margin:int, **kwargs) -> None:
        '''remove any components that are too close to the edge'''
        if self.noDF():
            return
        goodpts = (self.df.x0>margin)&(self.df.y0>margin)&(self.df.xf<self.w-margin)&(self.df.yf<self.h-margin)
        self.selectComponents(goodpts, **kwargs)
        
    def eraseBorderClingers(self, margin:int, **kwargs) -> None:
        '''remove any components that are fully within margin of the border'''
        goodpts = (self.df.xf>margin)&(self.df.yf>margin)&(self.df.x0<self.w-margin)&(self.df.y0<self.h-margin)
        self.selectComponents(goodpts, **kwargs)
        
    def eraseBorderTouchComponent(self, margin:int, border:str, **kwargs) -> None:
        '''erase components that are touching the border'''
        if self.noDF():
            return
        if border=='+x':
            goodpts = self.df.xf<self.w-margin
        elif border=='-x':
            goodpts = self.df.x0>margin
        elif border=='+y':
            goodpts = self.df.yf<self.h-margin
        elif border=='-y':
            goodpts = self.df.y0>margin
        self.selectComponents(goodpts, **kwargs)
        
    def eraseBorderLengthComponents(self, lcrit:int=400, acrit:int=1000, **kwargs) -> None:
        '''erase components that are touching the border too much'''
        if self.noDF():
            return
        df2 = self.df.copy()
        df2['bl'] = [0 for i in range(len(df2))]
        for i,row in (df2[df2.a>acrit]).iterrows():
            if self.touchingBorder(row):
                mask = self.singleMask(i)
                bl = self.borderLength(mask)
                df2.loc[i,'bl'] = bl
        goodpts = (df2.bl<lcrit)
        self.selectComponents(goodpts)
        
        
    def eraseFullWidthComponents(self, margin:int=0, **kwargs) -> None:
        '''remove components that are the full width of the image'''
        if self.noDF():
            return
        goodpts = (self.df.w<self.w-margin)
        self.selectComponents(goodpts, **kwargs)
        
    def eraseFullHeightComponents(self, margin:int=0, **kwargs) -> None:
        '''remove components that are the full height of the image'''
        if self.noDF():
            return
        goodpts = (self.df.h<self.h-margin)
        self.selectComponents(goodpts, **kwargs)
        
    def eraseFullWidthHeightComponents(self, margin:int=0, **kwargs) -> None:
        '''remove components that are the full width and height of the image'''
        if self.noDF():
            return
        goodpts = (self.df.w<self.w-margin)&(self.df.h<self.h-margin)
        self.selectComponents(goodpts, **kwargs)
        
    def eraseLeftRightBorder(self, margin:int=1, **kwargs) -> None:
        '''remove components that are touching the left or right border'''
        if self.noDF():
            return
        goodpts = ((self.df.x0>margin)&(self.df.xf<(self.w-margin)))
        self.selectComponents(goodpts, **kwargs)
        
    def eraseTopBottomBorder(self, margin:int=0, **kwargs) -> None:
        '''remove components that are touching the top or bottom border'''
        if self.noDF():
            return
        goodpts = (self.df.y0>margin)&(self.df.yf<self.h-margin)
        self.selectComponents(goodpts, **kwargs)
        
    def eraseTopBorder(self, margin:int=0, **kwargs) -> None:
        '''remove components that are touching the top or bottom border'''
        if self.noDF():
            return
        goodpts = (self.df.y0>margin)
        self.selectComponents(goodpts, **kwargs)
            
    def removeScragglies(self, **kwargs) -> None:
        '''if the largest object is smooth, remove anything with high roughness'''
        if self.numComponents<=1:
            return
        for i in self.df.index:
            mask = (self.labeledIm == i).astype("uint8") * 255 
            if np.max(mask)==255:
                cnt = co.getContours(mask)[0]
                self.df.loc[i, 'roughness'] = co.contourRoughness(cnt)
        if not self.df.idxmin()['roughness']==self.df.idxmax()['a']:
            # smoothest object is not the largest object
            return
        if self.df.roughness.min()>0.5:
            # smoothest object is pretty rough
            return
        goodpts = self.df.roughness<(self.df.roughness.min()+0.5)
        self.selectComponents(goodpts, **kwargs)
        
    def gapDistance(self, r10:pd.Series, r20:pd.Series) -> float:
        '''get the gap distance between the two objects'''
        r1 = dict(r10)
        r2 = dict(r20)
        r1['yf'] = r1['y0']+r1['h']
        r2['yf'] = r2['y0']+r2['h']
        r1['xf'] = r1['x0']+r1['w']
        r2['xf'] = r2['x0']+r2['w']
        if r1['y0']>r2['yf']:
            # 1 is above 2
            ygap = r1['y0']-r2['yf']
        elif r2['y0']>r1['yf']:
            # 2 is above 1
            ygap = r2['y0']-r1['yf']
        else:
            ygap = 0
        if r1['x0']>r2['xf']:
            # 1 is right of 2
            xgap = r1['x0']-r2['xf']
        elif r2['x0']>r1['xf']:
            # 2 is right of 1
            xgap = r2['x0']-r2['xf']
        else:
            xgap = 0
        return max(xgap, ygap)
        
    def selectCloseObjects(self, idealspx:dict) -> None:
        '''select objects that are close to the ideal position and remove anything that is too far away'''
        if self.numComponents<=1:
            return
        x = idealspx['xc']
        y = idealspx['yc']
        df2 = self.df.copy()
        for i,row in df2.iterrows():
            df2.loc[i,'dist'] = np.sqrt((row['xc']-x)**2+(row['yc']-y)**2)
            df2.loc[i,'keep'] = False
        df2.sort_values(by='dist', inplace=True)
        index = df2.index
        df2.loc[index[0], 'keep'] = True
        for i in range(1, len(df2)):
            row = df2.iloc[i]
            spaces = [self.gapDistance(row, row2) for j,row2 in df2[df2.keep].iterrows()]
            if min(spaces)<max(row['w'], row['h']):
                df2.loc[index[i],'keep'] = True
        df2.sort_index(inplace=True)
        goodpts = df2.keep==True
        self.selectComponents(goodpts)

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
        self.eraseLargeComponents(acrit=(self.w*self.h/2))
        self.eraseWhiteComponent()
        return 0  

    def singleMask(self, i:int) -> np.array:
        '''get a binary mask of a single component given as a row in df'''
        return (self.labeledIm == i).astype("uint8") * 255
            
    def reconstructMask(self, df:pd.DataFrame) -> np.array:
        '''construct a binary mask with all components labeled in the dataframe'''
        masks = [self.singleMask(i) for i in df.index]
        if len(masks)>0:
            componentMask = masks[0]
            if len(masks)>1:
                for mask in masks[1:]:
                    componentMask = cv.add(componentMask, mask)
            
        else:
            return np.zeros(self.filled.shape).astype(np.uint8)
        return componentMask
    
    def componentIsIn(self, mask:np.array) -> bool:
        '''determine if the component shown in the mask overlaps with the existing image'''
        both = cv.bitwise_and(mask, self.filled)
        return both.sum().sum()>0
    
    def commonMask(self, sdf, onlyOne:bool=False) -> np.array:
        '''get the mask of all components that overlap with the segments in sdf, another segmenterDF object'''
        mask = np.zeros(self.filled.shape).astype(np.uint8)
        if not hasattr(self, 'df'):
            return mask
        for i in self.df.index:
            m = self.singleMask(i)
            if sdf.componentIsIn(m):
                mask = cv.add(mask, m)
        return mask