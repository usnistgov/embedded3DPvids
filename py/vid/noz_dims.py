#!/usr/bin/env python
'''Functions for storing dimensions of the nozzle'''

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
import csv
import random
import time

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
from im.imshow import imshow
from im.contour import getContours
import im.morph as vm
import im.crop as vc
from tools.config import cfg
from tools.plainIm import *
import file.file_handling as fh
from v_tools import vidData
from noz_plots import *


# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)


#----------------------------------------------

class nozDims:
    '''holds dimensions of the nozzle'''
    
    def __init__(self, pfd:fh.printFileDict, importDims:bool=True):
        self.xL = -1
        self.xR = -1
        self.yB = -1
        self.xM = -1
        self.w = 790
        self.h = 590
        self.pfd = pfd
        self.pxpmm = self.pfd.pxpmm()
        self.nozDetected = False
        if importDims:
            self.importNozzleDims()
        
        
    def adjustForCrop(self, crops:dict) -> None:
        self.xL = int(self.xL + crops['x0'])
        self.xR = int(self.xR + crops['x0'])
        self.xM = int(self.xM + crops['x0'])
        self.yB = int(self.yB + crops['y0'])
        
    def padNozzle(self, left:int=0, right:int=0, bottom:int=0):
        self.xL = self.xL-left
        self.xR = self.xR+right
        self.yB = self.yB+bottom
        
    def nozDims(self) -> dict:
        '''get the nozzle dimensions
        yB is from top, xL and xR are from left'''
        return {'xL':self.xL, 'xR':self.xR, 'yB':self.yB}
    
    def nozDimsFN(self) -> str:
        '''file name of nozzle dimensions table'''
        # store noz dimensions in the subfolder
        return self.pfd.newFileName('nozDims', 'csv')
    
    def setSize(self, im:np.array) -> None:
        '''set the current dimensions equal to the dimensions of the image'''
        self.h, self.w = im.shape[:2]
        
    def setDims(self, d:dict) -> None:
        '''adopt the dimensions in the dictionary'''
        if 'xL' in d:
            self.xL = int(d['xL'])
        if 'xR' in d:
            self.xR = int(d['xR'])
        if 'yB' in d:
            self.yB = int(d['yB'])
        self.xM = int((self.xL+self.xR)/2)
        
    def copyDims(self, nd:nozDims) -> None:
        '''copy dimensions from another nozDims object'''
        self.yB = nd.yB
        self.xL = nd.xL
        self.xR = nd.xR
        self.xM = nd.xM
    
    
    def exportNozzleDims(self, overwrite:bool=False) -> None:
        '''export the nozzle location to file'''
        fn = self.nozDimsFN()  # nozzle dimensions file name
        if os.path.exists(fn) and not overwrite:
            return
        plainExpDict(fn, {'yB':self.yB, 'xL':self.xL, 'xR':self.xR, 'pxpmm':self.pxpmm, 'w':self.w, 'h':self.h})
        
        
    def importNozzleDims(self) -> int:
        '''find the target pressure from the calibration file. returns 0 if successful, 1 if not'''
        fn = self.nozDimsFN()      # nozzle dimensions file name
        if not os.path.exists(fn):
            self.nozDetected = False
            return 1
        tlist = ['yB', 'xL', 'xR', 'pxpmm', 'w', 'h']
        d,_ = plainImDict(fn, unitCol=-1, valCol=1)
        for st,val in d.items():
            setattr(self, st, int(val))
        if len(set(tlist)-set(d))==0:
            # we have all values
            self.xM = int((self.xL+self.xR)/2)
            self.nozDetected = True
            return 0
        else:
            return 1
        
        
    def absoluteCoords(self, d:dict) -> dict:
        '''convert the relative coordinates in mm to absolute coordinates on the image in px. y is from the bottom, x is from the left'''
        nc = [self.xM, self.yB]    # convert y to from the bottom
        out = {'x':nc[0]+d['dx']*self.pxpmm, 'y':nc[1]-d['dy']*self.pxpmm}
        return out
    
    def relativeCoords(self, x:float, y:float, reverse:bool=False) -> dict:
        '''convert the absolute coordinates in px to relative coordinates in px, where y is from the top and x is from the left. reverse=True to go from mm to px'''
        nx = self.xM
        ny = self.yB
        if reverse:
            return x*self.pxpmm+nx, ny-y*self.pxpmm
        else:
            return (x-nx)/self.pxpmm, (ny-y)/self.pxpmm
        
    def nozWidth(self):
        '''nozzle width in mm'''
        return (self.xR-self.xL)/self.pxpmm
    
    def nozCover(self, padLeft:int=0, padRight:int=0, padBottom:int=0, val:int=255, y0:int=0, color:bool=False, **kwargs) -> np.array:
        '''get a mask that covers the nozzle'''
        if type(val) is list:
            mask = np.zeros((self.h, self.w, len(val)), dtype="uint8")
        else:
            mask = np.zeros((self.h, self.w), dtype="uint8")
        if y0<0:
            y0 = self.yB+y0
        yB = int(self.yB + padBottom)
        xL = int(self.xL - padLeft)
        xR = int(self.xR + padRight)
        mask[y0:yB, xL:xR]=val
        if color and len(mask.shape)==2:
            mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        if 'crops' in kwargs:
            mask = vc.imcrop(mask, kwargs['crops'])
        return mask
    
    def dentHull(self, hull:list, crops:dict) -> list:
        '''conform the contour to the nozzle'''
        
        yB = int(self.yB-crops['y0'])
        xL = int(self.xL-crops['x0'])
        xR = int(self.xR-crops['x0'])
        
        image1 = np.zeros((self.h, self.w), dtype=np.uint8)
        cv.drawContours(image1, [hull], -1, 1, 1)
        hull2 = getContours(image1, cv.CHAIN_APPROX_NONE)[0]
        df = pd.DataFrame(hull2[:,0], columns=['x', 'y'])
        insidepts = df[(df.x>=xL)&(df.x<=xR)&(df.y<=yB)]

        if len(insidepts)==0 or (insidepts.y.min()==yB) or (insidepts.x.min()==xR) or (insidepts.x.max()==xL):
            # no points to dent
            return hull
        
        # find the points that intersect the nozzle
        leftedge = insidepts[insidepts.x==insidepts.x.min()]
        rightedge = insidepts[insidepts.x==insidepts.x.max()]
        li = leftedge[leftedge.y==leftedge.y.max()].iloc[0].name
        ri = rightedge[rightedge.y==rightedge.y.max()].iloc[-1].name     
        lpt = df.loc[li]   # point at left
        rpt = df.loc[ri]   # point at right
        
        # get the nozzle points
        nozpts = []
        if lpt.x==xL:
            # intersect with left edge of nozzle
            nozpts.append([xL, lpt.y])
            nozpts.append([xL, yB])
            if rpt.y==yB:
                # intersect with bottom edge of nozzle
                nozpts.append([rpt.x, yB])
            elif rpt.x==xR:
                # intersect with right edge of nozzle
                nozpts.append([xR, yB])
                nozpts.append([xR, rpt.y])
            elif rpt.x<xR:
                nozpts.append([rpt.x, rpt.y])
            else:
                
                raise ValueError('Unexpected intersection points')
        elif lpt.y==yB:
            nozpts.append([lpt.x, yB])
            if rpt.x==xR:
                # intersect with right edge of nozzle
                nozpts.append([xR, yB])
                nozpts.append([xR, rpt.y])
            else:
                raise ValueError('Unexpected intersection points')
        else:
            raise ValueError('Unexpected intersection points')
            
        # reconstitute the list of points
        if li>ri or ri>li:
            # points go counterclockwise
            nozpts.reverse()
            hull3 = np.vstack([hull2[:ri, 0, :], nozpts, hull2[li:, 0, :]])
        else:
            hull3 = np.vstack([hull2[:li, 0, :], nozpts, hull2[ri:, 0, :]])
            
        # simplify the list of points
        image2 = np.zeros((self.h, self.w), dtype=np.uint8)
        cv.drawContours(image2, [hull3], -1, 1, 1)
        hull4 = getContours(image2, cv.CHAIN_APPROX_SIMPLE)[0]
        return hull4[:,0,:]
            
            
#         if li>ri:
#             # points go counterclockwise
#             nozpts.reverse()
#             firsti = ri
#             lasti = li
#         else:
#             firsti = li
#             lasti = ri
          
#         df3 = pd.DataFrame(hull[:,0], columns=['x', 'y'])
#         firstpt = df.loc[firsti]
#         while firsti>0 and len(self.dfMatch(df3, firstpt))==0:
#             firsti = firsti-1
#             firstpt = df.loc[firsti]
#         if firsti>0:
#             ii = self.dfMatch(df3, firstpt).iloc[0].name
#         else:
#             ii = 0
            
#         lastpt = df.loc[lasti]
#         while lasti<len(df) and len(self.dfMatch(df3, lastpt))==0:
#             lasti = lasti+1
#             lastpt = df.loc[lasti]
#         if lasti<len(df):
#             jj = self.dfMatch(df3, lastpt).iloc[0].name
#             lastgroup = hull[jj:, 0, :]
#         else:
#             jj = len(hull)-1
 
#         print(firsti, lasti, li, ri, ii, jj, len(hull))
#         if ii==0 and jj==len(hull)-1:
#             nozpts.reverse()
#             return np.vstack([hull[:-1,0,:], nozpts])
#         elif jj<ii:
#             nozpts.reverse()
#             return np.vstack([hull[:jj, 0, :], nozpts, hull[ii:, 0, :]])
#         elif ii<jj:
#             if ii==0:
#                 ii = jj-2
#             print(hull[ii, 0, :], hull[jj, 0, :])
#             return np.vstack([hull[:ii, 0, :], nozpts, hull[jj:, 0, :]])
    
#     def dfMatch(self, df3:pd.DataFrame, pt:pd.Series, i:int=3) -> pd.DataFrame:
#         return df3[(df3.x>(pt.x-i))&(df3.x<(pt.x+i))&(df3.y>(pt.y-i))&(df3.y<(pt.y+i))]
    
        
        

#         blank = np.zeros((self.h, self.w))

#         # Copy each contour into its own image and fill it with '1'
#         image1 = cv.drawContours(blank.copy(), [hull], -1, 1, 1)
#         image2 = cv.drawContours(blank.copy(), [np.array([[xL, 0], [xL, yB], [xR, yB], [xR, 0]])],-1, 1, 1)

#         # Use the logical AND operation on the two images
#         # Since the two images had bitwise and applied to it,
#         # there should be a '1' or 'True' where there was intersection
#         # and a '0' or 'False' where it didnt intersect
#         intersection = cv.bitwise_and(image1, image2)
#         pts = cv.findNonZero(intersection)
        
#         print(pts)
#         imshow(image1, image2, intersection)
        
        
        
        
        # df = pd.DataFrame(hull[:,0] , columns=['x', 'y'])
        # if xL-10>df.x.max() or xL<df.x.min() or yB<df.y.min():
        #     # all points are left or right of the left edge of the nozzle or below the nozzle
        #     return hull
        # right = df[(df.y<yB)&(df.x>xR)]
        # under = df[(df.x<=xR)&(df.x>=xL)&(df.y>=yB)&(df.y<yB+10)]
        # ru = pd.concat([right, under])
        # if len(ru)==0:
        #     return hull
        # left = df[df.x<xL]
        # leftu = df[(df.x<xL+1)&(df.x>xL-10)&(df.y<yB)]
        # if len(left)==0:
        #     u1 = ru[ru.y<ru.y.min()+5]         # highest point on right/under
        #     u2 = u1[u1.x==u1.x.min()].iloc[0] # leftmost point at highest point
        #     i = int(u2.name)
        # else:
        #     # points go clockwise, so the under point will be after the left point
        #     # i = left.index.max()+1  # index of the transition point
        #     i = left[left.x==left.x.max()].index.max()+1
        #     xL = max(hull[i-1, 0, 0],xL)
        # if len(leftu)==0:
        #     # include left point at nozzle and bottom left corner
        #     pt = np.array([[xL, hull[i-1, 0, 1]], [xL, yB]])
        # else:
        #     # include just bottom left corner
        #     pt = np.array([[xL, yB]])
        # if len(right)>0:
        #     # points to the right of the nozzle. include bottom right corner
        #     pt = np.concatenate((pt, np.array([[xR, yB], [xR, right.y.min()]])))
        # hull2 = np.vstack([hull[:i, 0, :], pt, hull[i:, 0, :]])
        # if cv.contourArea(hull2)>cv.contourArea(hull):
        #     # new contour has larger area, so dent was in the wrong direction. just return the original hull
        #     return hull
        # else:
        #     return hull2
        