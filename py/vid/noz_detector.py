#!/usr/bin/env python
'''Functions for detecting the nozzle in an image'''

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
import im.morph as vm
import im.crop as vc
from tools.config import cfg
from tools.plainIm import *
import file.file_handling as fh
from v_tools import vidData
from noz_plots import *
from noz_dims import *
from background import *


# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)


#----------------------------------------------
def combineLines(df:pd.DataFrame) -> dict:
    '''combine groups of similar Hough transform lines into one line'''
    top = (df[df.y0==df.y0.min()]).iloc[0]
    bot = (df[df.yf==df.yf.max()]).iloc[0]
    return {'x0':top['x0'],'y0':top['y0'],'xf':bot['xf'],'yf':bot['yf']}

def lineIntersect(line1:pd.Series, line2:pd.Series) -> Tuple[float,float]:
    '''find intersection between two lines'''
    if line1['xf']-line1['x0']==0:
        # line 1 is vertical
        x = line1['xf']
        m1 = (line2['yf']-line2['y0'])/(line2['xf']-line2['x0'])
        line1['y0']=line2['y0']
        line1['x0']=line2['x0']
    elif line2['xf']-line2['x0']==0:
        # line 2 is vertical
        x = line2['xf']
        m1 = (line1['yf']-line1['y0'])/(line1['xf']-line1['x0'])
    else:
        m1 = (line1['yf']-line1['y0'])/(line1['xf']-line1['x0'])
        m2 = (line2['yf']-line2['y0'])/(line2['xf']-line2['x0'])
        x = (line2['y0']-line1['y0']-m2*line2['x0']-m1*line1['x0'])/(m1-m2)
    y = line1['y0']+m1*(x-line1['x0'])
    return int(x),int(y)


class nozDetector:
    '''for detecting the nozzle in an image'''
    
    def __init__(self, fs, pfd, printFolder:str, **kwargs):
        self.fs = fs   # frame selector
        self.np = nozPlotter(printFolder)
        self.crops = {'x0':0, 'y0':0}
        self.pfd = pfd
        self.pxpmm = self.pfd.pxpmm()
        self.printFolder = printFolder
        self.defineHoughParams(**kwargs)
        self.defineCritVals()
        
    def defineHoughParams(self, min_line_length:int = 50, max_line_gap:int=300, threshold:int=30, rho:int=0, critslope:float=0.1, theta:float=np.pi/180, hmax:int=400, **kwargs):
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.threshold = threshold
        if rho==0:
            self.rho = int(3*self.pxpmm/139)
        else:
            self.rho = rho
        self.critslope = critslope
        self.theta = theta
        self.hmax = hmax
        
    def defineCritVals(self):
        '''critical nozzle position values that indicate nozzle detection may have failed, so trigger an error'''
        
        # bounds in px for a 600x800 image
        self.xLmin = 200 # px
        self.xLmax = 500
        self.xRmin = 300
        self.xRmax = 700
        self.yBmin = 200
        self.yBmax = 430
        
        # bounds of size of nozzle in mm. for 20 gauge nozzle, diam should be 0.908 mm
        self.nozwidthMin = 0.75 # mm
        self.nozWidthMax = 1.2 # mm
        
    def defineCritValsImage(self, nd, crops:dict, xmargin:int=20, ymargin:int=20, yCropMargin:int=20, xCropMargin:int=20, **kwargs) -> None:
        '''define crit vals, where this is a cropped image and we already have approximate nozzle position'''
        self.crops = crops.copy()
        x0 = crops['x0']
        y0 = crops['y0']
        self.xLmin = nd.xL-x0-xmargin # px
        self.xLmax = nd.xL-x0+xmargin
        self.xRmin = nd.xR-x0-xmargin
        self.xRmax = nd.xR-x0+xmargin
        self.yBmin = nd.yB-y0-ymargin
        self.yBmax = nd.yB-y0+ymargin
        self.xLCrop = nd.xL-x0-xCropMargin
        self.xRCrop = nd.xR-x0+xCropMargin
        self.yBCrop = nd.yB-y0+yCropMargin

    
    def thresholdNozzle(self, frame:np.array) -> None:
        '''convert the image into an edge image'''
        self.np.line_image0 = np.copy(frame)              # copy of original frame to draw nozzle lines on
        
        # convert to gray, blurred, normalized
        if len(frame.shape)==3:
            gray2 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # convert to gray
        gray2 = vm.normalize(gray2)                   # normalize frame
        gray2 = cv.GaussianBlur(gray2,(5,5),0)        # blur edges
        self.np.lines_image = cv.cvtColor(gray2, cv.COLOR_GRAY2BGR) # blank thresholded image to draw all lines on
        
        # take edge
        thres2 = cv.Canny(gray2, 5, 80)             # edge detect
        thres2 = vm.dilate(thres2,3)                  # thicken edges
        
        # only include points above a certain threshold (nozzle is black, so this should get rid of most ink)
        _,threshmask = cv.threshold(gray2, 50,255,cv.THRESH_BINARY_INV)
        threshmask = vm.dilate(threshmask, 15)
        thres2 = cv.bitwise_and(thres2, thres2, mask=threshmask)
        self.edgeImage = thres2.copy()                # store edge image for displaying diagnostics
        self.np.edgeImage = self.edgeImage
        
        
    def nozzleLines0(self, im:np.array):
        theta = np.pi/180   # angular resolution in radians of the Hough grid
      # threshold is minimum number of votes (intersections in Hough grid cell)
        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv.HoughLinesP(self.edgeImage, self.rho, self.theta, self.threshold, np.array([]), self.min_line_length, self.max_line_gap)
        
        if lines is None or len(lines)==0:
            return [], []

        # convert to dataframe
        lines = pd.DataFrame(lines.reshape(len(lines),4), columns=['x0', 'y0', 'xf', 'yf'], dtype='int32')        
        lines['slope'] = abs(lines['x0']-lines['xf'])/abs(lines['y0']-lines['yf'])
        # find horizontal lines
        horizLines = lines[(lines['slope']>20)&(lines['y0']>self.yBmin)&(lines['y0']<self.yBmax)]
        lines0h = horizLines.copy()
        
        # only take nearly vertical lines, where slope = dx/dy
        lines = lines[(lines['slope']<self.critslope)&(lines['x0']>self.xLmin)&(lines['x0']<self.xRmax)]
        lines0 = lines.copy()
        # sort each line by y
        for i,row in lines0.iterrows():
            if row['yf']<row['y0']:
                lines0.loc[i,['x0','y0','xf','yf']] = list(row[['xf','yf','x0','y0']])
        lines0 = lines0.convert_dtypes()         # convert back to int
        lines0 = lines0[lines0.yf<self.hmax]      # only take lines that extend close to the top of the frame
        return lines0h, lines0
        
        
    def nozzleLines(self) -> pd.DataFrame:
        '''get lines from the stored edge image'''
        lines0h, lines0 = self.nozzleLines0(self.edgeImage)
        if len(lines0)==0:
            raise ValueError('Failed to detect any lines in nozzle')
        if len(lines0h)>0:
            self.lines0h = lines0h
        else:
            self.lines0h = pd.DataFrame([])
        if len(lines0)>0:
            self.lines0 = lines0
        else:
            self.lines0 = pd.DataFrame([])
        self.np.lines0h = self.lines0h
        self.np.lines0 = self.lines0
        
    def useHoriz(self) -> None:
        '''use horizontal line to find nozzle corners'''
        horizLine = self.lines0h.iloc[0]                        # dominant line
        xL, yL = lineIntersect(horizLine, self.lines.loc[0])    # find the point where the horizontal line and left vertical line intersect
        self.leftCorner = pd.Series({'xf':xL, 'yf':yL})
        xR, yR = lineIntersect(horizLine, self.lines.loc[1])    # find the point where the horizontal line and right vertical line intersect
        self.rightCorner = pd.Series({'xf':xR, 'yf':yR})
        
    def useVerticals(self) -> None:
        '''use vertical lines to find corners'''
        # corners are bottom points of each line
        self.leftCorner = self.lines.loc[0,['xf','yf']]         # take bottom point of left vertical line
        self.rightCorner = self.lines.loc[1,['xf', 'yf']]       # take bottom point of right vertical line
        
    def findNozzlePoints(self, mode:int=4, **kwargs) -> None:
        '''find lines and corners of nozzle from list of lines'''
        # based on line with most votes, group lines into groups on the right edge and groups on the left edge
        best = self.lines0.iloc[0]                # line with most votes
        dx = max(10,2*abs(best['xf']-best['x0'])) # margin of error for inclusion in the group
        nearbest = self.lines0[(self.lines0.x0<best['x0']+dx)&(self.lines0.x0>best['x0']-dx)]  # lines that are near the best line
        
        # group lines between 0.5 and 1.5 nozzles away on left and right side of best line
        margin = 0.45*self.pxpmm # half a nozzle
        right = self.lines0[(self.lines0.x0>best['x0']+margin)&(self.lines0.x0<best['x0']+3*margin)] # lines that are 1-3 margins to the left of the best line
        left = self.lines0[(self.lines0.x0<best['x0']-margin)&(self.lines0.x0>best['x0']-3*margin)]  # lines that are 1-3 margins to the right of the best line
    
        
        if len(right)>len(left):
            left = nearbest     # best was the left side, use nearbest as the left lines
        else:
            right = nearbest    # best was the right side, use nearbest as the right lines
            
        if len(left)==0 or len(right)==0:
            raise ValueError('Failed to detect left and right edges of nozzle')

        # combine all left lines into one line and combine all right lines into one line
        self.lines = pd.DataFrame([combineLines(left), combineLines(right)])
        self.np.lines = self.lines
        
        if len(self.lines0h)>0:
            # we have already defined horizontal lines. use horizontal lines to find corners
            self.useHoriz()
            if min([self.leftCorner['yf'],self.rightCorner['yf']]) > self.lines0.yf.max():
                # horiz line is far below verticals
                # throw out the horiz line and use the verticals
                self.useVerticals()
        else:
            # we do not have horizontal lines. Use the bottom of the verticals to defined the bottom edge
            self.useVerticals()
        
        # store left edge x, right edge x, and bottom edge midpoint
        xL = self.leftCorner['xf']   # left corner x
        xR = self.rightCorner['xf']  # right corner x
        
        if abs(self.leftCorner['yf']-self.rightCorner['yf'])>20:
            # if bottom edge is not horizontal, use bottommost point for y position of nozzle bottom
            yB = max([self.leftCorner['yf'],self.rightCorner['yf']])
        else:
            # otherwise, use mean of bottom point of left and right edges
            yB = (self.leftCorner['yf']+self.rightCorner['yf'])/2
        self.nd = nozDims(self.pfd, importDims=False)
        if mode==1:
            # means erase lateral movement of the nozzle, so need to expand the nozzle slightly
            xL = xL-10
        self.nd.setDims({'yB':yB, 'xL':xL, 'xR':xR})
        self.np.nd = self.nd
        self.nd.nozDetected = True

            
    def checkNozzleValues(self) -> None:
        '''check that nozzle values are within expected bounds'''
        # check values
        if self.nd.xL<self.xLmin or self.nd.xL>self.xLmax:
            raise ValueError(f'Detected left corner is outside of expected bounds: {self.nd.xL} ({self.xLmin}, {self.xLmax})')
        if self.nd.xR<self.xRmin or self.nd.xR>self.xRmax:
            raise ValueError(f'Detected right corner is outside of expected bounds: {self.nd.xR} ({self.xRmin}, {self.xRmax})')
        if self.nd.yB<self.yBmin or self.nd.yB>self.yBmax:
            raise ValueError(f'Detected bottom edge is outside of expected bounds: {self.nd.yB} ({self.yBmin}, {self.yBmax})')
        nozwidth = self.nd.nozWidth()
        if nozwidth<self.nozwidthMin:
            raise ValueError(f'Detected nozzle width is too small: {nozwidth} mm')
        if nozwidth>self.nozWidthMax:
            raise ValueError(f'Detected nozzle width is too large: {nozwidth} mm')
            
    
    
    def detectNozzle0(self, frame:np.array, diag:int=0, suppressSuccess:bool=False, overwrite:bool=False, export:bool=True, **kwargs) -> None:
        '''find the bottom corners of the nozzle. suppressSuccess=True to only print diagnostics if the run fails'''
        self.thresholdNozzle(frame)    # threshold the nozzle
        self.nozzleLines()            # edge detect and Hough transform to get nozzle edges as lines
        if len(self.lines0)==0:
            raise ValueError('Failed to detect any vertical lines in nozzle')
        self.findNozzlePoints(**kwargs)       # filter the lines to get nozzle coords
        self.checkNozzleValues()      # make sure coords make sense
        if export:
            self.nd.exportNozzleDims(overwrite=overwrite)       # save coords  
        if diag>0 and not suppressSuccess:
            self.np.drawDiagnostics(diag) # show diagnostics
            
    def detectNozzle1(self, frame:np.array, diag:int=0, margin:int=5, **kwargs) -> None:
        '''just detect the nozzle for this one frame, with error handling'''
        if hasattr(self, 'yBCrop'):
            # crop the frame to just the ROI
            x0 = self.xLCrop
            
            frame = frame[0:self.yBCrop+1, x0:self.xRCrop+1]
            self.defineHoughParams(max_line_gap=50)
            self.xLmin = self.xLmin-x0+margin # px
            self.xLmax = self.xLmax-x0+margin # px
            self.xRmin = self.xRmin-x0-margin # px
            self.xRmax = self.xRmax-x0-margin # px
            self.crops['x0'] = x0+self.crops['x0']
        try:
            self.detectNozzle0(frame, diag=diag, **kwargs)
        except ValueError as e:
            self.np.drawDiagnostics(diag) # show diagnostics
            raise e

        if 'xf' in self.crops:
            self.nd.adjustForCrop(self.crops)

    def detectNozzle(self, diag:int=0, **kwargs) -> None:
        '''find the bottom corners of the nozzle, trying different images. suppressSuccess=True to only print diagnostics if the run fails'''
        if 'modes' in kwargs:
            modes = kwargs['modes']
        else:
            if 'Horiz' in self.printFolder:
                modes = [4,1]
            else:
                modes = [1]
        
        for mode in modes: # min, median, then mean
            for i in range(3):
                try:
                    frame = self.fs.frame(mode=mode, numpics=10, **kwargs)           # get median or averaged frame
                    self.detectNozzle0(frame, diag=diag, mode=mode, **kwargs)
                except ValueError as e:
                    if diag>1:
                        print(f'{e}: Looping to next mode')
                    pass
                else:
                    return 0
            
        # if all modes failed:
        self.np.drawDiagnostics(diag) # show diagnostics
        raise ValueError('Failed to detect nozzle after 9 iterations')
    
    