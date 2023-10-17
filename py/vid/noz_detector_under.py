#!/usr/bin/env python
'''Functions for detecting the nozzle in an image shot from underneath the nozzle'''

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
from noz_detector_tools import *


# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)


#----------------------------------------------

class nozDetectorUnder(nozDetector):
    '''for detecting nozzle in an image from underneath'''
    
    def __init__(self, fs, pfd, printFolder:str, **kwargs):
        super().__init__(fs, pfd, printFolder, **kwargs) 
        self.nozwidthMin = self.nod-0.1 # mm
        self.nozWidthMax = self.nod+0.1 # mm
        
    def defineHoughParams(self, dp:float=1, min_dist:float=30, upper_thresh:float=80, thresh:float=5, **kwargs):
        '''define paramters for finding circular nozzle profile'''
        self.dp = dp
        self.min_dist = min_dist
        self.upper_thresh = upper_thresh
        self.thresh = thresh
        
    def defineCritVals(self):
        '''critical nozzle position values that indicate nozzle detection may have failed, so trigger an error'''        
        # bounds in px for a 600x800 image
        self.xCmin = 270
        self.xCmax = 460
        self.yCmin = 120
        self.yCmax = 300
        
    def defineCritValsImage(self, nd, crops:dict, xmargin:int=20, ymargin:int=20, yCropMargin:int=20, xCropMargin:int=20, **kwargs) -> None:
        '''define crit vals, where this is a cropped image and we already have approximate nozzle position'''
        self.crops = crops.copy()
        x0 = crops['x0']
        y0 = crops['y0']
        self.xCmin = nd.xC-x0-xmargin
        self.xCmax = nd.xC-x0+xmargin
        self.yCmin = nd.yC-y0-ymargin
        self.yCmax = nd.yC-y0+ymargin
        self.xCCrop = nd.xC-x0-xCropMargin
        self.yCCrop = nd.yC-y0-yCropMargin
        
    def thresholdNozzle(self, frame:np.array) -> None:
        '''convert the image into an edge image'''
        self.np.line_image0 = np.copy(frame)              # copy of original frame to draw nozzle lines on
        
        # convert to gray, blurred, normalized
        if len(frame.shape)==3:
            gray2 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # convert to gray
        gray2 = vm.normalize(gray2)                   # normalize frame
        self.gray2 = cv.GaussianBlur(gray2,(5,5),0)        # blur edges
        self.np.lines_image = cv.cvtColor(self.gray2, cv.COLOR_GRAY2BGR) # blank thresholded image to draw all lines on
        self.np.edgeImage = self.gray2
        
    def nozzleLines(self):
        '''find eligible circles'''
        minRpx = int(self.nozwidthMin*self.pxpmm/2)
        maxRpx = int(self.nozWidthMax*self.pxpmm/2)
        xmin = self.xCmin-maxRpx
        xmax = self.xCmax+maxRpx
        ymin = self.yCmin-maxRpx
        ymax = self.yCmax+maxRpx
        cropped = self.gray2[ymin:ymax, xmin:xmax]
        cropped = vm.normalize(cropped)
        self.np.lines_image = cropped
        circles = cv.HoughCircles(cropped, cv.HOUGH_GRADIENT
                                  , self.dp, self.min_dist
                                  , param1=self.upper_thresh, param2=self.thresh
                                  , minRadius = minRpx, maxRadius = maxRpx)
        if circles is None or len(circles)==0:
            raise ValueError(f'No circles found in {self.printFolder}')
 
        self.circles = pd.DataFrame(circles.reshape(len(circles[0]),3), columns=['xC', 'yC', 'r'], dtype='int32')
        self.circles['xC'] = [x+xmin for x in self.circles['xC']]
        self.circles['yC'] = [y+ymin for y in self.circles['yC']]
        self.np.circles = self.circles
        
    def findNozzlePoints(self, **kwargs):
        '''choose the first circle'''
        self.nd = nozDimsUnder(self.pfd, importDims=False)
        self.nd.setDims({'xC':self.circles.loc[0,'xC'], 'yC':self.circles.loc[0,'yC'], 'r':self.circles.loc[0,'r']})
        self.np.nd = self.nd
        self.nd.nozDetected=True
        
    def checkNozzleValues(self):
        return
        
        