#!/usr/bin/env python
'''Functions for collecting data from stills of single line horiz'''

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
import shutil
import subprocess
import copy

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
from file_unit import *
from file_horiz import *
from file_disturb import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)

#----------------------------------------------

def horizDisturbMeasureFile(file:str, **kwargs) -> Tuple[dict, dict]:
    return fileHorizDisturb(file, **kwargs).values()

def horizDisturbTestFile(fstr:str, fistr:str, **kwargs) -> None:
    '''test a single file and print diagnostics'''
    testFile(fstr, fistr, fileHorizDisturb, ['x0', 'y0', 'w', 'h'], **kwargs)

class fileHorizDisturb(fileHoriz, fileDisturb):
    '''for disturbed horizontal lines'''
    
    def __init__(self, file:str, diag:int=0, acrit:int=2500, f:float=0.4, f2:float=0.3, **kwargs):
        self.f = f
        self.f2 = f2
        super().__init__(file, diag=diag, acrit=acrit, **kwargs)
        

    def removeThreads(self, diag:int=0) -> np.array:
        '''remove zigzag threads from bottom left and top right part of binary image'''
        f = self.f
        f2 = self.f2
        thresh = self.segmenter.labelsBW.copy()
        if diag>0:
            thresh0 = thresh.copy()
            thresh0 = cv.cvtColor(thresh0, cv.COLOR_GRAY2BGR)
        h0,w0 = thresh.shape
        left = thresh[:, :int(w0*f)]
        right0 = int(w0*(1-f))
        right = thresh[:, right0:]
        for i,im in enumerate([left, right]):
            contours = cv.findContours(im, 1, 2)
            if int(cv.__version__[0])>=4:
                contours = contours[0]
            else:
                contours = contours[1]
            contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True) # select the largest contour
            if len(contours)>0:
                x,y,w,h = cv.boundingRect(contours[0])
                if i==0:
                    # mask the top if the right edge is tall
                    if thresh[:y, int(w0*(1-f2)):].sum(axis=0).sum(axis=0)>0:
                        thresh[:y-10, :] = 0
                        if diag>0:
                            thresh0 = cv.rectangle(thresh0, (0,0), (w0,y-10), (255,0,0), 2)
                            thresh0 = cv.rectangle(thresh0, (x,y), (x+w,y+h), (255,0,0), 2)
                else:
                    # mask the bottom on the left side if the left edge is tall
                    if thresh[h+y+10:, :int(w0*f2)].sum(axis=0).sum(axis=0)>0:
                        thresh[h+y+10:, :int(w0*f2)] = 0
                        if diag>0:
                            x = x+right0
                            thresh0 = cv.rectangle(thresh0, (0,h+y+10), (int(w0*f2),h0), (0,0,255), 2)
                            thresh0 = cv.rectangle(thresh0, (x,y), (x+w,y+h), (0,0,255), 2)                    
        if diag>0:
            imshow(thresh0, thresh)
        self.segmenter.filled = thresh
        self.segmenter.getConnectedComponents()
        
    def prepareImage(self):
        '''clean and crop the image'''
        if self.pv.ink.dye=='blue':
            self.im = self.nd.subtractBackground(self.im, diag=self.diag-2)  # remove background and nozzle
            self.im = vm.removeBlack(self.im)   # remove bubbles and dirt from the image
            self.im = vm.removeChannel(self.im,0) # remove the blue channel
        elif self.pv.ink.dye=='red':
            self.im = self.nd.maskNozzle(self.im)
            self.im = vm.removeChannel(self.im, 2)   # remove the red channel

        h,w,_ = self.im.shape
        self.scale = 1
        self.maxlen = w

        # segment components
        hc = 0
        if self.name[-1]=='o':
            # observing
            self.crop = {'y0':int(h/2), 'yf':int(h*6/6), 'x0':hc, 'xf':w-hc, 'w':w, 'h':h}   # crop the left and right edge to remove zigs
        else:
            # writing
            self.crop = {'y0':int(h/6), 'yf':int(h*5/6), 'x0':hc, 'xf':w-hc, 'w':w, 'h':h}
        self.im = ic.imcrop(self.im, self.crop)
    #     im = vm.removeDust(im)
        self.im = vm.normalize(self.im)


    def measure(self) -> None:
        '''measure disturbed horizontal lines'''
        self.nd.importNozzleDims()
        if not self.nd.nozDetected:
            raise ValueError(f'No nozzle dimensions in {self.nd.printFolder}')

        self.prepareImage()
        if 'water' in self.pv.ink.base:
            bt = 200
        else:
            bt = 90
            
        self.segmenter = segmenter(self.im, acrit=self.acrit, diag=max(0,self.diag-1), cutoffTop=0, topthresh=bt, removeBorder=False, nozData=self.nd, crops=self.crop, eraseMaskSpill=True)
        if not self.segmenter.success:
            return
        self.segmenter.eraseFullWidthComponents()
        self.segmenter.eraseTopBottomBorder()
        self.removeThreads(diag=self.diag-1)
        if self.diag>1:
            self.segmenter.display()
        self.dims()
        self.adjustForCrop(self.crop)
        self.gaps(self.pv.dEst)
        self.display()
        

