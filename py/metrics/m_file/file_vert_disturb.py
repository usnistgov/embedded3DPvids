#!/usr/bin/env python
'''Functions for collecting data from stills of disturbed single vertical lines'''

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
sys.path.append(os.path.dirname(os.path.dirname(currentdir)))
from file_disturb import *
from file_vert import *
from file_unit import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)

#----------------------------------------------

def vertDisturbMeasure(file:str, **kwargs) -> Tuple[dict, dict]:
    '''given a file name, measure the image and return the measured values'''
    return fileVertDisturb(file, **kwargs).values() 

def vertDisturbTestFile(fstr:str, fistr:str, **kwargs) -> None:
    '''test a single file and print diagnostics'''
    testFile(fstr, fistr, fileVertDisturb, ['x0', 'y0', 'w', 'h'], **kwargs)

class fileVertDisturb(fileVert, fileDisturb):
    '''for disturbed lines'''
    
    def __init__(self, file:str, diag:int=0, acrit:int=2500, **kwargs):
        super().__init__(file, diag=diag, acrit=acrit, **kwargs)
        
    def prepareImage(self) -> None:
        '''clean and crop the image'''
        if 'water' in self.pv.ink.base:
            self.im = self.nd.subtractBackground(self.im, diag=self.diag-2)  # remove background and nozzle
            self.im = vm.removeBlack(self.im)   # remove bubbles and dirt from the image
            self.im = vm.removeChannel(self.im,0) # remove the blue channel
        if self.pv.ink.dye=='red':
            self.im = self.nd.maskNozzle(self.im)
            self.im = vm.removeChannel(self.im, 2)   # remove the red channel

        h,w,_ = self.im.shape
        self.maxlen = h
        self.scale = 1

        # segment components
        hc = 0
        if self.name[-1]=='o':
            # observing
            self.crop = {'y0':hc, 'yf':h-hc, 'x0':200, 'xf':self.nd.xL+20, 'w':w, 'h':h}
        else:
            # writing
            self.crop = {'y0':hc, 'yf':h-hc, 'x0':self.nd.xL-100, 'xf':self.nd.xR+100, 'w':w, 'h':h}
        self.im = vc.imcrop(self.im, self.crop)
    #     im = vm.removeDust(im)
        self.im = cv.cvtColor(self.im,cv.COLOR_BGR2GRAY)  # convert to grayscale
        self.im = vm.normalize(self.im)

        
    def measure(self) -> None:
        '''measure disturbed vertical lines'''
        self.nd.importNozzleDims()
        if not self.nd.nozDetected:
            raise ValueError(f'No nozzle dimensions in {nd.printFolder}')

        self.prepareImage()
        if 'water' in self.pv.ink.base:
            bt = 210
        else:
            bt = 90
            
        self.segmenter = segmenter(self.im, acrit=self.acrit, diag=max(0,self.diag-1), cutoffTop=0, topthresh=bt, removeBorder=False, nozData=self.nd, crops=self.crop)
        if not self.segmenter.success:
            return # nothing to measure here
        self.segmenter.eraseLeftRightBorder()
        if not self.segmenter.success:
            return  # nothing to measure here

        self.dims()
        self.adjustForCrop(self.crop)
        self.gaps(self.pv.dEst)
        self.display()
        