#!/usr/bin/env python
'''Functions for collecting data from a still of a horizontal SDT line'''

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
from file_SDT import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)

#----------------------------------------------

def horizSDTMeasure(file:str, **kwargs) -> Tuple[dict, dict]:
    return fileHorizSDT(file, **kwargs).values() 

def horizSDTTestFile(fstr:str, fistr:str, **kwargs) -> None:
    '''test a single file and print diagnostics'''
    testFile(fstr, fistr, fileHorizSDT, ['emptiness','x0','segments'], **kwargs)
        
class fileHorizSDT(fileHoriz, fileSDT):
    '''for singledoubletriple lines'''
    
    def __init__(self, file:str, diag:int=0, acrit:int=1000, overrideSegment:bool=False, **kwargs):
        self.overrideSegment = overrideSegment
        self.maxlen = 800
        super().__init__(file, diag=diag, acrit=acrit, **kwargs)

                
    def findIntendedCoords(self) -> None:
        '''find the intended x0,y0,xc,and yc of the assembly'''
        rc1, rc2, w1, w2, l = self.intendedRC()
        for j in [['dx', 'w']]:
            coord = j[0][1]
            right = (rc2[j[0]]+w2/2)      # the left edge should be 1/2 diameter to the left of the first line center
            left = (rc1[j[0]]-w1/2)       # the right edge should be 1/2 diameter to the right of the last line center
            if coord=='y':
                y0 = left
                left = right
                right = y0
            self.ideals[f'{coord}0'] = left
            self.ideals[f'{coord}f'] = right
            self.ideals[j[1]] = abs(right - left)     # get the ideal width
            self.ideals[f'{coord}c'] = (right+left)/2  # get the ideal center
        w = self.ideals['w']
        r = w/2
        self.ideals['area'] = l*w
        self.ideals['v'] = (l-w)*np.pi*r**2 + 4/3*np.pi*r**3
        self.ideals['h'] = l
        
    def findDisplacement(self) -> None:
        '''find the displacement of the center and dimensions relative to the intended dimensions'''
        for s,ival in {'xc':'xc', 'x0':'x0', 'xf':'xf', 'dxprint':'xc'}.items():
            # ratio of size to intended size
            if s in self.stats:
                self.stats[s] = ((self.stats[s]-self.ideals[ival])/self.pv.dEst)
                self.units[s] = 'dEst'
        for s in ['dxprint', 'dx0', 'dxf', 'space_a', 'space_at', 'h']:
            if s in self.stats:
                self.stats[s] = self.stats[s]/self.pv.dEst
                self.units[s] = 'dEst'
        for s,ival in {'w':'w', 'meanT':'w', 'vest':'v', 'vintegral':'v'}.items():
            # ratio of size to intended size
            if s in self.stats:
                self.stats[s] = (self.stats[s]/self.ideals[ival])
                self.units[s] = 'intended'
        for s in ['area', 'vleak', 'aspect']:
            if s in self.stats:
                self.stats.pop(s)
                
        # remove length measurements for mid-print measurements 
        if not 'o' in self.tag:
            for s in ['h', 'vest', 'vintegral', 'w', 'xf', 'meanT', 'stdevT', 'minmaxT']:
                if s in self.stats:
                    self.stats.pop(s)
                    
    def initialize(self) -> None:
        '''initialize variables'''
        self.getProgRow()
        self.fillDilation = 0
        self.grayBlur = 1
        
    def getCrop(self, export:bool=True, overwrite:bool=False):
        '''get the crop position. only export if export=True and there is no existing row'''
        if not self.hasIm:
            h = 590
            w = 790
        else:
            h,w = self.im.shape[:2]
        self.importCrop()
        if len(self.crop)==0 or overwrite:
            # generate a new crop value
            rc = {'relative':True, 'w':250, 'h':800, 'wc':80, 'hc':400}
            self.crop = vc.relativeCrop(self.pg, self.nd, self.tag, rc)  # get crop position based on the actual line position
            self.crop = vc.convertCropHW(h,w, self.crop)    # make sure everything is in bounds
            self.cl.changeCrop(self.file, self.crop)
            if export:
                self.cl.export()
        
    def cropIm(self):
        self.im = self.nd.subtractBackground(self.im)   # remove the background and the nozzle
        self.im = vc.imcrop(self.im, self.crop)
        
    def segment(self) -> None:
        '''segment the foreground'''
        self.segmenter = segmenter(self.im, acrit=self.acrit, diag=max(0, self.diag-1)
                                   , fillMode=fillMode.fillByContours
                                   , nozData=self.nd, crops=self.crop, adaptive=[1,2]
                                   , addNozzle=False, removeSharp=True
                                   , addNozzleBottom=True, fillTop=True, openBottom=True, grayBlur=self.grayBlur
                                  , closing=self.fillDilation)
        self.segmenter.eraseFullWidthComponents(margin=2*self.fillDilation, checks=False) # remove glue or air
        self.segmenter.eraseLeftRightBorder(margin=1)   # remove components touching the left or right border
        self.segmenter.removeScragglies()  # remove scraggly components
        
    def reconcileImportedSegment(self):
        '''reconcile the difference between the imported segmentation files'''
        if hasattr(self, 'MLsegment'):
            self.MLsegment = self.nd.maskNozzle(self.MLsegment, crops=self.crop, invert=True)
            if hasattr(self, 'Usegment'):
                self.Usegment = self.nd.maskNozzle(self.Usegment, crops=self.crop, invert=True)
                self.segmenter = combineSegments(self.MLsegment, self.Usegment, self.acrit, diag=self.diag-1)
            else:
                self.segmenter = segmenterDF(self.MLsegment, acrit=self.acrit)
        elif hasattr(self, 'Usegment'):
            self.Usegment = self.nd.maskNozzle(self.Usegment, crops=self.crop, invert=True)
            self.segmenter = segmenterDF(self.Usegment, acrit=self.acrit)
            
    def padNozzle(self, left:int=0, right:int=0, bottom:int=0):
        '''expand the nozzle'''
        self.nd.padNozzle(left=left, right=right, bottom=bottom)
    
    def measure(self) -> None:
        '''measure vertical SDT line'''
        self.initialize()
        self.getCrop()
        self.generateIm0()
        
        # get the real nozzle position and pad it
        if not 'o' in self.tag:
            self.nd.adjustEdges(self.im0, self.crop, diag=self.diag-2)  # find the nozzle in the image and use that for future masking 
        self.padNozzle(left=1, right=30, bottom=2)
        
        self.cropIm()
        if self.overrideSegment:
            self.segment()
        else:
            self.importSegmentation()
            self.reconcileImportedSegment()
        
        if not hasattr(self, 'segmenter'):
            self.segment()
        if not self.segmenter.success:
            if self.diag>0:
                logging.warning(f'Segmenter failed on {self.file}')
            self.display()
            return
        self.dims(numLines=self.lnum, largest=False, getLDiff=('o' in self.tag))
        for s in ['y0', 'yc', 'yf']:
            self.stats.pop(s)
            self.units.pop(s)
        self.gaps(self.pv.dEst)
        if len(self.stats)==1:
            if self.diag>0:
                logging.warning(f'Measurement failed on {self.file}')
            self.display()
            return
        self.adjustForCrop(self.stats, self.crop)  # px, full image
        self.stats, self.units = self.makeMM(self.stats, self.units)
        self.makeRelative()           # mm, relative to nozzle
        
        self.findIntendedCoords()
        if self.diag>0:
            self.findIntendedPx()
            self.findNozzlePx()
        self.findDisplacement()  # dEst, relative to intended coords
        self.display()
        
