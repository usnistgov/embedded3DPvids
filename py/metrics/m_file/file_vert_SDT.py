#!/usr/bin/env python
'''Functions for collecting data from a still of a vertical SDT line'''

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
from file_unit import *
from file_vert import *
from file_SDT import *
from im.segment import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)

#----------------------------------------------

def vertSDTMeasure(file:str, **kwargs) -> Tuple[dict, dict]:
    return fileVertSDT(file, **kwargs).values() 

def vertSDTTestFile(fstr:str, fistr:str, **kwargs) -> None:
    '''test a single file and print diagnostics'''
    testFile(fstr, fistr, fileVertSDT, ['emptiness','x0','segments'], **kwargs)
    
def fileVertSDTFromTag(folder:str, tag:str, **kwargs):
    '''get the fileVertSDT from a string that is in the file name'''
    return fileMetricFromTag(fileVertSDT, folder, tag, **kwargs)
        
class fileVertSDT(fileVert, fileSDT):
    '''for singledoubletriple lines'''
    
    def __init__(self, file:str, acrit:int=500, **kwargs):
        self.maxlen = 800
        self.fillDilation = 0
        self.grayBlur = 1
        super().__init__(file, acrit=acrit, **kwargs)
        
    def addToTestFile(self) -> None:
        '''add the current measurements to the csv of intended measurements for XSSDT'''
        csv = testCSV('SDTVert')
        slist = ['emptiness','x0','segments']
        super().addToTestFile(csv, slist)
                
    def findIntendedCoords(self) -> None:
        '''find the intended x0,y0,xc,and yc of the assembly'''
        rc1, rc2, w1, w2, l, lprog = self.intendedRC()  # intended coords and widths of each line in mm
        for j in [['dx', 'w']]:
            coord = j[0][1]
            right = (rc2[j[0]]+w2/2)      # the left edge should be 1/2 diameter to the left of the first line center
            left = (rc1[j[0]]-w1/2)       # the right edge should be 1/2 diameter to the right of the last line center
            self.ideals[f'{coord}0'] = min(left, right)
            self.ideals[f'{coord}f'] = max(left, right)
            self.ideals[j[1]] = abs(right - left)     # get the ideal width
            self.ideals[f'{coord}c'] = (right+left)/2  # get the ideal center
        w = self.ideals['w']   # total width
        r = w/2
        self.ideals['area'] = l*w
        self.ideals['v'] = (l-w)*np.pi*r**2 + 4/3*np.pi*r**3
        self.ideals['h'] = l
        self.ideals['hn'] = lprog
        
    def findDisplacement(self) -> None:
        '''find the displacement of the center and dimensions relative to the intended dimensions'''
        if not 'h' in self.stats:
            return
        self.stats['hn'] = self.stats['h']
        self.units['hn'] = self.units['h']
        for s,ival in {'xc':'xc', 'x0':'x0', 'xf':'xf', 'dxprint':'xc'}.items():
            # ratio of size to intended size
            if s in self.stats:
                self.stats[s] = ((self.stats[s]-self.ideals[ival])/self.pv.dEst)
                self.units[s] = 'dEst'
        for s in ['dxprint', 'dx0', 'dxf', 'space_a', 'space_at', 'h']:
            if s in self.stats:
                self.stats[s] = self.stats[s]/self.pv.dEst
                self.units[s] = 'dEst'
        for s,ival in {'w':'w', 'meanT':'w', 'vest':'v', 'vintegral':'v', 'hn':'hn'}.items():
            # ratio of size to intended size
            if s in self.stats:
                self.stats[s] = (self.stats[s]/self.ideals[ival])
                self.units[s] = 'intended'
        for s in ['area', 'vleak', 'aspect']:
            if s in self.stats:
                self.stats.pop(s)
                
        # remove length measurements for mid-print measurements 
        if not 'o' in self.tag:
            for s in ['h', 'vest', 'vintegral', 'w', 'xf', 'meanT', 'stdevT', 'minmaxT', 'hn']:
                if s in self.stats:
                    self.stats.pop(s)
                    
    def getCrop(self, export:bool=True, overwrite:bool=False):
        '''get the crop position. only export if export=True and there is no existing row'''
        rc = {'relative':True, 'w':270, 'h':1000, 'wc':80, 'hc':500}
        self.makeCrop(rc, export=self.exportCropLocs, overwrite=self.overwriteCropLocs)
        
    def generateSegment(self, overwrite:bool=False):
        '''generate a new segmentation'''
        if not hasattr(self, 'im0'):
            self.generateIm0()
        self.cropIm()
        self.segment()
        self.componentMask = self.segmenter.labelsBW.copy()
        self.exportSegment(overwrite=overwrite)
        
    def segment(self) -> None:
        '''segment the foreground'''
        self.segmenter = segmenter(self.im, acrit=self.acrit, diag=max(0, self.diag-1)
                                   , fillMode=fi.fillMode.fillByContours
                                   , nozData=self.nd, crops=self.crop
                                   , segmentMode=[sMode.kmeans, sMode.adaptive]
                                   , nozMode=nozMode.full, removeSharp=True
                                   , closeTop=True, openBottom=True, grayBlur=self.grayBlur
                                  , closing=self.fillDilation)
        self.segmenter.eraseFullWidthComponents(margin=2*self.fillDilation, checks=False) # remove glue or air
        self.segmenter.eraseLeftRightBorder(margin=2, checks=False)   # remove components touching the left or right border
        #self.segmenter.removeScragglies()  # remove scraggly components
        self.segmenter.eraseBorderClingers(40)

    def measure(self) -> None:
        '''measure vertical SDT line'''
        if self.checkWhite(val=254):
            # white image
            if self.overrideSegment:
                self.getCrop()
                self.cropIm()
                self.im[:,:] = 0
                self.componentMask = self.im
                self.exportSegment(overwrite=False)              # export segmentation
            self.stats['error'] = 'white'
            return
        self.initialize()
        self.getCrop()
        if self.overrideSegment:
             # get the real nozzle position and pad it
            if not 'o' in self.tag:
                self.generateIm0()
                self.nd.adjustEdges(self.im0, self.crop, diag=self.diag-2)  # find the nozzle in the image and use that for future masking
            self.padNozzle(left=1, right=30, bottom=10)
            self.generateSegment(overwrite=True)
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
        self.dims(numLines=self.lnum, largest=False, getLDiff=('o' in self.tag and not ('w1' in self.tag or 'd1' in self.tag)))
        for s in ['y0', 'yc', 'yf']:
            if s in self.stats:
                self.stats.pop(s)
                self.units.pop(s)
        if not 'o' in self.tag:
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
            if not hasattr(self, 'im0'):
                self.generateIm0()
                self.cropIm()
        self.findDisplacement()  # dEst, relative to intended coords
        self.display()
        
