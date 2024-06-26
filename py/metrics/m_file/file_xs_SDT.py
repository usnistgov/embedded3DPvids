#!/usr/bin/env python
'''Functions for collecting data from stills of SDT xs'''

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
from file_xs import *
from file_SDT import *
from im.segment import *


# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)

#----------------------------------------------

def xsSDTMeasure(file:str, **kwargs) -> Tuple[dict, dict]:
    '''given a file name, measure the image and return the measured values'''
    return fileXSSDT(file, **kwargs).values() 

def xsSDTTestFile(fstr:str, fistr:str, **kwargs) -> None:
    '''test a single file and print diagnostics'''
    testFile(fstr, fistr, fileXSSDT, ['w', 'h', 'xc', 'yc'], **kwargs)
    
def fileXSSDTFromTag(folder:str, tag:str, **kwargs):
    '''get the fileVertSDT from a string that is in the file name'''
    return fileMetricFromTag(fileXSSDT, folder, tag, **kwargs)
    
class fileXSSDT(fileXS, fileSDT):
    '''for singledoubletriple lines'''
    
    def __init__(self, file:str, acrit:int=800, **kwargs):
        self.numLines = int(re.split('_', os.path.basename(file))[1])
        super().__init__(file, acrit=acrit, **kwargs)
        
    def addToTestFile(self) -> None:
        '''add the current measurements to the csv of intended measurements for XSSDT'''
        csv = testCSV('SDTXS')
        slist = ['w', 'h', 'xc', 'yc']
        super().addToTestFile(csv, slist)
        
                
    def findIntendedCoords(self) -> None:
        '''find the intended x0,y0,xc,and yc of the assembly. assume that the center of the first filament should be at the center of the nozzle tip'''
        rc1, rc2, w1, w2, _, _ = self.intendedRC(fixList=['y','z'])
        for j in [['dx', 'w'], ['dy', 'h']]:
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
        # self.ideals['area'] = self.progRows.a.max()
        self.ideals['area'] = np.pi*(self.progRows.wmax.max()/2)**2
        
    def findDisplacement(self) -> None:
        '''find the displacement of the center and dimensions relative to the intended dimensions'''
        for s in ['xc', 'yc', 'x0', 'y0', 'xf', 'yf']:
            # shift in position relative to intended, normalized by intended diameter
            if not s in self.ideals:
                raise ValueError(f'Missing value in ideals: {s}, {self.ideals}')
            if not s in self.stats:
                raise ValueError(f'Missing value in stats: {s}, {self.stats}')
            self.stats[s] = (self.stats[s]-self.ideals[s])/self.pv.dEst
            self.units[s] = 'dEst'
        for s in ['w', 'h']:
            # ratio of size to intended size
            self.stats[s] = (self.stats[s]/self.pv.pxpmm/self.ideals[s])
            self.units[s] = 'intended'
        self.stats['area'] = self.stats['area']/self.pv.pxpmm**2/self.ideals['area']
        self.units['area'] = 'intended'
        self.stats['aspectI'] = self.stats['h']/self.stats['w']   # ratio of aspect ratio to intended aspect ratio
        self.units['aspectI'] = 'intended'

        
    def getCrop(self, export:bool=True, overwrite:bool=False):
        '''get the crop position. only export if export=True and there is no existing row'''
        if '_1_' in os.path.basename(self.folder):
            rc = {'relative':True, 'w':200, 'h':250, 'wc':100, 'hc':180}
        else:
            if 'o' in self.tag:
                rc = {'relative':True, 'w':300, 'h':500, 'wc':100, 'hc':400}
            else:
                rc = {'relative':True, 'w':300, 'h':350, 'wc':100, 'hc':200}
        self.makeCrop(rc, export=self.exportCropLocs, overwrite=self.overwriteCropLocs)
                
    def segment(self) -> None:
        '''segment the foreground'''
        if 'water' in self.pv.ink.base:
            th = 140
        else:
            th = 130
        self.segmenter = segmenter(self.im, acrit=self.acrit, diag=max(0, self.diag-1)
                                   , topthresh=th
                                   , fillMode=fi.fillMode.removeBorder
                                   , nozData=self.nd, crops=self.crop
                                   , nozMode=nozMode.full
                                   , segmentMode=[sMode.threshold]
                                   , trimNozzle=(not 'o' in self.tag)
                                   , removeSharp=False
                                   )
        self.segmenter.eraseBorderComponents(10)
        self.segmenter.selectCloseObjects(self.idealspx)  # remove bubbles and debris that are far from the main object
        
    def generateSegment(self):
        '''generate a new segmentation'''
        if not hasattr(self, 'im0'):
            self.generateIm0()
        if not 'o' in self.tag:
            self.nd.adjustEdges(self.im0, self.crop, diag=self.diag-2, ymargin=3)  # find the nozzle in the image and use that for future masking
        self.padNozzle(left=1, right=10, bottom=2)       # cover the nozzle, with some extra wiggle room
        self.cropIm()
        self.segment()
        if self.segmenter.success:
            self.componentMask = self.segmenter.labelsBW.copy()
        else:
            self.componentMask = self.segmenter.filled.copy()
            self.componentMask[:,:] = 0
        self.exportSegment(overwrite=self.overrideSegment)
    
    def measure(self) -> None:
        '''measure cross-section of single disturbed line'''
        if self.checkWhite(val=254):
            # white image
            self.stats['error'] = 'white'
            return
        self.initialize()
        self.getCrop()
        if self.diag>0:
            self.findNozzlePx()
                
        if not self.overrideSegment:
            self.importSegmentation()
        self.findIntendedCoords()                        # find where the object should be
        self.findIntendedPx()
        if not hasattr(self, 'segmenter'):
            self.generateSegment()                                   # segment the image
        
        if not self.segmenter.success:
            if self.diag>0:
                logging.warning(f'Segmenter failed on {self.file}')
            self.display()
            return
        self.multiMeasure()      # px, cropped
        if len(self.stats)==1:
            if self.diag>0:
                logging.warning(f'Measurement failed on {self.file}')
            self.display()
            return
        self.adjustForCrop(self.stats, self.crop)   # px, full image
        self.makeRelative()     # mm, relative to nozzle
        
        self.findDisplacement()  # dEst, relative to intended coords
        self.renameY()
        self.display()
        