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
from file_unit import *
from file_xs import *
from file_SDT import *


# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)



#----------------------------------------------

def xsSDTMeasure(file:str, **kwargs) -> Tuple[dict, dict]:
    return fileXSSDT(file, **kwargs).values() 

def xsSDTTestFile(fstr:str, fistr:str, **kwargs) -> None:
    '''test a single file and print diagnostics'''
    testFile(fstr, fistr, fileXSSDT, ['w', 'h', 'xc', 'yc'], **kwargs)
    
class fileXSSDT(fileXS, fileSDT):
    '''for singledoubletriple lines'''
    
    def __init__(self, file:str, diag:int=0, acrit:int=100, **kwargs):
        self.numLines = int(re.split('_', os.path.basename(file))[1])
        super().__init__(file, diag=diag, acrit=acrit, **kwargs)
        
                
    def findIntendedCoords(self) -> None:
        '''find the intended x0,y0,xc,and yc of the assembly. assume that the center of the first filament should be at the center of the nozzle tip'''
        rc1, rc2, w1, w2, _ = self.intendedRC(fixList=['y','z'])
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
        self.ideals['area'] = self.progRows.a.sum()
        
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

    
    def measure(self) -> None:
        '''measure cross-section of single disturbed line'''
        # imshow(self.im)
        dilation = 4
        self.im = self.nd.subtractBackground(self.im, dilation)   # remove the background and the nozzle
        self.getProgDims()
        self.getProgRow()
        if '_1_' in os.path.basename(self.folder):
            rc = {'relative':True, 'w':200, 'h':250, 'wc':100, 'hc':180}
        else:
            if 'o' in self.tag:
                rc = {'relative':True, 'w':300, 'h':500, 'wc':100, 'hc':400}
            else:
                rc = {'relative':True, 'w':300, 'h':350, 'wc':100, 'hc':200}
        self.crop = vc.relativeCrop(self.pg, self.nd, self.tag, rc, fixList=['y', 'z'])  # get crop position based on the actual line position
        self.crop = vc.convertCrop(self.im, self.crop)    # make sure everything is in bounds
        if self.diag>0:
            self.findNozzlePx()

        self.im = vc.imcrop(self.im, self.crop)
        if 'water' in self.pv.ink.base:
            th = 140
        else:
            th = 110
        self.segmenter = segmenter(self.im, acrit=self.acrit, topthresh=th, nozData=self.nd, crops=self.crop, dilation=dilation, diag=max(0, self.diag-1))
        self.segmenter.eraseBorderComponents(10)
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
        self.findIntendedCoords() 
        if self.diag>0:
            self.findIntendedPx()
        self.findDisplacement()  # dEst, relative to intended coords
        self.renameY()
        self.display()
        