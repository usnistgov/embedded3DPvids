#!/usr/bin/env python
'''Functions for collecting data from stills of disturbed single lines in cross-section view'''

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
from file_xs import *
from file_disturb import *
from file_unit import *


# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)

#----------------------------------------------

def xsDisturbMeasure(file:str, **kwargs) -> Tuple[dict, dict]:
    '''given a file name, measure the image and return the measured values'''
    return fileXSDisturb(file, **kwargs).values()  

def xsDisturbTestFile(fstr:str, fistr:str, **kwargs) -> None:
    '''test a single file and print diagnostics'''
    testFile(fstr, fistr, fileXSDisturb, ['w', 'h', 'xc', 'yc'], **kwargs)

class fileXSDisturb(fileXS, fileDisturb):
    '''for disturbed lines'''
    
    def __init__(self, file:str, diag:int=0, acrit:int=100, **kwargs):
        super().__init__(file, diag=diag, acrit=acrit, **kwargs)
        self.pv = printVals(os.path.dirname(file))
        self.lineName()
        self.measure()
    
    def measure(self) -> None:
        '''measure cross-section of single disturbed line'''
        
        self.scale = 1
        self.title = os.path.basename(self.file)
        self.im = vm.normalize(self.im)

        # segment components
        h,w,_ = self.im.shape
        hc = 150
        crop = {'y0':hc, 'yf':h-hc, 'x0':170, 'xf':300}
        self.im = vc.imcrop(self.im, crop)

        if 'water' in self.pv.ink.base:
            th = 140
        else:
            th = 80
        self.segmenter = segmenter(self.im, acrit=self.acrit, topthresh=th, diag=max(0, self.diag-1))
        if not self.segmenter.success:
            return
        self.singleMeasure()
        self.adjustForCrop(self.stats, crop)
        