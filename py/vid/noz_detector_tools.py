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
from noz_dims import *
from background import *
from noz_plots import nozPlotter


# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)


#----------------------------------------------

class nozDetector:
    '''for detecting the nozzle in an image'''
    
    def __init__(self, fs, pfd, printFolder:str, **kwargs):
        self.failed = False
        self.fs = fs   # frame selector
        self.np = nozPlotter(printFolder)
        self.crops = {'x0':0, 'y0':0}
        self.pfd = pfd
        self.pxpmm = self.pfd.pxpmm()
        self.printFolder = printFolder
        self.defineHoughParams(**kwargs)
        self.defineCritVals()
        self.defineDimensions()
        
    def defineDimensions(self):
        '''bounds of size of nozzle in mm. for 20 gauge nozzle, diam should be 0.908 mm'''
        self.nod = 0.908 # mm
        if len(self.pfd.meta)>0:
            meta,u = plainImDict(self.pfd.meta[0], unitCol=1, valCol=2)
            if 'nozzle_outer_diameter' in meta:
                self.nod = float(meta['nozzle_outer_diameter'])

        self.nozwidthMin = self.nod-0.15 # mm
        self.nozWidthMax = self.nod+0.3 # mm


    def detectNozzle0(self, frame:np.array, diag:int=0, suppressSuccess:bool=False, overwrite:bool=False, export:bool=True, **kwargs) -> None:
        '''find the bottom corners of the nozzle. suppressSuccess=True to only print diagnostics if the run fails'''
        self.thresholdNozzle(frame)    # threshold the nozzle
        self.nozzleLines()            # edge detect and Hough transform to get nozzle edges as lines
        self.findNozzlePoints(**kwargs)       # filter the lines to get nozzle coords
        self.checkNozzleValues()      # make sure coords make sense
        if export:
            self.nd.exportNozzleDims(overwrite=overwrite)       # save coords  
        if diag>0 and not suppressSuccess:
            self.np.drawDiagnostics(diag) # show diagnostics


    def detectNozzle(self, diag:int=0, **kwargs) -> None:
        '''find the bottom corners of the nozzle, trying different images. suppressSuccess=True to only print diagnostics if the run fails'''
        if 'modes' in kwargs:
            modes = kwargs['modes']
        else:
            if 'Horiz' in self.printFolder:
                modes = [4,1]
            else:
                modes = [1]
        
        if 'frameGetMode' in kwargs and kwargs['frameGetMode']==frameGetModes.snap:
            loops = 1
        else:
            loops = 3
        
        for mode in modes: # min, median, then mean
            for i in range(loops):
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
        raise ValueError(f'Failed to detect nozzle after {loops*len(modes)} iterations')
        