#!/usr/bin/env python
'''Functions for collecting measurements from a single image of a disturbed line'''

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
import time

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
sys.path.append(os.path.dirname(os.path.dirname(currentdir)))
from file_metric import *
from crop_locs import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 4)
pd.set_option('display.max_rows', 500)


#----------------------------------------------

class fileDisturb(fileMetric):
    '''collect measurements of segments in disturbed prints'''
    
    def __init__(self, file:str, diag:int=0, acrit:int=2500, measure:bool=True, **kwargs):
        super().__init__(file, diag=diag, acrit=acrit, **kwargs)
        self.scale = 1
        if 'nd' in kwargs:
            self.nd = kwargs['nd']
        else:
            self.nd = nozData(os.path.dirname(file))   # detect nozzle
        self.pfd = self.nd.pfd
        if 'pv' in kwargs:
            self.pv =  kwargs['pv']
        else:
            self.pv = printVals(os.path.dirname(file), pfd = self.pfd, fluidProperties=False)
        if 'pg' in kwargs:
            self.pg = kwargs['pg']
        else:
            self.getProgDims()
        if 'cl' in kwargs:
            self.cl = kwargs['cl']
        else:
            self.getCropLocs()
        self.title = os.path.basename(self.file)
        self.lineName()
        if measure:
            self.measure()
        
        
    def lineName(self) -> None:
        '''get the name of a singleDisturb, or tripleLine line'''
        if not 'vstill' in self.file:
            raise ValueError(f'Cannot determine line name for {self.file}')
        spl = re.split('_', re.split('vstill_', os.path.basename(self.file))[1])
        self.name = f'{spl[0]}_{spl[1]}'  # this is the full name of the pic, e.g. HOx2_l3w2o1
        self.tag = spl[1]                 # this is the name of the pic, e.g. l1wo
        self.gname = self.tag[:2]     # group name, e.g. l3
        self.ltype = self.tag[2:4]
        self.printName = self.tag[:4]
        self.picName = self.tag[-2:]
        
        
    def getProgRow(self):
        '''get the progDims rows for this line, where it's all the written lines in this group'''
        progRows = pd.concat([self.pg.progLine(i+1, self.gname) for i in range(self.lnum)])
        self.progRows = progRows
