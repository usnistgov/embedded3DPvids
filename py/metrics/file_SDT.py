#!/usr/bin/env python
'''Functions for collecting data from stills of single lines, for a single image'''

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
from file_metric import *
from file_disturb import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 4)
pd.set_option('display.max_rows', 500)


#----------------------------------------------

class fileSDT(fileDisturb):
    '''singleDoubleTriple single files'''
    
    def __init__(self, file:str, diag:int=0, acrit:int=2500,  **kwargs):
        super().__init__(file, diag=diag, acrit=acrit, **kwargs)
        
    def initialize(self):
        self.getProgRow() 
        self.getProgTime()
        
    def getProgTime(self):
        '''get the time of this picture relative to the time when the endpoint was written'''
        actPoint = self.pg.progDims[(self.pg.progDims.name.str.contains(self.printName))&(self.pg.progDims.name.str.contains('p'))].iloc[0]
        thisPoint = self.pg.progDims[(self.pg.progDims.name==self.tag)].iloc[-1]
        writePoint = self.progRows.iloc[-1]
        self.stats['time'] = thisPoint['tpic']-actPoint['tf']   # time since the last action ended
        self.units['time'] = self.pg.progDimsUnits['tpic']
        self.stats['wtime'] = thisPoint['tpic']-writePoint['tf']  # time since the last write ended
        self.units['wtime'] = self.pg.progDimsUnits['tpic']
        self.stats['zdepth'] = thisPoint['zpic']
        self.units['zdepth'] = self.pg.progDimsUnits['zpic']
    
    def lineName(self) -> None:
        '''get the name of a singleDisturb, or tripleLine line'''
        if not 'vstill' in self.file:
            raise ValueError(f'Cannot determine line name for {self.file}')
        self.numLines = int(re.split('_', os.path.basename(self.file))[1])
        spl = re.split('_', re.split('vstill_', os.path.basename(self.file))[1])
        self.name = f'{spl[0]}_{spl[1]}'  # this is the full name of the pic, e.g. HOx2_l3w2o1
        lt = re.split('o', re.split('_', self.name)[1][2:])[0]
        if lt=='d' or lt=='w':
            # get the last line
            self.lnum = self.numLines
        else:
            self.lnum = int(lt[1])
        self.tag = spl[1]                 # this is the name of the pic, e.g. l3w2o1
        self.gname = self.tag[:2]     # group name, e.g. l3
        self.ltype = self.tag[2:4]
        self.printName = self.tag[:4]
        self.picName = self.tag[-2:]
        self.stats['gname'] = self.gname
        self.stats['ltype'] = self.ltype
        self.stats['pr'] = self.printName
        self.stats['pname'] = self.picName
        self.units['gname'] = ''
        self.units['ltype'] = ''
        self.units['pr'] = ''
        self.units['pname'] = ''
        
    

    
        
        
    def addToTraining(self, trainFolder:str=r'singleDoubleTriple\trainingVert', s:str='componentMask', openPaint:bool=False):
        '''add the original image and the segmented image to the training dataset'''
        folder = os.path.join(cfg.path.fig, trainFolder)
        fnorig = os.path.join(folder, 'orig', self.title)
        cv.imwrite(fnorig, self.im0)
        fnseg = os.path.join(folder, 'segmented', self.title)
        if s=='thresh':
            img = self.segmenter.thresh
            ccl = segmenterDF(img)   # remove little segments
            ccl.eraseFullWidthComponents()
            img = ccl.labelsBW
        else:
            img = getattr(self, s)
        cv.imwrite(fnseg, img)
        if s=='thresh' or openPaint:
            openInPaint(fnseg)
        print(f'Exported {self.title} to training data')
        

        