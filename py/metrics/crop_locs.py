#!/usr/bin/env python
'''Functions for collecting data from stills of single lines, for a single image'''

# external packages
import os, sys
import traceback
import logging
import pandas as pd
from typing import List, Dict, Tuple, Union, Any, TextIO
import re
import numpy as np
import cv2 as cv
import shutil
import time

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
import file.file_handling as fh
from im.imshow import imshow
from tools.plainIm import *
from val.v_print import printVals
from progDim.prog_dim import getProgDims, getProgDimsPV
from vid.noz_detect import nozData
from m_tools import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 4)
pd.set_option('display.max_rows', 500)


#----------------------------------------------


class cropLocs:
    '''a class for holding the locations of cropped images. this is useful for making sure that all cropped images are cropped to the same region'''
    
    def __init__(self, folder:str, overwrite:bool=False, **kwargs):
        self.folder = folder
        if 'pfd' in kwargs:
            self.pfd = kwargs['pfd']
        else:
            self.pfd = fh.printFileDict(folder)
        self.fn = self.pfd.newFileName('cropLocs', '.csv')
        if os.path.exists(self.fn) and not overwrite:
            self.df,_ = plainIm(self.fn, ic=0)
        else:
            if len(self.pfd.vstill)==0:
                self.pfd.findVstill()
            self.df = pd.DataFrame({'vstill':self.pfd.vstill})
        self.units = {'vstill':'', 'x0':'px', 'xf':'px', 'y0':'px', 'yf':'px'}
            
    def getCrop(self, file:str) -> dict:
        '''get the crop dimensions from the file'''
        row = self.df[self.df.vstill==file]
        if len(row)==0:
            raise ValueError(f'Cannot find {file} in cropLocs')
        d = dict(row.iloc[0])
        d.pop('vstill')
        for key,val in d.items():
            if pd.isna(val):
                return {}
            else:
                d[key] = int(val)
        return d
    
    def sameCrop(self, file:str, d2:dict) -> bool:
        '''check if the crop values are the same'''
        d = self.getCrop(file)
        for key,val in d.items():
            if not d2[key]==val:
                return False
        return True
    
    def changeCrop(self, file:str, crop:dict) -> None:
        '''change the value of the crop in the table'''
        row = self.df[self.df.vstill==file]
        if len(row)==0:
            raise ValueError(f'Cannot find {file} in cropLocs')
        i = (row.iloc[0]).name
        for key,val in crop.items():
            self.df.loc[i,key] = val
            
    def export(self, overwrite:bool=False):
        '''export the values to file'''
        if os.path.exists(self.fn) and not overwrite:
            return
        plainExp(self.fn, self.df, self.units)
