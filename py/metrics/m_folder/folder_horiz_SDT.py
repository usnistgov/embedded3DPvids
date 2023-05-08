#!/usr/bin/env python
'''Functions for collecting data and summarizing stills of single line xs for a whole folder'''

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
from folder_SDT import *
from m_file.file_horiz_SDT import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)

#----------------------------------------------

class folderHorizSDT(folderSDT):
    '''for a horizSDT folder, measure the disturbed lines'''
    
    def __init__(self, folder:str, **kwargs) -> None:
        super().__init__(folder, **kwargs)
        if not 'disturbHoriz' in os.path.basename(self.folder):
            raise ValueError(f'Wrong folderSDT class called for {self.folder}')
        
    def measureFolder(self) -> None:
        '''measure all cross-sections in the folder and export table'''
        self.measure(fileHorizSDT)

    #-----------------------------------------------------------------
    # summaries

    def summarize(self, **kwargs) -> Tuple[dict,dict]:
        '''summarize xsical measurements in the folder and export table'''
        r = self.summaryHeader()
        if r==0:
            return self.summary, self.summaryUnits, self.failures
        elif r==2:
            return pd.DataFrame([]), {}, pd.DataFrame([])
  
        # dependent variables for observe images. use different measurements for w1 and all other lines
        ovars = ['segments', 'yBot', 'yTop', 'w', 'h', 'yc', 'roughness', 'emptiness', 'meanT', 'stdevT', 'minmaxT', 'ldiff']
        ovars1 = ['segments', 'yBot', 'yTop', 'w', 'h', 'yc', 'roughness', 'emptiness', 'meanT', 'stdevT', 'minmaxT']   # for the 1st line
        opairvars = ['segments', 'yBot', 'yTop', 'w', 'h', 'yc', 'roughness', 'emptiness', 'meanT', 'stdevT', 'minmaxT', 'ldiff']     # for pairs of observe lines
        # dependent variables for progress images. use different measurements for w1 and all other lines
        pvars = ['yBot', 'segments', 'roughness', 'emptiness', 'dy0l', 'dy0lr', 'dyfl', 'dyflr', 'space_l', 'space_b']
        pvars1 = ['yBot', 'dy0l', 'dyfl', 'dyflr']
        ppairvars = ['yBot']   # for pairs of progress lines
        tunits = self.pg.progDimsUnits['tpic']
        
        # average observed values
        for single in self.singles():
            if 'o' in single:
                if '1' in single:
                    dv = ovars1
                else:
                    dv = ovars
            else:
                if '1' in single and not 'd' in single:
                    dv = pvars1
                else:
                    dv = pvars
            self.addSingle(single, dv)     
            
            # get slopes
            if 'o' in single:
                for var in dv:
                    self.addSlopes(single, var, tunits, rcrit=0)
                    
        # changes between pairs
        for pair in self.pairs():
            if 'p' in pair[0] or 'p' in pair[1]:
                dv = ppairvars
            else:
                dv = opairvars
            self.addPair(dv, pair)                 

        self.convertValuesAndExport()
        
        if self.diag>0:
            self.printAll()

        return self.summary, self.summaryUnits, self.failures
