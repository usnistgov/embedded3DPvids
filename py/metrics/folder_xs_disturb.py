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
from folder_disturb import *
from file_xs_disturb import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)

#----------------------------------------------

class folderXSDisturb(folderDisturb):
    '''for a xsDisturb folder, measure the disturbed lines'''
    
    def __init__(self, folder:str, overwrite:bool=False, **kwargs) -> None:
        super().__init__(folder, overwrite=overwrite, **kwargs)
        if not 'disturbXS' in os.path.basename(self.folder):
            raise ValueError(f'Wrong folderDisturb class called for {self.folder}')

    def measureFolder(self) -> None:
        '''measure all cross-sections in the folder and export table'''
        self.measure(fileXSDisturb)

    #-----------------------------------------------------------------
    # summaries

    def summarize(self, **kwargs) -> Tuple[dict,dict]:
        '''summarize xsical measurements in the folder and export table'''
        r = self.summaryHeader()
        if r==0:
            return self.summary, self.summaryUnits
        elif r==2:
            return {}, {}
        
        # find changes between observations
        aves = {}
        aveunits = {}
        for num in range(4):
            wodf = self.df[self.df.line.str.contains(f'l{num}wo')]
            dodf = self.df[self.df.line.str.contains(f'l{num}do')]
            if len(wodf)==1 and len(dodf)==1:
                wo = wodf.iloc[0]
                do = dodf.iloc[0]
                for s in ['aspect', 'yshift', 'xshift']:
                    try:
                        addValue(aves, aveunits, f'delta_{s}', difference(do, wo, s), self.du[s])
                    except ValueError:
                        pass
                for s in ['h', 'w']:
                    try:
                        addValue(aves, aveunits, f'delta_{s}_n', difference(do, wo, s)/wo[s], '')
                    except ValueError:
                        pass
                for s in ['xc']:
                    try:
                        addValue(aves, aveunits, f'delta_{s}_n', difference(do, wo, s)/self.pxpmm/self.pv.dEst, 'dEst')
                    except ValueError:
                        pass


        ucombine = aveunits 
        lists = aves
        self.convertValuesAndExport(ucombine, lists)
        return self.summary, self.summaryUnits