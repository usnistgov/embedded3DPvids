#!/usr/bin/env python
'''Functions for collecting data and summarizing stills of single line disturbed horiz for a whole folder'''

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
from folder_disturb import *
from m_file.file_horiz_disturb import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)

#----------------------------------------------

class folderHorizDisturb(folderDisturb):
    '''for a horizDisturb folder, measure the disturbed lines
        export a table of values (Measure)
        export a list of failed files (Failures)
        export a row of summary values (Summary)'''
    
    def __init__(self, folder:str, **kwargs) -> None:
        super().__init__(folder, **kwargs)
        if not 'disturbHoriz' in os.path.basename(self.folder):
            raise ValueError(f'Wrong folderDisturb class called for {self.folder}')
    
    def measureFolder(self) -> None:
        '''measure all cross-sections in the folder and export table'''
        self.measure(fileHorizDisturb)

    
    def summarize(self) -> Tuple[dict,dict]:
        '''summarize measurements in the folder and export table'''
        errorRet = {},{}
        r = self.summaryHeader()
        if r==0:
            return self.summary, self.summaryUnits
        elif r==2:
            return errorRet

        # find changes between observations
        aves = {}
        aveunits = {}
        for num in range(4):
            wodf = self.df[self.df.line==f'HOh_l{num}wo']
            dodf = self.df[self.df.line==f'HOh_l{num}do']
            if len(wodf)==1 and len(dodf)==1:
                wo = wodf.iloc[0]
                do = dodf.iloc[0]
                for s in ['segments', 'roughness']:
                    try:
                        addValue(aves, aveunits,f'delta_{s}', difference(do,wo,s), self.du[s])
                    except:
                        pass
                for s in ['totlen', 'meanT']:
                    try:
                        addValue(aves, aveunits,f'delta_{s}_n', difference(do,wo,s)/wo[s], '')
                    except:
                        pass
                for s in ['yc']:
                    try:
                        addValue(aves, aveunits, f'delta_{s}_n', difference(do, wo, s)/self.pxpmm/self.pv.dEst, 'dEst')
                    except ValueError:
                        pass

        # find displacements
        disps = {}
        dispunits = {}
        dlist = ['dy0l', 'dy0r', 'dy0lr', 'space_b']
        for num in range(4):
            wdf = self.df[self.df.line==f'HOh_l{num}w']
            ddf = self.df[self.df.line==f'HOh_l{num}d']
            for s in dlist:
                for vdf in [wdf,ddf]:
                    if len(vdf)>0:
                        v = vdf.iloc[0]
                        if hasattr(v, s):
                            sii = str(v.line)[-1]
                            si = f'{sii}_{s}'
                            if not si in ['w_dy0r', 'w_dy0lr', 'w_space_b']:
                                val = v[s]/self.pxpmm/self.pv.dEst
                                addValue(disps, dispunits, si, val, 'dEst')

        ucombine = {**aveunits, **dispunits} 
        lists = {**aves, **disps}
        self.convertValuesAndExport(ucombine, lists)
        return self.summary, self.summaryUnits
