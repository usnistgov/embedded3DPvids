#!/usr/bin/env python
'''Functions for collecting data and summarizing stills of single line horiz for a whole folder'''

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
import im.crop as vc
import im.morph as vm
from im.segment import *
from im.imshow import imshow
from tools.plainIm import *
from val.v_print import *
from vid.noz_detect import nozData
from m_tools import *
from m_folder import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)

#----------------------------------------------




class horizDisturbMeasures(disturbMeasures):
    '''for a horizDisturb folder, measure the disturbed lines'''
    
    def __init__(self, folder:str, overwrite:bool=False, **kwargs) -> None:
        super().__init__(folder, overwrite=overwrite, lineType='horiz', **kwargs)
    
    def measureFolder(self) -> None:
        '''measure all cross-sections in the folder and export table'''
        if not 'disturbHoriz' in os.path.basename(self.folder):
            return 1


        if 'lines' in self.kwargs:
            lines = self.kwargs['lines']
        else:
            lines = [f'l{i}{s}{s2}' for i in range(4) for s in ['w', 'd'] for s2 in ['', 'o']]
        self.measure(lines, horizDisturbMeasure)

    
    def summarize(self) -> Tuple[dict,dict]:
        '''summarize measurements in the folder and export table'''
        errorRet = {},{}
        if not 'disturbHoriz' in os.path.basename(self.folder):
            return errorRet
        
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
    
def horizDisturbMeasureSummarize(topFolder:str, overwrite:bool=False, **kwargs) -> Tuple[dict, dict]:
    return horizDisturbMeasures(topFolder, overwrite=overwrite, **kwargs).summarize()

def horizDisturbSummariesRecursive(topFolder:str, overwrite:bool=False, **kwargs) -> None:
    '''recursively go through folders'''
    s = summaries(topFolder, horizDisturbMeasureSummarize, overwrite=overwrite, **kwargs)
    return s.out, s.units
    
def horizDisturbSummaries(topFolder:str, exportFolder:str, overwrite:bool=False, **kwargs) -> None:
    '''measure all cross-sections in the folder and export table'''
    s = summaries(topFolder, horizDisturbMeasureSummarize, overwrite=overwrite, **kwargs)
    s.export(os.path.join(exportFolder, 'horizDisturbSummaries.csv'))
    