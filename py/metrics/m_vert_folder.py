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
import im.crop as vc
import im.morph as vm
from im.segment import *
from im.imshow import imshow
from tools.plainIm import *
from val.v_print import *
from vid.noz_detect import nozData
from m_tools import *
from m_folder import *
from m_vert_file import vertSDTMeasure, vertSDTTestFile

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)

#----------------------------------------------


class vertDisturbMeasures(disturbMeasures):
    '''for a vertDisturb measure, measure the disturbed lines'''
    
    def __init__(self, folder:str, overwrite:bool=False, **kwargs) -> None:
        super().__init__(folder, overwrite=overwrite, **kwargs)
    
    def measureFolder(self) -> None:
        '''measure all cross-sections in the folder and export table'''
        if not 'disturbVert' in os.path.basename(self.folder):
            return 1
        self.fn = self.pfd.newFileName('vertMeasure', '.csv')

        if 'lines' in self.kwargs:
            lines = self.kwargs['lines']
        else:
            lines = [f'l{i}{s}{s2}' for i in range(4) for s in ['w', 'd'] for s2 in ['', 'o']]
        self.measure(lines, vertDisturbMeasure)


    def summarize(self) -> Tuple[dict,dict]:
        '''summarize vertical measurements in the folder and export table'''
        errorRet = {},{}
        if not 'disturbVert' in os.path.basename(self.folder):
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
            if num in [0,2]:
                ltype = 'bot'
            else:
                ltype = 'top'
            wodf = self.df[self.df.line==f'V_l{num}wo']
            dodf = self.df[self.df.line==f'V_l{num}do']
            if len(wodf)==1 and len(dodf)==1:
                wo = wodf.iloc[0]
                do = dodf.iloc[0]

                for s in ['segments', 'roughness']:
                    try:
                        addValue(aves, aveunits, f'{ltype}_delta_{s}', difference(do, wo, s), self.du[s])
                    except ValueError:
                        pass
                for s in ['h', 'meanT']:
                    try:
                        addValue(aves, aveunits, f'{ltype}_delta_{s}_n', difference(do, wo, s)/wo[s], '')
                    except ValueError:
                        pass
                for s in ['xc']:
                    try:
                        addValue(aves, aveunits, f'{ltype}_delta_{s}_n', difference(do, wo, s)/self.pxpmm/self.pv.dEst, 'dEst')
                    except ValueError:
                        pass

        # find displacements
        disps = {}
        dispunits = {}
        dlist = ['dxprint', 'dxf', 'space_at', 'space_a']
        for num in range(4):
            wdf = self.df[self.df.line==f'V_l{num}w']
            ddf = self.df[self.df.line==f'V_l{num}d']
            if num in [0,2]:
                ltype = 'bot'
            else:
                ltype = 'top'
            for s in dlist:
                for vdf in [wdf,ddf]:
                    if len(vdf)>0:
                        v = vdf.iloc[0]
                        if hasattr(v, s):
                            sii = str(v.line)[-1]
                            si = f'{sii}_{s}'
                            sifull = f'{ltype}_{si}'
                            if si not in ['w_dxf', 'w_space_a', 'w_space_at']:
                                val = v[s]/self.pxpmm/self.pv.dEst
                                addValue(disps, dispunits, sifull, val, 'dEst')

        ucombine = {**aveunits, **dispunits} 
        lists = {**aves, **disps}
        self.convertValuesAndExport(ucombine, lists)
        return self.summary, self.summaryUnits
    
def vertDisturbMeasureSummarize(topFolder:str, overwrite:bool=False, **kwargs) -> Tuple[dict, dict]:
    return vertDisturbMeasures(topFolder, overwrite=overwrite, **kwargs).summarize()

def vertDisturbSummariesRecursive(topFolder:str, overwrite:bool=False, **kwargs) -> None:
    '''recursively go through folders'''
    s = summaries(topFolder, vertDisturbMeasureSummarize, overwrite=overwrite, **kwargs)
    return s.out, s.units
    
def vertDisturbSummaries(topFolder:str, exportFolder:str, overwrite:bool=False, **kwargs) -> None:
    '''measure all cross-sections in the folder and export table'''
    s = summaries(topFolder, vertDisturbMeasureSummarize, overwrite=overwrite, **kwargs)
    s.export(os.path.join(exportFolder, 'vertDisturbSummaries.csv'))
    