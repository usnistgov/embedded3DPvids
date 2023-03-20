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
from m_xs_file import xsSDTMeasure, xsSDTTestFile

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)

#----------------------------------------------

class xsDisturbMeasures(disturbMeasures):
    '''for a xsDisturb folder, measure the disturbed lines'''
    
    def __init__(self, folder:str, overwrite:bool=False, **kwargs) -> None:
        if '+y' in folder:
            self.dire = '+y'
        elif '+z' in folder:
            self.dire = '+z'
        else:
            raise ValueError(f'Could not find direction for {folder}')
        super().__init__(folder, overwrite=overwrite, lineType=f'xs{self.dire}', **kwargs)

    def measureFolder(self) -> None:
        '''measure all cross-sections in the folder and export table'''
        if not 'disturbXS' in os.path.basename(self.folder):
            return
        if 'lines' in self.kwargs:
            lines = self.kwargs['lines']
        else:
            lines = [f'l{i}{s}{s2}' for i in range(4) for s in ['w', 'd'] for s2 in ['o']]
        self.measure(lines, xsDisturbMeasure)


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
    
    
def xsDisturbMeasureSummarize(topFolder:str, overwrite:bool=False, **kwargs) -> Tuple[dict, dict]:
    return xsDisturbMeasures(topFolder, overwrite=overwrite, **kwargs).summarize()

def xsDisturbSummariesRecursive(topFolder:str, dire:str, overwrite:bool=False, **kwargs) -> None:
    '''recursively go through folders'''
    s = summaries(topFolder, xsDisturbMeasureSummarize, overwrite=overwrite, mustMatch=['xs', dire], **kwargs)
    return s.out, s.units
    
def xsDisturbSummaries(topFolder:str, exportFolder:str, overwrite:bool=False, **kwargs) -> None:
    '''measure all cross-sections in the folder and export table'''
    for dire in ['+y', '+z']:
        s = summaries(topFolder, xsDisturbMeasureSummarize, overwrite=overwrite, mustMatch=['xs', dire], **kwargs)
        s.export(os.path.join(exportFolder, f'xs{dire}DisturbSummaries.csv'))
        
#-----------------------------------------------------------------------------------        
        
        
class xsSDTMeasures(SDTMeasures):
    '''for a xsSDT folder, measure the disturbed lines'''
    
    def __init__(self, folder:str, overwrite:bool=False, **kwargs) -> None:
        super().__init__(folder, overwrite=overwrite, **kwargs)
        if '+y' in folder:
            self.dire = '+y'
        elif '+z' in folder:
            self.dire = '+z'
        else:
            raise ValueError(f'Could not find direction for {folder}')
        super().__init__(folder, overwrite=overwrite, lineType=f'xs_{self.dire}',  **kwargs)
        
    def measureFolder(self) -> None:
        '''measure all cross-sections in the folder and export table'''
        self.measure(self.lines, xsSDTMeasure)


    #-----------------------------------------------------------------
    # summaries

    def summarize(self, **kwargs) -> Tuple[dict,dict]:
        '''summarize xsical measurements in the folder and export table'''
        r = self.summaryHeader()
        if r==0:
            return self.summary, self.summaryUnits, self.failures
        elif r==2:
            return pd.DataFrame([]),{}, pd.DataFrame([])
  
        depvars = self.depvars()
        writedepvars = ['yBot', 'xLeft']
        tunits = self.pg.progDimsUnits['tpic']
        
        # find changes between observations
        aves = {}
        aveunits = {}
        for pair in self.pairs():
            p1 = pair[0]
            p2 = pair[1]
            title = pair[2]
            for num in range(4):
                wo = self.dfline(f'l{num}{p1}')  # get measurements of images
                do = self.dfline(f'l{num}{p2}')  
                time = self.pairTime([f'l{num}{p1}', f'l{num}{p2}'])  # get time between images
                if len(wo)>0 and len(do)>0:
                    for s in depvars:
                        try:
                            u = self.du[s]
                            self.addValue(aves, aveunits, f'delta_{s}_{title}', difference(do, wo, s)/time, f'{u}/{tunits}')
                        except ValueError:
                            pass
        for single in self.singles():
            lines = self.dflines(single)
            if 'o' in single:
                dv = depvars
            else:
                dv = writedepvars
            for var in dv:
                self.addValues(aves, aveunits, f'{var}_{single}', list(lines[var]), self.du[var])

        self.convertValuesAndExport(aveunits, aves)
        
        if self.diag>0:
            self.printAll()

        return self.summary, self.summaryUnits, self.failures
    
    
def xsSDTMeasureSummarize(topFolder:str, overwrite:bool=False, **kwargs) -> Tuple[dict, dict]:
    return xsSDTMeasures(topFolder, overwrite=overwrite, **kwargs).summarize(**kwargs)

def xsSDTSummariesRecursive(topFolder:str, dire:str, overwrite:bool=False, **kwargs) -> None:
    '''recursively go through folders'''
    s = summaries(topFolder, xsSDTMeasureSummarize, overwrite=overwrite, mustMatch=['XS', dire], **kwargs)
    return s.out, s.units

def xsSDTFailureFile(dire:str):
    exportFolder = os.path.join(cfg.path.fig, 'singleDoubleTriple')
    return os.path.join(exportFolder, f'xs{dire}SDTFailures.csv')
    
def xsSDTSummaries(topFolder:str, exportFolder:str, overwrite:bool=False, **kwargs) -> None:
    '''measure all cross-sections in the folder and export table'''
    for dire in ['+y', '+z']:
        s = summaries(topFolder, xsSDTMeasureSummarize, overwrite=overwrite, mustMatch=['XS', dire], **kwargs)
        s.export(os.path.join(exportFolder, f'xs{dire}SDTSummaries.csv'))
        s.exportFailures(xsSDTFailureFile(dire))
    
class xsSDTFailureTest(failureTest):
    
    def __init__(self, dire):
        super().__init__(xsSDTFailureFile(dire), xsSDTTestFile)
        self.dire = dire
