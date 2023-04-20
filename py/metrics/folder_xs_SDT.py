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
from file_xs_SDT import *
from folder_SDT import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)

#----------------------------------------------

class folderXSSDT(folderSDT):
    '''for a xsSDT folder, measure all of the disturbed lines'''
    
    def __init__(self, folder:str, overwrite:bool=False, **kwargs) -> None:
        if not 'disturbXS' in os.path.basename(self.folder):
            raise ValueError(f'Wrong folderDisturb class called for {self.folder}')
        super().__init__(folder, overwrite=overwrite, **kwargs)
        
    def measureFolder(self) -> None:
        '''measure all cross-sections in the folder and export table'''
        self.measure(self.lines, fileXSSDT)

    #-----------------------------------------------------------------
    # summaries

    def summarize(self) -> Tuple[dict,dict, pd.DataFrame]:
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