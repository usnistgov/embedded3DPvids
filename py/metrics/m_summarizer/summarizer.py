#!/usr/bin/env python
'''Functions for collecting data from stills of single lines, for a whole folder'''

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
from tools.plainIm import *
from tools.config import cfg
from val.v_print import printVals
from progDim.prog_dim import getProgDims
import file.file_handling as fh
from m_tools import *
from failureTest import *
from full_sequence import SDTWorkflow

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)
pd.set_option('display.max_rows', 500)


#----------------------------------------------
 
class summarizer(fh.folderLoop):
    '''recursively create measures, summaries, failuares, and collect all of the summaries into a table. measureClass is a class definition for a folderMetric class. failures will be a list of files'''
    
    def __init__(self, folders:Union[str,list], measureClass, overwrite:bool=False, overwriteMeasure:bool=False, overwriteSummary:bool=False, **kwargs):
        super().__init__(folders, self.summarize, **kwargs)
        self.overwrite = overwrite
        self.overwriteMeasure = overwriteMeasure
        self.overwriteSummary = overwriteSummary
        self.measureClass = measureClass
        self.out = []
        self.units = {}
        self.failures = pd.DataFrame([])
        
    def summarize(self, folder:str, **kwargs) -> None:
        '''get summaries from a single folder and add them to the running list'''
        summary = []
        failures = []
        pfd = fh.printFileDict(folder)
        # go through the full workflow if there are no nozzle dimensions
        runFull = True
        try:
            pfd.nozDims
        except:
            runFull = True
        else:
            pfd.findVstill()
            runFull=(len(pfd.vstill)==0)

        if runFull:
            sw = SDTWorkflow(folder)
            sw.run()

        cl = self.measureClass(folder, overwrite=self.overwrite, overwriteMeasure=self.overwriteMeasure, overwriteSummary=self.overwriteSummary, exportCrop=False,  **self.kwargs)
        if self.overwriteMeasure:
            cl.measureFolder()
        if self.overwriteSummary or not os.path.exists(pfd.summary):
            summary, units, failures = cl.summarize()
        else:
            cl.summaryHeader()
            summary, units, failures = cl.summaryValues()

        if len(summary)>0:
            self.units = {**self.units, **units}
            if 'bn' in summary:
                self.out.append(summary)
            else:
                self.out = self.out+list(summary.values())
        if len(failures)>0:
            flist = []
            for i,row in failures.iterrows():
                if len(self.failures)==0 or not row['file'] in self.failures['file']:
                    if 'error' in row:
                        err = row['error']
                    else:
                        err = 'unknown'
                    flist.append({'file':row['file'], 'error':err})
            self.failures = pd.concat([self.failures, pd.DataFrame(flist)])
            self.failures.reset_index(inplace=True, drop=True)

    def export(self, fn:str) -> None:
        df = pd.DataFrame(self.out)
        
        if 'gname' in df:
            plainExp(fn, df[df.gname=='total'], self.units, index=False)   # export averaged values
            plainExp(fn.replace('.csv', '_gname.csv'), df[~(df.gname=='total')], self.units, index=False)  # export groups individually
        else:
            plainExp(fn, df, self.units, index=False)
        
    def exportFailures(self, fn:str) -> None:
        '''export a list of failed files'''
        plainExp(fn, self.failures, {'file':'', 'error':''}, index=False)
        if len(self.folderErrorList)>0:
            plainExp(fn.replace('Failures', 'Errors'), pd.DataFrame(self.folderErrorList), {}, index=False)
            
    def runFailure(self, i:int) -> None:
        self.summarize(self.folderErrorList[i]['folder'])
            
    def run(self):
        self.out = []
        self.units = {}
        self.failures = pd.DataFrame([])
        super().run()
        
#---------------------------------------------------
        
