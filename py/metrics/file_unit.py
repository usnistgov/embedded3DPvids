#!/usr/bin/env python
'''Functions for testing data collection on single images'''

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
import file.file_handling as fh
from im.imshow import imshow
from tools.plainIm import *
from tools.config import cfg
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

def testFile(fstr:str, fistr:str, func, slist:list, diag:int=4, **kwargs) -> dict:
    '''test a single file, for any print type given a metricSegment class '''
    folder = os.path.join(cfg.path.server, fstr)
    file = os.path.join(folder, fistr)
    d,u = func(file, diag=diag, **kwargs).values()
    out = f'{fstr},{fistr}'
    olist = {'folder': fstr, 'file':fistr}
    for s in slist:
        if s in d:
            v = d[s]
        else:
            v = -1
        out = f'{out},{v}'
        olist[s] =v
    return olist

def addToTestFile(csv:str, fstr:str, fistr:str, func, slist:list, diag:int=4, **kwargs) -> None:
    l = testFile(fstr, fistr, func, slist, diag=diag, **kwargs)
    df, _ = plainIm(csv, ic=None)
    if len(df)==0:
        df = pd.DataFrame([l])
    else:
        if l['file'] in df.file:
            for key, val in l:
                df.loc[df.file==l['file'], key] = val
        else:
            df = pd.concat([df, pd.DataFrame([l])])
    plainExp(csv, df, {}, index=False)
    
class unitTester:
    '''this class lets you run unit tests and evaluate functions later. fn is the test name, e.g. SDTXS or disturbHoriz, func is the function that you run on a file to get values'''
    
    def __init__(self, fn:str, func):
        cdir = os.path.dirname(os.path.abspath(os.path.join('..')))
        self.testcsv = os.path.join(cdir, 'tests', f'test_{fn}.csv')  # the csv file for the test
        self.testpy = f'test_{fn}'   # the python file for the test
        self.func = func
        
    def run(self):
        currentdir = os.path.dirname(os.path.realpath(__file__))
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(currentdir)), 'tests'))
        print(self.testpy)
        tp = __import__(self.testpy)
        runner = tp.unittest.TextTestRunner()
        result = runner.run(tp.suite())
        self.failedFiles = [int(re.split(': ', str(s))[-1][:-4]) for s in result.failures]  # indices of failed files
        
    def importList(self):
        if not hasattr(self, 'testList'):
            self.testList = pd.read_csv(self.testcsv)
            
    def runTest(self, i:int, diag:int=0, **kwargs) -> Tuple[pd.Series, dict, list]:
        self.importList()
        if i>=len(self.testList):
            print(f'UnitTester has {len(self.testList)} entries. {i} not in list')
            return [],{},[]
        row = self.testList.loc[i]
        folder = row['folder']
        file = row['file']
        if diag>0:
            print(f'TEST {i} (excel row {i+2})\nFolder: {folder}\nFile: {file}')
        folder = os.path.join(cfg.path.server, folder)
        file = os.path.join(folder, file)
        d,u = self.func(file, diag=diag, **kwargs)
        cols = list(self.testList.keys())
        cols.remove('folder')
        cols.remove('file')
        return row, d, cols
        
    def compareTest(self, i:int, diag:int=1, **kwargs) -> None:
        '''print diagnostics on why a test failed. fn is the basename of the test csv, e.g. test_SDTXS.csv. i is the row number'''
        row, d, cols = self.runTest(i, diag=diag, **kwargs)
        if len(row)==0:
            return
        df = pd.DataFrame({})
        for c in cols:
            df.loc['expected', c] = row[c]
            if c in d:
                df.loc['actual', c] = d[c]
        pd.set_option("display.precision", 8)
        display(df)
        
    def compareAll(self):
        '''check diagnostics for all failed files'''
        c = 0
        if len(self.failedFiles)>5:
            print(f'{len(self.failedFiles)} failed files, showing 5')
        for i in self.failedFiles:
            self.compareTest(i)
            c = c+1
            if c==5:
                return
            
    def openCSV(self):
        subprocess.Popen([cfg.path.excel, self.testcsv]);
        
    def exportTestList(self):
        '''overwrite the list of tests with current values'''
        self.testList.to_csv(self.testcsv, index=False)
        logging.info(f'Exported {self.testcsv}')
        
    def keepTest(self, i:int, export:bool=True) -> None:
        '''overwrite the value in the csv file with the current values'''
        row, d, cols = self.runTest(i, diag=1)
        for c in cols:
            self.testList.loc[i, c] = d[c]
        if export:
            self.exportTestList()
        
    def keepAllTests(self) -> None:
        '''overwrite all failed values with the values found now'''
        for i in self.failedFiles:
            self.keepTest(i, export=False)
        self.exportTestList()
        
    def openExplorer(self, i:int):
        folder = self.testList.loc[i,'folder']
        folder = os.path.join(cfg.path.server, folder)
        fh.openExplorer(folder)
            
def runUnitTest(testName:str, func):
    ut = unitTester(testName, func)
    ut.run()
    ut.compareAll()

    
    
