#!/usr/bin/env python
'''Functions for handling tables of programmed timings'''

# external packages
import os, sys
import traceback
import logging
from typing import List, Dict, Tuple, Union, Any, TextIO
import re
import pandas as pd
import numpy as np
import csv

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
from tools.config import cfg
from pg_tools import *
from timeRewriteChecker import *
from progPosChecker import *
from progPosSplitter import *
from progDimsChecker import *
from progDimsLabeler import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)



#----------------------------------------------

class progDimsSDT(progDim):
    '''for programmed dimensions of single double triple prints'''
    
    def __init__(self, printFolder:str, pv:printVals, **kwargs):
        self.ppd = progPosData(printFolder)
        if 'XS' in printFolder:
            self.ppd.numOtimes = 2
        else:
            self.ppd.numOtimes = 8
        self.ppd.numPtimes = 5
        super().__init__(printFolder, pv, **kwargs)

    def getTimeRewrite(self, diag:int=0) -> int:
        '''overwrite the target points in the time file'''
        super().getTimeRewrite(diag=diag)
        self.rewriteChecker = timeRewriteChecker(self.printFolder, self.ftable)  # correct errors in the file
        return 0
        
    def getProgPos(self) -> None:
        super().getProgPos()
        self.progPosChecker = progPosChecker(self.printFolder, self.progPos)
        self.progPos = self.progPosChecker.progPos
        
    def initializeProgDims(self, xs:bool=False):
        '''initialize programmed dimensions table'''
        super().initializeProgDims()
        
        if '_1_' in os.path.basename(self.printFolder):
            # 1 write, 1 disturb
            self.ppd.ll = ['w1', 'd1']
            self.ppd.wnum = 1
            self.ppd.dnum = 1
        elif '_2_' in os.path.basename(self.printFolder):
            # 2 write, 1 disturb
            self.ppd.ll = ['w1', 'w2', 'd2']
            self.ppd.wnum = 2
            self.ppd.dnum = 1
        elif '_3_' in os.path.basename(self.printFolder):
            # 3 write, no disturb
            self.ppd.ll = ['w1', 'w2', 'w3']
            self.ppd.wnum = 3
            self.ppd.dnum = 0
        if xs:
            s3list = [''] + [f'o{i}' for i in range(1, self.ppd.numOtimes+1)]
        else:
            s3list = [f'p{i}' for i in range(1, self.ppd.numPtimes+1)] + [f'o{i}' for i in range(1, self.ppd.numOtimes+1)]
        self.progDims.name = [f'l{i}{s2}{s3}' for i in range(self.numLines) for s2 in self.ppd.ll for s3 in s3list] 
     
    #---------------------------------------
        
    def getPrimaryLines(self, c:str, pg:pd.DataFrame):
        '''get the list of printed lines that have this character, only taking one in-progress write line per written line'''
        if c=='o':
            return pg[(pg.name.str.contains('o1'))|(pg.name.str.contains(f'o{self.ppd.numOtimes}'))]
        else:
            p1 = pg[(pg.name.str.contains(c))&(~(pg.name.str.contains('o')))]
            p1p = p1[p1.name.str.contains('p3')]
            if len(p1p)==0:
                return p1
            else:
                return p1p
        
    def getNumLines(self, c:str):
        '''get the number of printed lines that have this character'''
        return len(self.getPrimaryLines(c, self.progDims))
    
    #---------------------------------------
        
    def splitProgPos(self, **kwargs):
        '''convert list of positions'''
        self.progPosSplitter = progPosSplitter(self.printFolder, self.ppd, self.flagFlip, self.progPos)
        self.vll = self.progPosSplitter.vll
    
    #---------------------------------------        


    def labelProgDims(self):
        self.progDimsLabeler = progDimsLabeler(self.ppd, self.progPos, self.progDims)
        self.progDims = self.progDimsLabeler.progDims


    def checkProgDims(self):
        '''create an object that raises errors if the progDims don't make sense'''
        self.progDimsChecker = progDimsChecker(self.printFolder, self.progDims, self.ppd, self.pv)

    def getProgDims(self, diag:bool=False):
        '''convert the full position table to a list of timings'''
        self.importProgPos()
        self.importFlagFlip()
        self.initializeProgDims('XS' in self.sbp)

        self.ppd.wprog = self.getNumLines('w')
        self.ppd.dprog = self.getNumLines('d')
        self.ppd.oprog = self.getNumLines('o')
        
        self.splitProgPos()
        self.labelProgDims()
        self.checkProgDims()  # check that all values are correct
        if diag:
            display(self.progDims)
   