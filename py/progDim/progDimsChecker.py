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
from progPosSplitter import progPosData

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)



#----------------------------------------------

class progDimsChecker:
    '''check that the programmed picture locations match expections. this only raises errors, doesn't fix anything'''
    
    def __init__(self, printFolder:str, progDims:pd.DataFrame, ppd:progPosData, pv:printVals):
        self.printFolder = printFolder
        self.progDims = progDims
        self.ppd = ppd
        self.pv = pv
        if not all(self.progDims.sort_values(by='tpic').index==self.progDims.index):
            print(self.progDims[self.progDims.sort_values(by='tpic').index!=self.progDims.index])
            raise ValueError('Times are out of order')
        if pd.isna(self.progDims.loc[0,'l']):
            raise ValueError('Missing values in progDims')
        bn = os.path.basename(self.printFolder)
        self.checkProgDimsT()
            
    #------------------------------------
    
    def compareZPositions(self, gp:pd.DataFrame, gnum:int) -> None:
        ''' all z positions in one group should be same for +y prints'''
        zlin = len(gp.zpic.unique())
        if zlin>1:
            raise ValueError(f'Too many zpic locations in {self.printFolder} group {gnum}: {zlin}')
            
    def compareOLines(self, gp:pd.DataFrame, gnum:int) -> None:
        '''all observe points in one group should be same'''
        olines = gp[gp.name.str.contains('o')]   # observe lines
        if self.ppd.ptype=='xs':
            l00 = ['ypic', 'zpic']
        else:
            l00 = ['xpic', 'ypic', 'zpic']
        for ss in l00:
            zlin = len(olines[ss].unique())
            if zlin>1:
                display(olines)
                raise ValueError(f'Too many {ss} observe locations in {self.printFolder} group {gnum}: {zlin}')
                
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
                
    def getdwlines(self, gp:pd.DataFrame) -> pd.DataFrame:
        '''get write and disturb lines for a specific group'''
        if self.ppd.ptype=='xs':
            dwlines = gp[~gp.name.str.contains('o')]
        else:
            dwlines = pd.concat([self.getPrimaryLines('w', gp), self.getPrimaryLines('d', gp)])
        return dwlines
                
    def compareWritePositions(self, gp:pd.DataFrame, gnum:int) -> None:
        '''determine if there are the correct number of write positions for horiz lines'''
        olines = gp[gp.name.str.contains('o')]   # observe lines
        for ss in ['xpic', 'ypic']:
            zlin = len(olines[ss].unique())
            if zlin>1:
                #display(dwlines)
                raise ValueError(f'Too many {ss} write locations in {self.printFolder} group {gnum}: {zlin}')
                
    def getSpacings(self, dwlines:pd.DataFrame, gnum:int) -> pd.Series:
        '''get spacings between pictures'''
        if self.ppd.ptype=='xs':
            if self.dire=='+y':
                self.spacings = dwlines.ypic.diff()
            elif self.dire=='+z':
                self.spacings = dwlines.zpic.diff()
                zlin = len(dwlines.ypic.unique())
                if zlin>1:
                    #display(dwlines)
                    raise ValueError(f'Too many ypic write locations in {self.printFolder} group {gnum}: {zlin}')
        elif self.ppd.ptype=='vert':
            self.spacings = dwlines.ypic.diff()
        elif self.ppd.ptype=='horiz':
            self.spacings = dwlines.zpic.diff()
    
    def checkProgDimsT(self) -> None:
        '''check programmed dimensions for a specific type of print'''
        bn = os.path.basename(self.printFolder)
        spl = re.split('_', bn)
        if self.ppd.moveDir=='-x':
            self.dire = spl[2]
            num = int(spl[1])
        else:
            num = int(spl[1])
            self.dire = ''
        spacing = float(spl[-1])
        intendedSpacing = self.pv.dEst*spacing
        scrit = 0.1
        
        for gnum in range(4):
            gp = self.progDims[self.progDims.name.str.startswith(f'l{gnum}')]
            
            if self.ppd.moveDir=='-x' and self.dire=='+y':
                self.compareZPositions(gp, gnum)
             
            self.compareOLines(gp, gnum)
            dwlines = self.getdwlines(gp)
            if self.ppd.ptype=='horiz':
                self.compareWritePositions(gp, gnum)
            self.getSpacings(dwlines, gnum)
            for sp in self.spacings:
                if sp>intendedSpacing+scrit or sp<intendedSpacing-scrit:
                    raise ValueError(f'Bad spacing in {self.printFolder} group {gnum}: {sp}/{intendedSpacing}')