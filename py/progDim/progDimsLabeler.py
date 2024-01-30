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

class progDimsLabeler:
    '''labels the lines in the progDims'''
    
    def __init__(self, ppd:progPosData, progPos:pd.DataFrame, progDims:pd.DataFrame):
        self.ppd = ppd
        self.progPos = progPos
        self.progDims = progDims
        self.otimes = self.ppd.otimes
        
        oi = 0
        # assign written lines
        wi = 0 # 
        di = 0
        for j in range(self.ppd.numLines):  
            for l in self.ppd.ll:
                if l[0]=='w':
                    wi, oi = self.labelSublist(j, l, wi, oi, self.ppd.wlines)
                else:
                    di, oi = self.labelSublist(j, l, di, oi, self.ppd.dlines)
    #--------------------------------------
                    
    def getFullLength(self, line:pd.Series) -> dict:
        '''get stats on the full length of a line where extrusion spanned several points'''
        n0 = line.name   # row number
        n = n0
        while n>0 and self.progPos.loc[n, 'l']>0:
            n = n-1
        n = n+1
        nmin = n
        row = {'l':0, 'vol':0, 't0':self.progPos.loc[n, 't0_flow']}
        while n<len(self.progPos) and self.progPos.loc[n, 'l']>0:
            for s in ['l', 'vol']:
                row[s] = row[s]+self.progPos.loc[n,s]
            n = n+1
        n = n-1
        row['tf'] = self.progPos.loc[n, 'tf_flow']
        if row['l']>0:
            row['a'] = row['vol']/row['l']
            row['w'] = 2*np.sqrt(row['a']/np.pi)
        row['t'] = row['tf']-row['t0']
        row['ltr'] = self.progPos.loc[nmin:n, 'dtr'].max()
        row['wmax'] = self.progPos.loc[nmin:n, 'wmax'].max()
        return row
    
    def labelProgLine(self, line:pd.Series, j:int, cha:str, gfl:dict) -> None:
        '''label a single progdims row
        j is the group number, 1 2 3 4
        cha is the line name, e.g. l1w1p3'''
        row = (self.progDims['name']==cha)
        
        # adopt the line's values
        for y in ['l', 'w', 'wmax', 't', 'a', 'vol', 't0', 'tf', 'speed']:
            self.progDims.loc[row,y] = line[y]
            
        # find the pic time
        if 'p' in cha:
            # this is an in-progress line
            frac = 0.1 + (int(cha[-1])-1)/self.ppd.numPtimes
        else:
            frac = 0.5
        
        dmax = max(abs(line['dx']), abs(line['dy']), abs(line['dz']))
        
        # find midpoint of line
        for cs in ['x', 'y', 'z']:
            dd = line[f'd{cs}']
            if not abs(dd)==dmax:
                dd = 0
            self.progDims.loc[row,f'{cs}pic'] = line[f'{cs}t']-dd*(1-frac)
        
        self.progDims.loc[row,'lprog'] = line['dprog']
        self.progDims.loc[row,'ltr'] = line['dtr']
        if cha[2]=='w':
            for key,val in gfl.items():
                self.progDims.loc[row,key] = val
        self.progDims.loc[row,'tpic'] = self.progDims.loc[row, 't0']+line['dprog']*frac/line['speed']
                
            
    def labelObserveLine(self, cha:str, oi:int) -> int:
        '''label a group of observe pics
        j is the group number
        cha is the full name of the line, e.g. l1w1o2
        k is the o
        '''
        tpic1 = self.otimes.iloc[oi]['time']
        oi = oi+1
        tpic2 = self.otimes.iloc[oi]['time']
        oi = oi+1
        
        # get the position where the image is taken
        lines = self.progPos[(self.progPos.t0<tpic1)&(self.progPos.tf>=tpic1)]
        if len(lines)==0:
            display(self.progPos)
            raise ValueError(f'Could not find observe line in progPos: {tpic}')
        line = lines.iloc[0]
        
        for i in range(1, self.ppd.numOtimes+1):
            cha2 = cha.replace('o1', f'o{str(i)}')
            frac = (i-1)/(self.ppd.numOtimes-1)
            tpic = tpic1 + (tpic2-tpic1)*frac
            row = (self.progDims['name']==cha2)
            self.progDims.loc[row,'tpic'] = tpic
            for cs in ['x', 'y', 'z']:
                self.progDims.loc[row,f'{cs}pic'] = line[f'{cs}t']
        
        return oi
                    
    def labelSublist(self, j:int, ss:str, i:int, oi:int, lines:pd.DataFrame) -> Tuple[int,int]:
        '''find all relevant pics in the sublist and put their stats in progDims. 
        j is the line number, 1, 2,3,4
        ss is a name for the write mode, e.g. w1, w2, d1
        i is the current index within lines
        oi is the current index within olines
        lines is the list of lines for this type, either wlines or dlines
        olines is the list of observe lines'''
        sublist = list(filter(lambda l: l[0]==ss , self.ppd.ll))   # all the names for this write mode, e.g. w1, w2, d1
        if i>=len(lines):
            # we've run out
            return 100,oi
        line = lines.iloc[i]
        gfl = self.getFullLength(line)
        i = i+1
        
        names = self.progDims[(self.progDims.name.str.contains(ss))&(self.progDims.name.str.contains(f'l{j}'))].name  
                      # all the line pic names with this line number and write format. includes all p lines and o lines

        for cha in names:
            if not 'o' in cha:
                self.labelProgLine(line, j, cha, gfl)
            elif 'o1' in cha:
                oi = self.labelObserveLine(cha, oi)  
        return i, oi
        