#!/usr/bin/env python
'''class for splitting progPos tables into lines'''

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

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)



#----------------------------------------------

class progPosData:
    '''holds metadata about progDims lines'''
    
    def __init__(self, printFolder:str):
        self.printFolder = printFolder
        if 'Horiz' in self.printFolder:
            self.moveDir = '+y'
            self.ptype = 'horiz'
        elif 'Vert' in self.printFolder:
            self.moveDir = '+z'
            self.ptype='vert'
        elif 'XS' in self.printFolder:
            self.moveDir = '-x'
            self.ptype = 'xs'
        elif 'Under' in self.printFolder:
            self.moveDir = '+x'
            self.ptype = 'under'
        self.numLines = 4


class progPosSplitter:
    '''for splitting progPos tables into lines'''
    
    def __init__(self, printFolder:str, ppd:progPosData, flagFlip:pd.DataFrame, progPos:pd.DataFrame, getBlips:bool=True, **kwargs):
        self.printFolder = printFolder
        self.ppd = ppd
        self.flagFlip = flagFlip
        self.progPos = progPos

        # get the snap times
        otimes = self.flagFlip[self.flagFlip.cam.str.contains('SNAP')&(~self.flagFlip.cam.str.contains('SNOFF'))] 
        otimes = otimes.copy()
        snofftimes = self.flagFlip[(~self.flagFlip.cam.str.contains('SNAP'))&(self.flagFlip.cam.str.contains('SNOFF'))] 
        otimes['snapdt'] = [snofftimes.iloc[i]['time']-otimes.iloc[i]['time'] for i in range(len(snofftimes))]
        if getBlips:
            otimes = otimes[otimes.snapdt>otimes.snapdt.mean()/2]  # remove very small snap times
        self.ppd.otimes = otimes
        
        # get the moves
        self.vLongLines()
        self.getwdlines()
        self.checkLongLines()  # fix errors in vll
        
    #-----------------------------------------------
        
    def vLongLines(self) -> None:
        '''filter out just the write and disturb moves from progPos'''
        if not hasattr(self, 'progPos'):
            raise ValueError(f'{self.printFolder}: progPos not created')
        pdp = self.progPos.copy()
        p = self.ppd.moveDir[0]
        md = self.ppd.moveDir[1:]
        if self.ppd.moveDir=='-x':  # xs
            vlines = pdp[(pdp.dx<0)&(pdp.zt<0)&(pdp.shift(-1).dx<0)]
            mval = vlines[f'd{md}'].min()
        elif self.ppd.moveDir=='+x': # under
            vlines = pdp[(pdp.dx>0)&(pdp.zt<0)&(pdp.shift(-1).dx>0)]
            mval = vlines[f'd{md}'].max()
        elif self.ppd.moveDir=='+y':  # horiz
            vlines = pdp[(pdp.dy>0)&(pdp.zt<0)&(pdp.dx==0)&(pdp.dz==0)&(pdp.shift(-1).dy>0)]
            mval = vlines[f'd{md}'].max()
        elif self.ppd.moveDir=='+z':  # vert
            vlines = pdp[(pdp.dz>0)&(pdp.zt<0)&(pdp.shift(-1).dz>0)]   # make sure there is an extension on the end of it
            vc = vlines.dprog.value_counts()   # number of moves per length      
            mval = vc[vc>=3].index.max()     # find the largest move with at least 8 instances

        self.vll = vlines[vlines[f'd{md}']==mval]
        if self.ppd.moveDir=='+y':
            self.vll = self.vll[self.vll.yt==self.vll.yt.min()]
            
    #-----------------------------------------------
            
    def getwdlines(self) -> None:
        '''get the number of write and disturb lines'''
        # split the moves into writes and disturbs
        self.vll.sort_values(by='t0', inplace=True)
        numper = (self.ppd.wnum+self.ppd.dnum)
        wlines = pd.concat([self.vll.iloc[i::numper] for i in range(self.ppd.wnum)])
        wlines.sort_values(by='t0', inplace=True)
        if self.ppd.dnum>0:
            dlines = self.vll.iloc[(self.ppd.wnum)::numper]
        else:
            dlines = []
        self.ppd.wlines = wlines
        self.ppd.dlines = dlines
        self.checkNumberMatching()
        
    def checkNumberMatching(self) -> None:
        '''check that the number of lines matches'''
        if not len(self.ppd.wlines)==self.ppd.wprog:
            print(self.vll)
            raise ValueError(f'Mismatch in write lines: {len(self.ppd.wlines)} found, {self.ppd.wprog} programmed')

        
        if not len(self.ppd.dlines)==self.ppd.dprog:
            raise ValueError(f'Mismatch in disturb lines: {len(self.ppd.dlines)} found, {self.ppd.dprog} programmed')

        if not len(self.ppd.otimes)==self.ppd.oprog:
            if len(self.ppd.otimes)==self.ppd.oprog+2:
                self.ppd.backgroundSnaps = 2
                self.ppd.otimes = self.ppd.otimes[2:]
            else:
                raise ValueError(f'Mismatch in observe lines: {len(self.ppd.otimes)} found, {self.ppd.oprog} programmed')
        
    #-----------------------------------------------
    
    def missingx(self) -> pd.DataFrame:
        bn = os.path.basename(self.printFolder)
        if '+y' in bn or 'Under' in self.printFolder:
            counts = self.vll.value_counts('zt')
        elif '+z' in bn:
            counts = self.vll.value_counts('yt')
        if 'Under' in self.printFolder:
            numCrit = 3
        else:
            numCrit = 4
        if not len(counts)==numCrit:
            raise ValueError(f'Wrong number of targets in {self.printFolder}: {len(counts)}')
        missing = counts[counts<counts.max()]
        return missing
    
    def checkLongLinesx(self) -> None:
        '''check that there are the right number of long lines for a -x move'''
        missing = self.missingx()
        if len(missing)>0:
            # we have at least one missing target
            print(self.vll)
            raise ValueError(f'Missing targets in {self.printFolder}: {len(missing)}')
            
        # dy and dz should be 0 for x moves. remove any diagonals that show up because of missing targets in time file
        self.vll.loc[:, 'dy'] = 0
        self.vll.loc[:, 'dz'] = 0
        self.vll.loc[:, 'dprog'] = abs(self.vll.iloc[0]['dx'])
        
    def missingy(self) -> pd.DataFrame:
        counts = self.vll.value_counts('xt')
        if not len(counts)==4:
            raise ValueError(f'Wrong number of targets in {self.printFolder}: {len(counts)}')
        missing = counts[counts<counts.max()]
        return missing
    
    def checkLongLinesy(self) -> None:
        '''check that there are the right number of long lines for a +y move'''
        missing = self.missingy()
        if len(missing)>1:
            # we have at least one missing target
            raise ValueError(f'Multiple missing targets in {self.printFolder}: {len(missing)}')
        elif len(missing)==1:
            xt = missing.index[0]
            if xt==self.vll.xt.min():
                # this is the last target. just cut off the line
                logging.warning(f'Missing line in {self.printFolder}. Cutting off last line')
                tmax = self.vll[self.vll.xt==xt].t0.min()
                self.ppd.otimes = self.ppd.otimes[self.ppd.otimes.time<tmax]   # remove late observations
                self.vll = self.vll[self.vll.xt>xt]               # remove late moves
                self.ppd.numLines = 3
                self.initializeProgDims()                        # reinitialize progdims without the last line
                return
            zd = self.vll.zt.diff()
            counts = round(zd,2).value_counts()
            dzmax = counts.index.max()
            if counts[dzmax]<3:
                # missing big shift
                raise ValueError(f'Missing big shift to {xt}')
            dzsmall = counts[counts==counts.max()].index[0]
            if counts[dzsmall]<8:
                # missing big shift
                raise ValueError(f'Missing small shift to {xt}')  
                
    def checkExtraY(self):
        '''check for extra rows in longlinesz'''
        for x in self.vll.xt.unique():
            sub = self.vll[self.vll.xt==x]
            if len(sub.yt.unique())<len(sub):
                # duplicate yt values
                for y in sub.yt.unique():
                    suby = sub[sub.yt==y]
                    for i,row in suby[suby.dtr<suby.dtr.max()].iterrows():
                         self.vll.drop(i,inplace=True)
    
    def missingz(self) -> pd.DataFrame:
        counts = self.vll.value_counts('xt')
        if not len(counts)==2:
            raise ValueError(f'Wrong number of targets in {self.printFolder}: {len(counts)}')
                    
        missing = counts[counts<counts.max()]
        return missing
        
    def checkLongLinesz(self) -> None:
        '''check that there are the right number of long lines for a +z move'''
        self.checkExtraY()
        missing = self.missingz()

        if len(missing)>1:
            # we have at least one missing target
            raise ValueError(f'Multiple missing targets in {self.printFolder}: {len(missing)}')
        elif len(missing)==1:
            # find coordinates of missing target
            xt = missing.index[0]
            counts2 = self.vll.value_counts('yt')
            yt = counts2[counts2<counts2.max()].index[0]
            counts3 = self.vll.value_counts('zt')
            print(self.vll)
            zt = counts3[counts3<counts3.max()].index[0]
            matchline = self.vll[(self.vll.xt==xt)&(self.vll.zt==zt)&(abs(self.vll.yt-yt)<2)]   # the corresponding write or disturb line
            otherlines = self.vll[(self.vll.xt==xt)&(self.vll.zt==zt)&(abs(self.vll.yt-yt)>2)]  # another set of write and disturb lines
            if len(otherlines)==0:
                raise ValueError(f'Missing target {xt}, {yt}, {zt} in {self.printFolder}')
            dtdw = otherlines.iloc[1]['tf'] - otherlines.iloc[0]['tf']   # difference in time between write and disturb
            didw = otherlines.index.to_series().diff().max()
            if matchline.iloc[0]['yt']>yt:
                # missing write line
                dtdw = -dtdw    
            t0 = round(matchline.iloc[0]['t0'] + dtdw,1)
            tf = round(matchline.iloc[0]['tf'] + dtdw,1)
            raise ValueError(f'Missing target {xt}, {yt}, {zt} in {self.printFolder}, t0 between {t0-1} and {t0+1}, tf between {tf-1} and {tf+1}, at row {matchline.index[0]-didw}')
        
    def checkLongLines(self) -> None:
        '''check that there is the right number of long lines'''
        # check for missing moves
        if self.ppd.moveDir=='+y':
            self.checkLongLinesy()
        elif self.ppd.moveDir=='+z':
            self.checkLongLinesz()
        elif self.ppd.moveDir=='-x' or self.ppd.moveDir=='+x':
            self.checkLongLinesx()

    