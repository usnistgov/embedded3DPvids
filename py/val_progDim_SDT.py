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
from config import cfg
from val_progDim_tools import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)



#----------------------------------------------

class progDimsSDT(progDim):
    '''for programmed dimensions of single double triple prints'''
    
    def __init__(self, printFolder:str, pv:printVals, **kwargs):
        super().__init__(printFolder, pv, **kwargs)
        
    def getOvershoots(self) -> pd.DataFrame:
        return self.ftable[((self.ftable.targetLine)>(self.ftable.targetLine.shift(-1)))&(~self.ftable.trusted)]
        
    def getTimeRewrite(self, diag:int=0) -> int:
        '''overwrite the target points in the time file'''
        super().getTimeRewrite(diag=diag)
        self.ftable.loc[0, 'targetLine'] = 0
        overshoots = self.getOvershoots()
        dummy = 0
        while len(overshoots)>0 and dummy<10:
            for i,row in overshoots.iterrows():
                if i>0:
                    bads = self.ftable[(self.ftable.targetLine==row['targetLine'])&(self.ftable.time<=row['time'])]
                    for s in ['xt', 'yt', 'zt', 'targetLine']:
                        self.ftable.loc[bads.iloc[0].name:bads.iloc[-1].name, s] = self.ftable.loc[row.name+1, s]   # overwrite these with the next target
            overshoots = self.getOvershoots()
            dummy = dummy+1
            
        self.ftable.drop(columns=['xt_orig', 'yt_orig', 'zt_orig'], inplace=True)
        
        self.rewritten = True
        return 0
        
    def initializeProgDims(self):
        '''initialize programmed dimensions table'''
        super().initializeProgDims()
        if '_1_' in os.path.basename(self.printFolder):
            # 1 write, 1 disturb
            self.ll = ['w', 'd']
            self.wnum = 1
            self.dnum = 1
        elif '_2_' in os.path.basename(self.printFolder):
            # 2 write, 1 disturb
            self.ll = ['w1', 'w2', 'd']
            self.wnum = 2
            self.dnum = 1
        elif '_3_' in os.path.basename(self.printFolder):
            # 3 write, no disturb
            self.ll = ['w1', 'w2', 'w3']
            self.wnum = 3
            self.dnum = 0
        self.progDims.name = [f'l{i}{s2}{s3}' for i in range(self.numLines) for s2 in self.ll for s3 in ['', 'o1', 'o2'] ] 
        
    def vLongLines(self, moveDir:str, diag:bool=False) -> None:
        '''filter out just the write and disturb moves from progPos'''
        if not hasattr(self, 'progPos'):
            raise ValueError(f'{self.printFolder}: progPos not created')
        pdp = self.progPos.copy()
        p = moveDir[0]
        md = moveDir[1:]
        if moveDir=='-x':
            vlines = pdp[(pdp.dx<0)&(pdp.zt<0)]
            mval = vlines[f'd{md}'].min()
        elif moveDir=='+y':
            vlines = pdp[(pdp.dy>0)&(pdp.zt<0)&(pdp.dx==0)&(pdp.dz==0)]
            mval = vlines[f'd{md}'].max()
        elif moveDir=='+z':
            vlines = pdp[(pdp.dz>0)&(pdp.zt<0)]
            vc = vlines.dprog.value_counts()   # number of moves per length      
            mval = vc[vc>=3].index.max()     # find the largest move with at least 8 instances

        self.vll = vlines[vlines[f'd{md}']==mval]
        if moveDir=='+y':
            self.vll = self.vll[self.vll.yt==self.vll.yt.min()]
                
        return 
    
    def checkLongLines(self, moveDir:str, diag:bool=False) -> None:
        '''check that there is the right number of long lines'''
        # check for missing moves
        if moveDir=='+y':
            counts = self.vll.value_counts('xt')
            if not len(counts)==4:
                raise ValueError(f'Wrong number of targets in {self.printFolder}: {len(counts)}')
            missing = counts[counts<counts.max()]
            if len(missing)>1:
                # we have at least one missing target
                raise ValueError(f'Multiple missing targets in {self.printFolder}: {len(missing)}')
            elif len(missing)==1:
                xt = missing.index[0]
                if xt==self.vll.xt.min():
                    # this is the last target. just cut off the line
                    logging.warning(f'Missing line in {self.printFolder}. Cutting off last line')
                    tmax = self.vll[self.vll.xt==xt].t0.min()
                    self.otimes = self.otimes[self.otimes.time<tmax]   # remove late observations
                    self.vll = self.vll[self.vll.xt>xt]               # remove late moves
                    self.numLines = 3
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
        elif moveDir=='+z':
            counts = self.vll.value_counts('xt')
            if not len(counts)==2:
                raise ValueError(f'Wrong number of targets in {self.printFolder}: {len(counts)}')
            missing = counts[counts<counts.max()]
            if len(missing)>1:
                # we have at least one missing target
                raise ValueError(f'Multiple missing targets in {self.printFolder}: {len(missing)}')
            elif len(missing)==1:
                # find coordinates of missing target
                xt = missing.index[0]
                counts2 = self.vll.value_counts('yt')
                yt = counts2[counts2<counts2.max()].index[0]
                counts3 = self.vll.value_counts('zt')
                zt = counts3[counts3<counts3.max()].index[0]
                matchline = self.vll[(self.vll.xt==xt)&(self.vll.zt==zt)&(abs(self.vll.yt-yt)<2)]   # the corresponding write or disturb line
                otherlines = self.vll[(self.vll.xt==xt)&(self.vll.zt==zt)&(abs(self.vll.yt-yt)>2)]  # another set of write and disturb lines
                dtdw = otherlines.iloc[1]['tf'] - otherlines.iloc[0]['tf']   # difference in time between write and disturb
                didw = otherlines.index.to_series().diff().max()
                if matchline.iloc[0]['yt']>yt:
                    # missing write line
                    dtdw = -dtdw    
                t0 = round(matchline.iloc[0]['t0'] + dtdw,1)
                tf = round(matchline.iloc[0]['tf'] + dtdw,1)
                raise ValueError(f'Missing target {xt}, {yt}, {zt} in {self.printFolder}, t0 between {t0-1} and {t0+1}, tf between {tf-1} and {tf+1}, at row {matchline.index[0]-didw}')
        return
                

        
    def splitProgPos(self, moveDir:str, diag:bool=False):
        '''convert list of positions for horizontal lines'''
        
        # get the snap times
        self.otimes = self.flagFlip[self.flagFlip.cam.str.contains('SNAP')] 
        
        # get the moves
        self.vLongLines(moveDir, diag)
        self.checkLongLines(moveDir, diag)

        # split the moves into writes and disturbs
        self.vll.sort_values(by='t0', inplace=True)
        numper = (self.wnum+self.dnum)
        wlines = pd.concat([self.vll.iloc[i::numper] for i in range(self.wnum)])
        wlines.sort_values(by='t0', inplace=True)
        if self.dnum>0:
            dlines = self.vll.iloc[(self.wnum)::numper]
        else:
            dlines = []
        
        return wlines, dlines
    
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
        return row

    def getProgDims(self, diag:bool=False):
        '''convert the full position table to a list of timings'''
        self.importProgPos()
        self.importFlagFlip()
        self.initializeProgDims()
    
        
        if 'Horiz' in self.sbp:
            wlines, dlines = self.splitProgPos('+y', diag=diag)
        elif 'Vert' in self.sbp:
            wlines, dlines = self.splitProgPos('+z', diag=diag)
        elif 'XS' in self.sbp:
            wlines, dlines = self.splitProgPos('-x', diag=diag)
        otimes = self.otimes
        
        wprog = len(self.progDims[(self.progDims.name.str.contains('w'))&(~(self.progDims.name.str.contains('o')))])
        if not len(wlines)==wprog:
            raise ValueError(f'Mismatch in write lines: {len(wlines)} found, {wprog} programmed')

        dprog = len(self.progDims[self.progDims.name.str[-1]=='d'])
        if not len(dlines)==dprog:
            raise ValueError(f'Mismatch in disturb lines: {len(dlines)} found, {dprog} programmed')
            
        oprog = len(self.progDims[self.progDims.name.str.contains('o')])
        if not len(otimes)==oprog:
            raise ValueError(f'Mismatch in observe lines: {len(otimes)} found, {oprog} programmed')
        
        oi = 0
        # assign written lines
        wi = 0 # 
        di = 0
        for j in range(self.numLines):        
            for ss in ['w', 'd']:
                for cha in list(filter(lambda l: l[0]==ss , self.ll)):
                    if ss=='w':  
                        if wi>=len(wlines):
                            # we've run out
                            return
                        line = wlines.iloc[wi]
                        wi = wi+1
                    else:
                        if di>=len(dlines):
                            # we've run out
                            return
                        line = dlines.iloc[di]
                        di = di+1
                    for y in ['l', 'w', 't', 'a', 'vol', 't0', 'tf']:
                        self.progDims.loc[self.progDims['name']==f'l{j}{cha}',y] = line[y]
                                    # determine where to take pic
                    tpic = line['t0']+line['dprog']*0.5/line['speed']
                    self.progDims.loc[self.progDims['name']==f'l{j}{cha}','tpic'] = tpic
                    self.progDims.loc[self.progDims['name']==f'l{j}{cha}','lprog'] = line['dprog']
                    self.progDims.loc[self.progDims['name']==f'l{j}{cha}','ltr'] = line['dtr']
                    self.progDims.loc[self.progDims['name']==f'l{j}{cha}','speed'] = line['speed']
                    if ss=='w':
                        gfl = self.getFullLength(line)
                        for key,val in gfl.items():
                            self.progDims.loc[self.progDims['name']==f'l{j}{cha}',key] = val
                    
                    for k in range(2):
                        self.progDims.loc[self.progDims['name']==f'l{j}{cha}o{k+1}','tpic'] = otimes.iloc[oi]['time']
                        oi+=1
        if not all(self.progDims.sort_values(by='tpic').index==self.progDims.index):
            print(self.progDims)
            raise ValueError('Times are out of order')
        if pd.isna(self.progDims.loc[0,'l']):
            raise ValueError('Missing values in progDims')