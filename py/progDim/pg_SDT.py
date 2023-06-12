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

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)



#----------------------------------------------

class progDimsSDT(progDim):
    '''for programmed dimensions of single double triple prints'''
    
    def __init__(self, printFolder:str, pv:printVals, **kwargs):
        if 'XS' in printFolder:
            self.numOtimes = 2
        else:
            self.numOtimes = 8
        self.numPtimes = 5
        super().__init__(printFolder, pv, **kwargs)
        
    def getOvershoots(self, trust:bool=True) -> pd.DataFrame:
        f = self.ftable
        if trust:
            return f[((f.targetLine)>(f.targetLine.shift(-1)))&(~f.trusted)]
        else:
            return f[((f.targetLine)>(f.targetLine.shift(-1)))]
        
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
        if len(self.getOvershoots(trust=False))>0:
            # we still have some out of order points. steal from another folder
            self.stealTargets()
        self.rewritten = True
        return 0
        
    def initializeProgDims(self, xs:bool=False):
        '''initialize programmed dimensions table'''
        super().initializeProgDims()
        if '_1_' in os.path.basename(self.printFolder):
            # 1 write, 1 disturb
            self.ll = ['w1', 'd1']
            self.wnum = 1
            self.dnum = 1
        elif '_2_' in os.path.basename(self.printFolder):
            # 2 write, 1 disturb
            self.ll = ['w1', 'w2', 'd2']
            self.wnum = 2
            self.dnum = 1
        elif '_3_' in os.path.basename(self.printFolder):
            # 3 write, no disturb
            self.ll = ['w1', 'w2', 'w3']
            self.wnum = 3
            self.dnum = 0
        if xs:
            s3list = [''] + [f'o{i}' for i in range(1, self.numOtimes+1)]
        else:
            s3list = [f'p{i}' for i in range(1, self.numPtimes+1)] + [f'o{i}' for i in range(1, self.numOtimes+1)]
        self.progDims.name = [f'l{i}{s2}{s3}' for i in range(self.numLines) for s2 in self.ll for s3 in s3list] 
        
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
    
    def checkLongLinesx(self) -> None:
        '''check that there are the right number of long lines for a -x move'''
        if '+y' in os.path.basename(self.printFolder):
            counts = self.vll.value_counts('zt')
        else:
            counts = self.vll.value_counts('yt')
        
        if not len(counts)==4:
            raise ValueError(f'Wrong number of targets in {self.printFolder}: {len(counts)}')
        missing = counts[counts<counts.max()]
        if len(missing)>0:
            # we have at least one missing target
            raise ValueError(f'Missing targets in {self.printFolder}: {len(missing)}')
            
        # dy and dz should be 0 for x moves. remove any diagonals that show up because of missing targets in time file
        self.vll.loc[:, 'dy'] = 0
        self.vll.loc[:, 'dz'] = 0
        self.vll.loc[:, 'dprog'] = abs(self.vll.iloc[0]['dx'])
    
    def checkLongLinesy(self) -> None:
        '''check that there are the right number of long lines for a +y move'''
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
                
    def checkLongLinesz(self) -> None:
        '''check that there are the right number of long lines for a +z move'''
        self.checkExtraY()
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
    
    def checkLongLines(self, moveDir:str) -> None:
        '''check that there is the right number of long lines'''
        # check for missing moves
        if moveDir=='+y':
            self.checkLongLinesy()
        elif moveDir=='+z':
            self.checkLongLinesz()
        elif moveDir=='-x':
            self.checkLongLinesx()
        return

        
    def splitProgPos(self, moveDir:str, diag:bool=False):
        '''convert list of positions'''
        
        # get the snap times
        otimes = self.flagFlip[self.flagFlip.cam.str.contains('SNAP')&(~self.flagFlip.cam.str.contains('SNOFF'))] 
        otimes = otimes.copy()
        snofftimes = self.flagFlip[(~self.flagFlip.cam.str.contains('SNAP'))&(self.flagFlip.cam.str.contains('SNOFF'))] 
        otimes['snapdt'] = [snofftimes.iloc[i]['time']-otimes.iloc[i]['time'] for i in range(len(snofftimes))]
        otimes = otimes[otimes.snapdt>otimes.snapdt.mean()/2]  # remove very small snap times
        self.otimes = otimes
        
        # get the moves
        self.vLongLines(moveDir, diag)
        self.checkLongLines(moveDir)

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
        row['ltr'] = self.progPos.loc[nmin:n, 'dtr'].max()
        return row
    
    def labelProgLine(self, line:pd.Series, j:int, cha:str, gfl:dict) -> None:
        '''label a single progdims row
        j is the group number, 1 2 3 4
        cha is the line name, e.g. l1w1p3'''
        row = (self.progDims['name']==cha)
        
        # adopt the line's values
        for y in ['l', 'w', 't', 'a', 'vol', 't0', 'tf', 'speed']:
            self.progDims.loc[row,y] = line[y]
            
        # find the pic time
        if 'p' in cha:
            # this is an in-progress line
            frac = 0.1 + (int(cha[-1])-1)/self.numPtimes
        else:
            frac = 0.5
        tpic = line['t0']+line['dprog']*frac/line['speed']
        
        # find midpoint of line
        for cs in ['x', 'y', 'z']:
            self.progDims.loc[row,f'{cs}pic'] = line[f'{cs}t']-line[f'd{cs}']*(1-frac)
        self.progDims.loc[row,'tpic'] = tpic
        self.progDims.loc[row,'lprog'] = line['dprog']
        self.progDims.loc[row,'ltr'] = line['dtr']
        if cha[2]=='w':
            for key,val in gfl.items():
                self.progDims.loc[row,key] = val
            
    def labelObserveLine(self, cha:str, otimes:pd.DataFrame, oi:int) -> int:
        '''label a group of observe pics
        j is the group number
        cha is the full name of the line, e.g. l1w1o2
        k is the o
        '''
        tpic1 = otimes.iloc[oi]['time']
        oi = oi+1
        tpic2 = otimes.iloc[oi]['time']
        oi = oi+1
        
        # get the position where the image is taken
        lines = self.progPos[(self.progPos.t0<tpic1)&(self.progPos.tf>=tpic1)]
        if len(lines)==0:
            display(self.progPos)
            raise ValueError(f'Could not find observe line in progPos: {tpic}')
        line = lines.iloc[0]
        
        for i in range(1, self.numOtimes+1):
            cha2 = cha.replace('o1', f'o{str(i)}')
            frac = (i-1)/(self.numOtimes-1)
            tpic = tpic1 + (tpic2-tpic1)*frac
            row = (self.progDims['name']==cha2)
            self.progDims.loc[row,'tpic'] = tpic
            for cs in ['x', 'y', 'z']:
                self.progDims.loc[row,f'{cs}pic'] = line[f'{cs}t']
        
        return oi
    
    def labelSublist(self, j:int, ss:str, i:int, oi:int, lines:pd.DataFrame, otimes:pd.DataFrame) -> Tuple[int,int]:
        '''find all relevant pics in the sublist and put their stats in progDims. 
        j is the line number, 1, 2,3,4
        ss is a name for the write mode, e.g. w1, w2, d1
        i is the current index within lines
        oi is the current index within olines
        lines is the list of lines for this type, either wlines or dlines
        olines is the list of observe lines'''
        sublist = list(filter(lambda l: l[0]==ss , self.ll))   # all the names for this write mode, e.g. w1, w2, d1
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
                oi = self.labelObserveLine(cha, otimes, oi)  
        return i, oi
    
    
    def getPrimaryLines(self, c:str, pg:pd.DataFrame):
        '''get the list of printed lines that have this character, only taking one in-progress write line per written line'''
        if c=='o':
            return pg[(pg.name.str.contains('o1'))|(pg.name.str.contains(f'o{self.numOtimes}'))]
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
        

    def getProgDims(self, diag:bool=False):
        '''convert the full position table to a list of timings'''
        self.importProgPos()
        self.importFlagFlip()
        self.initializeProgDims('XS' in self.sbp)

        if 'Horiz' in self.sbp:
            wlines, dlines = self.splitProgPos('+y', diag=diag)
        elif 'Vert' in self.sbp:
            wlines, dlines = self.splitProgPos('+z', diag=diag)
        elif 'XS' in self.sbp:
            wlines, dlines = self.splitProgPos('-x', diag=diag)
        otimes = self.otimes
        
        wprog = self.getNumLines('w')
        if not len(wlines)==wprog:
            raise ValueError(f'Mismatch in write lines: {len(wlines)} found, {wprog} programmed')

        dprog = self.getNumLines('d')
        if not len(dlines)==dprog:
            raise ValueError(f'Mismatch in disturb lines: {len(dlines)} found, {dprog} programmed')
            
        oprog = self.getNumLines('o')
        if not len(otimes)==oprog:
            raise ValueError(f'Mismatch in observe lines: {len(otimes)} found, {oprog} programmed')
        
        oi = 0
        # assign written lines
        wi = 0 # 
        di = 0
        for j in range(self.numLines):  
            for l in self.ll:
                if l[0]=='w':
                    wi, oi = self.labelSublist(j, l, wi, oi, wlines, otimes)
                else:
                    di, oi = self.labelSublist(j, l, di, oi, dlines, otimes)
        self.checkProgDims()  # check that all values are correct
        if diag:
            display(self.progDims)
            
            
    def checkProgDimsXS(self):
        '''check progdims for xs'''
        bn = os.path.basename(self.printFolder)
        spl = re.split('_', bn)
        dire = spl[2]
        num = int(spl[1])
        spacing = float(spl[-1])
        intendedSpacing = self.pv.dEst*spacing
        scrit = 0.1
        if len(self.progDims.xpic.unique())>1:
            raise ValueError(f'Too many xpic locations in {self.printFolder}')
            
        for gnum in range(4):
            gp = self.progDims[self.progDims.name.str.startswith(f'l{gnum}')]
            
            # all z positions in one group should be same for +y prints
            if dire=='+y':
                zlin = len(gp.zpic.unique())
                if zlin>1:
                    raise ValueError(f'Too many zpic locations in {self.printFolder} group {gnum}: {zlin}')
            
            # all observe points in one group should be same
            olines = gp[gp.name.str.contains('o')]   # observe lines
            for ss in ['ypic', 'zpic']:
                zlin = len(olines[ss].unique())
                if zlin>1:
                    raise ValueError(f'Too many {ss} observe locations in {self.printFolder} group {gnum}: {zlin}')
                    
            dwlines = gp[~gp.name.str.contains('o')]
            if dire=='+y':
                spacings = dwlines.ypic.diff()
            elif dire=='+z':
                spacings = dwlines.zpic.diff()
                zlin = len(dwlines.ypic.unique())
                if zlin>1:
                    raise ValueError(f'Too many ypic write locations in {self.printFolder} group {gnum}: {zlin}')
                
            for sp in spacings:
                if sp>intendedSpacing+scrit or sp<intendedSpacing-scrit:
                    raise ValueError(f'Bad spacing in {self.printFolder} group {gnum}: {sp}/{intendedSpacing}')

        
    def checkProgDimsVert(self):
        '''check progdims for Vert'''
        bn = os.path.basename(self.printFolder)
        spl = re.split('_', bn)
        num = int(spl[1])
        spacing = float(spl[-1])
        intendedSpacing = self.pv.dEst*spacing
        scrit = 0.1
            
        for gnum in range(4):
            gp = self.progDims[self.progDims.name.str.startswith(f'l{gnum}')]
            
            # all observe points in one group should be same
            olines = gp[gp.name.str.contains('o')]   # observe lines
            for ss in ['xpic', 'ypic', 'zpic']:
                zlin = len(olines[ss].unique())
                if zlin>1:
                    print(olines)
                    raise ValueError(f'Too many {ss} observe locations in {self.printFolder} group {gnum}: {zlin}')
                    
            # dwlines = gp[~gp.name.str.contains('o')]
            dwlines = pd.concat([self.getPrimaryLines('w', gp), self.getPrimaryLines('d', gp)])
            
            spacings = dwlines.ypic.diff()
            for sp in spacings:
                if sp>intendedSpacing+scrit or sp<intendedSpacing-scrit:
                    raise ValueError(f'Bad spacing in {self.printFolder} group {gnum}: {sp}/{intendedSpacing}')
        
    def checkProgDimsHoriz(self):
        '''check progdims for horiz'''
        bn = os.path.basename(self.printFolder)
        spl = re.split('_', bn)
        num = int(spl[1])
        spacing = float(spl[-1])
        intendedSpacing = self.pv.dEst*spacing
        scrit = 0.1
            
        for gnum in range(4):
            gp = self.progDims[self.progDims.name.str.startswith(f'l{gnum}')]
            
            # all observe points in one group should be same
            olines = gp[gp.name.str.contains('o')]   # observe lines
            for ss in ['xpic', 'ypic', 'zpic']:
                zlin = len(olines[ss].unique())
                if zlin>1:
                    raise ValueError(f'Too many {ss} observe locations in {self.printFolder} group {gnum}: {zlin}')
                    
            dwlines = pd.concat([self.getPrimaryLines('w', gp), self.getPrimaryLines('d', gp)])
            for ss in ['xpic', 'ypic']:
                zlin = len(olines[ss].unique())
                if zlin>1:
                    raise ValueError(f'Too many {ss} write locations in {self.printFolder} group {gnum}: {zlin}')
            spacings = dwlines.zpic.diff()
            for sp in spacings:
                if sp>intendedSpacing+scrit or sp<intendedSpacing-scrit:
                    raise ValueError(f'Bad spacing in {self.printFolder} group {gnum}: {sp}/{intendedSpacing}')
            
            
    def checkProgDims(self):
        '''check that the programmed picture locations match expections'''
        if not all(self.progDims.sort_values(by='tpic').index==self.progDims.index):
            print(self.progDims[self.progDims.sort_values(by='tpic').index!=self.progDims.index])
            raise ValueError('Times are out of order')
        if pd.isna(self.progDims.loc[0,'l']):
            raise ValueError('Missing values in progDims')
        bn = os.path.basename(self.printFolder)
        if 'disturbXS2' in bn:
            self.checkProgDimsXS()
        elif 'disturbVert2' in bn:
            self.checkProgDimsVert()
        elif 'disturbHoriz3' in bn:
            self.checkProgDimsHoriz()
        else:
            logging.warning(f'Cannot check prog dims: unexpected shopbot file name {bn}')
            
    def findAnotherFolder(self) -> pd.DataFrame:
        '''find another folder to steal from'''
        topfolder = os.path.dirname(os.path.dirname(os.path.dirname(self.printFolder)))  # singleDoubleTriple\\SO_S85-0.05
        for f1 in os.listdir(topfolder):
            f1f = os.path.join(topfolder, f1)   # I_SO8-S85-0.05_S_4.00
            if not f1f in self.printFolder:
                for f2 in os.listdir(f1f):
                    f2f = os.path.join(f1f, f2)  # I_SO8-S85-0.05_S_4.00_230511
                    f3f = os.path.join(f2f, os.path.basename(self.printFolder))
                    if os.path.exists(f3f):
                        pfd2 = fh.printFileDict(f3f)
                        if os.path.exists(pfd2.timeRewrite):
                            # print(f3f)
                            ft2,_ = plainIm(pfd2.timeRewrite)
                            overshoots = ft2[((ft2.targetLine)>(ft2.targetLine.shift(-1)))]
                            if len(overshoots)==0:
                                return ft2
        raise ValueError('No folder to steal from')
        
    def targetdxdydz(self, pt1:pd.Series, pt2:pd.Series) -> dict:
        '''difference between target points of pt1 and pt2'''
        return dict([[s, pt1[f'{s}_target']-pt2[f'{s}_target']] for s in ['x', 'y', 'z']])
    
    def sameDirection(self, row:pd.Series, dxdydz:dict, pt:pd.Series) -> bool:
        '''determine if this row is within the move'''
        dvals = [row[f'{s}d']-pt[f'{s}_target'] for s in ['x', 'y', 'z']]
        if abs(dvals[0])<0.01 and abs(dvals[1])<0.01 and abs(dvals[2])<0.01:
            # we hit the point
            return False
        for s in ['x', 'y', 'z']:
            dd = dxdydz[s]
            if dd>0.01:
                if row[f'{s}d']-pt[f'{s}_target']>dd+0.01:
                    # difference between display and target is bigger than size of move
                    return False
            elif dd<-0.01:
                if row[f'{s}d']-pt[f'{s}_target']<dd-0.01:
                    # difference between display and target is bigger than size of move
                    return False
            else:
                if abs(row[f'{s}d']-pt[f'{s}_target'])>0.01:
                    return False
        return True
    
    def overwriteSection(self, tl:int, jj:int, ii:int, lastBad:int, ft2:pd.DataFrame, ft1:pd.DataFrame) -> int:
        pt = ft2[ft2.targetLine==tl].iloc[0]
        prevpt = ft2[ft2.targetLine==(ft2[ft2.targetLine<tl].targetLine.max())].iloc[0]
        dxdydz = self.targetdxdydz(prevpt, pt)
        while self.sameDirection(ft1.loc[jj], dxdydz, pt) and jj<lastBad:
            jj = jj+1
        # print(ii,jj, dxdydz, tl)
        for s in ['x', 'y', 'z']:
            # overwrite the target values
            self.ftable.loc[ii:jj, f'{s}t'] = pt[f'{s}_target']
        self.ftable.loc[ii:jj, 'targetLine'] = tl
        self.ftable.loc[ii:jj, 'speed'] = pt['speed']
        return jj
            
    def stealTargets(self, mode:int=0):
        '''previous attempts have failed, and the time table can't be corrected without outside knowledge. copy it from another folder'''
        ft2 = self.findAnotherFolder()
        ft2 = ft2[ft2.z_target<0]
        ft1 = self.ftable[self.ftable.zt<0]
        tlu1 = ft1.targetLine.unique()
        tlu2 = ft2.targetLine.unique()
#         print(tlu1)
#         print(tlu2)
        
        # print(overshoots)
        if mode==0:
            # overwrite only bad points
            overshoots = self.getOvershoots(trust=False)
            for i,row in overshoots.iterrows():
                tlnext = self.ftable.loc[i+1, 'targetLine']
                bad = self.ftable.loc[:i]
                bad = bad[bad.targetLine>tlnext]
                firstBad = bad.iloc[0].name          # find index in ftable of first bad row. i is the last bad row
                tlprev = self.ftable.loc[firstBad-1, 'targetLine']
                lastBad = bad.iloc[-1].name

                # print(tlprev, tlnext,  i)
                missing = tlu2[(tlu2>tlprev)&(tlu2<tlnext)]   # find missing steps
                # print(missing)
                ii = firstBad

                for tl in missing:
                    jj = ii
                    jj = self.overwriteSection(tl, jj, ii, lastBad, ft2, ft1)
                    ii = jj+1
        elif mode==1: 
            # overwrite all points
            ii = ft1.iloc[0].name

            for tl in tlu2[(tlu2>=min(tlu1))&(tlu2>min(tlu2))]:
                jj = ii
                pt = ft2[ft2.targetLine==tl].iloc[0]
                prevpt = ft2[ft2.targetLine==(ft2[ft2.targetLine<tl].targetLine.max())].iloc[0]
                dxdydz = self.targetdxdydz(prevpt, pt)
                while self.sameDirection(ft1.loc[jj], dxdydz, pt) and jj<len(ft1):
                    jj = jj+1
                # print(ii,jj, dxdydz, tl)
                for s in ['x', 'y', 'z']:
                    # overwrite the target values
                    self.ftable.loc[ii:jj, f'{s}t'] = pt[f'{s}_target']
                self.ftable.loc[ii:jj, 'targetLine'] = tl
                self.ftable.loc[ii:jj, 'speed'] = pt['speed']
                ii = jj+1
        # print(self.ftable.targetLine.unique())
            
        