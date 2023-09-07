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

class timeRewriteChecker:
    '''this is for checking and correcting errors in timeRewrite files'''
    
    def __init__(self, printFolder:str, ftable:pd.DataFrame, getBlips:bool=True, **kwargs):
        self.printFolder = printFolder
        self.ftable = ftable
        self.ftable.loc[0, 'targetLine'] = 0
        if getBlips:
            self.getBlips()
        self.checkOvershoots()
            
        self.ftable.drop(columns=['xt_orig', 'yt_orig', 'zt_orig'], inplace=True)
        if len(self.getOvershoots(trust=False))>0:
            # we still have some out of order points. steal from another folder
            self.stealTargets()
        self.rewritten = True
        
    def getTargetList(self, stopPoints:list, imin:int, imax:int):
        '''get the endpoint of the current run for replacement'''
        stopPoint = min(list(filter(lambda x:x>imin, stopPoints)))
        stopPoint = min(stopPoint-1, imax-1)
        #stopPoint = stopPoint-1
        targets = list(self.ftable.loc[imin:stopPoint].targetLine.unique())
        return stopPoint, targets
    
    def findAnotherFolder(self, doc:str='timeRewrite') -> pd.DataFrame:
        '''find another folder to steal from'''
        topfolder = os.path.dirname(os.path.dirname(os.path.dirname(self.printFolder)))  # singleDoubleTriple\\SO_S85-0.05
        for f1 in os.listdir(topfolder):
            f1f = os.path.join(topfolder, f1)   # I_SO8-S85-0.05_S_4.00
            if not f1f in self.printFolder:
                ll = os.listdir(f1f)
                ll.reverse()
                for f2 in ll:
                    f2f = os.path.join(f1f, f2)  # I_SO8-S85-0.05_S_4.00_230511
                    f3f = os.path.join(f2f, os.path.basename(self.printFolder))
                    if os.path.exists(f3f):
                        pfd2 = fh.printFileDict(f3f)
                        if os.path.exists(getattr(pfd2, doc)):
                            # print(f3f)
                            ft2,_ = plainIm(getattr(pfd2, doc))
                            if doc=='timeRewrite':
                                overshoots = ft2[((ft2.targetLine)>(ft2.targetLine.shift(-1)))]
                                if len(overshoots)==0:
                                    return ft2
                            else:
                                return ft2
        raise ValueError('No folder to steal from')
    
    def getStolenTargets(self, overwrite:bool=False):
        '''get a list of targets to steal'''
        if hasattr(self, 'stolenTargets') and not overwrite:
            return
        ft2 = self.findAnotherFolder()
        self.stolenTargets = ft2.groupby(['x_target', 'y_target', 'z_target', 'targetLine', 'speed']).size().reset_index(name='Freq')
        self.stolenTargets.rename(columns={'x_target':'xt', 'y_target':'yt', 'z_target':'zt'}, inplace=True)
        self.stolenTargets.sort_values(by='targetLine', inplace=True)
        
    def correctMovement(self, j0:int, jf:int) -> None:
        '''add movements to the target list'''
        
        f1 = self.ftable.loc[j0-1:jf+2].copy()
        for s in ['x', 'y', 'z']:
            f1[f'd{s}'] = np.sign(f1[f'{s}d'].shift(-1)-f1[f'{s}d'])   # sign of the change in value
        f1 = f1[(~((f1.dx==0)&(f1.dy==0)&(f1.dz==0)))]
        for s in ['x', 'y', 'z']:    
            f1[f's{s}'] = abs(f1[f'd{s}'].shift()-f1[f'd{s}'])       # difference between this change and the previous change
        f1['s'] = f1['sx']+f1['sy']+f1['sz']
        pivots = f1[((f1.s>0))&(~((f1.dx==0)&(f1.dy==0)&(f1.dz==0)))]  # points where the stage changes direction
        i0 = j0
        if len(pivots)>0:
            self.getStolenTargets()
        else:
            return
        st = self.stolenTargets.copy()
        target0 = self.ftable.loc[j0-1,'targetLine']
        targetf = self.ftable.loc[jf+1,'targetLine']
        st = st[(st.targetLine>=target0)&(st.targetLine<=targetf)]
        if len(st)==0:
            return
        st = st.copy()
        for i,row in pivots.iterrows():
            for s in ['x', 'y', 'z']:
                st[f'd{s}'] = [x-row[f'{s}d'] for x in st[f'{s}t']]
            st['distance'] = st['dx']**2+st['dy']**2+st['dz']**2
            closest = st[st.distance==st.distance.min()]
            if closest.iloc[0]['distance']<2:
                # print(row['xd'], row['yd'], row['zd'])
                # print(closest)
                for s in ['xt', 'yt', 'zt', 'targetLine', 'speed']:
                    self.ftable.loc[i0:i, s] = closest.iloc[0][s]
            i0 = i+1
        for s in ['xt', 'yt', 'zt', 'targetLine', 'speed']:
            self.ftable.loc[i:jf, s] = closest.iloc[0][s]
            
    def checkForMovement(self, j0:int, jf:int) -> None:
        '''check if there is stage movement where the stage is currently labeled as staying still'''
        t1 = self.ftable.loc[j0:jf+1]
        for s in ['x', 'y', 'z']:
            if not len(t1[f'{s}t'].unique())==len(t1[f'{s}d'].unique()) or abs(t1.iloc[0][f'{s}t']-t1.iloc[0][f'{s}d'])>1:
                return self.correctMovement(j0, jf)
            
            
    def correctBlip(self, blip:pd.Series):
        '''erase the effect of the blip on the targets by going from the back and pulling targets forward'''
        fc = self.flagChanges
        j = blip.name
        flag = blip['flag']
        nexts = fc[(fc.flag==flag)].loc[j:]
        others = fc[(fc.flag.isin(self.onFlags.difference(set([flag]))))]
        stopPoints = list(others.i0) + [nexts.iloc[-1]['i0']]
        # go from the back and pull targets forward
        i = len(nexts)-2
        if i<0:
            return
        while i>=0:
            if i==0:
                row = fc.loc[j]
                prev = {'i0':row['i0']-2}
            else:
                row = nexts.iloc[i]
                prev = nexts.iloc[i-1]
            nex = nexts.iloc[i+1]
            stopPoint, targets = self.getTargetList(stopPoints, row['i0'], nex['i0'])
            prevStop, prevTargets = self.getTargetList(stopPoints, prev['i0'], row['i0'])
            for t in targets:
                # steal the target from the previous point
                tspan = list(self.ftable[self.ftable.targetLine==t].index)
                tspan = list(filter(lambda x:(x>=row['i0'])&(x<=stopPoint), tspan))
                if len(prevTargets)>0:
                    pt = prevTargets.pop(0)
                    pt0 = self.ftable[self.ftable.targetLine==pt].loc[prev['i0']:prevStop]
                    pt0 = pt0.iloc[0]
                if pt0['targetLine']==60:
                    print(pt0)
                for s in ['targetLine', 'xt', 'yt', 'zt']:
                    self.ftable.loc[tspan, s]=[pt0[s] for i in tspan]
            st = str(self.ftable.loc[row['i0'],'status'])
            if 'SNAP' in st or 'SNOFF' in st:
                # this is 2 snaps in a row
                self.checkForMovement(row['i0'], stopPoint)
            i = i-1
        # get rid of annotation
        self.ftable.loc[row['i0'], 'status'] = 'erased: 8 flag on'
        i = row['i0']+1
        while not 'SNOFF' in str(self.ftable.loc[i, 'status']) and i<nex['i0']:
            i = i+1
        if 'SNOFF' in str(self.ftable.loc[i,'status']):
            self.ftable.loc[i, 'status'] = 'erased: 8 flag off'

    def getBlips(self):
        '''find places where the flag only turned on briefly in error and shift the targets all the way down'''
        fc = []  # table of points where the flag changes
        f1 = self.ftable
        for i, g in f1.groupby([(f1.flag != f1.flag.shift()).cumsum()]):
            fc.append({'flag':g.iloc[0]['flag'], 'n':len(g), 'i0':g.iloc[0].name})
        fc = pd.DataFrame(fc)
        fc = fc[(fc.i0>0)&(fc.flag>0)]
        self.flagChanges = fc
        blips = fc[fc.n<3]
        if len(blips)==0:
            return
        self.flags = fc.flag.unique()
        self.off = min(self.flags)
        self.onFlags = set(self.flags).difference(set([self.off]))
        for i,blip in blips.iterrows():
            self.correctBlip(blip)
            
    #--------------------------------------------
            
        
    def getOvershoots(self, trust:bool=True) -> pd.DataFrame:
        f = self.ftable.loc[1:]
        if trust:
            return f[((f.targetLine)>(f.targetLine.shift(-1)))&(~f.trusted)]
        else:
            return f[((f.targetLine)>(f.targetLine.shift(-1)))]
        
    def checkOvershoots(self) -> None:
        '''check for points where the shopbot has backtracked on targets'''
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
        
    #--------------------------------------------
        
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
        
        