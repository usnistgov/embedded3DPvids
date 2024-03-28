#!/usr/bin/env python
'''Functions for handling tables of programmed timings for disturbed single lines'''

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

class progDimsSingleDisturb(progDim):
    '''for programmed dimensions of single disturb prints'''
    
    def __init__(self, printFolder:str, pv:printVals, **kwargs):
        super().__init__(printFolder, pv)
        
    def importSBPPoints(self) -> None:
        '''find the programmed shopbot points'''
        file = self.pfd.sbpPointsFile()
        if not os.path.exists(file):
            self.sbpPoints = []
        else:
            self.sbpPoints = pd.read_csv(file, index_col=0)
            
    def samePoint(self, target:pd.Series, blank:pd.Series):
        '''determine if this is the same point'''
        dist = np.sqrt(sum([(target[f'{s}t']-blank[f'{s}t'])**2 for s in ['x', 'y', 'z']]))
        return dist<0.01
    
    
    def combineTargets(self, ti:int, si:int, targets:list, diag:int=0) -> None:
        '''combine these targets into a single dataframe'''
        timeT = self.timeT
        sT = self.sbpT
        if diag>0:
            print(pd.concat([timeT, sT], axis=1, keys=['time', 'sbp']))
            print('ti si tii sii tpoint\tsbppoint')
        while ti<len(timeT) and si<len(sT):
            tii = ti
            sii = si
            tp = [timeT.loc[tii]]
            sp = [sT.loc[sii]]
            while (not self.samePoint(tp[0],sp[-1])) and (not self.samePoint(tp[-1], sp[0])) and tii<len(timeT)-1 and sii<len(sT)-1:
                # loop until we find a match
                tii+=1
                sii+=1
                tp.append(timeT.loc[tii])
                sp.append(sT.loc[sii])
            if self.samePoint(tp[0],sp[-1]):
                # same point or extra shopbot points 
                if diag>0:
                    print(ti, si, ti, sii, list(tp[0]), list(sp[-1]), 'extra shopbot')
                targets = targets+sp
                ti+=1
                si+=len(sp)
            elif self.samePoint(tp[-1],sp[0]):
                # extra time points
                if diag>0:
                    print(ti, si, tii, si, list(tp[0]), list(sp[-1]), 'extra time')
                targets = targets + tp
                si+=1
                ti+=len(tp)
            else:
                if diag>0:
                    print(pd.concat([timeT, sT], axis=1, keys=['time', 'sbp']))
                    print(pd.DataFrame(targets))
                    print(self.mer)
                raise ValueError('Could not consolidate sbp points and time file')

        self.targetPoints = pd.DataFrame(targets)
        if si<len(sT):
            # extra shopbot points. add to end
            self.targetPoints = pd.concat([self.targetPoints, sT.loc[si+1:]])
                
        self.targetPoints.reset_index(drop=True, inplace=True)


    def getTargetPoints(self, diag:int=0) -> None:
        '''read the SBP programmed points into a pandas dataframe'''
        # get target points from fluigent table
        if not hasattr(self, 'ftable'):
            self.importTimeFile()
        timeT = uniqueConsecutive(self.ftable, ['xt_orig', 'yt_orig', 'zt_orig'])  # list of target points from time file
        timeT.rename(columns={'xt_orig':'xt', 'yt_orig':'yt', 'zt_orig':'zt'}, inplace=True)
        
        # get target points from sbp file
        self.importSBPPoints()
        self.sbpPoints.fillna({'x':timeT.loc[0, 'xt'], 'y':timeT.loc[0, 'yt'], 'z':timeT.loc[0, 'zt']}, inplace=True)
        sT = uniqueConsecutive(self.sbpPoints, ['x', 'y', 'z', 'speed'])   # list of target points from shopbot file
        sT.rename(columns={'x':'xt', 'y':'yt', 'z':'zt'}, inplace=True)
        
        # compare lists
        tg = timeT.groupby(['xt', 'yt', 'zt']).size().reset_index().rename(columns={0:'count'})
        sg = sT.groupby(['xt', 'yt', 'zt']).size().reset_index().rename(columns={0:'count'})
        self.mer = pd.merge(tg, sg, on=['xt', 'yt', 'zt'], how='outer', suffixes=('_f', '_s'))
        if self.mer['count_s'].isna().sum()>4:
            # lots of mismatched points. use target points from fluigent table
            self.targetPoints = timeT
            return
        elif self.mer['count_f'].isna().sum()>0 and self.mer['count_s'].isna().sum()==0:
            self.targetPoints = sT
            return
        
        
        self.timeT = timeT
        self.sbpT = sT
        
        try:
            self.combineTargets(0, 0, [], diag=diag)  
        except ValueError:
            if self.mer['count_s'].isna().sum()>0:
                # fluigent points that aren't in sbp: automatically adopt 1st point and run again
                self.combineTargets(1, 0, [timeT.loc[0]], diag=diag-1)  

        return self.targetPoints
    
    def rewriteFRows(self, diag:int=0) -> int:
        '''rewrite targets in the time table'''
        spi = 0
        cp = self.targetPoints.loc[spi]  # current target point
        r0 = 0
        dprev = 1000
        hit = False
        dprog = 0
        fi = 0
        while fi<len(self.ftable):
            fi+=1
            row = self.ftable.loc[fi]
            d = np.sqrt((row['xd']-cp['xt'])**2+(row['yd']-cp['yt'])**2+(row['zd']-cp['zt'])**2)
            self.ftable.loc[fi, 'ldt'] = d   # store distance
            if diag>1:
                print(fi, list(row[['xd', 'yd', 'zd']]), list(cp), d, dprev)
            if not hit and d<3 or dprev>3:
                hit=True
                
            if d>dprev and hit:
                if diag>0:
                    print(fi, list(row[['xd', 'yd', 'zd']]), list(cp), d, dprev)
                # started to move away. reset target
                for s in ['xt', 'yt', 'zt', 'speed']:
                    self.ftable.loc[r0:fi-1, s] = cp[s]
                    
                f11 = self.ftable.loc[r0:fi-1]
                if f11.flag.max()>2048 and f11[f11.flag>2048].index.min()<fi-5:
                    # flag on during this run that turns on more than 5 time steps before end of chunk. compensate
                    t0 = self.ftable.loc[r0, 'time']
                    dtrav = (self.ftable.loc[fi-1, 'time'] - t0)*cp['speed']
                    if dtrav<dprog:
                        # ended point too early
                        tf = t0 + (dprog+1)/cp['speed']  # anticipated final time
                        fi2 = (self.ftable['time']-tf).abs().argsort()[0]   # get row that is closest to that time
                        for s in ['xt', 'yt', 'zt', 'speed']:
                            self.ftable.loc[fi-1:fi2-1, s] = cp[s]
                        # check if we hit other points during the overwritten part
                        
                        check = True
                        fii=0
                        while check and fi2+fii<len(self.ftable):
                            spi+=1
                            
                            skip = self.ftable.copy()
                            skip = skip.loc[fi-1:fi2+fii-1]
                            cp2 = self.targetPoints.loc[spi]  # next target point
                            skip['ldt'] = np.sqrt((skip['xd']-cp2['xt'])**2+(skip['yd']-cp2['yt'])**2+(skip['zd']-cp2['zt'])**2)   # find distance to point
                            lmin = skip.ldt.min()
                            if lmin<3 and skip[skip.ldt==lmin].index.min()<fi2-1:
                                # hit point and then moved away. keep one point
                                for s in ['xt', 'yt', 'zt', 'speed']:
                                    self.ftable.loc[fi2+fii, s] = cp2[s]
                                fii+=1
                            else:
                                check = False
                        spi = spi-2                      
                        
                        fi = fi2+fii

                spi+=1
                
                # check position
                if spi==len(self.targetPoints)-1:
                    # we've hit the last point
                    self.ftable.loc[r0:, 'speed'] = cp['speed']
                    return 0
                elif spi>=len(self.targetPoints):
                    # reset the table and return
                    print(self.mer)
                    logging.error(f'Failed to rewrite targets in {self.printFolder}: ran out of targets')
                    self.importTimeFile()
                    return 1
                
                # get next point
                cp2 = self.targetPoints.loc[spi]  # current target point
                if np.sqrt((cp2['xt']-cp['xt'])**2+(cp2['yt']-cp['yt'])**2+(cp2['zt']-cp['zt'])**2)<0.01:
                    if cp2['speed']==0:
                        # mark pause using speed
                        f1 = self.ftable.loc[r0:fi-1]
                        self.ftable.loc[f1[f1.ldt<0.01].index, 'speed'] = 0
                    spi+=1   # target point is the same. skip
                    cprev = cp
                    cp = self.targetPoints.loc[spi]  # current target point
                    
                else:
                    cprev = cp
                    cp = cp2
                    
                # get programmed distance
                dprog = np.sqrt((cp['xt']-cprev['xt'])**2+(cp['yt']-cprev['yt'])**2+(cp['zt']-cprev['zt'])**2)
                    
                r0 = fi # reset row counter
                hit = False
                dprev = np.sqrt((row['xd']-cp['xt'])**2+(row['yd']-cp['yt'])**2+(row['zd']-cp['zt'])**2)
            else:
                dprev = d
                
        if spi+1<len(self.targetPoints):
            logging.error(f'Failed to rewrite targets in {self.printFolder}: did not hit last {len(self.targetPoints)-spi-1} targets')
            self.importTimeFile()
            return 1
                
    def getTimeRewrite(self, diag:int=0) -> int:
        '''overwrite the target points in the time file'''
        super().getTimeRewrite(diag=diag)
        self.getTargetPoints(diag=diag)
        out = self.rewriteFRows(diag=diag)
        if out>0:
            return out

#         # distance between displayed and target points
#         self.ftable['ldt'] = np.sqrt((self.ftable['xt']-self.ftable['xd'])**2 +(self.ftable['yt']-self.ftable['yd'])**2 +(self.ftable['zt']-self.ftable['zd'])**2 )
        
        self.rewritten = True
        return 0
        
    def initializeProgDims(self):
        '''initialize programmed dimensions table'''
        super().initializeProgDims()

        self.progDims.name = ['l0w', 'l0wo', 'l0d', 'l0do', 
                                  'l1w', 'l1wo', 'l1d', 'l1do', 
                                  'l2w', 'l2wo', 'l2d', 'l2do', 
                                  'l3w', 'l3wo', 'l3d', 'l3do']
        
    def correctZeroJog(self):
        '''remove the zero jog target point from the time table'''
        maxyt = self.ftable.yt.max()
        self.ftable.loc[self.ftable.yt==(maxyt-1),'yt'] = maxyt
        
    def splitProgPos(self, moveDir:str, diag:bool=False):
        '''convert list of positions for horizontal lines'''
        if not hasattr(self, 'progPos'):
            raise ValueError(f'{self.printFolder}: progPos not created')
        pdp = self.progPos.copy()
        p = moveDir[0]
        md = moveDir[1:]
        if moveDir=='-x':
            vlines = pdp[(pdp.dx<0)&(pdp.zt<0)]
            olines = pdp[(pdp.dx>0)&(pdp.zt<0)]
            mval = vlines[f'd{md}'].min()
            omval = -mval/2
        elif moveDir=='+y':
            vlines = pdp[(pdp.dy>0)&(pdp.zt<0)]
            olines = pdp[(pdp.dy<0)&(pdp.zt<0)]
            mval = vlines[f'd{md}'].max()
            omval = -mval/2
        elif moveDir=='+z':
            vlines = pdp[(pdp.dz>0)&(pdp.zt<0)]
            olines = vlines
            mval = vlines[f'd{md}'].max()
            omval = mval/2

        vlongLines = vlines[vlines[f'd{md}']==mval]
        olines = olines[olines[f'd{md}']==omval]
        if moveDir=='-x':
            olines = olines[olines.xt==olines.xt.min()]
        elif moveDir=='+y':
            olines = olines[olines.yt==olines.yt.max()]
        elif moveDir=='+z':
            if len(vlongLines)>8:
                zt1 = vlongLines.zt.mode()
                if len(zt1)==1:
                    print(self.printFolder)
                    print(vlongLines)
                    print(olines)
                    zt2 = vlongLines[vlongLines.zt!=zt1[0]].zt.mode()
                    zt1 = pd.concat(zt1, zt2)
                vlongLines = vlongLines[(vlongLines.zt==zt1[0])|(vlongLines.zt==zt1[1])]
        if not len(vlongLines)==8 or not len(olines)==8:
            if (len(vlongLines)<8 or len(olines)<8):
                # reset targets in time table and try again
#                 self.restorePosTargets()
                if not 'xt_orig' in self.ftable:
                    self.getTimeRewrite(diag=diag)
                    return self.splitProgPos(moveDir, diag=diag)
            if diag:
                print(vlongLines)
                print(olines)
            raise ValueError(f'Wrong number of lines: {len(vlongLines)} long, {len(olines)} observe')
            
        wlines = vlongLines.iloc[::2]
        dlines = vlongLines.iloc[1::2]
        
        return wlines, dlines, olines
        
    def getProgDims(self, diag:bool=False):
        '''convert the full position table to a list of timings'''
        if not hasattr(self, 'progPos'):
            self.importProgPos()
        
        if 'Horiz' in self.sbp:
            wlines, dlines, olines = self.splitProgPos('+y', diag=diag)
        elif 'Vert' in self.sbp:
            wlines, dlines, olines = self.splitProgPos('+z', diag=diag)
        elif 'XS' in self.sbp:
            wlines, dlines, olines = self.splitProgPos('-x', diag=diag)
            
        for x in [['w', wlines], ['d', dlines]]:
            j=0
            cha = x[0]
            for i,line in x[1].iterrows():
                # store values
                for y in ['l', 'w', 't', 'a', 'vol', 't0', 'tf']:
                    self.progDims.loc[self.progDims['name']==f'l{j}{cha}',y] = line[y]

                # determine where to take pic
                tpic = line['t0']+line['dprog']*0.5/line['speed']
                self.progDims.loc[self.progDims['name']==f'l{j}{cha}','tpic'] = tpic
                self.progDims.loc[self.progDims['name']==f'l{j}{cha}','lprog'] = line['dprog']
                self.progDims.loc[self.progDims['name']==f'l{j}{cha}','ltr'] = line['dtr']
                self.progDims.loc[self.progDims['name']==f'l{j}{cha}','speed'] = line['speed']
                
                # select observation line
                olines0 = olines[olines.t0>line['t0']]
                olines0 = olines0[olines0.t0==olines0.t0.min()]
                for y in ['l', 'w', 't', 'a', 'vol', 't0', 'tf']:
                    self.progDims.loc[self.progDims['name']==f'l{j}{cha}o',y] = olines0.iloc[0][y]
                self.progDims.loc[self.progDims['name']==f'l{j}{cha}o','tpic'] = olines0.iloc[0]['tf']-1
                j+=1
         