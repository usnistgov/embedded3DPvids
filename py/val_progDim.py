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
from plainIm import *
from val_fluid import *
import file_handling as fh
from val_pressure import pressureVals
from val_geometry import geometryVals
from val_print import printVals

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)



#----------------------------------------------

#--------------------    
#--------------------

def measureTRD(pp:pd.Series) -> float:
    '''measure the distance between the read and target distance'''
    return np.sqrt((pp['xd']-pp['xt'])**2+(pp['yd']-pp['yt'])**2+(pp['zd']-pp['zt'])**2)

def timeColumns(reverse:bool=False):
    d = {'Channel 0 pressure':'pressure', 'Channel_0_pressure':'pressure', 'x_disp':'xd', 'y_disp':'yd', 'z_disp':'zd', 'x_est':'xe', 'y_est':'ye', 'z_est':'ze','x_target':'xt','y_target':'yt','z_target':'zt', 'x':'xd', 'y':'yd', 'z':'zd'}
    if reverse:
        for s in ['x', 'y', 'z']:
            d.pop(s)
        return dict([[val,key] for key,val in d.items()])
    else:
        return d
    
def uniqueConsecutive(df:pd.DataFrame, cols:List[str]) -> pd.DataFrame:
    '''select unique values in the dataframe
    duplicates value can occur if the duplicates are not consecutive
    '''
    st = " | ".join(["(df['{0}'].shift(-1)!=df['{0}'])".format(col) for col in cols])
    df1 = df[eval(st)]
    df1.reset_index(inplace=True, drop=True)
    return df1.loc[:,cols]

class progDim:
    '''class that holds timing for the video'''
    
    def __init__(self, printFolder:str, pv:printVals, **kwargs):
        self.printFolder = printFolder
        self.pv = pv
        
        # dictionary of files in the printFolder
        self.pfd = pv.pfd
        self.geo = pv.geo
        self.press = pv.press
        self.rewritten=False
            
        self.sbp = self.pfd.sbpName()    # name of shopbot file
        self.progDims = pd.DataFrame(columns=['name','l','w','t','a','vol', 't0','tf'])

                
    #---------------------------------------------------
    # rewrite time file     
    
    def timeFile(self) -> str:
        '''get the name of the original pressure-time table'''
        if len(self.pfd.timeSeries)==0:
            return ''
        if len(self.pfd.timeSeries)==1:
            return self.pfd.timeSeries[0]
        if len(self.pfd.timeSeries)>1:
            l = []
            for f in self.pfd.timeSeries:
                if self.pfd.printType in ['singleDisturb', 'tripleLines']:
                    l.append(f)
                elif 'singleLinesN' in f and 'Fluigent' in f:
                    l.append(f)
            if len(l)==0:
                raise NameError(f'No fluigent file found in {self.printFolder}')
            l.sort()
            file = os.path.join(self.printFolder, l[-1]) # select last fluigent file
            return file
        
    def timeRewriteFile(self) -> str:
        '''programmed positions file name'''
        return self.pfd.newFileName('timeRewrite', 'csv')
        
    def importTimeFile(self) -> None:
        '''find and import the pressure-time table'''
        file = self.timeFile()
        if os.path.exists(file):
            self.ftable = pd.read_csv(file)
            self.ftable, self.ftableUnits = splitUnits(self.ftable)   # split the units out of the header      
            self.ftable.rename(columns=timeColumns(), inplace=True)  # shorten names
        else:
            self.ftable = []    
            self.ftableUnits = {}
        if self.sbp.startswith('disturbHoriz2'):
            # correct for zero jog point
            self.correctZeroJog()
        self.ftable.loc[:,'time'] = self.ftable.loc[:,'time']-self.ftable.time.min()
        self.createTargetOrig()   # put old targets in another column
        self.rewritten = False
        
    def importTimeRewriteFile(self) -> None:
        '''import the file with rewritten targets'''
        file = self.timeRewriteFile()
        if not os.path.exists(file):
            self.exportTime()
        self.ftable, self.ftableUnits = plainIm(file, ic=0)
        if len(self.ftable)>0:
            self.ftable.rename(columns=timeColumns(), inplace=True)  # shorten names
            self.rewritten=True
    
    
    def createTargetOrig(self) -> None:
        '''create a column for original targets'''
        if 'xt_orig' in self.ftable:
            return
        if not 'xt' in self.ftable:
            return
        for s in ['x', 'y', 'z']:
            self.ftable[f'{s}t_orig'] = self.ftable[f'{s}t']
            self.ftableUnits[f'{s}t_orig'] = self.ftableUnits[f'{s}_target']
        
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

            
    def rewriteTargets(self, diag:int=0) -> int:
        '''overwrite the target points in the time file'''
        self.importTimeFile()
        if not 'xt' in self.ftable:
            return 1
        
        self.getTargetPoints(diag=diag)
        lu = self.ftableUnits['x_target']
        tu = self.ftableUnits['time']
        self.ftableUnits['ldt'] = lu
        self.ftableUnits['speed'] = f'{lu}/{tu}'
        out = self.rewriteFRows(diag=diag)
        if out>0:
            return out

#         # distance between displayed and target points
#         self.ftable['ldt'] = np.sqrt((self.ftable['xt']-self.ftable['xd'])**2 +(self.ftable['yt']-self.ftable['yd'])**2 +(self.ftable['zt']-self.ftable['zd'])**2 )
        
        self.rewritten = True
        return 0
                
    def exportTime(self, overwrite:bool=False, diag:int=0):
        '''rewrite targets and export'''
        
        fn = self.timeRewriteFile()
        if os.path.exists and not overwrite:
            # file already exists
            return
        
        if not self.rewritten:
            self.rewriteTargets(diag=diag)
            
        ftable2 = self.ftable.copy()
        ftable2.rename(columns=timeColumns(reverse=True), inplace=True)
        plainExp(fn, ftable2, self.ftableUnits)
                
    
                
    #----------------------------------------------
    
    def progPosFile(self) -> str:
        '''programmed positions file name'''
        return self.pfd.newFileName('progPos', 'csv')
    
    def importProgPos(self, export:bool=True) -> None:
        '''import the programmed positions'''
        fnpos = self.progPosFile()
        if os.path.exists(fnpos):
            self.progPos, self.progPosUnits = plainIm(fnpos, ic=0)
        else:
            if export:
                self.writeProgPos() 
            else:
                self.progPos = []
                self.progPosUnits = {}

    def readProgGroup(self, v:pd.DataFrame) -> dict:
        '''summarize the group of time steps'''
        anoz = np.pi*(self.geo.di/2)**2 # inner cross-sectional area of nozzle           
        pp = self.pprev       # previous target point coords
        pt = v.iloc[0]        # current target point
        pospoints = v[v.pressure>0]  # get points where pressure is on
        translationSpeed = v.speed.max()
        
        # characterize step size, direction
        if 'xt' in pp:
            direc = [(pt['xt']-pp['xt']), (pt['yt']-pp['yt']), (pt['zt']-pp['zt'])]
            dist = np.sqrt((direc[0])**2+(direc[1])**2+(direc[2])**2)
        else:
            dist = np.nan
            direc = [np.nan, np.nan, np.nan]
#         t0 = v.time.min()
        t0 = self.t0
        tf = v.time.max()
        self.t0 = tf
        dtraveled = translationSpeed*(tf-t0)
        app = {'xt':pt['xt'], 'yt':pt['yt'], 'zt':pt['zt']
               , 'dx':direc[0], 'dy':direc[1], 'dz':direc[2]
               , 'dprog':dist, 'dtr':dtraveled, 't0':t0, 'tf':tf, 'speed':translationSpeed}
        if len(pospoints)>0:
            # we have flow
            tfflow = pospoints.time.max()
            t0flow = pospoints.time.min()
            ttot = pospoints.dt.sum()
            
            l = translationSpeed*ttot  
                # estimate length based on speed. 
                # don't use table estimates because they can be influenced by bad output from Sb3.exe
            volflux = [max(self.press.calculateSpeed(p['pressure'])*anoz*p['dt'], 0) for i,p in pospoints.iterrows()]  
                # convert pressure to volume flux using calibration curve

            vol = sum(volflux)
            if l==0:
                w = 2*(vol*3/4/np.pi)**1/3
                a = np.pi*(w/2)**2
            else:
                a = vol/l
                w = 2*np.sqrt(a/np.pi)
            app = {**app, **{'t0_flow':t0flow, 'tf_flow':tfflow, 'l':l, 'w':w, 't':ttot, 'a':a, 'vol':vol}}
        else:
            # no flow
            app = {**app, **{'t0_flow':np.nan, 'tf_flow':np.nan, 'l':0, 'w':0, 't':0, 'a':0, 'vol':0}}
        self.progPos.append(app)
        self.pprev = pt
        
    def getProgPos(self) -> None:
        '''read programmed dimensions from the fluigent table, where positions are listed in the table'''
        self.importTimeRewriteFile()
        if len(self.ftable)==0:
            return
        
        self.initializeProgDims()
        self.progPos = []
        
        self.ftable['dt'] = self.ftable['time']-self.ftable['time'].shift(1)  # get change in time
        
        # group points by target point, where steps must be consecutive
        grouping = self.ftable.groupby(((self.ftable['xt'].shift() !=self.ftable['xt'])|
                                        (self.ftable['yt'].shift() != self.ftable['yt'])|
                                        (self.ftable['zt'].shift() != self.ftable['zt'])).cumsum())
        self.pprev = []
        self.t0 = 0
        for k,v in grouping:
            self.readProgGroup(v)
        self.progPos = pd.DataFrame(self.progPos)
        
        # determine units
        lu = self.ftableUnits['x_target']
        tu = self.ftableUnits['time']
        self.progPosUnits = {'xt':lu, 'yt':lu, 'zt':lu, 'dx':lu, 'dy':lu, 'dz':lu, 'dprog':lu, 'dtr':lu,
                             't0':tu, 'tf':tu, 't0_flow':tu, 'tf_flow':tu, 'speed':f'{lu}/{tu}',
                            'l':lu, 'w':lu, 't':tu, 'a':f'{lu}^2', 'vol':f'{lu}^3'}
        
    def exportProgPos(self, overwrite:bool=False, diag:int=0) -> int:
        '''label programmed moves and export'''
        
        fn = self.progPosFile()
        if os.path.exists and not overwrite:
            # file already exists
            return 0
        
        if overwrite:
            self.getProgPos()
          
        if hasattr(self, 'progPos') and len(self.progPos)>0:
            plainExp(fn, self.progPos, self.progPosUnits)
            return 0
        else:
            return 1
        
        
    #----------------------
    
    def progDimsFile(self) -> str:
        '''get the name of the progDims file'''
        if len(self.pfd.progDims)>0:
            fn = self.pfd.progDims[0]
        else:
            fn = self.pfd.newFileName('progDims', 'csv')
        return fn
    
    def importProgDims(self, export:bool=True, overwrite:bool=False) -> str:
        '''import the progdims file'''
        fn = self.progDimsFile()
        if os.path.exists(fn):
            self.progDims, self.progDimsUnits = plainIm(fn, ic=0)
        else:
            if overwrite:
                self.writeProgDims()
            else:
                self.progDims = []
                self.progDimsUnits = {}
                
    def defineProgDimsUnits(self) -> None:
        '''define the progDims units'''
        if not hasattr(self, 'ftableUnits'):
            self.importTimeFile()
        if 'x_target' in self.ftableUnits:
            lu = self.ftableUnits['x_target']
        else:
            lu = 'mm'
        if 'time' in self.ftableUnits:
            tu = self.ftableUnits['time']
        else:
            tu = 's'
        self.progDimsUnits = {'t':tu, 't0':tu, 'tf':tu, 'tpic':tu, 'name':'', 'lprog':lu, 'ltr':lu,
                        'l':lu, 'w':lu, 'a':f'{lu}^2', 'vol':f'{lu}^3', 'speed':f'{lu}/{tu}'}
    
    def exportProgDims(self, overwrite:bool=False, diag:int=0) -> None:
        '''sort programmed moves into intended moves'''
        fn = self.progDimsFile()
        if os.path.exists and not overwrite:
            # file already exists
            return
        
        if overwrite:
            self.getProgDims()   # overridden function
            
        self.defineProgDimsUnits()
        plainExp(fn, self.progDims, self.progDimsUnits) 
        
    #---------------
    
    def exportAll(self, overwrite:bool=False, diag:int=0) -> None:
        '''export rewritten time, progPos, and progDims'''
        
        self.exportTime(overwrite=overwrite, diag=diag)
        self.exportProgPos(overwrite=overwrite,diag=diag)
        self.exportProgDims(overwrite=overwrite,diag=diag)
        

        
#----------------------------------------------  

class progDimsSingleLine(progDim):
    '''for programmed dimensions of single line prints'''
    
    def __init__(self, printFolder:str, pv:printVals, **kwargs):
        super().__init__(printFolder, pv)
        self.units = {'name':'', 'l':'mm','w':'mm','t':'s'
                      ,'a':'mm^2','vol':'mm^3','t0':'s', 'tf':'s'
                     ,'xt':'mm', 'yt':'mm', 'zt':'mm', 'dx':'mm', 'dy':'mm', 'dz':'mm', 'dprog':'mm'
                     , 't0_flow':'s', 'tf_flow':'s', 'tpic':'s'}
        self.fluigent()
        
    def progDimsSummary(self) -> Tuple[pd.DataFrame,dict]:
        if len(self.progDims)==0:
            self.importProgDims()
        if len(self.progDims)==0:
            return {},{}
        if 0 in list(self.progDims.vol):
            # redo programmed dimensions if there are zeros in the volume column
            self.redoSpeedFile()
            self.exportProgDims()
        df = self.progDims.copy()
        df2 = pd.DataFrame(columns=df.columns)
        for s in ['xs', 'vert', 'horiz']:
            df2.loc[s] = df[df.name.str.contains(s)].mean()
        df2 = df2.drop(columns=['name'])
        df2 = df2.T
        v = (df2).unstack().to_frame().sort_index(level=0).T
        v.columns = v.columns.map('_'.join) # single row dataframe
        vunits = dict([[k, self.units[re.split('_', k)[1]]] for k in v]) # units for this dataframe
        for s in ['bn', 'vink', 'vsup', 'sigma']:
            v.loc[0, s] = getattr(self, s)
        v.loc[0, 'pressure'] = self.press.targetPressure
        vunits['bn']=''
        vunits['vink']='mm/s'
        vunits['vsup']='mm/s'
        vunits['sigma']='mJ/m^2'
        vunits['pressure']='mbar'
        return v,vunits
    
    def fluigent(self, overwrite:bool=False) -> None:
        '''write dimensions and positions to file'''
        self.importTimeFile()
        self.importProgDims()
    
    def initializeProgDims(self):
        '''initialize programmed dimensions table'''

        self.progDims.name=['xs5','xs4','xs3','xs2','xs1',
                                'vert4','vert3','vert2','vert1',
                                'horiz2','horiz1','horiz0']
        
    def useDefaultTimings(self):
        '''use the programmed line lengths'''
        self.initializeProgDims()
        if 'singleLinesNoZig' in self.sbp:
            for s,row in self.progDims.iterrows():
                if 'xs' in row['name']:
                    self.progDims.loc[s,'l']=15         # intended filament length in mm
                elif 'vert' in row['name']:
                    self.progDims.loc[s,'l']=15
                elif 'horiz' in row['name']:
                    self.progDims.loc[s,'l']=22.5 
                self.progDims.loc[s,'w']=self.dEst  # intended filament diameter
                self.progDims.loc[s,'t']=self.progDims.loc[s,'l']/self.sup.v # extrusion time
                self.progDims.loc[s,'a']=np.pi*(self.dEst/2)**2
                self.progDims.loc[s,'vol']=self.progDims.loc[s,'l']*self.progDims.loc[s,'a']
                
    def storeProg(self, i:int, vals:dict) -> None:
        '''store programmed values into the table'''
        self.progDims.iloc[i]['l'] = vals['l']
        self.progDims.iloc[i]['vol'] = vals['vol']
        self.progDims.iloc[i]['t'] = vals['ttot']
        self.progDims.iloc[i]['tf'] = vals['tf']
        self.progDims.iloc[i]['t0'] = vals['t0']
        self.progDims.iloc[i]['a']= vals['vol']/vals['l']
        self.progDims.iloc[i]['w'] = 2*np.sqrt(self.progDims.iloc[i]['a']/np.pi) # a=pi*(w/2)^2
        if not hasattr(self, 'progDimsUnits'):
            lu = 'mm'
            tu = 's'
            self.progDimsUnits = {'t':tu, 't0':tu, 'tf':tu,
                            'l':lu, 'w':lu, 'a':f'{lu}^2', 'vol':f'{lu}^3'}
        
    
    def getProgDims(self, ftable:pd.DataFrame):
        '''read programmed line dimensions from fluigent table based on when pressure is turned on and off. 
        ftable is time-pressure table from file'''
        self.initializeProgDims()
        i = 0
        vol = 0
        ttot = 0
        l = 0
        a = np.pi*(self.pv.dEst/2)**2 # ideal cross-sectional area
        anoz = np.pi*(self.geo.di/2)**2 # inner cross-sectional area of nozzle
        for j,row in ftable.iterrows():
            if row['pressure']>0:
                if ttot==0:
                    t0 = row['time']
                # flow is on
                if j>0:
                    dt = (row['time']-ftable.loc[j-1,'time']) # timestep size
                else:
                    dt = (-row['time']+ftable.loc[j+1,'time']) 
                ttot = ttot+dt # total time traveled
                if self.pv.date<=210929 and 'vert' in self.progDims.loc[i,'name']:
                    # bug in initial code wasn't changing vertical velocity
                    dl = dt*5
                else:
                    dl = dt*self.pv.sup.v # length traveled in this timestep
                l = l+dl  # total length traveled
                if dt==0:
                    dvol=0
                else:
                    p = row['pressure']
                    vcalc = self.press.calculateSpeed(p)
                    if not vcalc==0:
                        flux = max(vcalc*anoz,0)
                        # actual flow speed based on calibration curve (mm/s) * area of nozzle (mm^2) = flux (mm^3/s)
                    else:
                        flux = p/self.press.targetPressure*anoz*self.pv.ink.v
                    dvol = flux*dt 
                vol = vol+dvol # total volume extruded
            else:
                # flow is off
                if l>0:
                    self.storeProg(i, {'l':l, 'vol':vol, 'ttot':ttot, 'a':a, 't0':t0, 'tf':row['time']})
                    ttot=0
                    vol=0
                    l=0
                    i = i+1
        if i<len(self.progDims):
            self.storeProg(i, {'l':l, 'vol':vol, 'ttot':ttot, 'a':a, 't0':t0, 'tf':row['time']})
        
        # start times at 0
        self.progDims.tf = self.progDims.tf-self.progDims.t0.min()
        self.progDims.t0 = self.progDims.t0-self.progDims.t0.min()
    
    
class progDimsSingleDisturb(progDim):
    '''for programmed dimensions of single disturb prints'''
    
    def __init__(self, printFolder:str, pv:printVals, **kwargs):
        super().__init__(printFolder, pv)
        
    def initializeProgDims(self):
        '''initialize programmed dimensions table'''


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
                    self.rewriteTargets()
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
                
#----------
    
    
class progDimsTripleLine(progDim):
    '''for programmed dimensions of triple line prints'''
    
    def __init__(self, printFolder:str, pv:printVals, **kwargs):
        super().__init__(printFolder, pv)
    
    def initializeProgDims(self):
        '''initialize programmed dimensions table'''

        if 'crossDoubleVert_0.5' in self.sbp:
            # wrote 2 sets of vertical lines and a zigzag
            self.progDims.name = ['v00', 'v01', 'v10', 'v11', 'zig']
        elif 'crossDoubleVert_0' in self.sbp:
            # wrote 2 sets of vertical lines with 6 crosses for each set
            self.progDims.name = ['v00', 'v01'
                                  , 'h00', 'h01', 'h02', 'h03', 'h04', 'h05'
                                  , 'v10', 'v11'
                                 , 'h10', 'h11', 'h12', 'h13', 'h14', 'h15']
        elif 'crossDoubleHoriz_0.5' in self.sbp:
            # wrote zigzag and 4 vertical lines
            self.progDims.name = ['zig', 'v0', 'v1', 'v2', 'v3']
        elif 'crossDoubleHoriz_0' in self.sbp:
            # wrote 3 zigzags with 4 cross lines each
            self.progDims.name = ['hz0', 'hc00', 'hc01', 'hc02', 'hc03'
                                 , 'hz1', 'hc10', 'hc11', 'hc12', 'hc13'
                                 , 'hz2', 'hc20', 'hc21', 'hc22', 'hc23']
        elif 'underCross_0.5' in self.sbp:
            # wrote 2 zigzags
            self.progDims.name = ['z1', 'z2']
        elif 'underCross_0' in self.sbp:
            # wrote 3 double line zigzag, then 3 sets of 4 vertical lines 
            self.progDims.name = ['zig'
                                 , 'v00', 'v01', 'v02', 'v03'
                                 , 'v10', 'v11', 'v12', 'v13'
                                 , 'v20', 'v21', 'v22', 'v23']
        elif 'tripleLines' in self.sbp:
            # wrote 4 groups with 3 lines each
            self.progDims.name = ['l00', 'l01', 'l02'
                                 ,'l10', 'l11', 'l12'
                                 ,'l20', 'l21', 'l22'
                                 ,'l30', 'l31', 'l32']
            
def getProgDims(folder:str):
    pfd = fh.printFileDict(folder)
    pv = printVals(folder)
    if pv.pfd.printType=='tripleLine':
        return progDimsTripleLine(folder, pv)
    elif pv.pfd.printType=='singleLine':
        return progDimsSingleLine(folder, pv)
    elif pv.pfd.printType=='singleDisturb':
        return progDimsSingleDisturb(folder, pv)     
    else:
        raise ValueError(f'No print type detected in {folder}')
    
def exportProgDims(folder:str, overwrite:bool=False) -> list:
    '''export programmed dimensions. returns list of bad folders'''
    errorList = []
    if not os.path.isdir(folder):
        return errorList
    if not fh.isPrintFolder(folder):
        for f1 in os.listdir(folder):
            errorList = errorList + exportProgDims(os.path.join(folder, f1), overwrite=overwrite)
        return errorList

    pv = printVals(folder)
    if not overwrite:
        if len(pv.pfd.progDims)>0 and len(pv.pfd.progPos)>0:
            return errorList
        
    try:
        pdim = getProgDims(folder)
        pdim.exportAll(overwrite=overwrite)
    except ValueError as e:
        errorList.append(folder)
        print(e)
        return errorList
    else:
        return errorList
    
    
#--------------

def progTableRecursive(topfolder:str, useDefault:bool=False, overwrite:bool=False, **kwargs) -> pd.DataFrame:
    '''go through all of the folders and summarize the programmed timings'''
    if isSubFolder(topfolder):
        try:
            pv = printVals(topfolder)
            if (not 'dates' in kwargs or pv.date in kwargs['dates']) and overwrite:
                pv.redoSpeedFile()
                pv.fluigent()
                pv.exportProgDims() # redo programmed dimensions
            if useDefault:
                pv.useDefaultTimings()
            t,u = pv.progDimsSummary()
        except:
            traceback.print_exc()
            logging.warning(f'failed to get programmed timings from {topfolder}')
            return {}, {}
        return t,u
    elif os.path.isdir(topfolder):
        tt = []
        u = {}
        for f in os.listdir(topfolder):
            f1f = os.path.join(topfolder, f)
            if os.path.isdir(f1f):
                t,u0=progTableRecursive(f1f, useDefault=useDefault, overwrite=overwrite, **kwargs)
                if len(t)>0:
                    if len(tt)>0:
                        tt = pd.concat([tt,t])
                    else:
                        tt = t
                    if len(u)==0:
                        u = u0
        return tt, u
    
def checkProgTableRecursive(topfolder:str, **kwargs) -> None:
    '''go through the folder recursively and check if the pressure calibration curves are correct, and overwrite if they're wrong'''
    if isSubFolder(topfolder):
        try:
            pv = printVals(topfolder)
            pv.importProgDims()
            if 0 in list(pv.progDims.a):
                pv.fluigent()
                pv.exportProgDims() # redo programmed dimensions
        except:
            traceback.print_exc()
            logging.warning(f'failed to get programmed timings from {topfolder}')
            return
        return 
    elif os.path.isdir(topfolder):
        for f in os.listdir(topfolder):
            f1f = os.path.join(topfolder, f)
            if os.path.isdir(f1f):
                checkProgTableRecursive(f1f, **kwargs)
        return

def progTable(topfolder:str, exportFolder:str, filename:str, **kwargs) -> pd.DataFrame:
    '''go through all the folders, get a table of the speeds and pressures, and export to filename'''
    tt,units = progTableRecursive(topfolder, **kwargs)
    tt = pd.DataFrame(tt)
    if os.path.exists(exportFolder):
        plainExp(os.path.join(exportFolder, filename), tt, units)
    return tt,units

