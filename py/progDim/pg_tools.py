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
import shutil

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
from tools.config import cfg
from tools.plainIm import *
import file.file_handling as fh
from val.v_print import printVals

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)



#----------------------------------------------


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
        self.numLines = 4
        self.initializeProgDims()
        self.sbp = self.pfd.sbpName()    # name of shopbot file

    def importGeneric(self, s:str, export:bool=True) -> None:
        '''import a csv and export a new csv if it doesn't exist'''
#         print(f'import {s}')
        fn = getattr(self, f'{s}File')()
        scap = f'{s[0].capitalize()}{s[1:]}'
        if os.path.exists(fn):
            t,u = plainIm(fn, ic=0)
            setattr(self, s, t)
            setattr(self, f'{s}Units', u)
            
        else:
            if export:
                getattr(self, f'export{scap}')()  # export a new file
            else:
                setattr(self, s, [])
                setattr(self, f'{s}Units', {})
                
    def exportGeneric(self, s:str, overwrite:bool=False) -> None:
        '''generate data if not already found, and export'''
#         print(f'export {s}')
        fn = getattr(self, f'{s}File')()
        scap = f'{s[0].capitalize()}{s[1:]}'
        if os.path.exists(fn) and not overwrite:
            # file already exists
            return 0
        
        if overwrite or not hasattr(self, s):
            getattr(self, f'get{scap}')()
          
        if hasattr(self, s) and hasattr(self, f'{s}Units') and len(getattr(self, s))>0:
            table = getattr(self, s)
            if len(table)==0:
                getattr(self, f'get{scap}')()
                table = getattr(self, s)
                if len(table)==0:
                    return 1
            plainExp(fn, table, getattr(self, f'{s}Units'))
            return 0
        else:
            if hasattr(self, s) and not hasattr(self, f'{s}Units'):
                print(f'Missing units for {s}')
            return 1
                
    #---------------------------------------------------
    # import time file     
    
    def timeFile(self) -> str:
        '''get the name of the original pressure-time table'''
        if len(self.pfd.timeSeries)==0:
            return ''
        if len(self.pfd.timeSeries)==1:
            return self.pfd.timeSeries[0]
        if len(self.pfd.timeSeries)>1:
            l = []
            for f in self.pfd.timeSeries:
                if self.pfd.printType in ['singleDisturb', 'tripleLines', 'SDT']:
                    l.append(f)
                elif 'singleLinesN' in f and 'Fluigent' in f:
                    l.append(f)
            if len(l)==0:
                raise NameError(f'No fluigent file found in {self.printFolder}')
            l.sort()
            file = os.path.join(self.printFolder, l[-1]) # select last fluigent file
            return file
    
        
    def createTargetOrig(self) -> None:
        '''create a column for original targets'''
        if 'xt_orig' in self.ftable:
            return
        if not 'xt' in self.ftable:
            return
        for s in ['x', 'y', 'z']:
            self.ftable[f'{s}t_orig'] = self.ftable[f'{s}t']
            self.ftableUnits[f'{s}t_orig'] = self.ftableUnits[f'{s}_target']
        
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
    
    #---------------------------------------------------
    # rewrite time file   
    
    def timeRewriteFile(self) -> str:
        '''programmed positions file name'''
        return self.pfd.newFileName('timeRewrite', 'csv')
        
    def importTimeRewrite(self) -> None:
        '''import the file with rewritten targets'''
        if self.pfd.printType=='singleLine':
            self.importTimeFile()   
        else:
            self.importGeneric('timeRewrite')
            self.ftable = self.timeRewrite
            self.ftableUnits = self.timeRewriteUnits

        if len(self.ftable)>0:
            self.ftable.rename(columns=timeColumns(), inplace=True)  # shorten names
            self.rewritten=True

    def getTimeRewrite(self, diag:int=0) -> int:
        '''start to rewrite the target points'''
        self.importTimeFile()
        if not 'xt' in self.ftable:
            print(self.ftable)
            return 1
        
        lu = self.ftableUnits['x_target']
        tu = self.ftableUnits['time']
        self.ftableUnits['ldt'] = lu
        self.ftableUnits['speed'] = f'{lu}/{tu}'
   
    def exportTimeRewrite(self, overwrite:bool=False, diag:int=0) -> int:
        '''rewrite targets and export'''
        
        fn = self.timeRewriteFile()
        if os.path.exists(fn) and not overwrite:
            # file already exists
            return 0
        
        if not self.rewritten or overwrite:
            self.getTimeRewrite(diag=diag)
            
        if hasattr(self, 'ftable') and len(self.ftable)>0:
            ftable2 = self.ftable.copy()
            ftable2.rename(columns=timeColumns(reverse=True), inplace=True)
            plainExp(fn, ftable2, self.ftableUnits)
            return 0
        else:
            return 1
        
        
        
    #---------------------------------------------
    # flag flips: track when flags changed

    def flagFlipFile(self) -> str:
        '''flag flips file name'''
        return self.pfd.newFileName('flagFlip', 'csv')
    
    def importFlagFlip(self, export:bool=True) -> None:
        '''import the flag flips'''
        self.importGeneric('flagFlip', export)
        self.flagFlip.fillna('', inplace=True)
        
    def flagFlipRow(self, row:pd.Series) -> dict:
        '''turn the status into a dictionary indicating which devices are changing'''
        status = row['status']
        spl = re.split(', ', status)
        out = {'time':row['time']}
        for k in spl:
            spl2 = re.split(': ', k)
            if spl2[0] in out:
                out[spl2[0]] = out[spl2[0]] + ', '+ spl2[1]
            else:
                out[spl2[0]] = spl2[1]
        return out

                
    def getFlagFlip(self) -> None:
        '''get the times when flags flipped'''
        self.importTimeRewrite()
        if len(self.ftable)==0:
            return
        
        changes = self.ftable[~self.ftable.status.isna()]
        flags = changes[changes.status.str.contains('Flag')]
        self.flagFlip = pd.DataFrame([self.flagFlipRow(row) for i,row in flags.iterrows()])
        self.flagFlip.fillna('', inplace=True)
        self.flagFlipUnits = {}
        for f in self.flagFlip.columns.drop('time'):
            if self.flagFlip[f].str.contains('SNAP').sum()>0:
                self.flagFlip.rename(columns={f:'cam'}, inplace=True)
                self.flagFlipUnits['cam'] = ''
            elif self.flagFlip[f].str.contains('ON').sum()>0:
                self.flagFlip.rename(columns={f:'pressure'}, inplace=True)
                self.flagFlipUnits['pressure'] = ''
            else:
                self.flagFlipUnits[f] = ''
 
        self.flagFlipUnits['time']=self.ftableUnits['time']
        
    def exportFlagFlip(self, overwrite:bool=False) -> None:
        self.exportGeneric('flagFlip', overwrite)
                
    
                
    #----------------------------------------------
    # progPos: track changes in target point
    
    def progPosFile(self) -> str:
        '''programmed positions file name'''
        return self.pfd.newFileName('progPos', 'csv')
    
    def importProgPos(self, export:bool=True) -> None:
        '''import the programmed positions'''
        self.importGeneric('progPos')
                
    def getDTraveled(self, translationSpeed:float, tf:float, t0:float) -> float:
        return translationSpeed*(tf-t0)

    def readProgGroup(self, v:pd.DataFrame) -> dict:
        '''summarize the group of time steps'''
        anoz = np.pi*(self.geo.di/2)**2 # inner cross-sectional area of nozzle           
        pp = self.pprev       # previous target point coords 
        pt = v.iloc[0]        # current target point
        pospoints = v[v.pressure>0]  # get points where pressure is on
        t0 = self.t0
        tf = v.time.max()
        self.t0 = tf 
        # characterize step size, direction
        if 'xt' in pp:
            direc = [(pt['xt']-pp['xt']), (pt['yt']-pp['yt']), (pt['zt']-pp['zt'])]
            dist = np.sqrt((direc[0])**2+(direc[1])**2+(direc[2])**2)
        else:
            dist = np.nan
            direc = [np.nan, np.nan, np.nan]
            
        if 'speed' in v:
            translationSpeed = v.speed.max()
            dtraveled = self.getDTraveled(translationSpeed, tf, t0)
        else:
            translationSpeed = self.pv.sup.v
            dtraveled = self.getDTraveled(translationSpeed, tf, t0)
            if dtraveled>dist*0.45 and dtraveled<dist*0.55:
                # we're actually in a jog
                translationSpeed = translationSpeed*2
                dtraveled = self.getDTraveled(translationSpeed, tf, t0)

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
        self.importTimeRewrite()
            
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
        self.exportGeneric('progPos', overwrite=overwrite)
        
        
    #----------------------
    
    def progDimsFile(self) -> str:
        '''get the name of the progDims file'''
        fn = self.pfd.newFileName('progDims', 'csv')
        return fn
    
    def importProgDims(self, overwrite:bool=False) -> str:
        '''import the progdims file'''
        self.importGeneric('progDims', overwrite)
        
    def initializeProgDims(self) -> None:
        '''initialize the table'''
        cols = ['name', 'l', 'w', 't', 'a', 'vol', 't0', 'tf', 'tpic', 'lprog', 'ltr', 'speed', 'xpic', 'ypic', 'zpic']
        self.progDims = pd.DataFrame(columns=cols)
        
    
                
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
        self.progDimsUnits = {'name':'', 'l':lu, 'w':lu, 't':tu, 'a':f'{lu}^2', 'vol':f'{lu}^3', 
                              't0':tu, 'tf':tu, 'tpic':tu, 'lprog':lu, 'ltr':lu, 'speed':f'{lu}/{tu}',
                             'xpic':lu, 'ypic':lu, 'zpic':lu}
    
    def exportProgDims(self, overwrite:bool=False) -> None:
        '''sort programmed moves into intended moves'''
        self.defineProgDimsUnits()
        self.exportGeneric('progDims', overwrite=overwrite)
        
    #---------------
    
    def exportAll(self, overwrite:bool=False, diag:int=0) -> None:
        '''export rewritten time, progPos, and progDims'''
        
        self.exportTimeRewrite(overwrite=overwrite, diag=diag)
        self.exportFlagFlip(overwrite=overwrite)
        self.exportProgPos(overwrite=overwrite)
        self.exportProgDims(overwrite=overwrite)
        
    #---------------
    
    def progPosRow(self, lineName:str) -> dict:
        '''get the row from the progPos table that corresponds to the line with given name'''
        timeRow = self.progDims[self.progDims.name==lineName]
        time = float(timeRow.iloc[0]['tpic'])  # time when the picture was taken
        posRows = self.progPos[(self.progPos.t0<time)&(self.progPos.tf>time)]
        
        if len(posRows)==0:
            print(self.progPos)
            print(timeRow)
            raise ValueError('Cannot find line in table')

        return posRows.iloc[0]
    
    def lineName(self, writeLine:int, gname:str) -> str:
        '''get the name of the line of interest. if writeLine==1, find the first write line. otherwise find the given line. gname is the group name, e.g. l1, where there are 4 groups of n lines in this print'''
        if writeLine==1:
            if f'{gname}w1' in list(self.progDims.name):
                write = f'{gname}w1'
            elif f'{gname}w1p3' in list(self.progDims.name):
                write = f'{gname}w1p3'
            else:
                raise ValueError(f'Could not find first write line for {gname}, {writeLine}')
        else:
            if f'{gname}w{writeLine}' in list(self.progDims.name):
                write = f'{gname}w{writeLine}'
            elif f'{gname}w{writeLine}p3' in list(self.progDims.name):
                write = f'{gname}w{writeLine}p3'
            else:
                raise ValueError(f'Could not find write line for {gname}, {writeLine}')
        return write
    
    def progLine(self, writeLine:int, gname:str) -> pd.Series:
        '''get the row from the progDims table that correspond to that line'''
        if not hasattr(self, 'progDims') or pd.isna(self.progDims.iloc[0]['l']):
            self.importProgDims()
        lineName = self.lineName(writeLine, gname)
        timeRow = self.progDims[self.progDims.name==lineName]
        if len(timeRow)==0:
            raise ValueError(f'Could not find {lineName} in progDims')
        return timeRow
    
    def lineEndPoint(self, lineName:str) -> dict:
        '''get the target of the line'''
        posRow = self.progPosRow(lineName)
        end = {}
        for s in ['y', 'z']:
            end[s] = posRow[f'{s}t']
        return end
    
    def lineMidPoint(self, lineName:str, fixList:list=[]) -> dict:
        '''get the midpoint of the line'''
        posRow = self.progPosRow(lineName)
        prevRow = self.progPos.loc[posRow.name-1]
        # print(self.progPos.loc[(posRow.name-1):(posRow.name)])
        center = {}
        for s in ['y', 'z']:
            if s in fixList:
                center[s] = posRow[f'{s}t']
            else:
                center[s] = (posRow[f'{s}t']+prevRow[f'{s}t'])/2
        return center
    
    def linePicPoint(self, lineName:str) -> dict:
        '''get the coordinates of the picture from the progDims table'''
        sel = self.progDims[self.progDims.name==lineName]
        if len(sel)==0:
            print(self.progDims)
            raise ValueError(f'Cannot find line {lineName} in progDims')
        timeRow = sel.iloc[0]
        return {'y':timeRow['ypic'], 'z':timeRow['zpic']}

    
    def relativeCoords(self, tag:str, writeLine:int=1, fixList:list=[]) -> dict:
        '''get the position of the first written line in this set (e.g. l1w) relative to the nozzle at the time we took this picture (e.g. tag=l1wo). returns values in mm'''
        if not hasattr(self, 'progDims') or pd.isna(self.progDims.iloc[0]['l']):
            self.importProgDims()
        if not hasattr(self, 'progPos'):
            self.importProgPos()
        if not tag in list(self.progDims.name):
            print(self.printFolder)
            print(self.progDims)
            raise ValueError(f'Line {tag} is not in progDims')
        gname = tag[:2]   # l1, l2, etc.    
        # find the name of the first write line, e.g. l2w
        write = self.lineName(writeLine, gname)
        # print(fixList)
        # writePoint = self.lineMidPoint(write, fixList)
        writePoint = self.linePicPoint(write)
        d = {'dx':0, 'dy':0}   # change in coords from nozzle to the region we want to see
        if 'o' in tag:
            # put the first observe point in the center. find position of observe point
            center1 = f'{write[:4]}o1'
            # centerPoint1 = self.lineEndPoint(center1)  # where the origin should be
            centerPoint1 = self.linePicPoint(center1)
            
            if not center1==tag:
                centerPointNow = self.linePicPoint(tag)
                # centerPointNow = self.lineEndPoint(tag)    # where this line is
                # shift to get to the observe origin from this observe location
                d['dx'] = centerPointNow['y'] - centerPoint1['y']  
                d['dy'] = centerPointNow['z'] - centerPoint1['z']
                
            # shift to get to the written line from the observe origin
            d['dx']  = d['dx'] - (centerPoint1['y'] - writePoint['y'])
            d['dy']  = d['dy'] - (centerPoint1['z'] - writePoint['z'])
        else:
            # we are observing a line during writing
            if tag==write:
                return d
            centerPointNow = self.linePicPoint(tag)
            # centerPointNow = self.lineMidPoint(tag, fixList)    # where this line is
            # shift to get to the written line from the observe origin
            d['dx']  = d['dx'] - (centerPointNow['y'] - writePoint['y'])
            d['dy']  = d['dy'] - (centerPointNow['z'] - writePoint['z'])
        return d
