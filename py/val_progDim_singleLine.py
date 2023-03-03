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