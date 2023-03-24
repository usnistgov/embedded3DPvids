#!/usr/bin/env python
'''Functions for storing metadata about print folders'''

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
from tools.plainIm import *
import file.file_handling as fh

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)



#----------------------------------------------

            
            
class geometryVals:
    '''holds info about the print geometry'''
    
    def __init__(self, printFolder:str, **kwargs):
        self.printFolder = printFolder
        # dictionary of files in the printFolder
        if 'pfd' in kwargs:
            self.pfd = kwargs['pfd']
        else:
            self.pfd = fh.printFileDict(self.printFolder) 
        self.di = cfg.const.di    # nozzle inner diameter
        self.do = cfg.const.do    # nozzle outer diameter
        self.date = self.pfd.getDate()
        self.lNoz = 38  # nozzle length (mm)
        self.lTub = 280 # tubing length (mm)
        self.lBath = 12.5 # mm, depth of nozzle in bath
        self.units = {'di':'mm', 'do':'mm', 'date':'yymmdd', 'lNoz':'mm', 'lTub':'mm', 'lBath':'mm', 'pxpmm':'px/mm'}
        
        # guess camera magnification. if after 6/21/22, this will be in the metafile
        if self.date<220413:
            self.camMag = 1           # magnification
        else:
            self.camMag = 0.5
        self.camPosition = 'side' # camera position
        self.importMetaFile()
        self.pxpmm()
        
    def pxpmm(self):
        if self.camMag==1 and self.camPosition=='side':
            self.pxpmm = 139
        elif self.camMag == 0.5 and self.camPosition=='side':
            self.pxpmm = 71
        
    #-----------------
    # ShopbotPyQt after addition of _speed_ and _meta_ files
    def checkUnits(self, foundUnit:list, bn:str, expected:str, nam:str) -> None:
        '''check if the units are correct for this row'''
        if not foundUnit==expected:
            logging.warning(f'Bad units in {bn}: {nam}, {foundUnit}')
    
    def importMetaFile(self) -> int:
        '''find the metadata file. returns 0 if successful'''
        if not hasattr(self.pfd, 'meta') or len(self.pfd.meta)==0:
            return 1
        file = self.pfd.metaFile()
        bn = fh.twoBN(file)
        d,u = plainImDict(file, unitCol=1, valCol=2)
  
        nozkeys = dict([[key,val] for key,val in d.items() if key in ['&nid','nozzle_inner_diameter', 'nozzle_0_diameter']])
        if len(set(nozkeys.values()))>1:
            raise ValueError(f'Multiple nozzle diameter values found in {file}')
        else:
            k = list(nozkeys.keys())[0]
            self.di = float(d[k])
            self.checkUnits(u[k], bn, self.units['di'], k)
            
            
        for key,val in {'&nd':'do', 'camera_magnification':'', 'camera_position':'', 'nozzle_length':'lNoz', 'tubing_length':'lTub'}.items():
            if key in d:
                try:
                    self.do = float(d[key])
                except ValueError:
                    self.do = d[key]
                if val in self.units:
                    self.checkUnits(u[key], bn, self.units[val], key)
                
        return 0
    
    def metarow(self) -> Tuple[dict,dict]:
        '''row holding metadata'''
        mlist = ['di', 'do']
        meta = dict([[i,getattr(self,i)] for i in mlist])
        munits = dict([[i, self.units[i]] for i in mlist])
        
        return meta, munits