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
from config import cfg
from plainIm import *
from fluidVals import *
import fileHandling as fh

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
    def checkUnits(self, row:list, bn:str, expected:str) -> None:
        '''check if the units are correct for this row'''
        if not row[1]==expected:
            logging.warning(f'Bad units in {bn}: {row[0]}, {row[1]}')
    
    def importMetaFile(self) -> int:
        '''find the metadata file. returns 0 if successful'''
        if not hasattr(self.pfd, 'meta') or len(self.pfd.meta)==0:
            return 1
        file = self.pfd.meta[0]
        bn = fh.twoBN(file)
        with open(file, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                if row[0] in ['&nid','nozzle_inner_diameter', 'nozzle_0_diameter']:
                    self.di = float(row[2])
                    self.checkUnits(row, bn, self.units['di'])
                elif row[0]=='&nd':
                    self.do = float(row[2])
                    self.checkUnits(row, bn, self.units['do'])
                elif row[0]=='camera_magnification':
                    self.camMag = float(row[2])
                elif row[0]=='camera_position':
                    self.camPosition = row[2]
                elif row[0]=='nozzle_length':
                    self.lNoz = float(row[2])
                    self.checkUnits(row, bn, self.units['lNoz'])
                elif row[0]=='tubing_length':
                    self.lTub = float(row[2])
                    self.checkUnits(row, bn, self.units['lTub'])
                
        return 0
    
    def metarow(self) -> Tuple[dict,dict]:
        '''row holding metadata'''
        mlist = ['di', 'do']
        meta = [[i,getattr(self,i)] for i in mlist]
        munits = [[i, self.units[i]] for i in mlist]
        
        return out, units