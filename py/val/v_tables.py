#!/usr/bin/env python
'''Functions for storing tables that hold stats about the fluids'''

# external packages
import os, sys
import traceback
import logging
from typing import List, Dict, Tuple, Union, Any, TextIO
import re
import pandas as pd
import numpy as np
import csv
import time

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

class valTables:
    '''class the holds tables of data about many fluids'''
    
    def __init__(self, printType:str='', **kwargs):
        self.printType = printType
        
    def tableList(self, tableName:str, **kwargs) -> pd.DataFrame:
        '''get a list of tables to check'''
        if self.printType in ['singleLine', 'singleDisturb']:
            return self.importTable(cfg.path[tableName].single, **kwargs)
        elif self.printType in ['singleDoubleTriple', 'SDT']:
            return self.importTable(cfg.path[tableName].SDT, **kwargs)
        else:
            return pd.concat([self.importTable(t, **kwargs) for t in cfg.path[tableName].values()])
        
    def importTable(self, file:str, **kwargs) -> pd.DataFrame:
        '''import the table as a dataframe'''
        if not os.path.exists(file):
            return pd.DataFrame([])
        ext = os.path.splitext(file)[-1]
        if ext=='.xlsx':
            df = pd.read_excel(file)
        elif ext=='.csv':
            df,u = plainIm(file, **kwargs)
        else:
            print(ext)
            raise ValueError(f'Could not read table: {file}')
        return df
        
    def sigmaDF(self):
        '''find the sigma table dataframe'''
        if hasattr(self, 'sigt'):
            return self.sigt
        sigt = self.tableList('sigmaTable', ic=None)
        sigt = sigt.fillna('') 
        self.sigt = sigt
        return self.sigt
    
    def rheDF(self):
        '''find the rheology table dataframe'''
        if hasattr(self, 'rhet'):
            return self.rhet
        rhet = self.tableList('rheTable')
        rhet = rhet.fillna('') 
        self.rhet = rhet
        return self.rhet
    
    def densityDF(self):
        '''find the density table dataframe'''
        if hasattr(self, 'densityt'):
            return self.densityt
        t = self.tableList('densityTable', ic=None)
        t = t.fillna('') 
        self.densityt = t
        return self.densityt
        
        