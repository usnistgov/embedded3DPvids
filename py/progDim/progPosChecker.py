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

class progPosChecker:
    '''for checking and correcting errors in progPos files'''
    
    def __init__(self, printFolder, progPos:pd.DataFrame, **kwargs):
        self.printFolder = printFolder
        self.progPos = progPos
        if 'Horiz' in self.printFolder:
            self.moveDir = '+y'
        elif 'Vert' in self.printFolder:
            self.moveDir = '+z'
        elif 'XS' in self.printFolder:
            self.moveDir = '-x'
        self.checkExtensions()
        

    def checkExtensions(self):
        '''check that extended lines are split into the write and extend portions'''
        pdp = self.progPos.copy()
        # find lines that have an extension at the end
        if self.moveDir=='-x':  # xs
            vlines = pdp[(pdp.dx<0)&(pdp.zt<0)&(pdp.shift(-1).dx<0)]
            do = ['dy', 'dz']
            di = 'dx'
        elif self.moveDir=='+y':  # horiz
            vlines = pdp[(pdp.dy>0)&(pdp.zt<0)&(pdp.dx==0)&(pdp.dz==0)&(pdp.shift(-1).dy>0)]
            do = ['dx', 'dz']
            di = 'dy'
        elif self.moveDir=='+z':  # vert
            vlines = pdp[(pdp.dz>0)&(pdp.zt<0)&(pdp.shift(-1).dz>0)]  
            do = ['dx', 'dy']
            di = 'dz'
          
        # take the first line and get the next progPos step
        row = vlines.iloc[0]
        writed = float(row[di])     # write distance
        nextrow = self.progPos.loc[row.name+1]
        extendd = float(nextrow[di])   # extend distance
        totdz =  extendd + writed
        combined = pdp[(pdp[di]==totdz)]
        combined = combined.copy()
        if len(combined)==0:
            return
        extdf = combined.copy()
        
        for i,row in combined.iterrows():
            extdf.loc[i, di] = extendd   # set the distance equal to the extension length
            extdf.loc[i,'dprog'] = extendd
            combined.loc[i,di] = writed
            combined.loc[i,'dprog'] = writed
            tbreak = (row['t0']+(row['tf']-row['t0'])*writed/(extendd+writed))
            extdf.loc[i,'t0'] = tbreak
            combined.loc[i,'tf'] = tbreak
            if not np.isnan(row['tf_flow']):
                combined.loc[i,'tf_flow'] = tbreak
                extdf.loc[i,'t0_flow'] = tbreak
            
            if self.moveDir=='-x':
                combined.loc[i, 'xt'] = row['xt']+extendd   # move the target
            elif self.moveDir=='+y':
                combined.loc[i, 'yt'] = row['yt']-extendd 
            elif self.moveDir=='+z':
                combined.loc[i, 'zt'] = row['zt']-extendd
        
        
        combined['dtr'] = combined['speed']*(combined['tf']-combined['t0'])
        extdf['dtr'] = extdf['speed']*(extdf['tf']-extdf['t0']) 
        extdf.reset_index(drop=True, inplace=True)

        self.progPos.drop(combined.index, inplace=True)
        self.progPos = pd.concat([self.progPos, combined])
        self.progPos = pd.concat([self.progPos, extdf])
        self.progPos.sort_values(by='t0', inplace=True)
        self.progPos.reset_index(drop=True, inplace=True)
        
        display(combined)
        display(extdf)
    