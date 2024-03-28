#!/usr/bin/env python
'''Functions for manually characterizing stills of single double triple lines, for a whole folder'''

# external packages
import os, sys
import traceback
import logging
import pandas as pd
from typing import List, Dict, Tuple, Union, Any, TextIO
import re
import numpy as np
import cv2 as cv
import random

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
sys.path.append(os.path.dirname(os.path.dirname(currentdir)))
from folder_metric import *
from f_tools import *
from py.im.imshow import imshow
from tools.plainIm import *
from folder_loop import folderLoop

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)
pd.set_option('display.max_rows', 500)


#----------------------------------------------

class folderManualSDT(folderMetric):
    '''for a folder, manually describe the SDT lines
    export a table of values (manual)
    '''
    
    def __init__(self, folder:str, **kwargs) -> None:
        super().__init__(folder, splitGroups=True, **kwargs)
        if not f'disturb' in os.path.basename(self.folder):
            return
        if not hasattr(self, 'pg'):
            if 'pg' in kwargs:
                self.pg = kwargs['pg']
            else:
                self.pg = getProgDimsPV(self.pv)
                self.pg.importProgDims()
        self.lines = list(self.pg.progDims.name)  
        self.pfd.findVstill()
        self.mr, self.mu = self.pv.metarow()
        self.vals = {}
        self.fn = self.pfd.newFileName(f'manual', '.csv')
        
    def getName(self, l:int, s:str, j:int) -> str:
        '''get the English name of this value'''
        so = f'{s}{l}'
        if j==7:
            so = f'{so}relax'
        return so
        
    def displayLine(self, i:int) -> None:
        '''pull in representative images and display them'''
        ims = {}
        self.vals[i] = {}
        
        # assemble images
        for l in [1,2,3]:
            for s in ['w', 'd']:
                for j in [1,7]:
                    tag = f'l{i}{s}{l}o{j}'
                    if tag in self.lines:
                        file = self.pfd.findStillTag(tag)
                        if os.path.exists(file):
                            im = cv.imread(file)
                            name = self.getName(l,s,j)
                            ims[name]=im
                            if l==1:
                                self.vals[i][name] = 'no change'
                            elif l==2 or l==3:
                                if j==1 and s=='w':
                                    self.vals[i][name] = 'no fusion'   
                                else:
                                    self.vals[i][name] = 'no change'
        
        imshow(*(ims.values()), titles=list(ims.keys()), axesVisible=False)
        
    def setVal(self, i:int, name:str, val:str) -> None:
        '''characterize the line'''
        if not name in self.vals[i]:
            return
        self.vals[i][name] = val
        
    def export(self) -> None:
        '''export the values'''
        self.out = {**self.mr}
        self.u = {**self.mu}
        for l,d in self.vals.items():
            for name, val in d.items():
                n = f'l{l}{name}'
                self.out[n] = val
                self.u[n] = ''
        plainExpDict(self.fn, self.out, self.u)
        
        
class folderManualLoop(folderLoop):
    '''this finds folders to run folderManualSDT on'''
    
    def __init__(self, folders:Union[str, list], **kwargs):
        if 'fn' in kwargs:
            self.importFile(kwargs['fn'])
            findFolders = False
        else:
            findFolders = True
        super().__init__(folders, folderManualSDT, findFolders=findFolders, **kwargs)
        if findFolders:
            self.incomplete = self.folders.copy()
            self.complete = []
        
    def findFolder(self) -> folderManualSDT:
        '''find a folder'''
        folder = random.choice(self.incomplete)
        # check that the folder matches all keys
        if not allIn(self.mustMatch, folder):
            return self.findFolder()
        
        if not anyIn(self.canMatch, folder):
            return self.findFolder()
        fmi = folderManualSDT(folder)
        self.complete.append(folder)
        self.incomplete.remove(folder)
        if os.path.exists(fmi.fn):
            # already did manual characterization    
            return self.findFolder()
        else:
            return fmi
        
    def export(self, fn:str) -> None:
        '''export the folder statuses'''
        d = dict([[f, 'complete'] for f in self.complete]+[[f, 'incomplete'] for f in self.incomplete])
        plainExpDict(fn, d)
        
    def importFile(self, fn:str) -> None:
        '''import the folder statuses'''
        d,u = plainImDict(fn)
        self.incomplete = [k for k,v in d.items() if v=='incomplete']
        self.complete = [k for k,v in d.items() if v=='complete']   
        