#!/usr/bin/env python
'''Functions for handling files'''

# external packages
import os, sys
import re
import shutil
import time
from typing import List, Dict, Tuple, Union, Any, TextIO
import logging
import pandas as pd
import subprocess
import time

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
from tools.config import cfg
from f_tools import *


# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



#----------------------------------------------

def isSubPrintFolder(folder:str) -> bool:
    for f in ['raw', 'temp', 'crop', 'Usegment', 'MLsegment']:
        if f in folder:
            return True
    return False


class labelLevels:
    '''label the levels of the file hierarchy, with one characteristic file per level'''

    def __init__(self, file:str):
        if not os.path.exists(file):
            raise ValueError(f'File does not exist: {file}')
    
        levels = {}
        bottom = self.findBottom(file)

        # bottom is now a bottom level file
        if os.path.isdir(bottom):
            bfold = bottom
            bottom = ''
        else:
            bfold = os.path.dirname(bottom)
        if isSubPrintFolder(bfold):
            # subimage folder
            self.rawFile = bottom
            if os.path.basename(bfold)=='raw':
                self.rawFolder = bfold
                aboveRaw = os.path.dirname(self.rawFolder)
            elif os.path.basename(bfold)=='temp':
                self.tempFolder = bfold
                aboveRaw = os.path.dirname(self.tempFolder)
            else:
                if 'raw' in bfold:
                    self.rawLineFolder = bfold
                    self.rawFolder = os.path.dirname(bfold)
                    aboveRaw = os.path.dirname(self.rawFolder)
                else:
                    self.rawFolder = bfold
                    aboveRaw = os.path.dirname(self.rawFolder)

            if sampleInName(aboveRaw):
                self.file = os.path.join(aboveRaw, self.firstEntry(aboveRaw, directory=False)) # bottom level file inside sbpfolder
                if dateInName(aboveRaw):
                    # sample and date in name: this is a subfolder
                    self.subFolder = aboveRaw
                else:
                    # this is a sample folder. these files are misplaced
                    self.sampleFolder = aboveRaw
                    self.subFolder = 'generate' # need to generate a new subfolder
            else:
                # no sample in name: shopbot folder
                self.sbpFolder = aboveRaw
                self.subFolder = os.path.dirname(aboveRaw)
                self.file = os.path.join(aboveRaw, self.firstEntry(aboveRaw, directory=False)) # bottom level file inside sbpfolder
        else:
            # just a file. no raw folder, because we would have found it during recursion
            self.file = bottom
            if sampleInName(bfold):
                if dateInName(bfold):
                    # sample and date in name: this is a subfolder
                    self.subFolder = bfold
                else:
                    # this is a sample folder. these files are misplaced
                    self.sampleFolder = bfold
            else:
                # no sample in name: shopbot folder
                self.sbpFolder = bfold
                self.subFolder = os.path.dirname(bfold)           

        if not 'sampleFolder' in levels:
            sabove = os.path.dirname(self.subFolder)
            if not sampleInName(sabove):
                self.sampleFolder = 'generate' # need to generate a new sample folder
                self.sampleTypeFolder = sabove # the type is right above the subFolder
            else:
                self.sampleFolder = sabove
                self.sampleTypeFolder = os.path.dirname(sabove)
        else:
            self.sampleTypeFolder = os.path.dirname(self.sampleFolder)
        self.printTypeFolder = os.path.dirname(self.sampleTypeFolder)

        currentLevel = ''
        for key in self.__dict__:
            if getattr(self, key)==file:
                currentLevel = key
        self.currentLevel = currentLevel
        
    def firstEntry(self, folder:str, directory:bool=True) -> str:
        '''find the first entry in the folder. directory=True to find directories, False to find files'''
        ld = os.listdir(folder)
        if len(ld)==0:
            return ''
        d1 = ''
        while len(ld)>0 and (((os.path.isdir(d1) and not directory) or (not os.path.isdir(d1) and directory)) or 'Thumbs' in d1 or not os.path.exists(d1)):
            # go through files in the directory until you find another directory
            d1 = os.path.join(folder, ld.pop(0))
        if (os.path.isdir(d1) and directory) or (not os.path.isdir(d1) and not directory):
            return d1
        else:
            return ''

    def findBottom(self, bottom:str) -> str:
        '''find the bottom level file in the hierarchy'''
        # recurse until you hit the bottom level
        while os.path.isdir(bottom): # bottom is a directory
            newbot = self.firstEntry(bottom, directory=True) # find the first directory
            if os.path.exists(newbot):
                # this is a directory
                bottom = newbot
            else:
                # no directories in bottom. find first file
                bottom2 = self.firstEntry(bottom, directory=False) 
                if not os.path.exists(bottom2):
                    return bottom
                else:
                    return bottom2
        return bottom
        
        
    def printFolder(self) -> str:
        '''get the folder that the print files are in. note that this needed to be initialized at the printfolder or inside the print folder to get the right print folder, otherwise you'll get the first one in the subFolder or sampleFolder'''
        if hasattr(self, 'sbpFolder'):
            return self.sbpFolder
        else:
            return self.subFolder

    def printAll(self):
        spacer = '  '
        ii = 0
        for s in ['printTypeFolder', 'sampleTypeFolder', 'sampleFolder', 'subFolder']:
            if hasattr(self, s):
                if ii==0:
                    pout = getattr(self,s)
                else:
                    pout = os.path.basename(getattr(self,s))
                print(spacer*ii, s,': ', pout)
                ii+=1
        jj = 0
        for s in ['sbpFolder', 'file']:
            if hasattr(self, s):
                pout = os.path.basename(getattr(self,s))
                print(spacer*(ii+jj), s,': ', pout)
                jj+=1
        jj = 0
        for s in ['rawFolder', 'rawLineFolder', 'rawFile']:
            if hasattr(self, s):
                pout = os.path.basename(getattr(self,s))
                print(spacer*(ii+jj), s,': ', pout)
                jj+=1
                
