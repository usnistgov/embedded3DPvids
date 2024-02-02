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
import traceback

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
from tools.config import cfg
import f_tools as ft
import levels as le
import file_names as fn
from tools.plainIm import plainIm


# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

#----------------------------------------------

def isSubFolder(folder:str) -> bool:
    '''determine if the folder is a subfolder'''
    if ft.sampleInName(folder) and ft.dateInName(folder):
        return True
    else:
        return False

def subFolder(folder:str) -> str:
    '''name of the subfolder'''
    if isSubFolder(folder):
        return folder
    else:
        levels = le.labelLevels(folder)
        return levels.subFolder
    
def getPrintFolder(folder:str, **kwargs) -> str:
    if 'levels' in kwargs:
        return kwargs['levels'].printFolder()
    
    if fn.isPrintFolder(folder):
        return folder
    
    levels = le.labelLevels(folder)
    return levels.printFolder()
 
def printFolders(topFolder:str, tags:List[str]=[''], someIn:List[str]=[],  **kwargs) -> List[str]:
    '''Get a list of bottom level print folders in the top folder'''
    if 'folderFile' in kwargs:
        fostrlist, _ = plainIm(kwargs['folderFile'])
        if len(fostrlist)>0:
            return [os.path.join(cfg.path.server, fostr) for fostr in fostrlist['fostr']]
    if 'mustMatch' in kwargs:
        tags = kwargs['mustMatch']
    if 'canMatch' in kwargs:
        someIn = kwargs['canMatch']
    if fn.isPrintFolder(topFolder):
        if ft.allIn(tags, topFolder) and ft.anyIn(someIn, topFolder):
            folders = [topFolder]
        else:
            folders = []
    else:
        folders = []
        dirs = ft.listDirs(topFolder)
        for d in dirs:
            folders = folders+printFolders(d, tags=tags, someIn=someIn)
    return folders
