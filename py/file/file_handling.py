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
from fileNames import *
from printFileDict import *
from levels import *


# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



#----------------------------------------------

def isSubFolder(folder:str) -> bool:
    '''determine if the folder is a subfolder'''
    if sampleInName(folder) and dateInName(folder):
        return True
    else:
        return False

def subFolder(folder:str) -> str:
    '''name of the subfolder'''
    if isSubFolder(folder):
        return folder
    else:
        levels = labelLevels(folder)
        return levels.subFolder
    
def getPrintFolder(folder:str, **kwargs) -> str:
    if 'levels' in kwargs:
        return kwargs['levels'].printFolder()
    
    if isPrintFolder(folder):
        return folder
    
    levels = labelLevels(folder)
    return levels.printFolder()
 
    
def listDirs(folder:str) -> List[str]:
    '''List of directories in the folder'''
    return [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f)) ]

def anyIn(slist:List[str], s:str) -> bool:
    '''bool if any of the strings in slist are in s'''
    if len(slist)==0:
        return True
    for si in slist:
        if si in s:
            return True
    return False

def allIn(slist:List[str], s:str) -> bool:
    '''bool if all of the strings are in s'''
    if len(slist)==0:
        return True
    for si in slist:
        if not si in s:
            return False
    return True
    
def printFolders(topFolder:str, tags:List[str]=[''], someIn:List[str]=[], **kwargs) -> List[str]:
    '''Get a list of bottom level print folders in the top folder'''
    if isPrintFolder(topFolder):
        if allIn(tags, topFolder) and anyIn(someIn, topFolder):
            folders = [topFolder]
        else:
            folders = []
    else:
        folders = []
        dirs = listDirs(topFolder)
        for d in dirs:
            folders = folders+printFolders(d, tags=tags, someIn=someIn)
    return folders


def isStitch(file:str) ->bool:
    '''determine if the file is a stitched image'''
    if not '.png' in file:
        return False
    if '_vid_' in file or '_vstill_' in file:
        return False
    for st in allStFiles():
        if f'_{st}_' in file:
            return True
        
def searchFolder(topFolder:str, bottom:str)->str:
    '''find the full path within the folder'''
    for f in os.listdir(topFolder):
        file = os.path.join(topFolder, f, bottom)
        if os.path.exists(file):
            return file
    return ''
    
def findFullFN(bn:str, topFolder:str) -> str:
    '''given a basename, find the full file name'''
    if not '_vstill_' in bn or not '_I_' in bn or not '_S_' in bn:
        raise FileNotFoundError(f'Cannot find full file name for {bn}')
    b, ext = os.path.splitext(bn)
    spl = re.split('_', b)
    vs = spl.index('vstill')
    sbpname = '_'.join(spl[:vs])
    i = spl.index('I')
    s = spl.index('S')
    testname = '_'.join(spl[i:s+3])
    samplename = '_'.join(spl[i:s+2])
    bottom = os.path.join(samplename, testname, sbpname, bn)
    file = searchFolder(topFolder, bottom)
    if os.path.exists(file):
        return file
    for j in [2,3]:
        testname = '_'.join(spl[i:s+2]+[f'v{j}']+[spl[s+2]])
        bottom = os.path.join(samplename, testname, sbpname, bn)
        file = searchFolder(topFolder, bottom)
        if os.path.exists(file):
            return file
    raise FileNotFoundError(f'Could not find {bn} in {topFolder}')

#------------    

def checkFolders(topFolder:str) -> Tuple[list, list]:
    '''check the print folders in the top folder for extra files, mis-sorted files'''
    mismatch = []
    tooMany = []
    for f in printFolders(topFolder):
        pfd = printFileDict(f)
        pfd.check()               # check files
        if len(pfd.mismatch)>0:
            mismatch.append(f)
        if len(pfd.tooMany)>0:
            tooMany.append(f)
    return mismatch, tooMany

    
        
    

#------------------------------------------------------------


        
def putStillsAway(folder:str, debug:bool=False) -> None:
    '''identify the stills and put them in a folder'''
    if not os.path.exists(folder):
        return
    if "Thumbs" in folder or "DS_Store" in folder or 'README' in folder:
        return
    levels = labelLevels(folder)
    if levels.currentLevel in ['sampleTypeFolder', 'printTypeFolder', 'sampleFolder', 'subFolder']:
        for f in os.listdir(folder):
            putStillsAway(os.path.join(folder, f), debug=debug) # sort the folders inside sampleTypeFolder
    elif levels.currentLevel in ['sbpFolder']:    
        pfd = printFileDict(folder)
        snap = os.path.join(folder, 'raw')
        if not os.path.exists(snap):
            os.mkdir(snap)
        if debug:
            print('old names: ')
            print(pfd.still_unknown)
            print('new names: ')
            print([os.path.join(snap, os.path.basename(file)) for file in pfd.still_unknown])
        else:
            for file in pfd.still_unknown:
                os.rename(file, os.path.join(snap, os.path.basename(file)))

            
#------------------------------------------------------
      
def mkdirif(folder:str) -> None:
    if not os.path.exists(folder):
        os.mkdir(folder)

def fillDirsTriple(topFolder:str) -> None:
    '''make missing folders'''
    for s in ['0.500', '0.625', '0.750', '0.875', '1.000', '1.250']:
        for f in ['tripleLinesHoriz', 'tripleLinesVert']:
            mkdirif(os.path.join(topFolder, f'{f}_{s}'))
        for z in ['0', '0.5']:
            for f in ['crossDoubleHoriz', 'crossDoubleVert']:
                mkdirif(os.path.join(topFolder, f'{f}_{z}_{s}'))
        for z in ['+y', '+z']:
            for f in ['tripleLinesXS']:
                mkdirif(os.path.join(topFolder, f'{f}_{z}_{s}'))
    
#-------------------------------
# singleLines
def countFiles(topFolder:str, diag:bool=True) -> pd.DataFrame:
    '''Identify which print folders in the top folder are missing data'''
    folders = printFolders(topFolder)
    outlist = []
    for folder in folders:
        d = {'Folder':os.path.basename(folder), 'BasVideo':0, 'BasStills':0, 'PhoneCam':0, 'Fluigent':0}
        for f in os.listdir(folder):
            if ('Basler camera' in f or 'horiz' in f or 'xs' in f or 'vert' in f) and '.png' in f:
                d['BasStills']+=1
            elif 'Basler camera' in f and '.avi' in f:
                d['BasVideo']+=1
            elif 'phoneCam' in f:
                d['PhoneCam']+=1
            elif 'Fluigent' in f:
                d['Fluigent']+=1
        outlist.append(d)        
    df = pd.DataFrame(outlist)
    if diag:
        display(df[df['PhoneCam']==0])
        display(df[df['BasVideo']==0])
        display(df[df['BasStills']==0])
    return df

#--------------------------------
# looping functions

class folderLoop:
    '''loops a function over all printFolders in the topFolder. 
    the function needs to have only one arg, folder, and all other variables need to go in kwargs
    folders could be either the top folder to recurse into, or a list of folders'''
    
    def __init__(self, folders:Union[str, list], func, mustMatch:list=[], canMatch:list=[], printTraceback:bool=False, **kwargs):
        if type(folders) is list:
            # list of specific folders
            self.folders = folders
        else:
            # top folder, recurse
            self.topFolder = folders
            self.folders = printFolders(folders)
        self.func = func
        self.mustMatch = mustMatch
        self.canMatch = canMatch
        self.kwargs = kwargs
        self.printTraceback = printTraceback
        
    def runFolder(self, folder:str) -> None:
        '''run the function on one folder'''
        if not isPrintFolder(folder):
            return
        
        # check that the folder matches all keys
        if not allIn(self.mustMatch, folder):
            return
        
        if not anyIn(self.canMatch, folder):
            return

        try:
            self.func(folder, **self.kwargs)
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            self.errorList.append(folder)
            print(e)
            if self.printTraceback:
                traceback.print_exc()

        
    def run(self) -> list:
        '''apply the function to all folders'''
        self.errorList = []
        for folder in self.folders:
            self.runFolder(folder)
        return self.errorList

        
    