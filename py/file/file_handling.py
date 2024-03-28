#!/usr/bin/env python
'''Functions for sorting and labeling files'''

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
from file_names import *
from print_file_dict import *
from levels import *
from folder_loop import *
from print_folders import *


# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

#----------------------------------------------

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
    d = -1
    while not len(spl[d])==6 and abs(d)>=len(spl):
        d = d-1
    testname = '_'.join(spl[i:d-1])   # include day
    samplename = '_'.join(spl[i:s+2]) # just I_XXX_S_XXX
    bottom = os.path.join(samplename, testname, sbpname, bn)   # e.g. I_XXX_S_XXX/I_XXX_S_XXX_230101/disturbblahblah/*vstill*.png
    file = searchFolder(topFolder, bottom)
    if os.path.exists(file):
        return file
    else:
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
    if not os.path.isdir(folder):
        return
    if "Thumbs" in folder or "DS_Store" in folder or 'README' in folder:
        return
    levels = labelLevels(folder)
    if levels.currentLevel in ['sampleTypeFolder', 'printTypeFolder', 'sampleFolder', 'subFolder']:
        for f in os.listdir(folder):
            putStillsAway(os.path.join(folder, f), debug=debug) # sort the folders inside sampleTypeFolder
    elif levels.currentLevel in ['sbpFolder']: 
        pfd = printFileDict(folder)
        pfd.sort()
        snap = os.path.join(folder, 'raw')
        if not os.path.exists(snap):
            os.mkdir(snap)
        if debug:
            print('old names: ')
            print(pfd.still)
            print('new names: ')
            print([os.path.join(snap, os.path.basename(file)) for file in pfd.still])
        else:
            for file in pfd.still:
                newname = os.path.join(snap, os.path.basename(file))
                if not newname==file:
                    os.rename(file, newname)
                
def labelAndSort(folder:str, debug:bool=False) -> None:
    '''label the levels and create folders where needed'''
    levels = labelLevels(folder)
    if levels.sampleFolder =='generate':
        # generate a sample folder
        sample = sampleName(levels.subFolder, formatOutput=True)
        sampleFolder = os.path.join(levels.sampleTypeFolder, sample)
        if not os.path.exists(sampleFolder):
            if debug:
                logging.debug(f'Create {sampleFolder}')
            else:
                os.mkdir(sampleFolder)
        levels.sampleFolder = sampleFolder
        file = levels.subFolder
        newname = os.path.join(levels.sampleFolder, os.path.basename(file))
        if debug:
            logging.info(f'Move {file} to {newname}')
        else:
            if not newname==file:
                os.rename(file, newname)
            logging.info(f'Moved {file} to {newname}')    
    if levels.subFolder=='generate':
        # generate a subfolder
        fd = fileDate(levels.file)
        sample = os.path.basename(levels.sampleFolder) 
        bn = f'{sample}_{fd}'
        subfolder = os.path.join(levels.sampleFolder, bn)


def sortRecursive(folder:str, debug:bool=False) -> None:
    '''given any folder or file, sort and rename all the files inside'''
    if not os.path.exists(folder):
        return
    if "Thumbs" in folder or "DS_Store" in folder or 'README' in folder:
        return
    levels = labelLevels(folder)
    if levels.currentLevel =='sampleTypeFolder':
        for f in os.listdir(levels.sampleTypeFolder):
            labelAndSort(os.path.join(levels.sampleTypeFolder, f), debug=debug) # sort the folders inside sampleTypeFolder
    elif levels.currentLevel=='printTypeFolder':
        for f in os.listdir(levels.printTypeFolder):
            sortRecursive(os.path.join(levels.printTypeFolder, f), debug=debug)
    elif levels.currentLevel in ['sampleFolder', 'subFolder', 'sbpFolder']:
        labelAndSort(folder, debug=debug)

            
#------------------------------------------------------
      
def mkdirif(folder:str) -> None:
    '''make a directory if it doesn't already exist'''
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


        
        
