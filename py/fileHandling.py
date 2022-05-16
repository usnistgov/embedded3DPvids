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

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
from config import cfg

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# info
__author__ = "Leanne Friedrich"
__copyright__ = "This data is publicly available according to the NIST statements of copyright, fair use and licensing; see https://www.nist.gov/director/copyright-fair-use-and-licensing-statements-srd-data-and-software"
__credits__ = ["Leanne Friedrich"]
__license__ = "NIST"
__version__ = "1.0.0"
__maintainer__ = "Leanne Friedrich"
__email__ = "Leanne.Friedrich@nist.gov"
__status__ = "Development"

#----------------------------------------------

def fileTime(file:str) -> str:
    '''get the time from the file, where the time is in the filename'''
    split = re.split('_', os.path.basename(file))
    time = split[-1]
    if len(time)!=6:
        time = split[-2]
    return time

def fileDate(file:str) -> str:
    '''Get the creation date from the file'''
    split = re.split('_', os.path.basename(file))
    if len(split)>1:
        date = split[-2] # get the date from the file name (useful for files created by ShopbotPyQt GUI)
    else:
        date = ''
    if not len(date)==6:
        # if the date is not valid, choose the beginning of the file name (useful for cell phone pics)
        date = split[0]
        if len(date)==8:
            date = date[2:]
        if not len(date)==6:
            # if the date is not a valid date, get the date from the file modified time
            date = time.strftime("%y%m%d", time.gmtime(os.path.getctime(file)))
    return date

def formatSample(s:str) -> str:
    '''convert a string for use in I_#_S_# format'''
    while s[0]=='_':
        s = s[1:]
    while s[-1]=='_':
        s = s[0:-1]
    try:
        sf = float(s)
    except:
        pass
    else:
        s = "{:.2f}".format(sf)
    return s

def mlapSample(file:str, formatOutput:bool=True) -> str:
    '''determine the sample name, where the sample was named as M#_Lap# instead of IM#_S#'''
    parent = os.path.basename(file)
    split1 = re.split('M', parent)[1]
    split2 = re.split('Lap', split1)
    ink = split2[0]
    split3 = re.split('_', split2[1])
    if len(split3[0])==0:
        sup = '_'+split3[1]
    else:
        sup = split3[0]
    if formatOutput:
        sample = f'I_M{formatSample(ink)}_S_{formatSample(sup)}'
    else:
        sample = f'M{ink}Lap{sup}'
    return sample

def sampleRecursive(file:str) -> str:
    '''get the string that contains the sample name'''
    bn = os.path.basename(file)
    if 'I' in bn and '_S' in bn:
        return bn
    else:
        parent = os.path.dirname(file)
        bn = os.path.basename(parent)
        if 'I' in bn and '_S' in bn:
            return bn
        else:
            parent = os.path.dirname(parent)
            bn = os.path.basename(parent)
            if 'I' in bn and '_S' in bn:
                return bn
            else:
                raise f'Could not determine sample name for {file}'

def sampleName(file:str, formatOutput:bool=True) -> str:
    '''Get the name of the sample from a file name or its parent folder'''

    parent = sampleRecursive(file)
    split1 = re.split('I', parent)[1]
    split2 = re.split('_S', split1)
    ink = split2[0]
    split3 = re.split('_', split2[1])
    if len(split3[0])==0:
        sup = f'_{split3[1]}'
    else:
        sup = split3[0]
    if formatOutput:
        sample = f'I_{formatSample(ink)}_S_{formatSample(sup)}'
    else:
        sample = f'I{ink}_S{sup}'
    return sample

#---------------------------------------------  

    
#-----------------------------------------------------

def dateInName(file:str) -> bool:
    '''this file basename ends in a date'''
    entries = re.split('_', os.path.basename(file))
    date = entries[-1]
    if not len(date)>=6:
        return False
    yy = float(date[0:2])
    mm = float(date[2:4])
    dd = float(date[4:])
    if mm>12:
        # bad month
        return False
    if dd>31:
        # bad day
        return False
    
    # passed all tests
    return True

def sampleInName(file:str) -> bool:
    '''this file basename contains a sample name'''
    entries = re.split('_', os.path.basename(file))
    if entries[0]=='I' and entries[2]=='S':
        return True
    else:
        return False
    
def firstEntry(folder:str, directory:bool=True) -> str:
    '''find the first entry in the folder. directory=True to find directories, False to find files'''
    ld = os.listdir(folder)
    if len(ld)==0:
        return ''
    d1 = ''
    while len(ld)>0 and not (os.path.isdir(d1) and directory):
        # go through files in the directory until you find another directory
        d1 = os.path.join(folder, ld.pop(0))
    if (os.path.isdir(d1) and directory) or (not os.path.isdir(d1) and not directory):
        return d1
    else:
        return ''
    

def labelLevels(file:str) -> dict:
    '''label the levels of the file hierarchy, with one characteristic file per level'''
    levels = {}
    bottom = file
    
    # recurse until you hit the bottom level
    while os.path.isdir(bottom): # bottom is a directory
        newbot = firstEntry(bottom, directory=True) # find the first directory
        if os.path.exists(newbot):
            # this is a directory
            bottom = newbot
        else:
            # no directories in bottom. find first file
            bottom = firstEntry(bottom, directory=False) 
        
            
    # bottom is now a bottom level file
    bfold = os.path.dirname(bottom)
    if 'raw' in bfold:
        # raw image folder
        levels['rawFile'] = bottom
        if os.path.basename(bfold)=='raw':
            levels['rawFolder'] = bfold
        else:
            levels['rawLineFolder'] = bfold
            levels['rawFolder'] = os.path.dirname(bfold)
        aboveRaw = os.path.dirname(levels['rawFolder'])
        if sampleInName(aboveRaw):
            levels['file'] = os.path.join(aboveRaw, firstEntry(aboveRaw, directory=False)) # bottom level file inside sbpfolder
            if dateInName(aboveRaw):
                # sample and date in name: this is a subfolder
                levels['subFolder'] = aboveRaw
            else:
                # this is a sample folder. these files are misplaced
                levels['sampleFolder'] = aboveRaw
                levels['subFolder'] = 'generate' # need to generate a new subfolder
        else:
            # no sample in name: shopbot folder
            levels['sbpFolder'] = aboveRaw
            levels['subFolder'] = os.path.dirname(aboveRaw)
            levels['file'] = os.path.join(aboveRaw, firstEntry(aboveRaw, directory=False)) # bottom level file inside sbpfolder
    else:
        # just a file. no raw folder, because we would have found it during recursion
        levels['file'] = bottom
        if sampleInName(bfold):
            if dateInName(bfold):
                # sample and date in name: this is a subfolder
                levels['subFolder'] = bfold
            else:
                # this is a sample folder. these files are misplaced
                levels['sampleFolder'] = bfold
        else:
            # no sample in name: shopbot folder
            levels['sbpFolder'] = bfold
            levels['subFolder'] = os.path.dirname(bfold)           
            
    if not 'sampleFolder' in levels:
        sabove = os.path.dirname(levels['subFolder'])
        if not sampleInName(sabove):
            levels['sampleFolder'] = 'generate' # need to generate a new sample folder
            levels['sampleTypeFolder'] = sabove # the type is right above the subFolder
        else:
            levels['sampleFolder'] = sabove
            levels['sampleTypeFolder'] = os.path.dirname(sabove)
    else:
        levels['sampleTypeFolder'] = os.path.dirname(levels['sampleFolder'])
    levels['printTypeFolder'] = os.path.dirname(levels['sampleTypeFolder'])
    
    for key in levels:
        if levels[key]==file:
            currentLevel = key
    levels['currentLevel'] = currentLevel
    
    return levels

#------------------------------------------------------------

def labelAndSort(folder:str, debug:bool=False) -> None:
    '''label the levels and create folders where needed'''
    levels = labelLevels(folder)
    if levels['sampleFolder'] =='generate':
        # generate a sample folder
        sample = sampleName(levels['subFolder'], formatOutput=False)
        sampleFolder = os.path.join(levels['sampleTypeFolder'], sample)
        if not os.path.exists(sampleFolder):
            if debug:
                logging.debug(f'Create {sampleFolder}')
            else:
                os.mkdir(sampleFolder)
        levels['sampleFolder'] = sampleFolder
        file = levels['subFolder']
        newname = os.path.join(levels['sampleFolder'], os.path.basename(file))
        if debug:
            logging.info(f'Move {file} to {newname}')
        else:
            os.rename(file, newname)
            logging.info(f'Moved {file} to {newname}')
    if levels['subFolder']=='generate':
        # generate a subfolder
        fd = fileDate(levels['file'])
        sample = os.path.basename(levels['sampleFolder']) 
        bn = f'{sample}_{fd}'
        subfolder = os.path.join(levels['sampleFolder'], bn)


def sortRecursive(folder:str, debug:bool=False) -> None:
    '''given any folder or file, sort and rename all the files inside'''
    if not os.path.exists(folder):
        return
    if "Thumbs" in folder or "DS_Store" in folder or 'README' in folder:
        return
    levels = labelLevels(folder)
    if levels['currentLevel'] =='sampleTypeFolder':
        for f in os.listdir(levels['sampleTypeFolder']):
            labelAndSort(os.path.join(levels['sampleTypeFolder'], f), debug=debug) # sort the folders inside sampleTypeFolder
    elif levels['currentLevel']=='printTypeFolder':
        for f in os.listdir(levels['printTypeFolder']):
            sortRecursive(os.path.join(levels['printTypeFolder'], f), debug=debug)
    elif levels['currentLevel'] in ['sampleFolder', 'subFolder', 'sbpFolder']:
        labelAndSort(folder, debug=debug)

            
#------------------------------------------------------
        
def listDirs(folder:str) -> List[str]:
    '''List of directories in the folder'''
    return [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f)) ]

def anyIn(slist:List[str], s:str) -> bool:
    '''bool if any of the strings in slist are in s'''
    for si in slist:
        if si in s:
            return True
    return False
            
def subFolders(topFolder:str, tags:List[str]=[''], **kwargs) -> List[str]:
    '''Get a list of bottom level subfolders in the top folder'''
    if isSubFolder(topFolder):
        if anyIn(tags, topFolder):
            folders = [topFolder]
        else:
            folders = []
    else:
        folders = []
        dirs = listDirs(topFolder)
        for d in dirs:
            folders = folders+subFolders(d, tags=tags)
    return folders
            
            
def countFiles(topFolder:str, diag:bool=True) -> pd.DataFrame:
    '''Identify which subfolders in the top folder are missing data'''
    folders = subFolders(topFolder)
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
        
    