#!/usr/bin/env python
'''Functions for handling files'''

# external packages
import os
import re
import shutil
import time
from typing import List, Dict, Tuple, Union, Any, TextIO
import logging
import pandas as pd

# local packages
from config import cfg

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# info
__author__ = "Leanne Friedrich"
__copyright__ = "This data is publicly available according to the NIST statements of copyright, fair use and licensing; see https://www.nist.gov/director/copyright-fair-use-and-licensing-statements-srd-data-and-software"
__credits__ = ["Leanne Friedrich"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Leanne Friedrich"
__email__ = "Leanne.Friedrich@nist.gov"
__status__ = "Development"

#----------------------------------------------

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
        sample = 'I_M'+formatSample(ink)+'_S_'+formatSample(sup)
    else:
        sample = 'M'+ink+'Lap'+sup
    return sample

def sampleName(file:str, formatOutput:bool=True) -> str:
    '''Get the name of the sample from a file name or its parent folder'''

    if 'I' in file and '_S' in os.path.basename(file):
        parent = os.path.basename(file)
    else:
        if formatOutput:
            parent = os.path.basename(os.path.dirname(file))
            if not ('I' in parent and '_S' in parent):
                raise ValueError(f'Could not determine sample name for {file}')
        else:
            # we're not formatting the output, so we need the exact sample name for this file
            if 'M' in os.path.basename(file) and 'Lap' in os.path.basename(file):
                return mlapSample(file, formatOutput)
            else:
                raise ValueError(f'Could not determine sample name for {file}')
    split1 = re.split('I', parent)[1]
    split2 = re.split('_S', split1)
    ink = split2[0]
    split3 = re.split('_', split2[1])
    if len(split3[0])==0:
        sup = '_'+split3[1]
    else:
        sup = split3[0]
    if formatOutput:
        sample = 'I_'+formatSample(ink)+'_S_'+formatSample(sup)
    else:
        sample = 'I'+ink+'_S'+sup
    return sample

#---------------------------------------------            

def renameSubFolder(folder:str, includeDate:bool=True, debug:bool=False) -> str:
    '''Given a subfolder inside of a sample designation, rename it to include the date, or set includeDate=False to not include the date. Returns new name'''
    
    if not os.path.isdir(folder):
        return folder
    f2 = os.listdir(folder)
    if len(f2)==0:
        return folder
    else:
        if f2[0].startswith('Thumbs'):
            if len(f2)>1:
                f2 = f2[1]
            else:
                return folder
        else:
            f2 = f2[0]      
    
    parent = os.path.dirname(folder)
    basename = os.path.basename(folder)

    # rename sample for consistent formatting
    sample = sampleName(folder, formatOutput=False)
    formattedSample = sampleName(folder, formatOutput=True)
    basename = basename.replace(sample, formattedSample)
        
    if includeDate:    
        date = fileDate(f2)
        if not date in basename:
            basename = basename+'_'+date
    newname = os.path.join(parent, basename)
    
    # if the new name is different, rename the folder
    if not newname==folder:
        if debug:
            logging.debug(folder, '\n', newname)
        else:
            if os.path.exists(newname):
                # if the new folder already exists, combine
                for f in os.listdir(folder):
                    ff = os.path.join(folder, f)
                    shutil.move(ff, newname)
                try:
                    os.rmdir(folder)
                except:
                    logging.debug(f'Cannot remove {folder}')
            else:
                # if the new folder doesn't exist, rename
                os.rename(folder, newname)
    return newname

def renameFile(file:str, debug:bool=False) -> str:
    '''Find new name for file to follow convention. Returns the new full path'''
    basename = os.path.basename(file)
    dirname = os.path.dirname(file)
    
    # format the sample name, if there is one
    try:
        sample = sampleName(file, formatOutput=False)
        formattedSample = sampleName(file, formatOutput=True)
        basename = basename.replace(sample, formattedSample)
    except:
        pass
    
    if '.jpg' in file:
        # cell phone pic
        sample = sampleName(file, formatOutput=True)
        date = fileDate(file)
        if not (sample in basename and date in basename):
            time = re.split('_', os.path.splitext(basename)[0])
            if len(time)>2:
                raise NameError(f'File name {file} does not follow phone format')
            else:
                time = time[1]
            basename = f'phoneCam_{sample}_{date}_{time}.jpg'
    newname = os.path.join(dirname, basename)
    if not file==newname:
        if debug:
            logging.debug(file,'\n', newname)
        else:
            if os.path.exists(newname):
                os.remove(file)
            else:
                os.rename(file, newname)
    return newname
    
#-----------------------------------------------------

def isSubFolder(file:str, debug:bool=False) -> bool:
    '''determine if the folder is a subfolder'''
    if not os.path.exists(file):
        return False # path must exist
    if not os.path.isdir(file):
        return False # path must be a directory
    f = os.path.basename(file)
    for f1 in os.listdir(file):
        if not f1 in ['raw']:
            if os.path.isdir(os.path.join(file, f1)):
                return False # this folder contains directories. is not a subfolder
    entries = re.split('_', f)
    if len(entries[-1])==6 and entries[-1][0]=='2':
        # this is a date. file is named as a subfolder.
        return True
    else:
        # file is not named as a subfolder, but should be.
        renameSubFolder(file, includeDate=True, debug=debug)
        return True

def putInSampleFolder(file:str, debug:bool=False) -> str:
    '''puts the subfolder in a sample folder. returns name of sample folder'''
    sample = sampleName(file, formatOutput=False)
    sampleFolder = os.path.join(os.path.dirname(file), sample)
    if not os.path.exists(sampleFolder):
        os.mkdir(sampleFolder)
    newname = os.path.join(sampleFolder, os.path.basename(file))
    if debug:
        logging.debug(f'Old name: {file}\n  New name:{newname}')
    else:
        os.rename(file, newname)
    return sampleFolder
        
def putInSubFolder(file:str) -> None:
    '''Put a file in a sample folder into a subfolder with the date'''
    if not os.path.exists(file):
        raise NameError(f'File does not exist: {file}')
        
    # correct file name if needed
    file = renameFile(file)
        
    # create subfolders
    parent = os.path.dirname(file)
    sample = sampleName(file, formatOutput=True)
    subfolder = sample +'_'+ fileDate(file)
    if os.path.basename(parent)==subfolder:
        # file is already in subfolder
        return
    subfolder = os.path.join(parent, subfolder)
    # create the subfolder if it doesn't already exist
    if not os.path.exists(subfolder):
        os.makedirs(subfolder, exist_ok=True)
       
    newname = os.path.join(subfolder, os.path.basename(file))
    os.rename(file, newname)
    logging.info(f'Moved {file} to {newname}')

#------------------------------------------------------------

def sortSubFolder(folder:str) -> None:
    '''sort and rename the files in the folder'''
    if not os.path.isdir(folder):
        return
    renameSubFolder(folder)
    for file in os.listdir(folder):
        renameFile(os.path.join(folder, file))
    
def sortSampleFolder(folder:str) -> None:
    '''Sort a sample folder (e.g. I_2.25_S_2.25) into subfolders based on date'''
    if not os.path.isdir(folder):
        return
    folder = renameSubFolder(folder, includeDate=False)
    if not os.path.exists(folder):
        return
    for file in os.listdir(folder):
        filefull = os.path.join(folder, file)
        if os.path.isdir(filefull):
            sortSubFolder(filefull)
        else:
            putInSubFolder(filefull)
    logging.info(f'Done sorting {folder}')
            
def sortGroupFolder(folder:str) -> None:
    '''Hierarchically sort all files in the folder (e.g. LapRD LapRD) into sample folders (I_2.25_S_2.25) and subfolders (I_2.25_S_2.25_210101)'''
    if not os.path.isdir(folder):
        return
    for f in os.listdir(folder):
        ffull = os.path.join(folder, f) # should be sample folder
        if isSubFolder(ffull): 
            ffull = putInSampleFolder(ffull) # if it's subfolder, put in sample folder
        sortSampleFolder(ffull)
    logging.info(f'Done sorting {folder}')
    
def sortDataFolder(folder:str) -> None:
    '''sort/rename all files in the data folder (e.g. singleLines)'''
    if not os.path.isdir(folder):
        return
    for f in os.listdir(folder):
        groupfolder = os.path.join(folder, f)
        fh.sortGroupFolder(groupfolder)
            
#------------------------------------------------------
        
def listDirs(folder:str) -> List[str]:
    '''List of directories in the folder'''
    return [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f)) ]
            
def subFolders(topFolder:str) -> List[str]:
    '''Get a list of bottom level subfolders in the top folder'''
    dirs = listDirs(topFolder)
    if len(dirs)==0:
        folders = [topFolder]
    else:
        folders = []
        for d in dirs:
            folders = folders+subFolders(d)
    return folders
            
            
def countFiles(topFolder:str, diag:bool=True) -> pd.DataFrame:
    '''Identify which subfolders in the top folder are missing data'''
    folders = subFolders(topFolder)
    outlist = []
    for folder in folders:
        d = {'Folder':os.path.basename(folder), 'BasVideo':0, 'BasStills':0, 'PhoneCam':0, 'Fluigent':0}
        for f in os.listdir(folder):
            if 'Basler camera' in f and '.png' in f:
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
        
    