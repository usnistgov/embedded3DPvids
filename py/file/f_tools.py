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


# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



#----------------------------------------------

def openExplorer(folder:str) -> None:
    '''open the folder in explorer'''
    subprocess.Popen(['explorer', folder.replace(r"/", "\\")], shell=True);

def twoBN(file:str) -> str:
    '''get the basename and folder it is in'''
    if len(file)==0:
        return ''
    return os.path.join(os.path.basename(os.path.dirname(file)), os.path.basename(file))

def numeric(s:str) -> bool:
    '''determine if the string is a number'''
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True
    
def dateOrTime(s:str) -> bool:
    '''determine if the string starts with a date'''
    if len(s)<6:
        return False
    if not numeric(s[:6]):
        return False
    if int(s[0])>2:
        # let me just introduce a Y3K bug into my silly little program
        return False
    return True

def toInt(s:str) -> Union[str,int]:
    '''convert to int if possible'''
    try:
        return int(s)
    except ValueError:
        return s


def fileDateAndTime(file:str, out:str='str') -> Tuple[Union[str,int]]:
    '''get the date and time of the file'''
    errorRet = '','',''
    if len(file)==0:
        return errorRet
    split = re.split('_|\.', os.path.basename(file))
    i = len(split)-1
    while not dateOrTime(split[i]):
        i=i-1
        if i<0:
            levels = labelLevels(file)
            if not levels.currentLevel in ['printTypeFolder', 'sampleTypeFolder', 'sampleFolder']:
                return fileDateAndTime(os.path.dirname(file))
            else:
                return errorRet
    if len(split[i-1])==6:
        # valid time and date
        time = split[i][0:6]
        date = split[i-1][0:6]
        if len(split)>i+1:
            v = split[i+1]
            try:
                int(v)
            except ValueError:
                v = ''
        else:
            v = ''
    else:
        # only date
        date = split[i][0:6]
        time = ''
        v = ''
    
    if out=='int':
        date = toInt(date)
        time = toInt(time)
        v = toInt(v)
    return date,time, v

def fileTime(file:str, out:str='str') -> str:
    '''get the time from the file, where the time is in the filename'''    
    return fileDateAndTime(file, out=out)[1]


def fileTimeV(file:str) -> str:
    '''get the time and number from the file'''
    errorRet = ''
    _,t,v = fileDateAndTime(file)
    if not len(t)==6:
        return errorRet
    if len(v)==0:
        return t
    else:
        return f'{t}_{v}'

def fileDate(file:str, out:str='str') -> str:
    '''Get the creation date from the file'''
    date = fileDateAndTime(file, out=out)[0]
    if (type(date) is str and not len(date)==6):
        # if the date is not a valid date, get the date from the file modified time
        date = time.strftime("%y%m%d", time.gmtime(os.path.getctime(file)))

    return date


def fileScale(file:str) -> str:
    '''get the scale from the file name'''
    if 'vid' in os.path.basename(file):
        return '1'
    if 'stitch' in os.path.basename(file):
        spl = re.split('stitch', os.path.basename(file))[-1]
        scale = re.split('_', spl)[1]
        return str(float(scale))
    try:
        scale = re.split('_', os.path.basename(file))[-2]
        if len(scale)==6:
            # this is a date
            return 1
        else:
            try:
                s2 = float(scale)
            except:
                return 1
            else:
                return s2
    except:
        scale = 1
    return str(scale)

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
    '''go up folders until you get a string that contains the sample name'''
    bn = os.path.basename(file)
    if 'I_' in bn and '_S' in bn:
        return bn
    else:
        parent = os.path.dirname(file)
        bn = os.path.basename(parent)
        if 'I_' in bn and '_S' in bn:
            return bn
        else:
            parent = os.path.dirname(parent)
            bn = os.path.basename(parent)
            if 'I_' in bn and '_S' in bn:
                return bn
            else:
                raise f'Could not determine sample name for {file}'

def sampleName(file:str, formatOutput:bool=True) -> str:
    '''Get the name of the sample from a file name or its parent folder'''

    parent = sampleRecursive(file)
    if '_I_' in parent:
        istr = '_I_'
    else:
        istr = 'I_'
    split1 = re.split(istr, parent)[1]
    split2 = re.split('_S_', split1)
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
    
def splitName(fname:str) -> List[str]:
    '''drop the version number from sbp name in files'''
    spl = re.split('_', fname)
    while spl[0][-1].isnumeric():
        spl[0] = spl[0][:-1]
    return spl

def splitFileName(file:str) -> str:
    exspl = os.path.splitext(os.path.basename(file))
    ext = exspl[-1]
    fname = exspl[0]
    spl = splitName(fname)
    return spl

def sbpPath(topFolder:str, name:str) -> str:
    '''find the full path of the sbp file, where name is just the file name, no extension'''
    if '.sbp' in name:
        name = re.split('.sbp', name)[0]
    fn = os.path.join(topFolder, f'{name}.sbp')
    if os.path.exists(fn):
        # found the file in this folder
        return fn
    else:
        # go through subfolders and look for the file
        for d in os.listdir(topFolder):
            dfull = os.path.join(topFolder, d)
            if os.path.isdir(dfull):
                s = sbpPath(dfull, name)
                if len(s)>0:
                    return s
    # didn't find any file
    return ''

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
