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

def allSBPFiles():
    return {**singleDisturbSBPfiles(), **SDTSBPfiles(), **tripleLineSBPfiles(), **singleLineSBPfiles()}

def allStFiles():
    return singleLineStN()+tripleLineSt()+singleDisturbSt()+SDTSt()


def isTripleLineStill(file):
    spl = splitFileName(file)
    return spl[0] in tripleLineSBPPicfiles()

def isSingleLineStill(file):
    spl = splitFileName(file)
    return spl[0] in singleLineSBPPicfiles()

def isSDTStill(file):
    spl = splitFileName(file)
    return spl[0] in SDTSBPPicfiles()

def isVidStill(file:str) ->bool:
    '''determine if the file is a video still'''
    if not '.png' in file:
        return False
    if '_vstill_' in file:
        return True
    for st in allStFiles():
        if f'_vid_{st}' in file:
            return True
    return False

def isStill(file:str) -> bool:
    '''determine if the file is an unstitched image'''
    return isTripleLineStill(file) or isSingleLineStill(file) or isSDTStill(file)

def isStitch(file:str) -> bool:
    '''determine if the file is a stitched image'''
    if not '.png' in file:
        return False
    if '_stitch_' in file:
        return True
    if not 'singleLine' in file:
        return False
    bn = os.path.basename(file)
    if 'xs' in bn or 'horiz' in bn or 'vert' in bn:
        return True
    return False


def printType(bn:str) -> str:
    if bn in tripleLineSBPfiles():
        return 'tripleLine'
    elif bn in singleLineSBPfiles():
        printType='singleLine'
    elif bn in SDTSBPfiles():
        printType = 'SDT'
    elif bn in singleDisturbSBPfiles():
        printType = 'singleDisturb'
    else:
        raise ValueError(f'Unknown sbp file {bn} passed to printType') 
    return printType

#--------------

def singleDisturbSBPfiles() -> dict:
    '''get a dictionary of singleDisturb sbp file names and their shortcuts'''
    files = {'disturbHoriz':'DH', 'disturbVert':'DV', 'disturbXS':'DX'}
    return files

def singleDisturbSt() -> list:
    '''get a list of disturbed line object types'''
    return ['HIx', 'HOx', 'HOh', 'V']

def singleDisturb2FileDict() -> str:
    '''get the corresponding object name and sbp file'''
    d = {'HIx':'disturbXS_+y', 
        'HOx':'disturbXS_+z',
        'HOh':'disturbHoriz',
        'V':'disturbVert'}
    return d

def singleDisturbName(file:str) -> str:
    '''get the short name, given the sbp file name'''
    d = singleDisturb2FileDict()
    for key,val in d.items():
        if val in file:
            return key
    raise ValueError(f'Unexpected sbp file name: {file}')

#--------------

def SDTSBPfiles() -> dict:
    '''get a dictionary of singleDoubleTriple sbp file names and their shortcuts'''
    files = {'disturbHoriz':'DH', 'disturbVert':'DV', 'disturbXS':'DX'}
    return files

def SDTSt() -> list:
    '''get a list of disturbed line object types'''
    return [f'{s}{i}' for s in ['HIx', 'HOx', 'HOh', 'V'] for i in [1,2,3]]

def SDTSBPPicfiles() -> dict:
    '''get a dictionary of triple line pic file names and their shortcuts'''
    tls = SDTSBPfiles()
    files = dict([[f'{key}',f'{val}P'] for key,val in tls.items()])
    return files

def SDT2FileDict() -> str:
    '''get the corresponding object name and sbp file'''
    d = {}
    for i in [1,2,3]:
        d[f'HIx{i}'] = f'disturbXS2_{i}_+y'
        d[f'HOx{i}'] = f'disturbXS2_{i}_+z'
        d[f'HOh{i}'] = f'disturbHoriz3_{i}'
        d[f'V{i}'] = f'disturbVert2_{i}'
    return d

def SDTName(file:str) -> str:
    '''get the short name, given the sbp file name'''
    d = SDT2FileDict()
    for key,val in d.items():
        if val in file:
            return key
    raise ValueError(f'Unexpected sbp file name: {file}')


#---

def tripleLineSBPfiles() -> dict:
    '''get a dictionary of tripleLine sbp file names and their shortcuts'''

    # C = cross, D = double, H = horiz, V = vert, U = under, TL = triple line, X = cross-section
    files = {'crossDoubleHoriz':'CDH',
             'crossDoubleVert':'CDV',
             'crossUnder':'CU',
             'tripleLinesHoriz':'TLH',
             'tripleLinesXS':'TLX',
             'tripleLinesVert':'TLV',
             'tripleLinesUnder':'TLU'}
    return files

def tripleLineSBPPicfiles() -> dict:
    '''get a dictionary of triple line pic file names and their shortcuts'''
    tls = tripleLineSBPfiles()
    files = dict([[f'{key}Pics',f'{val}P'] for key,val in tls.items()])
    return files

def tripleLineSt() -> list:
    '''get a list of triple line object types'''
    # H = horiz, V = vertical
    # I = in layer, O = out of layer
    # P = parallel, B = bridge, C = cross
    return ['HIB', 'HIPh', 'HIPxs', 'HOB', 'HOC', 'HOPh', 'HOPxs', 'VB', 'VC', 'VP']

def tripleLine2FileDict() -> str:
    '''get the corresponding object name and sbp file'''
    d = {'HOB':'crossDoubleHoriz_0.5',
         'HOC':'crossDoubleHoriz_0',
         'VB':'crossDoubleVert_0.5',
         'VC':'crossDoubleVert_0',
         'HIC':'underCross_0.5',
         'HIB':'underCross_0',
         'HIPxs':'tripleLinesXS_+y',
         'HOPxs':'tripleLinesXS_+z',
         'HIPh':'tripleLinesUnder',
         'HOPh':'tripleLinesHoriz',
         'VP':'tripleLinesVert'}
    return d

def tripleLineName(file:str) -> str:
    '''get the short name, given the sbp file name'''
    d = tripleLine2FileDict()
    for key,val in d.items():
        if val in file:
            return key
    raise ValueError(f'Unexpected sbp file name: {file}')

#---

def singleLineSBPfiles() -> dict:
    '''get a dictionary of singleLine sbp file names and their shortcuts'''
    files = {'singleLinesNoZig':'SLNZ',
            'singleLines':'SL'}
    return files

def singleLineSBPPicfiles() -> dict:
    '''get a dictionary of singleLine pic sbp file names and their shortcuts'''
    files = dict([[f'singleLinesPics{i}',f'SLP{i}'] for i in range(10)])
    files['singleLinesPics'] = 'SLP'
    return files

def singleLineSt() -> list:
    '''get a list of single line object types'''
    return ['horiz', 'vert', 'xs']

def singleLineStN() -> list:
    '''get a list of single line stitch names'''
    return ['horizfull', 'vert1', 'vert2', 'vert3', 'vert4', 'xs1', 'xs2', 'xs3', 'xs4', 'xs5', 'horiz0', 'horiz1', 'horiz2']

def singleLineStPics(vertLines:int, xsLines:int) -> List[str]:
    '''get a list of strings that describe the stitched pics'''
    l = ['horizfull']
    l = l+[f'vert{c+1}' for c in range(vertLines)]
    l = l+[f'xs{c+1}' for c in range(xsLines)]
    return l
    
def isSBPFolder(folder:str) -> bool:
    '''determine if the folder is a sbpfolder'''
    bn = os.path.basename(folder)
    if 'Pics' in bn:
        return False
    for s in allSBPFiles():
        if s in bn:
            return True
    
def isPrintFolder(folder:str) -> bool:
    '''determine if the folder is the print folder'''
    if os.path.exists(os.path.join(folder, 'raw')):
        return True  

    # has tripleLines SBP name in basename
    if isSBPFolder(folder):
        return True

    if sampleInName(folder) and dateInName(folder) and 'singleLines' in folder:
        # could be a printFolder if singleLine
        return True

    return False

