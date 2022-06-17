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



#----------------------------------------------


def twoBN(file:str) -> str:
    '''get the basename and folder it is in'''
    if len(file)==0:
        return ''
    return os.path.join(os.path.basename(os.path.dirname(file)), os.path.basename(file))

def fileTime(file:str) -> str:
    '''get the time from the file, where the time is in the filename'''
    if len(file)==0:
        return ''
    split = re.split('_', os.path.basename(file))
    time = split[-1]
    if len(time)!=6:
        time = split[-2]
    return time


def fileTimeV(file:str) -> str:
    '''get the time and number from the file'''
    if len(file)==0:
        return ''
    try:
        bn = os.path.basename(file)
        if len(bn)==0:
            return ''
        spl = re.split('_', bn)
        n = re.split('\.', spl[-1])[0]
        ftv = f'{spl[-2]}_{n}'
    except:
        print(file)
    return ftv

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


def fileScale(file:str) -> str:
    '''get the scale from the file name'''
    if 'vid' in os.path.basename(file):
        return '1'
    if 'stitch' in os.path.basename(file):
        spl = re.split('stitch', os.path.basename(file))[-1]
        scale = re.split('_', spl)[1]
        return str(float(scale))
    try:
        scale = float(re.split('_', os.path.basename(file))[-2])
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
    while len(ld)>0 and ((os.path.isdir(d1) and not directory) or (not os.path.isdir(d1) and directory)) or 'Thumbs' in d1 or not os.path.exists(d1):
        # go through files in the directory until you find another directory
        d1 = os.path.join(folder, ld.pop(0))
    if (os.path.isdir(d1) and directory) or (not os.path.isdir(d1) and not directory):
        return d1
    else:
        return ''
    

class labelLevels:
    '''label the levels of the file hierarchy, with one characteristic file per level'''

    def __init__(self, file:str):
    
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
        if 'raw' in bfold or 'temp' in bfold:
            # raw image folder
            self.rawFile = bottom
            if os.path.basename(bfold)=='raw':
                self.rawFolder = bfold
                aboveRaw = os.path.dirname(self.rawFolder)
            elif os.path.basename(bfold)=='temp':
                self.tempFolder = bfold
                aboveRaw = os.path.dirname(self.tempFolder)
            else:
                self.rawLineFolder = bfold
                self.rawFolder = os.path.dirname(bfold)
                aboveRaw = os.path.dirname(self.rawFolder)

            if sampleInName(aboveRaw):
                self.file = os.path.join(aboveRaw, firstEntry(aboveRaw, directory=False)) # bottom level file inside sbpfolder
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
                self.file = os.path.join(aboveRaw, firstEntry(aboveRaw, directory=False)) # bottom level file inside sbpfolder
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

def isSubFolder(folder:str) -> bool:
    '''determine if the folder is a subfolder'''
    levels = labelLevels(folder)
    if levels['currentLevel']=='subFolder':
        return True
    else:
        return False
    
def subFolder(folder:str) -> str:
    '''name of the subfolder'''
    levels = labelLevels(folder)
    return levels['subFolder']
    
def isSBPFolder(folder:str) -> bool:
    '''determine if the folder is a sbpfolder'''
    levels = labelLevels(folder)
    if levels['currentLevel']=='sbpFolder':
        return True
    else:
        return False
    
#------------

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
    
def singleLineSBPfiles() -> dict:
    '''get a dictionary of singleLine sbp file names and their shortcuts'''
    files = {'singleLinesNoZig':'SLNZ',
            'singleLines':'SL'}
    return files

def singleLineSBPPicfiles() -> dict:
    '''get a dictionary of singleLine pic sbp file names and their shortcuts'''
    files = dict([[f'singleLinesPics{i}',f'SLP{i}'] for i in range(10)])
    return files
    
def singleLineSt() -> list:
    '''get a list of single line object types'''
    return ['horiz', 'vert', 'xs']

def singleLineStN() -> list:
    '''get a list of single line stitch names'''
    return ['horizfull', 'vert1', 'vert2', 'vert3', 'vert4', 'xs1', 'xs2', 'xs3', 'xs4', 'xs5', 'horiz0', 'horiz1', 'horiz2']

    
#------------

def isStitch(file:str) ->bool:
    '''determine if the file is a stitched image'''
    if not '.png' in file:
        return False
    if '_vid_' in file or '_vstill_' in file:
        return False
    for st in singleLineStN():
        if f'_{st}_' in file:
            return True
    
def isVidStill(file:str) ->bool:
    '''determine if the file is a video still'''
    if not '.png' in file:
        return False
    if '_vstill_' in file:
        return True
    for st in singleLineStN():
        if f'_vid_{st}' in file:
            return True
    return False

def isStill(file:str) -> bool:
    '''determine if the file is an unstitched image'''
    exspl = os.path.splitext(os.path.basename(file))
    ext = exspl[-1]
    fname = exspl[0]
    spl = re.split('_', fname)
    if spl[0] in tripleLineSBPPicfiles() or spl[0] in singleLineSBPPicfiles():
        return True
    else:
        return False
    
#------------    

class printFileDict:
    '''get a dictionary of the paths for each file inside of the print folder'''
    
    def __init__(self, printFolder:str):
        levels = labelLevels(printFolder)
        if not levels['currentLevel'] in ['subFolder', 'sbpFolder']:
            raise ValueError(f'Input to labelPrintFiles must be subfolder or sbp folder')

        printFolder = levels['currentLevel']
        self.printFolder=levels[printFolder]
        self.vid_unknown=[]
        self.csv_unknown=[]
        self.csv_delete=[]
        self.still=[]
        self.stitch=[]
        self.vstill=[]
        self.still_unknown=[]
        self.unknown=[]
        self.phoneCam = []
        tls = tripleLineSBPfiles()
        sls = singleLineSBPfiles()

        for f1 in os.listdir(self.printFolder):
            ffull = os.path.join(self.printFolder, f1)
            if not os.path.isdir(ffull) and not 'Thumbs' in f1:
                exspl = os.path.splitext(f1)
                ext = exspl[-1]
                fname = exspl[0]
                spl = re.split('_', fname)
                if ext=='.avi':
                    if 'Basler camera' in fname:
                        if spl[0] in tls:
                            self.printType='tripleLine'
                            self.vid = ffull
                        elif spl[0] in sls:
                            self.printType='singleLine'
                            self.vid = ffull
                        else:
                            self.vid_unknown.append(ffull)
                    else:
                        self.vid_unknown.append(ffull)
                elif ext=='.csv':
                    if spl[0] in tls or spl[0] in sls:
                        if 'Fluigent' in fname:
                            self.fluigent = ffull
                        elif 'speeds' in fname:
                            self.speeds=ffull
                        else:
                            print(spl[0])
                            self.csv_unknown.append(ffull)
                    elif spl[0] in singleLineSBPPicfiles():
                        # extraneous fluigent or speed file from pic
                        self.csv_delete.append(ffull)
                    else:
                        # summary or analysis file
                        setattr(self, spl[-1], ffull)
                elif ext=='.png':
                    if isStill(f1):
                        if 'Basler camera' in fname:
                            # raw still
                            self.still.append(ffull)
                        else:
                            self.still_unknown.append(ffull)
                    elif isStitch(f1):
                        # stitched image
                        self.stitch.append(ffull)
                    elif isVidStill(f1):
                        self.vstill.append(ffull)
                    else:
                        self.still_unknown.append(ffull)
                elif ext=='.jpg':
                    self.phoneCam.append(ffull)
                else:
                    self.unknown.append(ffull) 
                    
                    
    def printAll(self) -> None:
        '''print all values'''
        for key,value in self.__dict__.items():
            if type(value) is list:
                if len(value)>0:
                    if os.path.exists(value[0]):
                        print(f'\n{key}:\t{[os.path.basename(file) for file in value]}\n')
                    else:
                        print(f'{key}:\t{value}')
            else:
                if os.path.exists(value) and not key=='printFolder':
                    print(f'{key}:\t{os.path.basename(value)}')
                else:
                    print(f'{key}:\t{value}')

    

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
        
    