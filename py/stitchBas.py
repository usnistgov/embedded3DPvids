#!/usr/bin/env python
'''Functions for stitching bascam stills'''

# external packages
import os, sys
import traceback
import logging
import pandas as pd
from typing import List, Dict, Tuple, Union, Any, TextIO

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
import fileHandling as fh
import stitching

# logging

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

STLIST = ['horiz', 'vert1', 'vert2', 'vert3', 'vert4', 'xs1', 'xs2', 'xs3', 'xs4', 'xs5']

class fileList:
    '''class that holds lists of files of different type'''
    
    def __init__(self, *args):
        '''args are folders'''
        
        self.folders = args
        self.resetList()
        self.stillGroups = [self.horizStill, self.vert1Still, self.vert2Still, self.vert3Still, self.vert4Still, self.xs1Still, self.xs2Still, self.xs3Still, self.xs4Still, self.xs5Still]
        for f in args:
            self.splitFiles(f)
            
    def resetList(self):
        '''empty the lists of files'''
        self.phoneStill = []
        self.basVideo = []
        self.basStill = []
        self.webcamVideo = []
        self.webcamStill = []
        self.fluigent = []
        
        self.horizStill = []
        self.vert1Still = []
        self.vert2Still = []
        self.vert3Still = []
        self.vert4Still = []
        self.xs1Still = []
        self.xs2Still = []
        self.xs3Still = []
        self.xs4Still = []
        self.xs5Still = []
        
        self.horizStitch = []
        self.vert1Stitch = []
        self.vert2Stitch = []
        self.vert3Stitch = []
        self.vert4Stitch = []
        self.xs1Stitch = []
        self.xs2Stitch = []
        self.xs3Stitch = []
        self.xs4Stitch = []
        self.xs5Stitch = []
        
    def splitFiles(self, folder:str) -> None:
        '''sort the files in the folder into the lists'''
        for f in os.listdir(folder):
            f1 = os.path.join(folder, f)
            if os.path.isdir(f1):
                self.splitFiles(f1)
            else:
                if 'Basler camera' in f:
                    if '.png' in f:
                        self.basStill.append(f1)
                    elif '.avi' in f:
                        self.basVideo.append(f1)
                elif 'phoneCam' in f:
                    self.phoneStill.append(f1)
                elif 'Fluigent' in f:
                    self.fluigent.append(f1)
                elif 'Nozzle camera' in f:
                    if '.avi' in f:
                        self.webcamVideo.append(f1)
                    elif '.png' in f:
                        self.webcamStill.append(f1)
                elif 'horiz' in f:
                    self.horizStitch.append(f1)
                elif 'vert1' in f:
                    self.vert1Stitch.append(f1)
                elif 'vert2' in f:
                    self.vert2Stitch.append(f1)
                elif 'vert3' in f:
                    self.vert3Stitch.append(f1)
                elif 'vert4' in f:
                    self.vert4Stitch.append(f1)
                elif 'xs1' in f:
                    self.xs1Stitch.append(f1)
                elif 'xs2' in f:
                    self.xs2Stitch.append(f1)
                elif 'xs3' in f:
                    self.xs3Stitch.append(f1)
                elif 'xs4' in f:
                    self.xs4Stitch.append(f1)
                elif 'xs5' in f:
                    self.xs5Stitch.append(f1)
        self.basStill.sort(key=fh.fileTime)
        if len(self.basStill)==49 and 'singleLinesPics' in self.basStill[0]:
            # we used the shopbot script to generate these images
            self.horizStill = self.basStill[0:9]
            self.vert1Still = self.basStill[9:14]
            self.vert2Still = self.basStill[14:19]
            self.vert3Still = self.basStill[19:24]
            self.vert4Still = self.basStill[24:29]
            self.xs1Still = self.basStill[29:33]
            self.xs2Still = self.basStill[33:37]
            self.xs3Still = self.basStill[37:41]
            self.xs4Still = self.basStill[41:45]
            self.xs5Still = self.basStill[45:49]
        elif len(self.basStill)==48 and 'singleLinesPics' in self.basStill[0]:
            # we used the shopbot script to generate these images
            self.horizStill = self.basStill[0:10]
            self.vert1Still = self.basStill[10:17]
            self.vert2Still = self.basStill[17:24]
            self.vert3Still = self.basStill[24:31]
            self.vert4Still = self.basStill[31:38]
            self.xs1Still = self.basStill[38:40]
            self.xs2Still = self.basStill[40:42]
            self.xs3Still = self.basStill[42:44]
            self.xs4Still = self.basStill[44:46]
            self.xs5Still = self.basStill[46:48]
        elif len(self.basStill)==58 and 'singleLinesPics' in self.basStill[0]:
            # we used the shopbot script to generate these images
            self.horizStill = self.basStill[0:10]
            self.vert1Still = self.basStill[10:17]
            self.vert2Still = self.basStill[17:24]
            self.vert3Still = self.basStill[24:31]
            self.vert4Still = self.basStill[31:38]
            self.xs1Still = self.basStill[38:42]
            self.xs2Still = self.basStill[42:46]
            self.xs3Still = self.basStill[46:50]
            self.xs4Still = self.basStill[50:54]
            self.xs5Still = self.basStill[54:58]
            
    def printGroups(self) -> None:
        '''print file dates in the groups'''
        for r in ['Still', 'Stitch']:
            for st in STLIST:
                files = getattr(self, st+r)
                if r=='Still':
                    times = [fh.fileTime(s) for s in files]
                else:
                    times = [os.path.basename(s) for s in files]
                logging.info(f'{st}:{times}')
                
    def countFiles(self) -> Dict:
        '''count the types of files in the folder'''
        c = [['folder',self.folders[0]]]
        for r in ['Still', 'Stitch']:
            for st in STLIST:
                files = getattr(self, st+r)
                c.append([st+r, len(files)])
        return dict(c)
    
    def stitchDone(self) -> bool:
        '''determine if the folder is done stitching'''
        for st in STLIST:
            files = getattr(self, st+'Stitch')
            if len(files)==0:
                return False
        else:
            return True
            
    def getFiles(self, st:str) -> Tuple[List, str, str]:
        '''get file list and directory name'''
        files = getattr(self, st+'Still')
        if len(files)==0:
            return [],'',''
        dirname = files[0]
        sample = ''
        while not ('I_' in sample and '_S_' in sample):
            # keep going up folders until you hit the sample subfolder
            dirname = os.path.dirname(dirname)
            sample = os.path.basename(dirname) # folder name
        return files, dirname, sample
            
    def archiveGroup(self, st:str, debug:bool=False, **kwargs) -> None:
        '''put files in an archive folder'''
        files, dirname, sample = self.getFiles(st)
        if 'raw' in files[0]:
            # already archived
            return
        rawfolder = os.path.join(dirname, 'raw')
        if not os.path.exists(rawfolder):
            if not debug:
                os.mkdir(rawfolder)
        linefolder = os.path.join(rawfolder, st)
        if not os.path.exists(linefolder):
            if not debug:
                os.mkdir(linefolder)
        newnames = []
        for f in files:
            newname = os.path.join(linefolder, os.path.basename(f))
            if debug:
                logging.info(f'Old: {f}, New: {newname}')
            else:
                os.rename(f, newname)
                newnames.append(newname)
        if not debug:
            setattr(self, st+"Still", newnames)
            
    def stitchGroup(self, st:str, archive:bool=True, **kwargs) -> int:
        '''stitch the group of files together and export. st must be horiz, vert1, vert2, vert3, vert4, xs1, xs2, xs3, xs4, or xs5. Returns 0 if stitched, 1 if not.'''
        files, dirname, sample = self.getFiles(st)
        if len(files)==0:
            return 1
        tag = sample+'_'+st
        s = stitching.Stitch(files)
        if 'xs' in st:
            s.matcher.setDefaults(4.24781116, -265.683797)
            s.matcher.resetLastH()
        elif 'vert' in st:
            s.matcher.setDefaults(-1, -277)
            s.matcher.resetLastH()
        elif 'horiz' in st:
            s.matcher.setDefaults(0, -280)
            s.matcher.resetLastH()
            
        try:
            # stitch images and export
            s.stitchTranslate(export=True, tag=tag, **kwargs)
        except:
            logging.warning('Stitching error')
            traceback.print_exc()
            return
        else:
            # archive
            if archive:
                self.archiveGroup(st, **kwargs)
            return 0
                
    def stitchGroups(self, archive:bool=True, **kwargs) -> None:
        '''stitch all groups and export'''
        if 'stlist' in kwargs:
            stlist = kwargs['stlist']
        else:
            stlist = STLIST
        for st in stlist:
            self.stitchGroup(st, archive=archive, **kwargs)
            
#--------------------------------------------------
            
def stitchSubFolder(folder:str, **kwargs) -> None:
    '''stitches images in the subfolder'''
    fl = fileList(folder)
    if not fl.stitchDone() and len(fl.xs1Still)>0:
        fl.stitchGroups(**kwargs)
            
def stitchRecursive(folder:str, **kwargs) -> None:
    '''for all folders in the folder, stitch images in the subfolders'''
    if not os.path.isdir(folder):
        return
    if fh.isSubFolder(folder):
        stitchSubFolder(folder,  **kwargs)
    else:
        for f in os.listdir(folder):
            stitchRecursive(os.path.join(folder, f), **kwargs)
                
                
def countFiles(folder:str, stills:bool=True, stitches:bool=True, subcall:bool=False) -> Union[dict, List[dict], pd.DataFrame]:
    '''count the types of files in each folder. stills=True to print any folders which are missing stills. stitches=True to print any folders which are missing stitched images. If this is the top call, returns a dataframe. If this is a child call to a sample folder or above, returns a list of dicts. If this is a call to a subfolder, returns a dict.'''

    if not os.path.isdir(folder):
        return
    if fh.isSubFolder(folder):
        fl = fileList(folder)
        return fl.countFiles() # returns a dictionary
    
    # parent folder
    c = [] # list of dictionaries
    for f in os.listdir(folder):
        c1 = countFiles(os.path.join(folder, f), stills=stills, stitches=stitches, subcall=True)
        if type(c1) is list:
            c = c+c1
        elif type(c1) is dict:
            c.append(c1) 
    if subcall:
        return c # return the list of dictionaries
    
    # top folder
    df = pd.DataFrame(c)
    
    missing = False
    
    if stills:
        for st in STLIST:
            tystr = 'Still'
            files = [os.path.basename(f) for f in df[df[st+tystr]==0]['folder']]
            if len(files)>0:
                missing=True
                logging.info(f'Missing {st} {tystr}: {files}')
                
    if stitches:
        for st in STLIST:
            tystr = 'Stitch'
            files = [os.path.basename(f) for f in df[(df[st+'Stitch']==0) & (df[st+'Still']>0)]['folder']]
            if len(files)>0:
                missing=True
                logging.info(f'Missing {st} {tystr}: {files}')
                
    if not missing:
        logging.info('No missing files')
        
    return df
    
            
