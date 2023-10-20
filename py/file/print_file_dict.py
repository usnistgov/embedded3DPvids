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
import file_names as fn
from levels import labelLevels


# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



#----------------------------------------------

class printFileDict:
    '''get a dictionary of the paths for each file inside of the print folder'''
    
    def __init__(self, printFolder:str, **kwargs):
        # t0 = time.time()
        if 'levels' in kwargs:
            self.levels = kwargs['levels']
            self.hasLevels = True
        else:
            self.hasLevels = False

        if fn.isPrintFolder(printFolder):
            self.printFolder = printFolder
        else:
            self.printFolderFromLevels(printFolder)

        # only label vid, printfolder, and printType for now. 10x speedup
        self.resetList()
        self.getPrintType()
        self.findVids()
        self.findTime()
        self.findMeta()
        self.getDate()
        # self.sort()
        # print(time.time()-t0)
        
    def resetList(self):
        '''reset the lists of files'''
        self.sorted = False
        self.vid_unknown=[]
        self.csv_unknown=[]
        self.csv_delete=[]
        self.still=[]
        self.stitch=[]
        self.vstill=[]
        self.MLsegment = []
        self.MLsegment2 = []
        self.Usegment = []
        self.vcrop = []
        self.vid = []
        self.meta = []
        self.timeSeries = []
        self.still_unknown=[]
        self.unknown=[]
        self.phoneCam = []
        self.confocal = []
        self.printType = ''
        
    def __getattr__(self, s:str, ext:str='csv') -> str:
        '''get a specific file from the folder that hasn't already been defined'''
        if s=='summary':
            return self.findSummary()
        elif s=='measure':
            return self.findMeasure()
        nfn = self.newFileName(s, ext)
        if os.path.exists(nfn):
            setattr(self, s, nfn)
            return nfn
        else:
            raise AttributeError


    def printFolderFromLevels(self, printFolder:str):
        '''get the printFolder from labeling the full file tree'''
        if not self.hasLevels:
            self.levels = labelLevels(printFolder)
            self.hasLevels = True
            if not self.levels.currentLevel in ['subFolder', 'sbpFolder']:
                raise ValueError(f'Input to labelPrintFiles must be subfolder or sbp folder. Given {printFolder}')
        self.printFolder=self.levels.printFolder()
        
    def getPrintType(self) -> None:
        '''get the print type from the folder hierarchy'''
        d = {'singleDisturb':'singleDisturb', 'singleLines':'singleLines', 'tripleLines':'tripleLine', 'SDT':'SDT'}
        if self.hasLevels:
            f = os.path.basename(self.levels.printTypeFolder)
            for key,val in d.items():
                if f==key:
                    self.printType=val
                    return  
                
            # printtype folder failed, try the printfolder
            logging.error('Could not determine print type from file hierarchy')
            bn = re.split('_',os.path.basename(self.printFolder))[0]
            self.printType = fn.printType(bn)
        else:
            # no levels, use the print folder
            for key,val in d.items():
                if key in self.printFolder:
                    self.printType=val
                    return
            raise ValueError('Could not determine print type')
            
    def findVids(self):
        '''find the videos'''
        sbp = fn.allSBPFiles() 
        for f1 in os.listdir(self.printFolder):
            if f1.endswith('.avi'):
                ffull = os.path.join(self.printFolder, f1)
                self.sortVid(ffull, sbp=sbp)
                
    def findTime(self):
        '''find the original time files'''
        sbp = fn.allSBPFiles() 
        for f1 in os.listdir(self.printFolder):
            if f1.endswith('.csv') and ('time' in f1 or 'Fluigent' in f1) and not ('timeRe' in f1):
                ffull = os.path.join(self.printFolder, f1)
                self.timeSeries.append(ffull)
                
    def findMeta(self):
        '''find the metadata files'''
        sbp = fn.allSBPFiles() 
        for f1 in os.listdir(self.printFolder):
            if f1.endswith('.csv') and ('meta' in f1 or 'speeds' in f1):
                ffull = os.path.join(self.printFolder, f1)
                self.meta.append(ffull)
                
    def findSummary(self) -> str:
        for f1 in os.listdir(self.printFolder):
            if f1.endswith('.csv') and 'summary' in f1.lower():
                ffull = os.path.join(self.printFolder, f1)
                self.summary = ffull
                return ffull
        return ''
    
    def findMeasure(self) -> str:
        for f1 in os.listdir(self.printFolder):
            if f1.endswith('.csv') and 'measure' in f1.lower():
                ffull = os.path.join(self.printFolder, f1)
                self.measure = ffull
                return ffull
        return ''
    
                
    def getDate(self):
        '''get the date of the folder'''
        if len(self.vid)>0:
            self.date = fileDate(self.vid[0], out='int')
        elif len(self.still)>0:
            self.date = fileDate(self.still[0], out='int')
        elif hasattr(self, 'levels'):
            self.date = fileDate(self.levels.subFolder, out='int')
        else:
            raise ValueError('Could not find date')
        return int(self.date)
            
            
    def newFileName(self, s:str, ext:str) -> str:
        '''generate a new file name'''
        if ext[0]=='.':
            ext = ext[1:]  # remove . from beginning
        if len(self.vid)>0:
            file = self.vid[0].replace('avi', ext)
            file = file.replace('Basler camera', s)
        elif len(self.still)>0:
            file = self.still[0].replace('png', ext)
            file = file.replace('Basler camera', s)
        else:
            file = os.path.join(self.printFolder, f'{s}.{ext}')
            if os.path.exists(file):
                ii = 0
                while os.path.exists(file):
                    file = os.path.join(self.printFolder, f'{s}_{ii}.{ext}')
                    ii+=1
        return file

    def first(self, s:str) -> str:
        '''return first entry in the list'''
        if not hasattr(self, s):
            return ''
        l = getattr(self, s)
        if type(l) is list:
            if len(l)==0:
                return ''
            return l[0]
        else:
            return l
        
    def vidFile(self):
        '''get the primary video file'''
        return self.first('vid')
    
    def timeFile(self):
        '''get the primary time file'''
        return self.first('timeSeries')

    
    def metaFile(self):
        '''get the primary time file'''
        return self.first('meta')

    
    def deconstructFileName(self, file:str) -> str:
        '''get the original string that went into the file name'''
        if len(self.vid)>0:
            spl2 = re.split('_', os.path.basename(self.vid[0]))
        elif len(self.still)>0:
            spl2 = re.split('_', os.path.basename(self.still[0]))
        else:
            return os.path.basename(os.path.splitext(file)[0])
        if 'Basler camera' in spl2:
            i = spl2.index('Basler camera')
            ir = len(spl2)-i-1
        else:
            raise ValueError('Failed to find original value')
        spl1 = re.split('_',os.path.basename(file))
        return '_'.join(spl1[i:-ir])

    
    def sbpName(self) -> str:
        '''gets the name of the shopbot file'''
        if hasattr(self, 'sbp'):
            return self.sbp
        
        # find sbp folder
        
        # labeled levels already
        if hasattr(self, 'levels'):
            self.sbp = os.path.basename(self.levels.sbpFolder)
            return self.sbp
        
        # folder matches known SBP names
        if fn.isSBPFolder(self.printFolder):
            self.sbp = os.path.basename(self.printFolder)
            return self.sbp
        
        # find from the video file name
        if len(self.vid)>0:
            file = self.vid[0]
            if '_Basler' in file:
                spl = re.split('_Basler', os.path.basename(file))
                self.sbp = spl[0]
                return self.sbp
            else:
                raise ValueError(f'Unexpected video file {file}')
                
    def sbpFile(self) -> str:
        '''find the full path name of the sbp file'''
        self.sbpName()
        return sbpPath(cfg.path.pyqtFolder, self.sbp)
    
    def sbpPointsFile(self) -> str:
        '''get the full path name of the sbp points csv'''
        file = self.sbpFile()
        return file.replace('.sbp', '.csv')
       
    def sortVid(self, ffull:str, fname:str='', spl:list=[], sbp:dict={}) -> None:
        '''label an avi video'''
        if len(fname)==0:
            fname, ext, spl = self.splitFile(os.path.basename(ffull))
        if len(sbp)==0:
            sbp = fn.allSBPFiles() 
        if 'Basler camera' in fname and spl[0] in sbp:
            self.vid.append(ffull)
        else:
            self.vid_unknown.append(ffull)
            
    def sortCSV(self, ffull:str, fname:str, spl:list, sbp:dict) -> None:
        '''label a csv file'''
        if spl[0] in sbp:
            if 'timeRewrite' in fname:
                self.timeRewrite=ffull
            elif 'Fluigent' in fname or 'time' in fname:
                self.timeSeries.append(ffull)
            elif 'speeds' in fname or 'meta' in fname:
                self.meta.append(ffull)
            elif 'failure' in fname.lower():
                self.failures = ffull
            elif 'summary' in fname.lower():
                self.summary = ffull
            elif 'measure' in fname.lower():
                self.measure = ffull
                setattr(self, self.deconstructFileName(ffull), ffull)
            else:
                setattr(self, self.deconstructFileName(ffull), ffull)
        elif spl[0] in fn.singleLineSBPPicfiles():
            # extraneous fluigent or speed file from pic
            self.csv_delete.append(ffull)
        else:
            # summary or analysis file
            if hasattr(self, spl[-1]):
                lprev = getattr(self, spl[-1])
                if len(lprev)>0 and type(lprev) is str:
                    l = [lprev]
                else:
                    l = []
                l.append(ffull)
                setattr(self, spl[-1], l)
            else:
                setattr(self, spl[-1], ffull)
                
    def sortPNG(self, ffull:str, fname:str) -> None:
        '''put the png in the right list'''
        if 'background' in fname:
            self.background = ffull
        elif fn.isStill(ffull):
            if 'Basler camera' in fname:
                # raw still
                self.still.append(ffull)
            elif 'vcrop' in fname:
                self.vcrop.append(ffull)
            elif 'MLsegment2' in fname:
                self.MLsegment2.append(ffull)
            elif 'MLsegment' in fname:
                self.MLsegment.append(ffull)
            elif 'Usegment' in fname:
                self.Usegment.append(ffull)
            elif fn.isVidStill(ffull):
                self.vstill.append(ffull)
            else: 
                self.still_unknown.append(ffull)
        elif fn.isStitch(ffull):
            # stitched image
            self.stitch.append(ffull)
        elif fn.isVidStill(ffull):
            self.vstill.append(ffull)
        else:
            self.still_unknown.append(ffull)
            
    def splitFile(self, bn) -> Tuple[str, str, list]:
        exspl = os.path.splitext(bn)
        ext = exspl[-1]
        fname = exspl[0]
        spl = splitName(fname)
        return fname, ext, spl
        
    def sortFiles(self, folder:str):
        '''sort and label files in the given folder'''
        sbp = fn.allSBPFiles() 
        for f1 in os.listdir(folder):
            ffull = os.path.join(folder, f1)
            if os.path.isdir(ffull):
                # recurse
                self.sortFiles(ffull)
            elif not 'Thumbs' in f1:
                fname, ext, spl = self.splitFile(f1)
                if ext=='.avi':
                    self.sortVid(ffull, fname, spl, sbp)
                elif ext=='.csv':
                    self.sortCSV(ffull, fname, spl, sbp)
                elif ext=='.png':
                    self.sortPNG(ffull, fname)
                elif ext=='.jpg':
                    self.phoneCam.append(ffull)
                elif ext=='.lif':
                    self.confocal.append(ffull)
                else:
                    self.unknown.append(ffull) 
        self.sorted = True
                    
                    
    def sort(self) -> None:
        '''sort and label files in the print folder'''
        self.resetList()
        self.sortFiles(self.printFolder)
        self.getPrintType()
        self.getDate()
        
    def findVstill(self) -> None:
        self.vstill = []
        for f1 in os.listdir(self.printFolder):
            if 'vstill' in f1 and 'png' in f1:
                ffull = os.path.join(self.printFolder, f1)
                self.vstill.append(ffull)
    
    def findMLsegment(self) -> None:
        self.MLsegment = []
        folder = os.path.join(self.printFolder, 'MLsegment')
        if not os.path.exists(folder):
            return
        for f1 in os.listdir(folder):
            if 'MLsegment' in f1 and 'png' in f1:
                ffull = os.path.join(folder, f1)
                self.MLsegment.append(ffull)
                
    def findMLsegment2(self) -> None:
        self.MLsegment2 = []
        folder = os.path.join(self.printFolder, 'MLsegment2')
        if not os.path.exists(folder):
            return
        for f1 in os.listdir(folder):
            if 'MLsegment2' in f1 and 'png' in f1:
                ffull = os.path.join(folder, f1)
                self.MLsegment2.append(ffull)
                
    def findUsegment(self) -> None:
        self.Usegment = []
        folder = os.path.join(self.printFolder, 'Usegment')
        if not os.path.exists(folder):
            return
        for f1 in os.listdir(folder):
            if 'Usegment' in f1 and 'png' in f1:
                ffull = os.path.join(folder, f1)
                self.Usegment.append(ffull)
                
    def findVcrop(self) -> None:
        self.vcrop = []
        folder = os.path.join(self.printFolder, 'crop')
        if not os.path.exists(folder):
            return
        for f1 in os.listdir(folder):
            if 'vcrop' in f1 and 'png' in f1:
                ffull = os.path.join(folder, f1)
                self.vcrop.append(ffull)
        
    def printAll(self) -> None:
        '''print all values'''
        for key,value in self.__dict__.items():
            if key in ['levels']:
                pass
            elif type(value) is list:
                if len(value)>0:
                    if os.path.exists(value[0]):
                        print(f'\n{key}:\t{[os.path.basename(file) for file in value]}\n')
                    else:
                        print(f'{key}:\t{value}')
            elif type(value) is str:
                if os.path.exists(value):
                    print(f'{key}:\t{os.path.basename(value)}')
                else:
                    print(f'{key}:\t{value}')
                    
    def check(self) -> list:
        '''check that the sample names match and that there is only one video per print folder'''
        self.mismatch=[]
        self.tooMany=[]
        bn = twoBN(self.printFolder)
        sname = sampleName(self.printFolder)
        for l in [self.vid, self.still, self.stitch, self.timeSeries, self.meta]:
            for file in l:
                if os.path.exists(file):
                    try:
                        s2 = sampleName(file)
                    except IndexError:
                        print(file)
                        raise IndexError
                    if not sname==s2:
                        logging.warning(f'Mismatched sample in {bn}: {os.path.basename(file)}')
                        self.mismatch.append(file)
        if len(self.vid)>1:
            logging.warning(f'Too many videos in {bn}: {len(self.vid)}')
            self.tooMany = self.vid
#         self.diagnoseMismatch()     # diagnose problems
#         self.fixMismatch()          # fix problems
        return 
    
    def datesAndTimes(self, s:str) -> List[str]:
        '''get a list of dates and times for the given list'''
        dates = []
        times = []
        if hasattr(self, s):
            sl = getattr(self, s)
            for si in sl:
                d,t = fileDateAndTime(si)
                dates.append(d)
                times.append(t)
        dates = [int(d) for d in list(set(dates))]
        times = [int(d) for d in list(set(times))]
        return dates,times
    
    def diagnoseMismatch(self):
        '''diagnose why there is a sample mismatch in the folder, and fix it'''
        self.newFolderFiles = []        # files to move to a new folder
        self.renameFiles = []          # files to rename
        if len(self.mismatch)==0:
            return
        stitchDates, stitchTimes = self.datesAndTimes('stitch')  # times and dates of stitches
        if len(stitchDates)==0:
            logging.info(f'Could not diagnose mismatch in {twoBN(self.printFolder)}: no stitches')
            return
        if len(stitchDates)>1:
            logging.info(f'Could not diagnose mismatch in {twoBN(self.printFolder)}: too many stitches')
        stitchDate = stitchDates[0]
        for m in self.mismatch:
            d,t = fileDateAndTime(m, out='int')
            if (not d==stitchDate) or t>max(stitchTimes):
                # different date from stitch, or vid taken after the stitch
                self.newFolderFiles.append(m)
            else:
                self.renameFiles.append(m)
        return
    
    def fixMismatch(self):
        '''fix mismatched files'''
        sname = sampleName(self.printFolder)
        for file in self.renameFiles:
            # rename file
            s = sampleName(file)
            newname = file.replace(s, sname)
            os.rename(file, newname)
            
            # remove from bad lists
            self.renameFiles.remove(file)
            self.mismatch.remove(file)
            
            # print status
            logging.info(f'Renamed {os.path.basename(file)} to {os.path.basename(newname)}')
            
        self.sort()
        
    def pxpmm(self):
        '''get pixels per mm'''
        if len(self.meta)==0:
            meta = {}
        else:
            df = pd.read_csv(self.meta[0], header=0, names=['var', 'units', 'val'])
            meta = dict(zip(df['var'], df['val']))
        if 'camera_magnification' in meta:
            cm = float(meta['camera_magnification'])
        else:
            cm = 1
        d = {0.5:71, 1:139}
        if cm in d:
            pxpmm = d[cm]
            return pxpmm
        else:
            raise ValueError(f'Unexpected camera magnification in {self.folder}: {cm}')
        