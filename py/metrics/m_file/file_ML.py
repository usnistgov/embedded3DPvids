#!/usr/bin/env python
'''Functions for working with machine learning segmentation models'''

# external packages
import os, sys
import traceback
import logging
import pandas as pd
from matplotlib import pyplot as plt
from typing import List, Dict, Tuple, Union, Any, TextIO
import re
import numpy as np
import cv2 as cv
import shutil
import subprocess
import time
import datetime

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
sys.path.append(os.path.dirname(os.path.dirname(currentdir)))
import file.file_handling as fh
from im.imshow import imshow
from plainIm import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 4)
pd.set_option('display.max_rows', 500)


#----------------------------------------------

def convertFileToBW(ffull:str, diag:bool=False) -> None:
    '''convert a single file to black and white with 1 as max'''
    im = cv.imread(ffull)
    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    if im.max().max()==1:
        return
    _,im = cv.threshold(im, 120, 255, cv.THRESH_BINARY)
    im[im==255]=1
    if diag:
        print(im.shape, np.unique(im), im[0,0], f)
    else:
        cv.imwrite(ffull, im)

def convertFilesToBW(folder:str, diag:bool=False) -> None:
    '''convert all the files in the folder to black and white'''
    for f in os.listdir(folder):
        if not 'Thumbs' in f:
            ffull = os.path.join(folder, f)
            convertFileToBW(ffull, diag)
            
def removeOme(folder:str):
    '''remove .ome from all of the file names'''
    for f in os.listdir(folder):
        ffull = os.path.join(folder, f)
        new = ffull.replace('.ome', '')
        os.rename(ffull, new)
            
class segmentCompare:
    '''for comparing pre-segmented images to freshly segmented images. func should be a metricSegment class definition '''
    
    def __init__(self, segFolder:str, serverFolder:str, origFolder:str, func):
        self.segFolder = segFolder
        self.serverFolder = serverFolder
        self.origFolder = origFolder
        self.func = func
        self.images = {}
        self.df = pd.DataFrame({'bn':os.listdir(self.segFolder)})
        
    def export(self, fn:str):
        '''export dataframe to file'''
        plainExp(fn, self.df, {}, index=False)
        
    def compare(self):
        '''iterate through all images in the segmentation folder and compare them to the current code'''
        for i in self.df.index:
            if '.png' in self.df.loc[i]['bn']:
                self.compareFile(i)
            
    def getSegIdeal(self, i:int, bn:str) -> np.array:
        '''get the manually segmented image'''
        segfile = os.path.join(self.segFolder, bn)
        self.df.loc[i, 'segfile'] = segfile
        segIdeal = cv.imread(segfile, cv.IMREAD_GRAYSCALE)
        return segIdeal
    
    def getSegReal(self, i:int, bn:str, diag:int=0, **kwargs) -> np.array:
        '''segment the image using the current algorithm'''
        try:
            fn = fh.findFullFN(bn, self.serverFolder)
        except FileNotFoundError:
            self.df.loc[i, 'result'] = 'FileNotFound'
            return []
        ob = self.func(fn, diag=diag, **kwargs) 
        if not hasattr(ob, 'componentMask'):
            self.df.loc[i, 'result'] = 'NoMask'
            return []
        segReal = ob.componentMask
        return segReal
    
    def getSegMachine(self, i:int, bn:str) -> np.array:
        '''get the ML segmented image from a folder'''
        try:
            fn = fh.findFullFN(bn, self.serverFolder)
        except FileNotFoundError:
            self.df.loc[i, 'result'] = 'MLNotFound'
            return []
        fn = os.path.join(os.path.dirname(fn), 'MLsegment', bn.replace('vstill', 'MLsegment'))
        if not os.path.exists(fn):
            self.df.loc[i, 'result'] = 'MLNotFound'
            return []
        seg = cv.imread(fn, cv.IMREAD_GRAYSCALE)
        return seg
    
    def getSegU(self, i:int, bn:str) -> np.array:
        '''get the unsupervised segmented image from a folder'''
        try:
            fn = fh.findFullFN(bn, self.serverFolder)
        except FileNotFoundError:
            self.df.loc[i, 'result'] = 'UNotFound'
            return []
        fn = os.path.join(os.path.dirname(fn), 'Usegment', bn.replace('vstill', 'Usegment'))
        seg = cv.imread(fn, cv.IMREAD_GRAYSCALE)
        return seg
    
    def getOrig(self, bn:str) -> np.array:
        '''get the original image'''
        fn = os.path.join(self.origFolder, bn)
        if not os.path.exists(fn):
            print(bn)
            try:
                fn = fh.findFullFN(bn.replace('.ome', '').replace('vcrop', 'vstill'), self.origFolder)
            except FileNotFoundError:
                return []
        orig = cv.imread(fn)
        return orig
            
    def compareFile(self, i:int, diag:int=0, diffCrit:float=0.01, **kwargs) -> dict:
        '''segfile is the pre-segmented file'''
        bn = self.df.loc[i, 'bn']
        self.df.loc[i, 'difference'] = 1
        
        # find original file
        segIdeal = self.getSegIdeal(i, bn)
        
        if type(self.func) is str:
            # get the file from the folder
            if self.func=='ML':
                segReal = self.getSegMachine(i, bn)
            elif self.func=='U':
                segReal = self.getSegU(i,bn)
            else:
                raise ValueError(f'Unexpected func {self.func}')
        else:
            # segment the file
            segReal = self.getSegReal(i, bn, diag=diag, **kwargs)
        if len(segReal)==0:
            return
        
        # compare segmentation to ideal
        if not segIdeal.shape==segReal.shape:
            self.df.loc[i, 'result'] = 'ShapeMismatch'
            return
        diff, difference = self.measureDifference(bn, i, segIdeal, segReal)
        if difference>diffCrit:
            self.createImDiag(bn, segIdeal, segReal, diff, diag=diag)
        
        
    def measureDifference(self, bn:str, i:int, segIdeal:np.array, segReal:np.array) -> None:
        '''measure the number of pixels that are different between the two images'''
        diff = cv.bitwise_xor(segIdeal, segReal)
        self.df.loc[i, 'result'] = 'Found'
        shape = segIdeal.shape
        a = shape[0]*shape[1]
        difference = diff.sum(axis=0).sum(axis=0)/255/a  # fraction of different pixels
        self.df.loc[i, 'difference'] = difference
        return diff, difference
        
    def createImDiag(self, bn:str, segIdeal:np.array, segReal:np.array, diff:np.array, diag:int=0, title1:str='ideal', title2:str='real') -> None:
        '''create a diagnostic image that compares the found image to the manual segmented image and shows differences in color'''
        imdiag = cv.cvtColor(segIdeal, cv.COLOR_GRAY2BGR)
        r0 = cv.bitwise_and(segIdeal, segReal)
        imdiag[(r0==255)] = [100,100,100]
        r1 = cv.subtract(segIdeal, segReal)
        imdiag[(r1==255)] = [0,255,0]
        r2 = cv.subtract(segReal, segIdeal)
        imdiag[(r2==255)] = [0,0,255]
        if diag>0:
            orig = self.getOrig(bn)
            if len(orig)>0:
                imshow(imdiag, segIdeal, segReal, orig, titles=['diff', title1, title2, 'orig'])
            else:
                imshow(imdiag, segIdeal, segReal, diff, titles=['diff', title1, title2, 'difference'])
        self.images[bn] = imdiag 
        
    def showWorstSegmentation(self, n:int=6) -> None:
        '''show the segmented images with the worst diff values'''
        print(f'Success rate: {self.successRate():0.3f}, Average diff: {self.averageDiff():0.3f}')
        df2 = self.df.sort_values(by='difference', ascending=False)
        ims = []
        print(df2.iloc[:n*2][['result', 'difference']])
        print('Red = algorithm, green = manual')
        i = 0
        j = 0
        while j<n and i<len(df2):
            bn = df2.iloc[i]['bn']
            if bn in self.images:
                ims.append(self.images[bn])
                j = j+1
            i = i+1
        imshow(*ims)
        
    def successRate(self, dcrit:float=0.01) -> float:
        '''get the percentage of files that have a diff less than dcrit'''
        return (len(self.df[self.df.difference<dcrit])/len(self.df))
    
    def averageDiff(self) -> float:
        '''get the average difference'''
        return self.df.difference.mean()
    
#-----------------------

class modelCompare(segmentCompare):
    '''for comparing the performance of 2 ML models with results in 2 folders'''
    
    def __init__(self, folder1:str, folder2:str, origFolder:str):
        super().__init__(folder1, '', origFolder, None)
        self.folder1 = folder1
        self.folder2 = folder2
        
    def getImage(self, folder:str, i:int, bn:str, tag:str) -> np.array:
        '''get the manually segmented image'''
        file = os.path.join(folder, bn)
        self.df.loc[i, tag] = file
        im = cv.imread(file, cv.IMREAD_GRAYSCALE)
        return im
        
    def compareFile(self, i:int, diag:int=0, diffCrit:float=0.01, **kwargs) -> dict:
        '''segfile is the pre-segmented file'''
        bn = self.df.loc[i, 'bn']
        self.df.loc[i, 'difference'] = 1
        
        # find original file
        seg1 = self.getImage(self.folder1, i, bn, 'folder1')
        seg2 = self.getImage(self.folder2, i, bn, 'folder2')
        
        # compare segmentation to ideal
        if not seg1.shape==seg2.shape:
            self.df.loc[i, 'result'] = 'ShapeMismatch'
            return
        diff, difference = self.measureDifference(bn, i, seg1, seg2)
        if difference>diffCrit:
            self.createImDiag(bn, seg1, seg2, diff, diag=diag, title1='folder 1', title2='folder 2')
            
    def showWorstSegmentation(self, n:int=6) -> None:
        '''show the segmented images with the worst diff values'''
        print(f'Fraction same: {self.successRate():0.3f}, Average diff: {self.averageDiff():0.3f}')
        df2 = self.df.sort_values(by='difference', ascending=False)
        ims = []
        print(df2.iloc[:n*2][['result', 'difference']])
        print('Red = folder 1, green = folder 2')
        i = 0
        j = 0
        while j<n and i<len(df2):
            bn = df2.iloc[i]['bn']
            if bn in self.images:
                ims.append(self.images[bn])
                j = j+1
            i = i+1
        imshow(*ims)

#-----------------------
    
class trainingGenerator:
    '''a class for generating training data for ML models'''
    
    def __init__(self, topFolder:str, excludeFolders:list=[], mustMatch:list=[], canMatch:list=[]):
        self.topFolder = topFolder
        self.excludeFolders = excludeFolders
        self.printFolders = fh.printFolders(topFolder, tags=mustMatch, someIn=canMatch)
        self.numFolders = len(self.printFolders)
        
    def randomFolder(self):
        '''get a random folder'''
        i = np.random.randint(self.numFolders)
        return self.printFolders[i]
    
    def excluded(self, bn:str) -> bool:
        '''check if the file basename is already in one of the excluded folders. return True if it should be excluded'''
        for f in self.excludeFolders:
            if os.path.exists(os.path.join(f, bn)):
                return True
        return False
    
    def randomFile(self, mustMatch:list=[], canMatch:list=[]) -> str:
        '''get a random vert vstill from the topFolder, but get a new one if it's already in one of the excludeFolders '''
        folder = self.randomFolder()
        pfd = fh.printFileDict(folder)
        pfd.findVstill()
        fileList = []
        for file in pfd.vstill:
            if fh.anyIn(canMatch, file) and fh.allIn( mustMatch, file):
                fileList.append(file)
        numstills = len(fileList)
        if numstills==0:
            return self.randomFile()
        excluded = True
        guessed = []
        while excluded:
            i = np.random.randint(numstills)
            if not i in guessed:
                guessed.append(i)
                file = fileList[i]
                excluded = self.excluded(os.path.basename(file))
            if len(guessed)==numstills:
                # we've guessed all of the files. remove this folder from contention and pick a different one
                self.printFolders.remove(folder)
                return self.randomFile()
        return file
    
#-----------------------
    
class resultMover:
    '''move machine learning results to the folder'''
    
    def __init__(self, segFolder:Union[List[str],str], serverFolder:str, diag:int=1, timing:int=100, tag:str='MLsegment'):
        if type(segFolder) is str:
            self.segFolders = [segFolder]
        elif type(segFolder) is list:
            self.segFolders = segFolder
        else:
            raise ValueError('Unexpected segFolder value passed to resultMover')
        self.serverFolder = serverFolder
        self.diag = diag
        self.timing = timing
        self.tag = tag
        self.error = []
        self.printi = -1
        self.copied = 0
        self.files = [os.path.join(sf, f) for sf in self.segFolders for f in os.listdir(sf) ]
        self.numFiles = len(self.files)
        if self.diag>0:
            logging.info(f'Copying {self.numFiles} files in {self.segFolders}')
        self.already = 0
        self.moveMLResults()
        
    def moveMLResult(self, file:str) -> None:
        '''move the segmented machine learning file into the folder'''
        origname = file.replace('vcrop', 'vstill')
        origname = origname.replace('.ome','')
        bn = os.path.basename(origname)
        fn = fh.findFullFN(bn, self.serverFolder)
        if not os.path.exists(fn):
            self.error.append(fn)
            return
        folder = os.path.dirname(fn)
        segfolder = os.path.join(folder, self.tag)
        if not os.path.exists(segfolder):
            os.mkdir(segfolder)
        newname = os.path.join(segfolder, bn.replace('vstill', self.tag))
        if os.path.exists(newname):
            self.already+=1
        else:
            shutil.copyfile(file, newname)
            if self.diag>1:
                logging.info(f'copied {os.path.basename(newname)}')
            self.copied+=1
        if file in self.error:
            self.error.remove(file)
        self.printi+=1
        if self.diag>0 and self.printi%self.timing==0:
            logging.info(f'Copied {self.copied}, already {self.already}, total {self.numFiles}')

    def moveMLResults(self):
        '''move all of the segmented images from segFolder into the appropriate folders in serverFolder'''
        for file in self.files:
            try:
                self.moveMLResult(file)
            except FileNotFoundError:
                self.error.append(file)
        if self.diag>0:
            logging.info(f'Finished copying, failed on {len(self.error)} files')

def copyToMLInputFolder(cropFolder:str, topFolder:str, mustMatch:list=[], reg:str='.*') -> None:
    '''copy any files that match the regex in folders that match mustMatch to the cropFolder that don't already have ML'''   
    newfolder = cropFolder
    for f in fh.printFolders(topFolder, mustMatch=mustMatch):
        crop = os.path.join(f, 'crop')
        if os.path.exists(crop):
            for f1 in os.listdir(crop):
                mlfn = os.path.join(f, 'MLsegment', os.path.basename(f1).replace('vcrop', 'MLsegment'))
                if not os.path.exists(mlfn):
                    if re.match(reg, f1):
                        newname = os.path.join(newfolder, f1)
                        if not os.path.exists(newname):
                            shutil.copyfile(os.path.join(crop, f1), newname)
                            # print(newname)
                        
def splitIntoSubFolders(cropFolder:str, size:int=1000) -> None:
    '''split the folder into equally sized subfolders for uploading'''
    i = 0
    j = 0
    files = os.listdir(cropFolder)
    while j<len(files):
        folder = os.path.join(cropFolder, str(i*size))
        if not os.path.exists(folder):
            os.mkdir(folder)
        i = i+1
        for k in range(size):
            if j>=len(files):
                return
            file = os.path.join(cropFolder, files[j])
            j = j+1
            if not os.path.isdir(file):
                newname = os.path.join(folder, os.path.basename(file))
                os.rename(file, newname)
                
                
def moveCrops(folder:str, newFolder:str) -> list:
    '''move cropped files to the new folder and return self if there are missing files'''
    pfd = fh.printFileDict(folder)
    pfd.findVstill()
    pfd.findMLsegment()
    missing = len(pfd.vstill)-len(pfd.MLsegment)
    if missing>0:
        print(folder, missing)
        if os.path.exists(newFolder):
            for vstill in pfd.vstill:
                ml = os.path.join(folder, 'MLsegment', os.path.basename(vstill).replace('vstill', 'MLsegment'))
                if not os.path.exists(ml):
                    cropname = os.path.join(folder, 'crop', os.path.basename(vstill).replace('vstill', 'vcrop'))
                    if os.path.exists(cropname):
                        newname = os.path.join(newFolder, os.path.basename(cropname))
                        shutil.copyfile(cropname, newname)
                    else:
                        print(f'Missing crop file {cropname}')
                    missing = missing-1
                    if missing==0:
                        return [folder]
        else:
            return [folder]
    else:
        return []
                
def findMissingML(topFolder:str, targetFolder:str, mustMatch:list=[]) -> list:
    '''find folders that are missing ML files'''
    flist = []
    if os.path.exists(targetFolder):
        newFolder = os.path.join(targetFolder, datetime.datetime.now().strftime("%y%m%d_%H%M"))
        os.mkdir(newFolder)
    for f in fh.printFolders(topFolder, mustMatch=mustMatch):
        flist = flist + moveCrops(f, newFolder)
    splitIntoSubFolders(newFolder)
    return flist
