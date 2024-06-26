#!/usr/bin/env python
'''class for testing failed files for all folders in a topfolder'''

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

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
sys.path.append(os.path.dirname(os.path.dirname(currentdir)))
from tools.plainIm import *
from tools.config import cfg
from val.v_print import printVals
from progDim.prog_dim import getProgDims
import file.file_handling as fh
from m_tools import *
from full_sequence import SDTWorkflow, fullSequencer

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)
pd.set_option('display.max_rows', 500)

#----------------------------------------------

class failureTest:
    '''for testing failed files for all folders in a topfolder. testFunc should be a fileMetric class definition'''
    
    def __init__(self, failureFile:str, testFunc):
        self.failureFile = failureFile
        self.testFunc = testFunc
        self.currFolder = ''
        self.failureChanged = False
        self.importFailures()
        
    def countFailures(self):
        '''count and print the different categories of file and folder'''
        self.bad = self.df[(self.df.approved==False)&(~(self.df.error==''))]
        self.approved = self.df[self.df.approved==True]
        self.unapproved = self.df[self.df.approved==False]
        self.failedLen = len(self.bad)
        self.failedFolders = len(self.bad.fostr.unique())
        self.uncheckedFolders = len(self.unapproved.fostr.unique())
        self.totalFolders = len(self.df.fostr.unique())
        print(f'{self.failedLen} failed files, {self.failedFolders} failed folders, {self.uncheckedFolders} unchecked folders, {self.totalFolders} total folders')
        
    def firstBadFile(self):
        '''find the name of the first bad file in the table'''
        if len(self.bad)==0:
            return 'No bad files'
        return self.bad.iloc[0].name
    
    def firstBadFolder(self):
        '''find the name of the first bad folder in the table'''
        if len(self.bad)==0:
            return 'No bad folders'
        return self.bad.iloc[0]['fostr']
    
    def firstUnapprovedFolder(self):
        '''find the name of the first unapproved folder in the table'''
        if len(self.unapproved)==0:
            return 'No unapproved folders'
        return self.unapproved.iloc[0]['fostr']
    
    def file(self, i):
        '''get the name of a file by index in the table'''
        row = self.df.loc[i]
        file = os.path.join(cfg.path.server, row['fostr'], row['fistr'])
        return file
    
    def folder(self, i):
        '''get the name of a folder by index of a file in the table'''
        row = self.df.loc[i]
        folder = os.path.join(cfg.path.server, row['fostr'])
        return folder
        
    def importFailures(self):
        '''import the table of failed files'''
        df, _ = plainIm(self.failureFile, ic=None)
        df.fillna('', inplace=True)
        if len(df)>0:
            for i,row in df.iterrows():
                folder = os.path.dirname(row['file'])
                df.loc[i, 'fostr'] = folder
                df.loc[i, 'file'] = os.path.join(cfg.path.server, df.loc[i, 'file'])
                df.loc[i, 'fistr'] = os.path.basename(row['file'])
                if not 'approved' in row:
                    df.loc[i, 'approved'] = False
                if row['error']=='white' or row['error']=='approved':
                    df.loc[i, 'approved'] = True  # approve all white images
        else:
            df = pd.DataFrame(df)
            df['approved'] = []
            df['fostr'] = []
            df['fistr'] = []
        self.df = df
        self.countFailures()
        
    def testFile(self, i:int, **kwargs):
        '''test out measurement on a single file, given by the row number'''
        row = self.df.loc[i]
        approved = row['approved']
        print(f'Row {i}, approved {approved}')
        file = os.path.join(cfg.path.server, row['fostr'], row['fistr'])
        if os.path.exists(file):
            self.testFunc(file, **kwargs)
        
    def testFolder(self, folder:str, testFailures:bool=False, **kwargs):
        '''initialize the SDTWorkflow object to test a specific folder by name'''
        if testFailures:
            ffiles = self.bad[self.bad.fostr==folder]
            for i,_ in ffiles.iterrows():
                self.testFile(i, **kwargs)
        else:
            ffiles = self.df[self.df.fostr==folder]
            i = ffiles.iloc[0].name
        folder = self.folder(i)
        fh.openExplorer(os.path.join(folder, 'Usegment'))
        folder2 = os.path.join(cfg.path.server, folder)
        self.sw = SDTWorkflow(folder2, **kwargs)
        self.sw.run()
        # if not testFailures:
        #     self.sw.showFailures()

    def resetFolder(self):
        '''clear out the folder attributes'''
        for s in ['failuredf', 'pfd']:
            if hasattr(self, s):
                delattr(self, s)
                
    def setFolderAtt(self, folder:str):
        '''set new folder attributes'''
        if self.currFolder==folder:
            return
        
        self.resetFolder()
        self.currFolder = folder
        self.failureChanged = False
        if not hasattr(self, 'failuredf'):
            if not hasattr(self, 'pfd'):
                self.pfd = fh.printFileDict(folder)
            if hasattr(self.pfd, 'failures'):
                self.failuredf, _ = plainIm(self.pfd.failures, ic=0)                
                
    def exportFolderFailures(self):
        '''export the failures for the current folder'''
        if not hasattr(self, 'pfd') or not hasattr(self, 'failuredf') or not self.failureChanged:
            return
        plainExp(self.pfd.failures, self.failuredf, {'file':'', 'error':''})
        
    def disableFile(self, i:int) -> None:
        '''overwrite the Usegment and MLsegment files so this file cannot be measured. useful if the function is measuring values from a bad image'''
        file = self.file(i)
        self.testFunc(file, measure=False).disableFile()
        
    def approveFile(self, i:int, export:bool=True, count:bool=True, whiteOut:bool=False):
        '''approve the file'''
 
        if self.df.loc[i, 'approved'] == True:
            return
        self.df.loc[i, 'approved'] = True
        folder = os.path.join(cfg.path.server, self.df.loc[i, 'fostr'])
        file = self.df.loc[i, 'file']
        if os.path.exists(file):
            if whiteOut:
                whiteoutAll(file)

            # set the current folder dataframe and file names to the current folder
            self.setFolderAtt(folder)

        # change the failure file in this file's folder
        if hasattr(self, 'failuredf') and len(self.failuredf)>0:
            self.failuredf.loc[self.failuredf.file==file, 'error'] = 'approved'
            self.failureChanged = True

        if export:
            self.exportFolderFailures()
        if count:
            self.countFailures()
            
    def approveFolder(self, fostr:str, whiteOut:bool=False, export:bool=True):
        '''approve all files in the folder'''
        # get the list of files to approve
        ffiles = self.df[self.df.fostr==fostr]
        for i,_ in ffiles.iterrows():
            self.approveFile(i, export=False, count=False, whiteOut=whiteOut)
        
        # if export:
        #     self.exportFolderFailures()
        self.sw.approve()
        self.countFailures()
            
    def approveFolderi(self, i:int, whiteOut:bool=False, export:bool=True) -> None:
        '''approve all files in folder given a folder number'''
        fostr = self.df.loc[i, 'fostr']
        self.approveFolder(fostr, whiteOut=whiteOut, export=export)
        
    def approveAllMatch(self, mustMatch:list=[], canMatch:list=[], export:bool=True, whiteOut:bool=False):
        '''approve and whiteout all images that match the strings'''
        for i in range(len(self.df)):
            file = self.file(i)
            if fh.allIn(mustMatch, file) and fh.anyIn(canMatch, file):
                self.approveFile(i, whiteOut=True, count=False, export=export)
            
    def openFolder(self, i:int) -> None:
        '''open the folder in explorer'''
        folder = self.folder(i)
        fh.openExplorer(folder)
        
    def adjustNozzle(self, fostr:str) -> None:
        '''open the nozDims spreadsheet and a writing line image'''
        folder = os.path.join(cfg.path.server, fostr)
        fh.openExplorer(folder)
        pfd = fh.printFileDict(folder)
        if os.path.exists(pfd.nozDims):
            openInExcel(pfd.nozDims)
        pfd.findVstill()
        for file in pfd.vstill:
            if 'w1p3' in file:
                openInPaint(file)
                return
        
    def openPaint(self, i:int) -> None:
        '''open a single file in paint'''
        file = self.file(i)
        openInPaint(file)  # from m_tools
        
    def openFolderInPaint(self, fostr:str) -> None:
        '''open all the bad files in the folder in paint'''
        ffiles = self.bad[self.bad.fostr==fostr]
        for i,_ in ffiles.iterrows():
            self.openPaint(i)
        
    def export(self):
        '''export the failure summary file'''
        plainExp(self.failureFile, self.df, {}, index=False)
        