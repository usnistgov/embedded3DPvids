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
import traceback

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
from tools.config import cfg
from f_tools import *
from print_folders import *
from file_names import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

#----------------------------------------------

class folderLoop:
    '''loops a function over all printFolders in the topFolder. 
    the function needs to have only one arg, folder, and all other variables need to go in kwargs
    folders could be either the top folder to recurse into, or a list of folders'''
    
    def __init__(self, folders:Union[str, list], func, mustMatch:list=[], canMatch:list=[], printTraceback:bool=False, printErrors:bool=True, folderDiag:int=0, **kwargs):
        if type(folders) is list:
            # list of specific folders
            self.folders = []
            for folder in folders:
                self.folders = self.folders + printFolders(folder, mustMatch=mustMatch, canMatch=canMatch)
        elif not os.path.exists(folders):
            self.topFolder = ''
            self.folders = []
        else:
            # top folder, recurse
            self.topFolder = folders
            self.folders = printFolders(folders)
        self.func = func
        self.mustMatch = mustMatch
        self.canMatch = canMatch
        self.kwargs = kwargs
        self.printTraceback = printTraceback
        self.printErrors = printErrors
        self.folderDiag = folderDiag
        
    def runFolder(self, folder:str) -> None:
        '''run the function on one folder'''
        if not isPrintFolder(folder):
            return
        
        # check that the folder matches all keys
        if not allIn(self.mustMatch, folder):
            return
        
        if not anyIn(self.canMatch, folder):
            return

        if self.folderDiag>0:
            print(folder)
        try:
            self.func(folder, **self.kwargs)
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            self.folderErrorList.append({'folder':folder, 'error':e})
            if self.printErrors:
                print(e)
            if self.printTraceback:
                traceback.print_exc()

        
    def run(self) -> list:
        '''apply the function to all folders'''
        self.folderErrorList = []
        for folder in self.folders:
            self.runFolder(folder)
        return self.folderErrorList
    
    def testFolderError(self, i:int, **kwargs) -> None:
        '''test a single file that threw an error. i is the row in self.folderErrorList'''
        if i>len(self.folderErrorList):
            print(f'{i} is greater than number of error files ({len(self.folderErrorList)})')
            return
        row = self.folderErrorList[i]
        print(row)
        self.func(row['folder'], **kwargs)
        
    def openErrorFolder(self, i:int) -> None:
        if i>len(self.folderErrorList):
            print(f'{i} is greater than number of error files ({len(self.folderErrorList)})')
            return
        row = self.folderErrorList[i]
        openExplorer(row['folder'])
        
    def exportErrors(self, fn:str) -> None:
        '''export the error list to file'''
        plainExp(fn, pd.DataFrame(self.folderErrorList), {'folder':'', 'error':''}, index=False)
    

class folderFileLoop(folderLoop):
    '''loops a function over all files in a printFolder, and loops over all printfolders that match the keys'''
    
    def __init__(self, folders:Union[str, list], **kwargs):
        super().__init__(folders, self.folderFunc, **kwargs)
        
        
    def folderFunc(self, folder, **kwargs):
        '''the function to run on a single folder. this should be overwritten for subclasses. folderFunc should call runFile on all files in the folder'''
        return
    
    def run(self) -> list:
        '''the function to run in order to run all folders'''
        self.fileErrorList = []
        return super().run()
        
    def runFile(self, file:str, **kwargs) -> None:
        '''run func on a single file'''
        try:
            self.fileFunc(file, **kwargs)
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            raiseError = True
            self.fileErrorList.append({'file':file, 'error':e}) 
            if self.printErrors:
                print(e)
    
    def testFileError(self, i:int, **kwargs) -> None:
        '''test a single file that threw an error. i is the row in self.fileErrorList'''
        if i>len(self.fileErrorList):
            print(f'{i} is greater than number of error files ({len(self.fileErrorList)})')
            return
        row = self.fileErrorList[i]
        try:
            self.fileFunc(row['file'], **kwargs)
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            raiseError = True
            self.fileErrorList[i]['error']=e 
            if self.printErrors:
                print(e)
            if self.printTraceback:
                traceback.print_exc()
        else:
            self.fileErrorList[i]['error'] = ''
        print(self.fileErrorList[i])