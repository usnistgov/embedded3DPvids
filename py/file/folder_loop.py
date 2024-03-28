#!/usr/bin/env python
'''class for looping through all subfolders in a folder and applying some operation'''

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
from tools.plainIm import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

#----------------------------------------------

class folderLoop:
    '''loops a function over all printFolders in the topFolder. 
    the function needs to have only one arg, folder, and all other variables need to go in kwargs
    folders could be either the top folder to recurse into, or a list of folders
    func is the function to run on all folders
    mustMatch is a list of strings that must be in the print folder name
    canMatch is a list of strings. If it's not empty, all print folders must have at least one of the strings in the list
    printTraceback is true to print the traceback message when an error is hit
    printErrors is true to print error messages from each folder
    folderDiag is the diagnostics printing level to feed into the function we're running on each file
    findFolders is false to use the folder list shown in the config file
    other kwargs get fed into the function to find folders and the function we're applying to each folder
    '''
    
    def __init__(self, folders:Union[str, list], func, mustMatch:list=[], canMatch:list=[], printTraceback:bool=False, printErrors:bool=True, folderDiag:int=0, findFolders:bool=True, **kwargs):
        if findFolders:
            if type(folders) is list:
                # list of specific folders
                self.folders = []
                for folder in folders:
                    self.folders = self.folders + printFolders(folder, mustMatch=mustMatch, canMatch=canMatch, **kwargs)
            elif not os.path.exists(folders):
                self.topFolder = ''
                self.folders = []
            else:
                # top folder, recurse
                self.topFolder = folders
                self.folders = printFolders(folders, mustMatch=mustMatch, canMatch=canMatch, **kwargs)
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
            # self.folderErrorList.append({'folder':folder, 'error':'not a print folder'})
            return
        
        # check that the folder matches all keys
        if not allIn(self.mustMatch, folder):
            # self.folderErrorList.append({'folder':folder, 'error':f'does not match {self.mustMatch}'})
            return
        
        if not anyIn(self.canMatch, folder):
            # self.folderErrorList.append({'folder':folder, 'error':f'does not match {self.canMatch}'})
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
    
    def testFolderError(self, i:int, openFolder:bool=False, **kwargs) -> None:
        '''test a single file that threw an error. i is the row in self.folderErrorList'''
        if i>len(self.folderErrorList):
            print(f'{i} is greater than number of error files ({len(self.folderErrorList)})')
            return
        row = self.folderErrorList[i]
        print(row)
        if openFolder:
            openExplorer(row['folder'])
        self.func(row['folder'], **kwargs)
        
    def openErrorFolder(self, i:int) -> None:
        '''open a folder in windows explorer, given an index in the list of folders that threw errors'''
        if i>len(self.folderErrorList):
            print(f'{i} is greater than number of error files ({len(self.folderErrorList)})')
            return
        row = self.folderErrorList[i]
        openExplorer(row['folder'])
        
    def exportErrors(self, fn:str) -> None:
        '''export the error list to file'''
        plainExp(fn, pd.DataFrame(self.folderErrorList), {'folder':'', 'error':''}, index=False)
    
#----------

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
