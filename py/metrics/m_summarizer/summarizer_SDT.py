#!/usr/bin/env python
'''Functions for collecting data from stills of disturbed lines'''

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

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
sys.path.append(os.path.dirname(os.path.dirname(currentdir)))
from m_folder.folder_xs_SDT import *
from m_folder.folder_vert_SDT import *
from m_folder.folder_horiz_SDT import *
from m_file.file_xs_SDT import *
from m_file.file_vert_SDT import *
from m_file.file_horiz_SDT import *
from summarizer import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)


#--------------------------------

class summarizerSDT(summarizer):
    '''recursively create summaries. measureClass is a class definition for a folderMetric class'''
    
    def __init__(self, topFolder:str, measureClass, printType:str, mustMatch:list=[], **kwargs):
        self.printType = printType
        super().__init__(topFolder, measureClass, mustMatch=mustMatch+[printType], **kwargs)
        
        
    def csvFN(self, s:str) -> str:
        '''get a filename for the csv'''
        if self.printType=='XS':
            f1 = f'{self.printType}{self.dire}'
        else:
            f1 = self.printType
        return os.path.join(cfg.path.fig, 'SDT', 'summaries', f'{f1}SDT{s}.csv')
    
    def summaryFN(self) -> str:
        return self.csvFN('Summaries')
    
    def failureFN(self) -> str:
        return self.csvFN('Failures')
        
    def export(self) -> None:
        '''export the big data summary and the failure summary'''
        fn1 = self.summaryFN()
        fn2 = self.failureFN()
        super().export(fn1)
        super().exportFailures(fn2)
    

class summarizerXSSDT(summarizerSDT):
    '''recursively create summaries. measureClass is a class definition for a folderMetric class'''
    
    def __init__(self, topFolder:str, dire:str, mustMatch:list=[], **kwargs):
        self.dire = dire
        super().__init__(topFolder, folderXSSDT, 'XS', mustMatch=mustMatch+[dire], **kwargs)
        
class summarizerVertSDT(summarizerSDT):
    '''recursively create summaries. measureClass is a class definition for a folderMetric class'''
    
    def __init__(self, topFolder:str, **kwargs):
        super().__init__(topFolder, folderVertSDT, 'Vert', **kwargs)
        
class summarizerHorizSDT(summarizerSDT):
    '''recursively create summaries. measureClass is a class definition for a folderMetric class'''
    
    def __init__(self, topFolder:str, **kwargs):
        super().__init__(topFolder, folderHorizSDT, 'Horiz', **kwargs)
        
    
#--------------------------------
    
class xsSDTFailureTest(failureTest):
    '''for testing failed files'''
    
    def __init__(self, dire):
        s = summarizerXSSDT('', dire)
        super().__init__(s.failureFN(), xsSDTTestFile)
        
class vertSDTFailureTest(failureTest):
    '''for testing failed files'''
    
    def __init__(self, dire):
        s = summarizerVertSDT('')
        super().__init__(s.failureFN(), vertSDTTestFile)
        
class horizSDTFailureTest(failureTest):
    '''for testing failed files'''
    
    def __init__(self, dire):
        s = summarizerHorizSDT('')
        super().__init__(s.failureFN(), horizSDTTestFile)