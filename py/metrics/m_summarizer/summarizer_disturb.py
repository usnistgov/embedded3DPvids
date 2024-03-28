#!/usr/bin/env python
'''Collect data from all folders into a single folder, for disturbed single lines'''

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
from m_folder.folder_xs_disturb import *
from m_folder.folder_vert_disturb import *
from m_folder.folder_horiz_disturb import *
from m_file.file_xs_disturb import *
from m_file.file_vert_disturb import *
from m_file.file_horiz_disturb import *
from summarizer import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)


#--------------------------------

class summarizerDisturb(summarizer):
    '''recursively create summaries. measureClass is a class definition for a folderMetric class'''
    
    def __init__(self, topFolder:str, measureClass, printType:str, mustMatch:list=[], **kwargs):
        super().__init__(topFolder, measureClass, mustMatch=mustMatch+[printType], **kwargs)
        
        
    def csvFN(self, s:str) -> str:
        '''get a filename for the csv'''
        return os.path.join(cfg.path.fig, 'singleDoubleTriple', 'summaries', f'{self.printType}Disturb{s}.csv')
    
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
    

class summarizerXSdisturb(summarizerDisturb):
    '''recursively create summaries for disturbed XS lines. measureClass is a class definition for a folderMetric class'''
    
    def __init__(self, topFolder:str, dire:str, **kwargs):
        self.dire = dire
        super().__init__(topFolder, folderXSDisturb, 'XS', mustMatch=mustMatch+[dire], **kwargs)
        
class summarizerVertdisturb(summarizerDisturb):
    '''recursively create summaries for disturbed vertical lines. measureClass is a class definition for a folderMetric class'''
    
    def __init__(self, topFolder:str, **kwargs):
        super().__init__(topFolder, folderVertDisturb, 'Vert', **kwargs)
        
class summarizerHorizdisturb(summarizerDisturb):
    '''recursively create summaries for disturbed horizontal lines. measureClass is a class definition for a folderMetric class'''
    
    def __init__(self, topFolder:str, **kwargs):
        super().__init__(topFolder, folderHorizDisturb, 'Horiz', **kwargs)
        
    
#--------------------------------
    
class xsdisturbFailureTest(failureTest):
    '''for testing failed disturbed XS files'''
    
    def __init__(self, dire):
        s = summarizerXSdisturb('', dire)
        super().__init__(s.failureFN(), xsDisturbTestFile)
        
class vertdisturbFailureTest(failureTest):
    '''for testing failed disturbed vertical files'''
    
    def __init__(self, dire):
        s = summarizerVertdisturb('')
        super().__init__(s.failureFN(), vertDisturbTestFile)
        
class horizdisturbFailureTest(failureTest):
    '''for testing failed disturbed horizontal files'''
    
    def __init__(self, dire):
        s = summarizerHorizdisturb('')
        super().__init__(s.failureFN(), horizDisturbTestFile)
        