#!/usr/bin/env python
'''Functions for collecting data from stills of single vertical lines'''

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
import copy

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
sys.path.append(os.path.dirname(os.path.dirname(currentdir)))
from file_Vert import *
from file_Single import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)

#----------------------------------------------

class fileVertSingle(fileVert, fileVert):
    '''for single vertical lines'''
    
    def __init__(self, file, progDims:pd.DataFrame, diag:int=0, acrit:int=2500, **kwargs):
        super().__init__(file, diag=diag, acrit=acrit, **kwargs)
        self.progDims = progDims
        self.measure()

    def measure(self) -> None:
        '''measure vertical lines'''
        self.name = int(self.lineName('vert'))
        self.maxlen = self.progDims[self.progDims.name==(f'vert{self.name}')].iloc[0]['l']
        self.maxlen = int(self.maxlen/self.scale)
        # label connected components

        for s in ['im', 'scale', 'maxlen']:
            if not hasattr(self, s):
                raise ValueError(f'{s} undefined for {self.file}')

        self.segmenter = segmenterSingle(self.im, acrit=self.acrit, diag=max(0,self.diag-1))
        if not self.segmenter.success:
            return 
 
        self.segmenter.eraseBorderComponents(10)
            # remove anything too close to the border
        if not self.segmenter.success:
            return
        
        self.dims(selectInline=True)
        self.display()
