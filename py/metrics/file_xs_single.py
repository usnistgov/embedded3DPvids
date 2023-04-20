#!/usr/bin/env python
'''Functions for collecting data from stills of single line xs'''

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
from file_xs import *
from file_single import *


# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)

#----------------------------------------------

class fileXSSingle(fileXS, fileSingle):
    '''collects data about single line XS segments'''
    
    def __init__(self, file:str, diag:int=0, acrit:int=2500, **kwargs):
        super().__init__(file, diag=diag, acrit=acrit, **kwargs)

    def xsSegment() -> None:
        '''im is imported image. 
        s is is the scaling of the stitched image compared to the raw images, e.g. 0.33 
        title is the title to put on the plot
        name is the name of the line, e.g. xs1
        acrit is the minimum segment size to be considered a cross-section
        '''
        self.segmenter = segmenter(self.im, acrit=self.acrit, diag=max(0, self.diag-1))
        if not self.segmenter.success:
            return
        self.singleMeasure()
    
    def measure(self) -> None:
        '''import image, filter, and measure cross-section'''
        self.name = self.lineName('xs')
        if 'I_M' in self.file or 'I_PD' in self.file:
            self.im = vc.imcrop(self.im, 10)
        # label connected components
        self.title = os.path.basename(self.file)
        self.im = vm.normalize(self.im)
        self.xsSegment()
        