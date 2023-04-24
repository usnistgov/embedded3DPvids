#!/usr/bin/env python
'''Functions for collecting data from stills of triple line xs'''

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
from file_xs import *


# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)



#----------------------------------------------


class fileXSTriple(fileXS):
    '''colleges data about triple line XS segments'''
    
    def __init__(self, file:str, diag:int=0, acrit:int=2500, **kwargs):
        super().__init__(file, diag=diag, acrit=acrit, **kwargs)
    
    
    def display(self) -> None:
        if self.diag==0:
            return
        cm = self.labelsBW.copy()
        cm = cv.cvtColor(cm,cv.COLOR_GRAY2RGB)
        cv.drawContours(cm, [self.hull], -1, (110, 245, 209), 6)
        cv.drawContours(cm, self.cnt, -1, (186, 6, 162), 6)
        imshow(cm)
        if hasattr(self, 'title'):
            plt.title(self.title)
    
    def measure(self) -> None:
        '''measure cross-section of 3 lines'''
        spl = re.split('xs', os.path.basename(self.file))
        name = re.split('_', spl[0])[-1] + 'xs' + re.split('_', spl[1])[1]
        self.title = os.path.basename(self.file)
        self.im = vm.normalize(self.im)

        # segment components
        if 'LapRD_LapRD' in file:
            # use more aggressive segmentation to remove leaks
            self.segmented = segmenter(self.im, acrit=self.acrit, topthresh=75, diag=max(0, self.diag-1))
        else:
            self.segmented = segmenter(self.im, acrit=self.acrit, diag=max(0, self.diag-1))
        self.filterXSComponents()
        if not self.segmented.success:
            return 