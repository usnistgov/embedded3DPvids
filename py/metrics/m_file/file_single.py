#!/usr/bin/env python
'''Functions for collecting data from stills of single lines, for a single image'''

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
from file_metric import fileMetric

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 4)
pd.set_option('display.max_rows', 500)


#----------------------------------------------

class fileSingle(fileMetric):
    '''collect measurements of segments in singleLine prints'''
    
    def __init__(self, file:str):
        super().__init__(file)
        
    def lineName(self, tag:str) -> float:
        '''for single lines, get the number of the line from the file name based on tag, e.g. 'vert', 'horiz', 'xs'. '''
        spl = re.split('_',os.path.basename(self.file))
        for st in spl:
            if tag in st:
                return float(st.replace(tag, ''))
        return -1