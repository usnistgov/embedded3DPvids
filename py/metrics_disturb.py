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
from pic_stitch_bas import stitchSorter
from file_handling import isSubFolder, fileScale
import im_crop as vc
import im_morph as vm
from tools.imshow import imshow
from tools.plainIm import *
from val_print import *
from vid_noz_detect import nozData
from metrics_xs import *
from metrics_vert import *
from metrics_horiz import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)


#--------------------------------


def measureFolders(topFolder:str, overwrite:bool=False, **kwargs) -> List[str]:
    '''measure all stills in folders recursively from the top folder'''
    errorFolders = []
    if not fh.isPrintFolder(topFolder):
        for f in os.listdir(topFolder):
            errorFolders = errorFolders + measureFolders(os.path.join(topFolder, f), overwrite=overwrite, **kwargs)
        return errorFolders
    
    try:
        if 'Horiz' in topFolder:
            horizDisturbMeasures(topFolder, overwrite=overwrite, **kwargs)
        elif 'XS' in topFolder:
            xsDisturbMeasures(topFolder, overwrite=overwrite, **kwargs)
        elif 'Vert' in topFolder:
            vertDisturbMeasures(topFolder, overwrite=overwrite, **kwargs)
    except Exception as e:
        errorFolders.append(topFolder)
    return errorFolders