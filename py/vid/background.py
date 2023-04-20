#!/usr/bin/env python
'''Functions for collecting background of video'''

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
import csv
import random
import time

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
from im.imshow import imshow
import im.morph as vm
import im.crop as vc
from tools.config import cfg
from tools.plainIm import *
import file.file_handling as fh
from v_tools import vidData
from noz_plots import *
from noz_dims import *
from noz_frame import frameSelector


# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)


#----------------------------------------------

class background:
    '''holds information about the background'''
    
    def __init__(self, printFolder:str, **kwargs):
        self.printFolder = printFolder
        if 'pfd' in kwargs:
            self.pfd = kwargs['pfd']  # print file dict
        else:
            self.pfd = printFileDict(printFolder)
        if 'fs' in kwargs:
            self.fs = kwargs['fs']
        else:
            self.fs = frameSelector(self.printFolder, self.pfd)  # frame selector
    
    def backgroundFN(self):
        return self.pfd.newFileName('background', 'png')
    
    def exportBackground0(self, diag:int=0):
        fn = self.backgroundFN()
        cv.imwrite(fn, self.background)
        logging.info(f'Exported {fn}')
        if diag>0:
            imshow(self.background)
            
    def exportBackground(self, overwrite:bool=False, diag:int=0, **kwargs) -> None:
        '''create a background file'''
        fn = self.backgroundFN()
        if not os.path.exists(fn) or overwrite:
            self.background = self.fs.frame(mode=2, diag=diag-1, overwrite=True, **kwargs)
            self.background = cv.medianBlur(self.background, 5)
            self.exportBackground0(diag=diag)
    
    def importBackground(self, overwrite:bool=False) -> None:
        '''import the background from file or create one and export it'''
        if hasattr(self, 'background') and not overwrite:
            # already have a background
            return
        fn = self.pfd.newFileName('background', 'png')
        if not os.path.exists(fn):
            # create a background
            self.exportBackground()
            return
        
        # import background from file
        self.background = cv.imread(fn)
        return
    
    def subtractBackground(self, im:np.array, diag:int=0) -> np.array:
        '''subtract the nozzle frame from the color image'''
        self.importBackground()
        bg = self.background
        bg = cv.medianBlur(bg, 5)
        subtracted = 255-cv.absdiff(im, bg)
        return subtracted

                
    def stealBackground(self, diag:int=0) -> None:
        '''steal a background from another folder in this series'''
        spacing = re.split('_', os.path.basename(self.printFolder))[-1]
        for n in ['0.625', '0.750', '0.875', '1.000']:
            newfolder = self.printFolder.replace(spacing, n)
            if os.path.exists(newfolder) and not newfolder==self.printFolder:
                nd = background(newfolder)
                nd.importBackground()
                if hasattr(nd, 'background'):
                    print(f'Stealing background from {newfolder}')
                    self.background = nd.background
                    self.exportBackground0(diag=diag)
                    return
    
