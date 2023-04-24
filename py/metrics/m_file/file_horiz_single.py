#!/usr/bin/env python
'''Functions for collecting data from stills of single line horiz'''

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
from file_horiz import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)

#----------------------------------------------

class horizSingleMeasures:
    '''for measuring all 3 lines from a stitched singleLine image'''
    
    def __init__(self, file:str, diag:int=0, acrit:int=1000, overwrite:bool=False, **kwargs):
        self.segmentOrig = fileHoriz(file, diag=diag, acrit=acrit, **kwargs)
        self.overwrite = overwrite
        self.folder = os.path.dirname(file)
        self.pfd = fh.printFileDict(self.folder)
        self.progDims = getProgDims(self.folder)

    def splitLines(self, margin:float=80, **kwargs) -> list:
        '''split the table of segments into the three horizontal lines. 
        margin is the max allowable vertical distance in px from the line location to be considered part of the line'''

        linelocs = [275, 514, 756] # expected positions of lines
        ylocs = [-1000,-1000,-1000] # actual positions of lines
        df0 = self.segmented.df
        # get position of largest segment
        if len(df0)==0:
            return df0
        largesty = float(df0[df0.a==df0.a.max()]['yc'])

        # take segments that are far away
        df = df0[(df0.yc<largesty-100)|(df0.yc>largesty+100)]
        if len(df)>0:
            secondy = float(df[df.a==df.a.max()]['yc'])
            df = df[(df.yc<secondy-100)|(df.yc>secondy+100)]
            if len(df)>0:
                thirdy = float(df[df.a==df.a.max()]['yc'])
                ylocs = ([largesty, secondy, thirdy])
                ylocs.sort()
            else:
                # only 2 lines
                largestI = closestIndex(largesty, linelocs)
                secondI = closestIndex(secondy, linelocs)
                if secondI==largestI:
                    if secondI==2:
                        if secondy>largesty:
                            largestI = largestI-1
                        else:
                            secondI = secondI-1
                    elif secondI==0:
                        if secondy>largesty:
                            secondI = secondI+1
                        else:
                            largestI = largestI+1
                    else:
                        if secondy>largesty:
                            secondI = secondI+1
                        else:
                            secondI = secondI-1
                ylocs[largestI] = largesty
                ylocs[secondI] = secondy
        else:
            # everything is in this line
            largestI = closestIndex(largesty, linelocs)
            ylocs[largestI] = largesty

        if diag>1:
            logging.info(f'ylocs: {ylocs}')
        self.dflist = [df0[(df0.yc>yloc-margin)&(df0.yc<yloc+margin)] for yloc in ylocs]


    def measure(self) -> None:
        '''segment the image and take measurements
        progDims holds timing info about the lines
        s is is the scaling of the stitched image compared to the raw images, e.g. 0.33
        acrit is the minimum segment size in px to be considered part of a line
        satelliteCrit is the min size of segment, as a fraction of the largest segment, to be considered part of a line
        '''
        self.fn = self.pfd.newFileName(f'horizSummary', '.csv')
        if os.path.exists(self.fn) and not self.overwrite:
            return
        self.segmentOrig.im = vm.removeBorders(self.segmentOrig.im)
        self.segmenter = segmenterSingle(self.im, acrit=self.acrit, diag=max(0, self.diag-1), removeVert=True)
        self.segmenter.eraseSmallestComponents(**self.kwargs)
        self.splitLines(**kwargs)
        self.units = {}
        self.out = []
        for j,df in enumerate(self.dflist):
            if len(df)>0:
                maxlen = self.progDims[self.progDims.name==f'horiz{j}'].iloc[0]['l']  # length of programmed line
                segment = copy.deepcopy(self.segmentOrig)
                segment.selectLine(df, maxlen, j)
                r,cmu = segment.values()
                self.out.append(r)
                if len(cmu)>len(self.units):
                    self.units = cmu
        self.df = pd.DataFrame(self.out)
        plainExp(self.fn, self.df, self.units)