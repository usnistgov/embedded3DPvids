#!/usr/bin/env python
'''Functions for checking that Usegment and MLsegment sizes are correct'''

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
from crop_locs import *
from m_tools import *
from tools.plainIm import *
from val.v_print import printVals
from progDim.prog_dim import getProgDims, getProgDimsPV
import file.file_handling as fh
from vid.noz_detect import *
from tools.timeCounter import timeObject
import tools.regression as reg

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)
pd.set_option('display.max_rows', 500)


#----------------------------------------------
 
class sizeChecker:
    '''for a folder, check that the Usegment and MLsegment sizes are correct'''
    
    def __init__(self, folder:str, diag:int=1):
        self.folder = folder
        self.diag = diag
        self.pfd = fh.printFileDict(folder)
        self.cl = cropLocs(self.folder, pfd=self.pfd)
        self.pfd.findVstill()
        self.out = []
        for file in self.pfd.vstill:
            self.checkSizes(file)
        self.df = pd.DataFrame(self.out)
        if self.diag>1 and len(self.df[self.df.fail])>0:
            display(self.df[self.df.fail][['bn', 'mw', 'mh', 'uw', 'uh', 'fail']])
        
    def fn(self, bn:str, s:str) -> str:
        '''machine learning segmentation file name'''
        return os.path.join(self.folder, f'{s}segment', bn.replace('vstill', f'{s}segment'))
    
    def cropfn(self, bn:str) -> str:
        '''cropped file name'''
        return os.path.join(self.folder, 'crop', bn.replace('vstill', 'vcrop'))
            
    def checkSizes(self, file:str) -> None:
        '''check sizes of ML and Usegment files'''
        bn = os.path.basename(file)
        ml = self.fn(bn, 'ML')
        mw,mh = get_image_size(ml)
        ul = self.fn(bn, 'U')
        uw,uh = get_image_size(ul)
        crop = self.cl.getCrop(file)
        w = crop['xf']-crop['x0']
        h = crop['yf']-crop['y0']
        d = {'file':file, 'bn':bn, 'mw':mw==w, 'mh':mh==h, 'uw':uw==w, 'uh':uh==h}
        d['fail'] = not(d['mw']&d['mh']&d['uw']&d['uh'])
        self.out.append(d)
        
    def correct(self, cropFolder:str, func) -> None:
        '''correct all of the incorrectly sized files. cropFolder is a folder to export cropped files to for ML. func is the fileMetric class to use to create new segmentation'''
        nd = nozData(self.folder, pfd=self.pfd)
        pv = printVals(self.folder, pfd = self.pfd, fluidProperties=False)
        pg  = getProgDimsPV(pv)      
        
        for i,row in self.df[self.df.fail].iterrows():
            file = row['file']
            bn = row['bn']
            vs = func(file, overrideSegment=True, measure=False, pfd=self.pfd, cl=self.cl, nd=nd, pv=pv, pg=pg)
            if not (row['mh'] and row['mw']):
                # redo ML
                ml = self.fn(bn, 'ML')
                if os.path.exists(ml):
                    os.remove(ml)
                    if self.diag>0:
                        logging.info(f'Removed {os.path.basename(ml)}')
                vs.getCrop(export=False)
                vs.exportCrop(overwrite=True, diag=self.diag)
                cropfn = self.cropfn(bn)
                if os.path.exists(cropfn):
                    newname = os.path.join(cropFolder, os.path.basename(cropfn))
                    shutil.copyfile(cropfn, newname)
                    if self.diag>0:
                        logging.info(f'Copied {os.path.basename(cropfn)}')
            if not (row['uh'] and row['uw']):
                # redo unsupervised
                vs.measure()
                vs.exportSegment(overwrite=True, diag=self.diag)
                