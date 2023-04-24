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
from crop_locs import *
from m_file.file_metric import *
from file.file_handling import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 4)
pd.set_option('display.max_rows', 500)


#----------------------------------------------

class cropExporter(folderFileLoop):
    '''for exporting cropped images and csvs of crop locations. folders is either a list of folders or the top folder. fileMetricFunc is a class definition for a fileMetric object. overwrite=True to overwrite crop images.'''
    
    def __init__(self, folders:Union[str, list], fileMetricFunc, overwrite:bool=False, exportDiag:int=0, **kwargs):
        super().__init__(folders, **kwargs)
        self.overwrite=overwrite
        self.fileMetricFunc = fileMetricFunc
        self.exportDiag = exportDiag
        
    def fileFunc(self, file:str, **kwargs):
        '''crop and export a single file'''
        if not self.overwrite:
            folder = os.path.basename(file)
            bn = os.path.basename(file)
            file2 = os.path.join(folder, 'crop', bn.replace('vstill', 'vcrop'))
            if os.path.exists(file2):
                return

        vs = self.fileMetricFunc(file, measure=False, **kwargs)
        vs.getCrop(export=False)
        vs.exportCrop(overwrite=self.overwrite, diag=self.exportDiag)
        
    def folderFunc(self, folder, **kwargs):
        '''the function to run on a single folder'''
        pfd = fh.printFileDict(folder)
        pfd.findVstill()
        cl = cropLocs(folder, pfd=pfd)
        for file in pfd.vstill:
            self.runFile(file, pfd=pfd, cl=cl)
        cl.export(overwrite=self.overwrite)
        
        
class cropLocExporter(folderFileLoop):
    '''for exporting csvs of crop locations. folders is either a list of folders or the top folder. fileMetricFunc is a class definition for a fileMetric object. overwrite=True to overwrite crop images.'''
    
    def __init__(self, folders:Union[str, list], fileMetricFunc, overwrite:bool=False, exportDiag:int=0, **kwargs):
        super().__init__(folders, **kwargs)
        self.overwrite=overwrite
        self.fileMetricFunc = fileMetricFunc
        self.exportDiag = exportDiag
        
    def fileFunc(self, file:str, **kwargs):
        '''crop and export a single file'''
        vs = self.fileMetricFunc(file, measure=False, **kwargs)
        vs.exportCropDims(overwrite=self.overwrite, export=False)
        
    def folderFunc(self, folder, **kwargs):
        '''the function to run on a single folder'''
        pfd = fh.printFileDict(folder) 
        cl = cropLocs(folder, pfd=pfd)
        if not self.overwrite and 'xf' in cl.df and not cl.df.xf.isnull().values.any():
            return
        pfd.findVstill()
        for file in pfd.vstill:
            self.runFile(file, pfd=pfd, cl=cl)
        cl.export(overwrite=self.overwrite)


class segmentExporter(folderFileLoop):
    '''for exporting segmented images. folders is either a list of folders or the top folder. fileMetricFunc is a class definition for a fileMetric object. overwrite=True to overwrite segmented images.'''
    
    def __init__(self, folders:Union[str, list], fileMetricFunc, overwrite:bool=False, exportDiag:int=0, **kwargs):
        super().__init__(folders, **kwargs)
        self.overwrite = overwrite
        self.fileMetricFunc = fileMetricFunc
        self.exportDiag = exportDiag
        
        
    def fileFunc(self, file:str, **kwargs) -> None:
        '''segment the image'''
        if not self.overwrite:
            folder = os.path.basename(file)
            bn = os.path.basename(file)
            file2 = os.path.join(folder, 'Usegment', bn.replace('vstill', 'Usegment'))
            if os.path.exists(file2):
                return
        
        kwargs['nd'].resetDims()   # reset the nozzle dimensions so we're going off the same thing every time
        vs = self.fileMetricFunc(file, measure=False, overrideSegment=True, **kwargs)
        vs.measure()
        vs.exportSegment(overwrite=self.overwrite, diag=self.exportDiag)
        
    def folderFunc(self, folder:str, **kwargs) -> None:
        '''create all segmented images in the folder'''
        pfd = fh.printFileDict(folder)
        pfd.findVstill()
        pfd.findUsegment()
        if not self.overwrite and len(pfd.vstill)>0 and len(pfd.vstill)==len(pfd.Usegment):
            return
        nd = nozData(folder, pfd=pfd)
        pv = printVals(folder, pfd = pfd, fluidProperties=False)
        pg  = getProgDimsPV(pv)
        cl = cropLocs(folder, pfd=pfd)
        for file in pfd.vstill:
            self.runFile(file, pfd=pfd, nd=nd, pv=pv, pg=pg, cl=cl, **kwargs)            
            