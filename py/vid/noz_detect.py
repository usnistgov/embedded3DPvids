#!/usr/bin/env python
'''Functions for holding data about the nozzle'''

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
import copy

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
from background import *
from noz_detector import *
from timeCounter import *


# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)


#----------------------------------------------

class nozData(timeObject):
    '''holds metadata about the nozzle'''
    
    def __init__(self, folder:str, maskPad:int=0, bgmode:int=fcModes.lightest, **kwargs):
        super().__init__()
        self.printFolder = folder
        if 'pfd' in kwargs:
            self.pfd = kwargs['pfd']
        else:
            self.pfd = fh.printFileDict(self.printFolder)
        self.maskPadLeft = maskPad
        self.maskPadRight = maskPad
        self.pxpmm = self.pfd.pxpmm()
        
        if 'Under' in self.printFolder:
            self.ndGlobal = nozDimsUnder(self.pfd)
            self.nd = nozDimsUnder(self.pfd)
        else:
            self.ndGlobal = nozDimsSide(self.pfd)
            self.nd = nozDimsSide(self.pfd)
        self.fs = frameSelector(self.printFolder, self.pfd)
        
        self.bg = background(self.printFolder, pfd=self.pfd, fs=self.fs, mode=bgmode)
        
    def nozDims(self):
        '''get the nozzle xL, xR, yB'''
        return self.nd.nozDims()
    
    def resetDims(self):
        '''reset the noz dims to the imported dims'''
        self.nd.copyDims(self.ndGlobal)

    #-----------------------------

    def subtractBackground(self, im:np.array, dilate:int=0, diag:int=0, **kwargs) -> np.array:
        subtracted = self.bg.subtractBackground(im, diag=diag)
        subtracted = self.maskNozzle(subtracted, dilate=dilate, **kwargs)
        if diag>0:
            imshow(im, bg, subtracted)
        return subtracted
    
    def exportBackground(self, **kwargs):
        self.bg.exportBackground(nd=self.nd, **kwargs)
        
    def __getattr__(self, s):
        return getattr(self.nd, s)
        

    #--------------------------------------------------------------------
    
    def detectNozzle(self, overwrite:bool=False, **kwargs) -> None:
        '''find the bottom corners of the nozzle, trying different images. suppressSuccess=True to only print diagnostics if the run fails'''
#         logging.info(f'Detecting nozzle in {self.printFolder}')
        if not overwrite:
            out = self.nd.importNozzleDims()
            if out==0:
                # we already detected the nozzle
                return 0
            
        if not 'max_line_gap' in kwargs:
            kwargs['max_line_gap'] = 100
        self.detector = self.createDetector(**kwargs)
        self.detector.detectNozzle(overwrite=overwrite, **kwargs)
        self.nd = self.detector.nd
        
    def createDetector(self, **kwargs):
        if 'Under' in self.printFolder:
            return nozDetectorUnder(self.fs, self.pfd, self.printFolder, **kwargs)
        else:
            return nozDetectorSide(self.fs, self.pfd, self.printFolder, **kwargs)

    #------------------------------------------------------------------------------------

    def maskNozzle(self, frame:np.array, dilate:int=0, ave:bool=False, invert:bool=False, normalize:bool=True, bottomOnly:bool=False, bottomDilate:int=0, **kwargs) -> np.array:
        '''block the nozzle out of the image. 
        dilate is number of pixels to expand the mask. 
        ave=True to use the background color, otherwise use white
        invert=False to draw the nozzle back on
        '''
        s = frame.shape
        kwargs2 = {}
        if ave:
            # mask with the average value of the frame
            nc = np.copy(frame)*0 # create mask
            average = frame.max(axis=0).mean(axis=0)
            if len(s)==3:
                average = [int(i) for i in average]
            else:
                average = int(average)
            kwargs2['val'] = average
        if bottomOnly:
            # only mask the bottom of the nozzle
            kwargs2['y0'] = -1
        if 'crops' in kwargs:
            kwargs2['crops'] = kwargs['crops']
        if len(frame.shape)==3:
            kwargs2['color'] = True
        
        if 'nd' in kwargs:
            nd = kwargs['nd']
        else:
            nd = self.nd
        mask = nd.nozCover(dilate, dilate, bottomDilate, **kwargs2)
        mask[:,-1] = 0  # make sure right edge is left open
        
        if invert:
            out = cv.subtract(frame, mask)
        else:
            out = cv.add(frame, mask) 

        if normalize:
            norm = np.zeros(out.shape)
            out = cv.normalize(out,  norm, 0, 255, cv.NORM_MINMAX) # normalize the image
        return out

    def adjustEdges(self, im:np.array, crops:dict, **kwargs) -> None:
        '''adjust the boundaries of the nozzle for this specific image, knowing dimensions should be close to stored dimensions'''
        detector = self.createDetector(**kwargs)
        detector.defineCritValsImage(self.nd, crops, **kwargs)
        try:
            detector.detectNozzle1(im, export=False, **kwargs)
        except ValueError as e:
            return 
        else:
            nd1 = detector.nd.nozDims()
            if nd1['xL']>self.nd.xL:
                detector.nd.setDims({'xL':self.nd.xL})
            if nd1['xR']<self.nd.xR:
                detector.nd.setDims({'xR':self.nd.xR})
            self.nd = detector.nd
        
        
    def absoluteCoords(self, d:dict) -> dict:
        return self.nd.absoluteCoords(d)
    
    def relativeCoords(self, x:float, y:float, reverse:bool=False) -> dict:
        return self.nd.relativeCoords(x,y,reverse=reverse)
    
        
    def dentHull(self, hull:list, crops:dict) -> list:
        '''conform the contour to the nozzle'''
        return self.nd.dentHull(hull, crops)
    
    def padNozzle(self, **kwargs):
        return self.nd.padNozzle(**kwargs)


        
#--------------------------------------------

def exportNozDims(folder:str, overwrite:bool=False, **kwargs) -> None:
    pfd = fh.printFileDict(folder)
    if not overwrite and hasattr(pfd, 'nozDims') and hasattr(pfd, 'background'):
        return
    nv = nozData(folder)
    nv.detectNozzle(overwrite=overwrite, **kwargs)
    nv.exportBackground(overwrite=overwrite)

def exportNozDimsRecursive(folder:str, overwrite:bool=False, **kwargs) -> list:
    '''export stills of key lines from videos'''
    fl = fh.folderLoop(folder, exportNozDims, overwrite=overwrite, **kwargs)
    fl.run()
    return fl

def exportNozDimsRetry(folder:str, lcrit:int=15, overwrite:bool=False) -> None:
    error = True
    e0 = None
    i = 0
    while error and i<lcrit:
        i = i+1
        try: 
            nd = nt.nozData(folder)
            nd.detectNozzle(diag=2, overwrite=overwrite)
            nd.nozDims()
        except KeyboardInterrupt:
            return
        except Exception as e:
            if 'Failed' in e or 'Detect' in e:
                error = True
                e0 = e
            else:
                raise e
        else:
            error = False
    if error:
        raise e0
    
    
def checkBackground(folder:str, diag:bool=False) -> float:
    '''check if the background is good or bad'''
    nd = nozData(folder)
    nd.bg.importBackground()
    empty = nd.maskNozzle(nd.bg.background, dilate=10, ave=True, invert=False, normalize=False)
    mm = empty.min(axis=0).min(axis=0).min(axis=0)
    if diag:
        imshow(empty)
    return mm
    
def findBadBackgrounds(topFolder:str, exportFolder:str) -> pd.DataFrame:
    '''find the bad backgrounds in the folder'''
    out = []
    for folder in fh.printFolders(topFolder):
        out.append({'folder':folder, 'mm':checkBackground(folder)})
    df = pd.DataFrame(out)
    df.sort_values(by='mm', inplace=True)
    df.reset_index(inplace=True, drop=True)
    fn = os.path.join(exportFolder, 'badBackgrounds.csv')
    df.to_csv(fn)
    logging.info(f'Exported {fn}')
    return df

def fixBackground(folder:str, diag:int=0) -> int:
    '''try to fix the background image in this folder'''
    nv = nozData(folder)
    mm = 1
    count = 0
    mcrit =120
    while mm<mcrit and count<3:
        nv.bg.exportBackground(diag=diag, overwrite=True)
        mm = checkBackground(folder)
        count+=1
        if diag>0:
            print(f'Count {count} mm {mm}')
    if mm<mcrit:
        nv.bg.stealBackground(diag=diag)