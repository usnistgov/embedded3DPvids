#!/usr/bin/env python
'''Functions for collecting frames from videos'''

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


# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)


#----------------------------------------------

class frameSelector:
    
    def __init__(self, printFolder:str, pfd:fh.printFileDict):
        self.frames = []
        self.printFolder = printFolder
        self.pfd = pfd
        
    def tlistFromProgPos(self, ymin:int=5, ymax:int=70, zmin:int=-20, numpics:int=6) -> list:
        '''get a random list of times from the progPos table'''
        prog,units = plainIm(self.pfd.progPos, ic=0)
        prog = prog[(prog.zt<0)&(prog.yt>ymin)&(prog.yt<ymax)&(prog.zt>zmin)]        # select moves with no extrusion that aren't close to the edge
        prog.reset_index(inplace=True, drop=True)
        tlist = list((prog['tf']+prog['t0'])/2)
        indices = random.sample(range(0, len(prog)), numpics-1)
        tlist = [self.randTime(prog.loc[i]) for i in indices]
        tlist = [prog.loc[0]['tf']-0.5] + tlist
        return tlist
    
    def tlistFromProgDims(self) -> list:
        '''get a random list of times from the progDims table'''
        self.prog, units = plainIm(self.pfd.progDims, ic=0)
        if len(self.prog)==0:
            raise ValueError('No programmed timings in folder')
        else:
            l0 = list(self.prog.loc[:10, 'tf'])     # list of end times
            l1 = list(self.prog.loc[1:, 't0'])      # list of start times
            ar = np.asarray([l0,l1]).transpose()    # put start times and end times together
            tlist = np.mean(ar, axis=1)             # list of times in gaps between prints
            tlist = np.insert(tlist, 0, 0)          # put a 0 at the beginning, so take a still at t=0 before the print starts

    def randTime(self, row:pd.Series) -> float:
        '''get a random time between these two times'''
        f = random.random()
        return row['t0']*f + row['tf']*(1-f)
    
    def lowpass(self, frame:np.array, dd:int=30) -> np.array:
        '''filter out very bright parts of the image'''
        h,w,_ = frame.shape
        center = np.median(frame[int(h/2):int(3*h/4), int(w/4):int(3*w/4), :]).astype(dtype=np.uint8)
        _, thresh = cv.threshold(frame,center+dd,255,cv.THRESH_TOZERO_INV)
        return thresh

    
    def frame(self, mode:int=0, diag:int=0, numpics:int=6, ymin:int=5, ymax:int=70, zmin:int=-20, overwrite:bool=False, useStills:bool=False, **kwargs) -> np.array:
        '''get an averaged frame from several points in the stream to blur out all fluid and leave just the nozzle. 
        mode=0 to use median frame, mode=1 to use mean frame, mode=2 to use lightest frame
        useStills=True to use stills from printing'''
        if useStills:
            if len(self.pfd.vstill)==0:
                self.pfd.findVstill()
            if len(self.pfd.vstill)>numpics:
                vstill = list(filter(lambda f: not 'p5' in f and not 'l3' in f, self.pfd.vstill))
                ilist = random.sample(range(0, len(vstill)), numpics)
                self.frames = [cv.imread(vstill[i]) for i in ilist]
        if len(self.frames)==0 or (overwrite and not useStills):
            if len(self.pfd.progPos)>0:
                tlist = self.tlistFromProgPos(ymin, ymax, zmin, numpics)
            else:
                if len(self.pfd.progDims)>0:
                    tlist = self.tlistFromProgDims()
                else:
                    raise ValueError('No programmed dimensions in folder')
            if not hasattr(self, 'vd'):
                self.vd = vidData(self.printFolder)
            self.frames = [vm.blackenRed(self.vd.getFrameAtTime(t)) for t in tlist]  # get frames in gaps between prints
            
        if mode==0:
            out = np.median(self.frames, axis=0).astype(dtype=np.uint8) # median frame
        elif mode==1:
            out = np.mean(self.frames, axis=0).astype(dtype=np.uint8)  # average all the frames
        elif mode==2:
            out = np.max(self.frames, axis=0).astype(dtype=np.uint8)   # lightest frame (do not do this for accurate nozzle dimensions)
        elif mode==3:
            medi = np.median(self.frames, axis=0).astype(dtype=np.uint8) # median frame
            maxi = np.max(self.frames, axis=0).astype(dtype=np.uint8)   # lightest frame (do not do this for accurate nozzle dimensions)
            _,thresh = cv.threshold(cv.cvtColor(medi, cv.COLOR_BGR2GRAY), (np.min(medi)+np.max(maxi))/2, 255, cv.THRESH_BINARY_INV)
            imshow(medi, maxi, thresh)
            out = cv.add(cv.bitwise_and(medi, medi, mask=cv.bitwise_not(thresh)), cv.bitwise_and(maxi, maxi, mask=thresh))
        elif mode==4:
            out = np.min(self.frames, axis=0).astype(dtype=np.uint8)   # darkest frame 
        if diag>0:
            imshow(*self.frames, numbers=True, perRow=10)
        return out