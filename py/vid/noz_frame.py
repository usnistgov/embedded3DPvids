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

class frameGetModes:
    '''modes for selecting a list of frames'''
    progPos = 0    # select certain points in time based on the progPos table
    still = 1     # use the vstills we already created
    snap = 2       # use the snaps that the shopbot created
    

class fcModes:
    '''modes for combining frames'''
    median = 0
    mean = 1
    lightest = 2
    medianLightest = 3
    darkest = 4    
    

class frameSelector:
    
    def __init__(self, printFolder:str, pfd:fh.printFileDict):
        self.frames = []
        self.printFolder = printFolder
        self.pfd = pfd
        
    def tlistFromProgPos(self, ymin:int=5, ymax:int=70, zmin:int=-20, numpics:int=6, **kwargs) -> list:
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
        l0 = list(self.prog.loc[:10, 'tf'])     # list of end times
        l1 = list(self.prog.loc[1:, 't0'])      # list of start times
        ar = np.asarray([l0,l1]).transpose()    # put start times and end times together
        tlist = np.mean(ar, axis=1)             # list of times in gaps between prints
        tlist = np.insert(tlist, 0, 0)          # put a 0 at the beginning, so take a still at t=0 before the print starts
        return tlist

    def randTime(self, row:pd.Series) -> float:
        '''get a random time between these two times'''
        f = random.random()
        return row['t0']*f + row['tf']*(1-f)
    
    def randomBounds(self, ymin:int=5, ymax:int=70, zmin:int=-20):
        '''get the bounds of when the nozzle is in the bath'''
        prog,units = plainIm(self.pfd.progPos, ic=0)
        prog = prog[(prog.zt<0)&(prog.yt>ymin)&(prog.yt<ymax)&(prog.zt>zmin)]        # select moves with no extrusion that aren't close to the edge
        self.tmin = prog.t0.min()
        self.tmax = prog.tf.max()
        if not hasattr(self, 'vd'):
            self.vd = vidData(self.printFolder)
        self.fmin = self.tmin/self.vd.duration
        self.fmax = self.tmax/self.vd.duration
    
    def randomFrame(self, inbath:bool=True) -> np.array:
        '''get a totally random frame'''
        t = random.uniform(self.tmin, self.tmax)
        return self.vd.getFrameAtTime(t)
    
    def lowpass(self, frame:np.array, dd:int=30) -> np.array:
        '''filter out very bright parts of the image'''
        h,w,_ = frame.shape
        center = np.median(frame[int(h/2):int(3*h/4), int(w/4):int(3*w/4), :]).astype(dtype=np.uint8)
        _, thresh = cv.threshold(frame,center+dd,255,cv.THRESH_TOZERO_INV)
        return thresh
    
    def flipFrame(self, frame:np.array) -> np.array:
        '''flip the bottom half of the frame'''
        frame2 = frame.copy()
        h,w,_ = frame.shape
        frame2[int(h/2):, :, :] = np.flip(frame2[int(h/2):, :, :], 1)
        return frame2
    
    def flipFrames(self) -> None:
        '''flip the bottom half of all of the frames and add that to the frame list'''
        self.frames = self.frames + [self.flipFrame(f) for f in self.frames]

        
    def getFramesStill(self, numpics:int=6, **kwargs) -> None:
        '''get frames from existing stills'''
        if len(self.pfd.vstill)==0:
            self.pfd.findVstill()
        if len(self.pfd.vstill)>numpics:
            vstill = list(filter(lambda f: not 'p5' in f and not 'l3' in f, self.pfd.vstill))
            ilist = random.sample(range(0, len(vstill)), numpics)
            self.frames = [cv.imread(vstill[i]) for i in ilist]
            
    def getFramesSnap(self, **kwargs) -> None:
        '''get the first two images in the snap folder'''
        folder = os.path.join(self.printFolder, 'raw')
        if not os.path.exists(folder):
            return
        self.frames = [cv.imread(os.path.join(folder, f))[5:-5,5:-5,:] for f in os.listdir(folder)[:2]]
            
    def getFramesProgPos(self, **kwargs) -> None:
        if len(self.pfd.progPos)>0:
            tlist = self.tlistFromProgPos(**kwargs)
        else:
            if len(self.pfd.progDims)>0:
                tlist = self.tlistFromProgDims()
            else:
                raise ValueError('No programmed dimensions in folder')
        if not hasattr(self, 'vd'):
            self.vd = vidData(self.printFolder)
        self.frames = [self.vd.getFrameAtTime(t) for t in tlist]  # get frames in gaps between prints
            
    def getFrameGetMode(self, **kwargs) -> int:
        if 'frameGetMode' in kwargs:
            frameGetMode = kwargs['frameGetMode']
        elif self.pfd.date>230807 and not 'XS' in self.printFolder:
            # use the snaps programmed into the print path after augst 7, 2023
            frameGetMode = frameGetModes.snap
        else:
            frameGetMode = frameGetModes.progPos
        return frameGetMode
        
    def getFrames(self, overwrite:bool=False, flip:bool=False, **kwargs) -> list:
        '''get a list of frames'''
        
        if len(self.frames)>0 and not overwrite:
            return
        
        if 'tlist' in kwargs:
            if not hasattr(self, 'vd'):
                self.vd = vidData(self.printFolder)
            self.frames = [self.vd.getFrameAtTime(t) for t in kwargs['tlist']]
            return
        frameGetMode = self.getFrameGetMode(**kwargs)
            
        if frameGetMode==frameGetModes.still:
            self.getFramesStill(**kwargs)
        elif frameGetMode==frameGetModes.snap:
            self.getFramesSnap(**kwargs)

        if len(self.frames)==0 or frameGetMode==frameGetModes.progPos:
            self.getFramesProgPos(**kwargs)
                
        if flip:
            self.flipFrames()
        return self.frames
    
    def frame(self, mode:int=0, diag:int=0, overwrite:bool=False, **kwargs) -> np.array:
        '''get an averaged frame from several points in the stream to blur out all fluid and leave just the nozzle. 
        mode=0 to use median frame, mode=1 to use mean frame, mode=2 to use lightest frame
        useStills=True to use stills from printing'''
        self.getFrames(overwrite=overwrite, **kwargs)
        if mode==fcModes.median:
            out = np.median(self.frames, axis=0).astype(dtype=np.uint8) # median frame
        elif mode==fcModes.mean:
            out = np.mean(self.frames, axis=0).astype(dtype=np.uint8)  # average all the frames
        elif mode==fcModes.lightest:
            out = np.max(self.frames, axis=0).astype(dtype=np.uint8)   # lightest frame (do not do this for accurate nozzle dimensions)
        elif mode==fcModes.medianLightest:
            medi = np.median(self.frames, axis=0).astype(dtype=np.uint8) # median frame
            maxi = np.max(self.frames, axis=0).astype(dtype=np.uint8)   # lightest frame (do not do this for accurate nozzle dimensions)
            _,thresh = cv.threshold(cv.cvtColor(medi, cv.COLOR_BGR2GRAY), (np.min(medi)+np.max(maxi))/2, 255, cv.THRESH_BINARY_INV)
            out = cv.add(cv.bitwise_and(medi, medi, mask=cv.bitwise_not(thresh)), cv.bitwise_and(maxi, maxi, mask=thresh))
        elif mode==fcModes.darkest:
            out = np.min(self.frames, axis=0).astype(dtype=np.uint8)   # darkest frame 
        if diag>0:
            imshow(*self.frames, numbers=True, perRow=10)
        return out
    
    
