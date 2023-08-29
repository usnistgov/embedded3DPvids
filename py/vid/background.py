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
from noz_frame import *


# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)


#----------------------------------------------

class background:
    '''holds information about the background'''
    
    def __init__(self, printFolder:str, mode:int=2, **kwargs):
        self.mode = mode
        self.printFolder = printFolder
        if 'pfd' in kwargs:
            self.pfd = kwargs['pfd']  # print file dict
        else:
            self.pfd = fh.printFileDict(printFolder)
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
            
    def exportBackground(self, overwrite:bool=False, diag:int=0, flip:bool=False, frameAve:bool=True, curveFit:bool=False, **kwargs) -> None:
        '''create a background file'''
        fn = self.backgroundFN()
        if not os.path.exists(fn) or overwrite:
            if frameAve:
                if 'mode' in kwargs:
                    self.mode = kwargs.pop('mode')

                self.background = self.fs.frame(mode=self.mode, diag=diag-1, overwrite=True, flip=flip, **kwargs)
                self.background = cv.medianBlur(self.background, 11)
            else:
                backSub = cv.createBackgroundSubtractorMOG2()
                frames = self.fs.getFrames(diag=diag-1, overwrite=True, flip=flip, **kwargs)
                for frame in frames:
                    backSub.apply(frame)
                self.fs.randomBounds(**kwargs)
                frame = self.fs.randomFrame()
                mask = cv.bitwise_not(backSub.apply(frame))
                self.background = cv.bitwise_and(frame, frame, mask=mask)
                count = 1
                while mask.mean()<255 and count<25:
                    count+=1
                    if count%5==0:
                        backSub = cv.createBackgroundSubtractorMOG2()
                    frame = self.fs.randomFrame()
                    newmask = cv.bitwise_not(backSub.apply(frame))
                    diffmask = cv.subtract(newmask, mask)
                    newbg = cv.bitwise_and(frame, frame, mask=diffmask)
                    mask = cv.add(mask, newmask)
                    if mask.mean()>0:
                        self.background = cv.add(self.background, newbg)
            if curveFit:
                self.fitBackground(**kwargs)
                self.background = self.curveBackground.copy()
                
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
        
    def region(self, x0:int, xf:int, y0:int, yf:int, channel:int, AA:np.array, BB:np.array) -> Tuple[np.array, np.array]:
        '''get a set of x,y,z lists for each region'''
        im = self.background[y0:yf, x0:xf, channel]
        h,w,_ = self.background.shape
        if x0<0:
            x0 = x0+w
        if xf<0:
            xf = xf+w
        if y0<0:
            y0 = y0+h
        if yf<0:
            yf = yf+h
        x = np.linspace(x0, xf-1, xf-x0)
        y = np.linspace(y0, yf-1, yf-y0)
        X, Y = np.meshgrid(x, y, copy=False)
        Z = im
        X = X.flatten()
        Y = Y.flatten()
        A = np.array([X*0+1, X, Y, X**2, X**2*Y, X**2*Y**2, Y**2, X*Y**2, X*Y, X**3, Y**3]).T
        B = Z.flatten()
        if len(AA)>0:
            AA = np.concatenate((AA, A), axis=0)
            BB = np.concatenate((BB,B), axis=0)
        else:
            AA = A
            BB = B
            
        return AA,BB
    
    def fitValue(self, coeff:list, x:int, y:int, channel:int) -> None:
        '''get the value of the curve fit'''
        basis = [x*0+1, x, y, x**2, x**2*y, x**2*y**2, y**2, x*y**2,x*y,x**3,y**3]
        out = sum(basis*coeff)
        self.curveBackground[y,x,channel] = max(0,min(255,int(out)))
        
    def fitChannel(self, channel:int, **kwargs) -> None:
        '''fit each channel to a 2D polynomial and reconstruct the image channel'''
        im = self.background[:,:,channel]
        A = []
        B = []
        h,w,_ = self.background.shape
        if 'nd' in kwargs:
            xL = kwargs['nd'].xL - 30
            xR = kwargs['nd'].xR + 30
            yB = kwargs['nd'].yB + 30
        else:
            xL = 200
            xR = w-200
            yB = 350
        regions = [{'x0':0, 'xf':xL, 'y0':0, 'yf':yB}, # left side 
                   {'x0':xR, 'xf':w-1, 'y0':0, 'yf':yB}, # right side
                   {'x0':0, 'xf':w-1, 'y0':yB, 'yf':h-1}] # bottom side
        for r in regions:
            A,B = self.region(r['x0'],r['xf'],r['y0'],r['yf'],channel, A, B)   
        coeff, ri, rank, s = np.linalg.lstsq(A, B, rcond=None)   # get the fit
        for r in [regions[2]]:
            [[self.fitValue(coeff, x, y, channel) for x in range(r['x0'], r['xf'])] for y in range(r['y0'],r['yf'])]  # update pixel vals
        
    def fitBackground(self, **kwargs) -> None:
        '''fit the background to a 2D polynomial and reconstruct the image'''
        self.curveBackground = self.background.copy()
        for channel in range(3):
            self.fitChannel(channel, **kwargs)
   
