#!/usr/bin/env python
'''Functions for collecting data from a still of a horizontal SDT line'''

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
from file_unit import *
from file_horiz import *
from file_SDT import *
from im.segment import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)

#----------------------------------------------

def horizSDTMeasure(file:str, **kwargs) -> Tuple[dict, dict]:
    return fileHorizSDT(file, **kwargs).values() 

def horizSDTTestFile(fstr:str, fistr:str, **kwargs) -> None:
    '''test a single file and print diagnostics'''
    testFile(fstr, fistr, fileHorizSDT, ['emptiness','x0','segments'], **kwargs)
        
class fileHorizSDT(fileHoriz, fileSDT):
    '''for singledoubletriple lines'''
    
    def __init__(self, file:str, diag:int=0, acrit:int=1000, overrideSegment:bool=False, **kwargs):
        self.overrideSegment = overrideSegment
        self.maxlen = 800
        self.fillDilation = 0
        self.grayBlur = 1
        self.segmentSteps = 0
        self.importedImages = False
        super().__init__(file, diag=diag, acrit=acrit, **kwargs)
        
    def addToTestFile(self) -> None:
        '''add the current measurements to the csv of intended measurements for XSSDT'''
        csv = testCSV('SDTHoriz')
        slist = ['emptiness','yBot', 'meanT', 'segments']
        super().addToTestFile(csv, slist)
                
    def findIntendedCoords(self) -> None:
        '''find the intended x0,y0,xc,and yc of the assembly'''
        rc1, rc2, w1, w2, l = self.intendedRC()  # intended coords in mm
        for j in [['dy', 'h']]:
            coord = j[0][1]
            bottom = (rc2[j[0]]+w1/2)      # the bottom edge should be 1/2 diameter below the first line center
            top = (rc1[j[0]]-w2/2)       # the top edge should be 1/2 diameter above the last line center
            self.ideals[f'{coord}f'] = min(top, bottom)
            self.ideals[f'{coord}0'] = max(top, bottom)
            self.ideals[j[1]] = abs(top-bottom)     # get the ideal width
            self.ideals[f'{coord}c'] = (top+bottom)/2  # get the ideal center
        h = self.ideals['h']
        r = h/2
        self.ideals['area'] = l*h
        self.ideals['v'] = (l-h)*np.pi*r**2 + 4/3*np.pi*r**3
        self.ideals['w'] = l
        
    def findDisplacement(self) -> None:
        '''find the displacement of the center and dimensions relative to the intended dimensions'''
        for s,ival in {'yc':'yc', 'y0':'y0', 'yf':'yf'}.items():
            # ratio of size to intended size
            if s in self.stats:
                self.stats[s] = ((self.stats[s]-self.ideals[ival])/self.pv.dEst)
                self.units[s] = 'dEst'
        for s in ['dy0l', 'dy0r', 'dy0lr', 'dyfl', 'dyfr', 'dyflr', 'space_l', 'space_r', 'space_b', 'w']:
            if s in self.stats:
                if not self.units[s]=='mm':
                    raise ValueError(f'{s} must be in mm')
                self.stats[s] = self.stats[s]/self.pv.dEst
                self.units[s] = 'dEst'
        for s,ival in {'h':'h', 'meanT':'h', 'vest':'v', 'vintegral':'v'}.items():
            # ratio of size to intended size
            if s in self.stats:
                self.stats[s] = (self.stats[s]/self.ideals[ival])
                self.units[s] = 'intended'
        for s in ['area', 'vleak', 'aspect']:
            if s in self.stats:
                self.stats.pop(s)
                
        # remove length measurements for mid-print measurements 
        if not 'o' in self.tag:
            for s in ['h', 'vest', 'vintegral', 'w', 'meanT', 'stdevT', 'minmaxT']:
                if s in self.stats:
                    self.stats.pop(s)
                    
    def getCrop(self, export:bool=True, overwrite:bool=False):
        '''get the crop position. only export if export=True and there is no existing row'''
        rc = {'relative':True, 'w':800, 'h':275, 'wc':400, 'hc':205}
        self.makeCrop(rc, export=export, overwrite=overwrite)
        
    def removeThreads(self) -> None:
        '''remove zigzag threads from top right part of binary image'''
        thresh = self.segmenter.labelsBW.copy()
        h0,w0 = thresh.shape
        cut = int(w0*0.9)
        right = thresh[:, cut:]
        if right.sum().sum()==0:
            return
        left = thresh[:, :cut]
        leco = co.getContours(left)
        if len(leco)==0:
            return
        rico = co.getContours(right)
        if len(rico)==0:
            return
        leftcnt = np.concatenate(leco)
        lx, ly, lw, lh = cv.boundingRect(leftcnt)
        rightcnt = np.concatenate(rico)
        rx, ry, rw, rh = cv.boundingRect(rightcnt)
        if rh>lh*1.5:
            # crop the top
            ymax = int(min(leftcnt[:,:,1]))           # top of left side
            thresh[:ymax, cut:] = 0                   # clear out anything above the top on the right side             
            self.segmenter.thresh = thresh.copy()     # save this part
            
            # find right edge
            right = thresh[:, cut:]
            rightcnt = np.concatenate(co.getContours(right))
            rightcnt[:,:,0] = rightcnt[:,:,0]+cut
            top = rightcnt[rightcnt[:,:,1]==min(rightcnt[:,:,1])]   # find points that are at top
            xmax = w0-(max(top[:,0]))
            
            # close right edge, fill, and remove right edge
            self.segmenter.thresh = self.segmenter.closeBorder(self.segmenter.thresh, '+x', 255, pos=xmax)
            self.segmenter.fill()
            self.segmenter.filled = self.segmenter.closeBorder(self.segmenter.filled, '+x', 0, pos=xmax)
            self.segmentComplete()
        
    def segment(self) -> None:
        '''segment the foreground'''
        self.segmenter = segmenter(self.im, acrit=self.acrit, diag=max(0, self.diag-1)
                                   , fillMode=fi.fillMode.fillByContours
                                   , nozData=self.nd, crops=self.crop, segmentMode=[sMode.adaptive, sMode.kmeans]
                                   , nozMode=nozMode.full, removeSharp=True
                                   , grayBlur=self.grayBlur, addLeftEdge=True, addRightEdge=True, trimNozzle=True
                                  , closing=self.fillDilation, complete=False)
        self.segmentSteps +=1
        self.segmenter.fill()
        self.segmentComplete()
        self.removeThreads()
        self.componentMask = self.segmenter.labelsBW.copy()
        
    def segmentComplete(self):
        '''fill the thresholded image, label components, and filter components'''
        self.segmenter.complete()
        self.segmentClean()
        
    def segmentClean(self):
        '''erase bad components'''
        self.segmenter.eraseFullHeightComponents(margin=2)
        self.segmenter.eraseBorderLengthComponents(lcrit=400)
        self.segmenter.eraseBorderTouchComponent(2, '-y')
        self.segmenter.eraseBorderTouchComponent(2, '+y')
        # self.segmenter.selectCloseObjects(self.idealspx)  # remove bubbles and debris that are far from the main object

    def generateSegment(self, overwrite:bool=False):
        '''generate a new segmentation'''
        self.segment()
        self.componentMask = self.segmenter.labelsBW.copy()
        self.exportSegment(overwrite=overwrite)
        
    def checkAndDims(self) -> int:
        '''check that the segmenter succeeded, get dims, and then check if dims are correct'''
        if not self.segmenter.success:
            return 1
        getLDiff = ('o' in self.tag and not ('w1' in self.tag or 'd1' in self.tag))
        self.dimsMulti(getLDiff=getLDiff)
        if (not 'emptiness' in self.stats or (self.stats['emptiness']>0.3)):
            return 1
        else:
            return 0
    
    def resetStats(self):
        '''reset stats to the initial state'''
        stat = {}
        for s in ['gname', 'ltype', 'pr', 'pname', 'time', 'wtime', 'zdepth']:
            stat[s]=self.stats[s]
        stat['line'] = ''
        self.stats = stat

    def measure(self) -> None:
        '''measure horizontal SDT line'''
        if self.checkWhite(val=254):
            # white image
            if self.overrideSegment:
                self.getCrop(overwrite=False)
                self.cropIm()
                self.im[:,:] = 0
                self.componentMask = self.im
                self.exportSegment(overwrite=False)              # export segmentation
            return
        self.initialize()
        self.getCrop(overwrite=False)
        self.generateIm0()
        # get the real nozzle position and pad it
        if not 'o' in self.tag:
            self.nd.adjustEdges(self.im0, self.crop, diag=self.diag-2, yCropMargin=0)  # find the nozzle in the image and use that for future masking
        self.padNozzle(left=3, right=3, bottom=1)
        self.cropIm()
        self.findIntendedCoords()                        # find where the object should be
        self.findIntendedPx()
        if self.overrideSegment:
            self.generateSegment(overwrite=True)
        else:
            self.importSegmentation()
            if hasattr(self, 'Usegment') and hasattr(self, 'MLsegment') and not self.Usegment.shape==self.MLsegment.shape:
                self.generateSegment(overwrite=True)
                self.Usegment = self.componentMask.copy()
            self.reconcileImportedSegment(eraseTop=False, largeCrit=100000)
            self.segmentClean()
        
        if not hasattr(self, 'segmenter'):
            self.segment()
            self.exportSegment()              # export segmentation
        
        o1 = self.checkAndDims()
        while (o1==1) and not self.importedImages and self.segmentSteps<5:
            self.segment()
            for s in ['cnt']:
                if hasattr(self, s):
                    delattr(self, s)   # reset contour
            o1 = self.checkAndDims()
        if not self.segmenter.success:
            if self.diag>0:
                logging.warning(f'Segmenter failed on {self.file}')
            self.display()
            self.resetStats()
            return
        if not 'o' in self.tag:
            self.gaps(self.pv.dEst)
        if len(self.stats)==1:
            if self.diag>0:
                logging.warning(f'Measurement failed on {self.file}')
            self.display()
            return
        self.adjustForCrop(self.stats, self.crop)  # px, full image
        self.stats, self.units = self.makeMM(self.stats, self.units)
        self.makeRelative()           # mm, relative to nozzle
        if self.diag>0:
            self.findNozzlePx()
        self.findDisplacement()  # dEst, relative to intended coords
        self.renameY()
        if 'w' in self.tag and not 'o' in self.tag:
            self.dropVariables(['vintegral', 'yTop', 'yc', 'space_r', 'dy0r', 'dyfr'])
        self.display()
        
    def export(self):
        '''overwrite this row in the measurement file'''
        if not hasattr(self.pfd, 'measure'):
            return
        df, du = plainIm(self.pfd.measure, ic=0)
        if not hasattr(self.pfd, 'failures'):
            failures = pd.DataFrame({'file':[]})
        else:
            failures, _ = plainIm(self.pfd.failures, ic=0)
        row = df[df.line==self.name]
        if len(row)==0:
            # this file is not in the measure table
            raise ValueError('This file is not in the measure table')
        i = row.iloc[0].name
        if len(self.stats['line'])<1:
            # this run failed
            if not self.file in failures.file:
                failures = pd.concat([failures, pd.DataFrame([{'file':self.file}])])
            for key in df.columns:
                if key in self.stats:
                    val = self.stats[key]
                else:
                    val = np.nan
                df.loc[i,key] = val
        else:
            # this run succeeded
            for key,val in self.stats.items():
                df.loc[i,key] = val
            if self.file in failures.file:
                failurerow = failures[failures.file==self.file].iloc[0].name
                failures.loc[failurerow, 'file'] = ''
        plainExp(self.pfd.failures, failures, {'file':''})
        plainExp(self.pfd.measure, df, du)
        
