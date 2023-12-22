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
from file_horiz_SDT import *
from im.segment import *
import im.morph as vm

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)

#----------------------------------------------

def underSDTMeasure(file:str, **kwargs) -> Tuple[dict, dict]:
    return fileUnderSDT(file, **kwargs).values() 

def underSDTTestFile(fstr:str, fistr:str, **kwargs) -> None:
    '''test a single file and print diagnostics'''
    testFile(fstr, fistr, fileUnderSDT, ['emptiness','x0','segments'], **kwargs)
    
def fileUnderSDTFromTag(folder:str, tag:str, **kwargs):
    '''get the fileUnderSDT from a string that is in the file name'''
    return fileMetricFromTag(fileUnderSDT, folder, tag, **kwargs)
        
class fileUnderSDT(fileHorizSDT):
    '''for singledoubletriple lines viewed from under. protocol very similar to fileHorizSDT, but with small changes'''
    
    def __init__(self, file:str, acrit:int=4000, segmentMode:str='gradient', gradientSlope:int=10, **kwargs):
        self.segmentMode = segmentMode
        self.fillDilation = 0
        self.gradientSlope = gradientSlope
        self.grayBlur = 2
        if 'topthresh' in kwargs:
            self.topthresh = kwargs['topthresh']
        super().__init__(file, acrit=acrit, **kwargs)
        
        
    def addToTestFile(self) -> None:
        '''add the current measurements to the csv of intended measurements for XSSDT'''
        csv = testCSV('SDTUnder')
        slist = ['emptiness','y0', 'meanT', 'segments']
        super().addToTestFile(csv, slist)
            
    def getCrop(self):
        '''get the crop position. only export if export=True and there is no existing row'''
        # rc = {'relative':True, 'w':800, 'h':275, 'wc':400, 'hc':205}
        rc = {'relative':True, 'w':1000, 'h':300, 'wc':500, 'hc':220}
        if 'p' in self.tag:
            rc['w'] = 800
            rc['wc'] = 450
        self.makeCrop(rc, export=self.exportCropLocs, overwrite=self.overwriteCropLocs)
        
    def removeGlare(self) -> np.array:
        _,thresh = cv.threshold(255-self.im,120,255,cv.THRESH_TRUNC)
        thresh = 255-normalize(thresh)
        return thresh
    
    def gradientMultiply(self, trunc:np.array, contrast:int) -> np.array:
        '''increase the brightness on the left side of the image'''
        h,w,_ = trunc.shape
        mask = np.array([[[(1-contrast/255*(w-x)/w) for i in range(3)] for x in range(w)] for y in range(h)])
        out = (np.multiply(trunc, mask)).astype(np.uint8)
        out = normalize(out)
        if self.diag>2:
            imshow((mask*255).astype(np.uint8), out, trunc, cv.absdiff(out,trunc))
        return out
        
    def horizContrast(self, trunc:np.array, segments:int, topthresh:int) -> np.array:
        '''increase contrast in each column of the image'''
        h,w,_ = trunc.shape
        cw = w/segments
        for i in range(segments):
            x0 = int(i*cw)
            xf = int((i+1)*cw)
            bw = cv.cvtColor(trunc[:, x0:xf], cv.COLOR_BGR2GRAY)
            if bw.min()<topthresh-10:
                trunc[:, x0:xf] = normalize(trunc[:, x0:xf])
        return trunc
    
    def makeSegmenter(self, trunc:np.array, topthresh:int):
        segmode = [ sMode.kmeans, sMode.threshold]
        si = segmenter(trunc, acrit=self.acrit, diag=max(0, self.diag-1)
                                   , fillMode=fi.fillMode.fillTiny
                                   , topthresh=topthresh
                                   , nozData=self.nd, crops=self.crop, segmentMode=segmode
                                   , nozMode=nozMode.full, removeSharp=True, closeTop=False
                                   , grayBlur=self.grayBlur, addLeftEdge=True, addRightEdge=True, trimNozzle=True
                                  , closing=self.fillDilation, complete=False, normalize=self.normalize)
        si.fill()
        si.filled = vm.openMorph(si.filled, 2)
        return si
    
    def halfhalf(self, trunc:np.array) -> None:
        '''split the image in half and process it separately'''
        topthreshes = [220, 230]
        segmenters = []
        for thresh in topthreshes:
            segmenters.append(self.makeSegmenter(trunc, thresh))
        self.segmenter = segmenters[0]
        self.segmenter.filled[:, :400] = segmenters[1].filled[:, :400]
        self.segmentComplete()
        
    def singleSegmenter(self, trunc:np.array) -> None:
        if not hasattr(self, 'topthresh'):  
            if 'o' in self.tag:
                self.topthresh = 220
            else:
                self.topthresh = 140
        self.segmenter = self.makeSegmenter(trunc, self.topthresh)
        if 'p' in self.tag:
            self.segmenter.filled[:, :20] = 0
        self.segmentComplete()

    def segment(self) -> None:
        '''segment the foreground'''
        trunc = self.im
        if self.segmentMode =='gradient' or self.segmentMode=='horizContrast':
            if 'o' in self.tag:
                if self.segmentMode=='gradient':
                    trunc = self.gradientMultiply(trunc, self.gradientSlope)
                elif self.segmentMode=='horizContrast':
                    trunc = self.horizContrast(trunc, 10, 210)
            self.singleSegmenter(trunc)
            if not self.segmenter.success and 'o' in self.tag:
                if self.diag>1:
                    logging.info('Segmenter failed. Switching to halfhalf')
                trunc = self.im
                self.halfhalf(trunc)
        elif self.segmentMode=='half':
            self.halfhalf(trunc)
        else:
            self.singleSegmenter(trunc)
        self.componentMask = self.segmenter.labelsBW.copy()
            

    def segmentClean(self):
        '''erase bad components'''
        for s in ['cnt', 'hull']:
            if hasattr(self, s):
                delattr(self, s)
        if not self.segmenter.success:
            return
        self.segmenter.eraseBorderLengthComponents(lcrit=400)
        #self.segmenter.eraseBorderTouchComponent(2, '-y', checks=False)
        #self.segmenter.eraseBorderTouchComponent(2, '+y', checks=False)
        self.segmenter.eraseBorderClingers(40)

        
    def measure(self) -> None:
        '''measure horizontal SDT line'''
        newSegment = False
        if self.checkWhite(val=254):
            # white image
            if self.overrideSegment:
                self.getCrop()
                self.cropIm(normalize=self.normalize)
                self.im[:,:] = 0
                self.componentMask = self.im
                self.exportSegment(overwrite=False)              # export segmentation
            self.stats['error'] = 'white'
            return
        self.initialize()
        self.getCrop()
        self.generateIm0()
        self.padNozzle(dr=10)
        self.cropIm(normalize=self.normalize)
        self.findIntendedCoords()                        # find where the object should be
        self.findIntendedPx()
        
        # use existing segmentation
        if not self.overrideSegment:
            self.importUsegment()
            if hasattr(self, 'Usegment'):
                if self.Usegment.sum().sum()>0:
                    self.segmenter = segmenterDF(self.Usegment, im=self.im, acrit=self.acrit, diag=self.diag)
                    self.componentMask = self.segmenter.filled
                else:
                    self.stats['error'] = 'white'
                    return
                
        # generate a new segment
        if not hasattr(self, 'segmenter'):
            newSegment=True
            self.generateSegment(overwrite=False)
            self.Usegment = self.componentMask.copy()
            
        self.segmentClean()
 
        if newSegment:
            self.exportSegment(overwrite=True)

        if not self.segmenter.success:
            if self.diag>0:
                logging.warning(f'Segmenter failed on {self.file}')
            self.display()
            self.resetStats()
            return
        getLDiff = ('o' in self.tag and not ('w1' in self.tag or 'd1' in self.tag))
        self.dimsMulti(getLDiff=getLDiff)  
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
        self.display()
        
