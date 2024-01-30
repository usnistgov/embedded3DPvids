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
    
def fileHorizSDTFromTag(folder:str, tag:str, **kwargs):
    '''get the filehorizSDT from a string that is in the file name'''
    return fileMetricFromTag(fileHorizSDT, folder, tag, **kwargs)
        
class fileHorizSDT(fileHoriz, fileSDT):
    '''for singledoubletriple lines'''
    
    def __init__(self, file:str, acrit:int=2000, **kwargs):
        self.maxlen = 800
        self.fillDilation = 0
        self.grayBlur = 1
        self.segmentSteps = 0
        self.importedImages = False
        super().__init__(file, acrit=acrit, **kwargs)
        
    def addToTestFile(self) -> None:
        '''add the current measurements to the csv of intended measurements for XSSDT'''
        csv = testCSV('SDTHoriz')
        slist = ['emptiness','yBot', 'meanT', 'segments']
        super().addToTestFile(csv, slist)
                
    def findIntendedCoords(self) -> None:
        '''find the intended x0,y0,xc,and yc of the assembly'''
        rc1, rc2, w1, w2, l, lprog = self.intendedRC()  # intended coords in mm
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
        self.ideals['wn'] = lprog
        
    def findDisplacement(self) -> None:
        '''find the displacement of the center and dimensions relative to the intended dimensions'''
        self.stats['wn'] = self.stats['w']
        self.units['wn'] = self.units['w']
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
        for s,ival in {'h':'h', 'meanT':'h', 'vest':'v', 'vintegral':'v', 'wn':'wn'}.items():
            # ratio of size to intended size
            if s in self.stats:
                self.stats[s] = (self.stats[s]/self.ideals[ival])
                self.units[s] = 'intended'
        for s in ['area', 'vleak', 'aspect']:
            if s in self.stats:
                self.stats.pop(s)
                
        # remove length measurements for mid-print measurements 
        if not 'o' in self.tag:
            for s in ['h', 'vest', 'vintegral', 'w', 'meanT', 'stdevT', 'minmaxT', 'wn']:
                if s in self.stats:
                    self.stats.pop(s)
                    
    def getCrop(self):
        '''get the crop position. only export if export=True and there is no existing row'''
        # rc = {'relative':True, 'w':800, 'h':275, 'wc':400, 'hc':205}
        rc = {'relative':True, 'w':1000, 'h':310, 'wc':500, 'hc':250}
        if 'p' in self.tag:
            rc['w'] = 800
            rc['wc'] = 450
        self.makeCrop(rc, export=self.exportCropLocs, overwrite=self.overwriteCropLocs)
        
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
        segmode = [sMode.adaptive, sMode.kmeans]
        #segmode = [sMode.kmeans]
        #segmode = [sMode.adaptive]
        self.segmenter = segmenter(self.im, acrit=self.acrit, diag=max(0, self.diag-1)
                                   , fillMode=fi.fillMode.fillByContours
                                   , nozData=self.nd, crops=self.crop, segmentMode=segmode
                                   , nozMode=nozMode.full, removeSharp=True, closeTop=False
                                   , grayBlur=self.grayBlur, addLeftEdge=True, addRightEdge=True, trimNozzle=True
                                  , closing=self.fillDilation, complete=False, normalize=self.normalize)
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
        for s in ['cnt', 'hull']:
            if hasattr(self, s):
                delattr(self, s)
        if not self.segmenter.success:
            return
        self.segmenter.eraseFullHeightComponents(margin=2)
        self.segmenter.eraseBorderLengthComponents(lcrit=400)
        self.segmenter.eraseBorderTouchComponent(2, '-y', checks=False)
        self.segmenter.eraseBorderTouchComponent(2, '+y', checks=False)
        # if 'o' in self.tag or 'p5' in self.tag or 'p4' in self.tag or 'p3' in self.tag:
        #     self.segmenter.eraseBorderTouchComponent(2, '+x', checks=True)
        # if 'o' in self.tag or 'p1' in self.tag or 'p2' in self.tag:
            # self.segmenter.eraseBorderTouchComponent(2, '-x', checks=True)
        self.segmenter.eraseBorderClingers(40)
        # self.segmenter.selectCloseObjects(self.idealspx)  # remove bubbles and debris that are far from the main object

    def generateSegment(self, overwrite:bool=False):
        '''generate a new segmentation'''
        self.segment()
        self.componentMask = self.segmenter.labelsBW.copy()
        # self.exportSegment(overwrite=overwrite)
        
    def checkAndDims(self) -> int:
        '''check that the segmenter succeeded, get dims, and then check if dims are correct'''
        for s in ['cnt']:
            if hasattr(self, s):
                delattr(self, s)   # reset contour
        if not self.segmenter.success:
            return 1
        getLDiff = ('o' in self.tag and not ('w1' in self.tag or 'd1' in self.tag))
        self.dimsMulti(getLDiff=getLDiff)
        if 'w1' in self.tag or 'd1' in self.tag:
            rcrit = 0.2
        elif 'w2' in self.tag or 'd2' in self.tag:
            if 'p1' in self.tag or 'p3' in self.tag or 'p2' in self.tag:
                rcrit = 1
            else:
                rcrit = 1
        elif 'w3' in self.tag or 'd3' in self.tag:
            if 'p1' in self.tag or 'p3' in self.tag or 'p2' in self.tag:
                rcrit = 1.5
            else:
                rcrit = 2
        if (not 'roughness' in self.stats or (self.stats['roughness']>rcrit)):
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
        
    def fattenEdges(self, r:int=10):
        '''dilate the right edge, fill and then erode. this is for lines where it thins out at the right and leaves holes so the filament can't be filled. r is the dilation/erosion size in px'''
        cm = self.componentMask.copy()
        m = 200
        
        # empty the middle
        cm[:, m:-m] =  np.zeros(cm[:, m:-m].shape)
        
        # dilate the ends
        cm = dilate(cm, r)
        
        # add back into the image and fill
        cm = cv.add(self.componentMask, cm)
        cm = fi.filler(cm).filled
        
        # split filled image into one image that's just ends and another that's just middle
        cmends = cm.copy()
        cmmid = cm.copy()
        rr = 0
        cmmid[:, :m-rr] = np.zeros(cmmid[:, :m-rr].shape)
        cmmid[:, -m+rr:] = np.zeros(cmmid[:, -m+rr:].shape)
        jj = 0
        cmends[:, m+r+jj:-m-r-jj] =  np.zeros(cmends[:, m+r+jj:-m-r-jj].shape)
        
        # erode the ends
        cmends = erode(cmends, r)
        
        # recombine middle and ends
        cm = cv.add(cmends, cmmid)
        self.componentMask = cm
        
        
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
        # get the real nozzle position and pad it
        if not 'o' in self.tag:
            self.nd.adjustEdges(self.im0, self.crop, diag=self.diag-2, yCropMargin=0)  # find the nozzle in the image and use that for future masking
        self.padNozzle(left=3, right=5, bottom=10)
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
            
        # use ML
        if self.forceML:
            self.importMLsegment()
            self.reconcileImportedSegment(func='horiz', smallCrit=500)
            newSegment=True
            
        self.segmentClean()
        
        # try fattening edges
        o1 = self.checkAndDims()
        if o1==1:
            # empty space. try to fill it
            self.fattenEdges(5)
            scopy = self.segmenter
            s2 = segmenterDF(self.componentMask, im=self.im, acrit=self.acrit, diag=self.diag)
            for s in ['im', 'gray', 'thresh']:
                if hasattr(scopy,s):
                    setattr(s2, s, getattr(scopy, s))
            self.segmenter = s2
            self.segmentClean()
            o1 = self.checkAndDims()
            if o1==0:
                newSegment=True
        
        # try machine learning
        if self.useML and not self.forceML:
            o1 = self.checkAndDims()
            if o1==1:
                self.importMLsegment()
                self.reconcileImportedSegment(func='horiz', smallCrit=500)
                self.segmentClean()
                self.checkAndDims()
                newSegment=True
 
        if newSegment:
            self.exportSegment(overwrite=True)

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
        
