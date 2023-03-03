#!/usr/bin/env python
'''Functions for collecting data from stills of single line xs'''

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
from pic_stitch_bas import stitchSorter
from file_handling import isSubFolder, fileScale
import im_crop as vc
import im_morph as vm
from im_segment import *
from tools.imshow import imshow
from tools.plainIm import *
from val_print import *
from vid_noz_detect import nozData
from metrics_tools import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)



#----------------------------------------------

class xsSegment(metricSegment):
    '''collects data about XS segments'''
    
    def __init__(self, file:str, diag:int=0, acrit:int=2500, **kwargs):
        super().__init__(file, diag=diag, acrit=acrit, **kwargs)
        
        
    def filterXSComponents(self) -> None:
        '''filter out cross-section components'''
        errorRet = [], []
        h,w = self.segmenter.labelsBW.shape[:2]
        xest = w/2 # estimated x
        if h>600:
            yest = h-300
            dycrit = 200
        else:
            yest = h/2
            dycrit = h/2
        if len(self.segmenter.df)>1:
            secondLargest = 2*list(self.segmenter.df.a.nlargest(2))[1]
        seg1 = copy.deepcopy(self.segmenter)    # make a copy of the segmenter in case we need to roll back changes  
        self.segmenter.eraseBorderComponents(10)  # remove anything too close to the border
        goodpts = (abs(self.segmenter.df.xc-xest)<100)&(abs(self.segmenter.df.yc-yest)<dycrit)
        self.segmenter.selectComponents(goodpts)
            # filter by location relative to expectation and area
        if not self.segmenter.success:
            self.segmenter = seg1
            self.segmenter.selectComponents(self.segmenter.df.a>1000)   # just filter by area
        if len(self.segmenter.df)>1 and self.segmenter.df.a.max() < secondLargest:
            # largest object not much larger than 2nd largest
            self.segmenter.success = False
        
    
    def dims(self) -> None:
        '''get the dimensions of the segments'''
        roughness = getRoughness(self.segmenter.labelsBW, diag=max(0,self.diag-1))
        m = self.segmenter.largestObject() # select largest object
        self.x0 = int(m['x0'])
        self.y0 = int(m['y0'])
        self.w = int(m['w'])
        self.h = int(m['h'])
        area = int(m['a'])
        self.xc = m['xc']
        self.yc = m['yc']
        aspect = self.h/self.w # height/width
        boxcx = self.x0+self.w/2 # x center of bounding box
        boxcy = self.y0+self.h/2 # y center of bounding box
        xshift = (self.xc-boxcx)/self.w
        yshift = (self.yc-boxcy)/self.h
        self.units = {'line':'', 'aspect':'h/w', 'xshift':'w', 'yshift':'h', 'area':'px'
                  , 'w':'px', 'h':'px'
                      , 'xc':'px', 'yc':'px', 'roughness':''} # where pixels are in original scale
        self.stats = {'line':self.name, 'aspect':aspect, 'xshift':xshift, 'yshift':yshift, 'area':area*self.scale**2
                  , 'w':self.w*self.scale, 'h':self.h*self.scale
                      , 'xc':self.xc*self.scale, 'yc':self.yc*self.scale, 'roughness':roughness}


    def display(self, title:str='') -> None:
        if self.diag==0:
            return
        # show the image with annotated dimensions
        im2 = cv.cvtColor(self.segmenter.labelsBW,cv.COLOR_GRAY2RGB)
        imgi = self.im.copy()
        cv.rectangle(imgi, (self.x0,self.y0), (self.x0+self.w,self.y0+self.h), (0,0,255), 1)   # bounding box
        cv.circle(imgi, (int(self.xc), int(self.yc)), 2, (0,0,255), 2)     # centroid
        cv.circle(imgi, (self.x0+int(self.w/2),self.y0+int(self.h/2)), 2, (0,255,255), 2) # center of bounding box
        imshow(imgi, im2, self.statText())
        if hasattr(self, 'title'):
            plt.title(self.title)
        
    
    def singleMeasure(self) -> None:
        '''measure a single cross section'''
        self.filterXSComponents()
        if not self.segmenter.success:
            return 
        self.dims()
        self.display()
        
    def multiMeasure(self) -> None:
        '''measure multiple cross sections'''
        # find contours
        contours = cv.findContours(self.segmenter.labelsBW,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        self.cnt = np.vstack(contours[0])
        self.hull = cv.convexHull(self.cnt)

        # measure components
        hullArea = cv.contourArea(self.hull)
        filledArea = self.segmenter.df.a.sum()
        porosity = 1-(filledArea/hullArea)

        perimeter = 0
        for cnti in contours[0]:
            perimeter+=cv.arcLength(cnti, True)
        hullPerimeter = cv.arcLength(self.hull, True)
        excessPerimeter = perimeter/hullPerimeter - 1

        self.x0,self.y0,self.w,self.h = cv.boundingRect(self.hull)
        aspect = self.h/self.w

        M = cv.moments(self.cnt)
        self.xc = int(M['m10']/M['m00'])
        self.yc = int(M['m01']/M['m00'])
        boxcx = self.x0+self.w/2 # x center of bounding box
        boxcy = self.y0+self.h/2 # y center of bounding box
        xshift = (self.xc-boxcx)/self.w
        yshift = (self.yc-boxcy)/self.h

        self.units = {'line':'', 'segments':'', 
                      'aspect':'h/w', 'xshift':'w', 'yshift':'h', 'area':'px'
                      , 'x0':'px', 'y0':'px'
                        , 'xc':'px', 'yc':'px'
                         , 'w':'px', 'h':'px'
                      , 'porosity':'', 'excessPerimeter':''} # where pixels are in original scale
        self.stats = {'line':self.name, 'segments':len(self.segmenter.df)
                      , 'aspect':aspect, 'xshift':xshift, 'yshift':yshift, 'area':filledArea*self.scale**2
                      , 'x0':self.x0*self.scale, 'y0':self.y0*self.scale
                      , 'xc':self.xc*self.scale, 'yc':self.yc*self.scale
                     , 'w':self.w*self.scale, 'h':self.h*self.scale
                      , 'porosity':porosity, 'excessPerimeter':excessPerimeter}
        
        

class xsSegmentSingle(xsSegment):
    '''collects data about single line XS segments'''
    
    def __init__(self, file:str, diag:int=0, acrit:int=2500, **kwargs):
        super().__init__(file, diag=diag, acrit=acrit, **kwargs)

    def xsSegment() -> None:
        '''im is imported image. 
        s is is the scaling of the stitched image compared to the raw images, e.g. 0.33 
        title is the title to put on the plot
        name is the name of the line, e.g. xs1
        acrit is the minimum segment size to be considered a cross-section
        '''
        self.segmenter = segmenter(self.im, acrit=self.acrit, diag=max(0, self.diag-1))
        if not self.segmenter.success:
            return
        self.singleMeasure()
    
    def measure(self) -> None:
        '''import image, filter, and measure cross-section'''
        self.name = self.lineName('xs')
        if 'I_M' in self.file or 'I_PD' in self.file:
            self.im = vc.imcrop(self.im, 10)
        # label connected components
        self.title = os.path.basename(self.file)
        self.im = vm.normalize(self.im)
        self.xsSegment()

#--------


class xsSegmentTriple(xsSegment):
    '''colleges data about triple line XS segments'''
    
    def __init__(self, file:str, diag:int=0, acrit:int=2500, **kwargs):
        super().__init__(file, diag=diag, acrit=acrit, **kwargs)
    
    
    def display(self) -> None:
        if self.diag==0:
            return
        cm = self.labelsBW.copy()
        cm = cv.cvtColor(cm,cv.COLOR_GRAY2RGB)
        cv.drawContours(cm, [self.hull], -1, (110, 245, 209), 6)
        cv.drawContours(cm, self.cnt, -1, (186, 6, 162), 6)
        imshow(cm)
        if hasattr(self, 'title'):
            plt.title(self.title)
    
    def measure(self) -> None:
        '''measure cross-section of 3 lines'''
        spl = re.split('xs', os.path.basename(self.file))
        name = re.split('_', spl[0])[-1] + 'xs' + re.split('_', spl[1])[1]
        self.title = os.path.basename(self.file)
        self.im = vm.normalize(self.im)

        # segment components
        if 'LapRD_LapRD' in file:
            # use more aggressive segmentation to remove leaks
            self.segmented = segmenter(self.im, acrit=self.acrit, topthresh=75, diag=max(0, self.diag-1))
        else:
            self.segmented = segmenter(self.im, acrit=self.acrit, diag=max(0, self.diag-1))
        self.filterXSComponents()
        if not self.segmented.success:
            return 
        
        
        

class xsSegmentDisturb(xsSegment):
    '''for disturbed lines'''
    
    def __init__(self, file:str, diag:int=0, acrit:int=100, **kwargs):
        super().__init__(file, diag=diag, acrit=acrit, **kwargs)
        self.pv = printVals(os.path.dirname(file))
        self.lineName()
        self.measure()
    
    def measure(self) -> None:
        '''measure cross-section of single disturbed line'''
        
        self.scale = 1
        self.title = os.path.basename(self.file)
        self.im = vm.normalize(self.im)

        # segment components
        h,w,_ = self.im.shape
        hc = 150
        crop = {'y0':hc, 'yf':h-hc, 'x0':170, 'xf':300}
        self.im = vc.imcrop(self.im, crop)

        if 'water' in self.pv.ink.base:
            th = 140
        else:
            th = 80
        self.segmenter = segmenter(self.im, acrit=self.acrit, topthresh=th, diag=max(0, self.diag-1))
        if not self.segmenter.success:
            return
        self.singleMeasure()
        self.adjustForCrop(crop)
     
    
class xsSegmentSDT(xsSegment, segmentDisturb):
    '''for singledoubletriple lines'''
    
    def __init__(self, file:str, diag:int=0, acrit:int=100, **kwargs):
        self.numLines = int(re.split('_', os.path.basename(file))[1])
        super().__init__(file, diag=diag, acrit=acrit, **kwargs)
        
    def makeRelative(self) -> None:
        '''convert the coords to relative coordinates'''
        for s in ['c', '0']:
            xs = f'x{s}'
            ys = f'y{s}'
            if xs in self.stats and ys in self.stats:
                self.stats[xs], self.stats[ys] = self.nd.relativeCoords(self.stats[xs], self.stats[ys])
                self.units[xs] = 'mm'
                self.units[ys] = 'mm'
                
    def findIntendedCoords(self) -> None:
        '''find the intended x0,y0,xc,and yc of the assembly'''
        rc1 = self.pg.relativeCoords(self.tag)   # position of line 1 in mm, relative to the nozzle
        if self.numLines>1:
            lt = re.split('o', re.split('_', self.name)[1][2:])[0]
            if lt=='d' or lt==f'w{self.numLines}':
                # get the last line
                rc2 = self.pg.relativeCoords(self.tag, self.numLines)
            else:
                lnum = int(lt[1])
                rc2 = self.pg.relativeCoords(self.tag, lnum)

            
        print(rc1, rc2)
    
    def measure(self) -> None:
        '''measure cross-section of single disturbed line'''
        self.im = self.nd.subtractBackground(self.im, 10)   # remove the background and the nozzle
        self.getProgDims()
        rc = {'relative':True, 'w':250, 'h':250, 'wc':50, 'hc':170}
        self.crop = vm.relativeCrop(self.pg, self.nd, self.tag, rc)  # get crop position based on the actual line position
        self.crop = vc.convertCrop(self.im, self.crop)    # make sure everything is in bounds
        self.im = vc.imcrop(self.im, self.crop)
        if 'water' in self.pv.ink.base:
            th = 140
        else:
            th = 120
        self.segmenter = segmenter(self.im, acrit=self.acrit, topthresh=th, diag=max(0, self.diag-1))
        if not self.segmenter.success:
            return
        self.multiMeasure()
        self.adjustForCrop(self.crop)
        self.makeRelative()
        self.display()
        self.findIntendedCoords()
        
        
#------------------------------------------------------------------------------

        
                    
def xsDisturbMeasure(file:str, **kwargs) -> Tuple[dict, dict]:
    return xsSegmentDisturb(file, **kwargs).values()           


class xsDisturbMeasures(disturbMeasures):
    '''for a xsDisturb folder, measure the disturbed lines'''
    
    def __init__(self, folder:str, overwrite:bool=False, **kwargs) -> None:
        super().__init__(folder, overwrite=overwrite, **kwargs)

    def measureFolder(self) -> None:
        '''measure all cross-sections in the folder and export table'''
        if not 'disturbXS' in os.path.basename(folder):
            return
        self.fn = self.pfd.newFileName('xsMeasure', '.csv')
        if 'lines' in self.kwargs:
            lines = self.kwargs['lines']
        else:
            lines = [f'l{i}{s}{s2}' for i in range(4) for s in ['w', 'd'] for s2 in ['o']]
        self.measure(lines, xsDisturbMeasure)


    #-----------------------------------------------------------------
    # summaries

    def summarize(self, dire:str='+y', **kwargs) -> Tuple[dict,dict]:
        '''summarize xsical measurements in the folder and export table'''
        if not dire in os.path.basename(self.folder):
            return {}, {}
        
        r = self.summaryHeader('xs')
        if r==0:
            return self.summary, self.summaryUnits
        elif r==2:
            return 
        
        # find changes between observations
        aves = {}
        aveunits = {}
        for num in range(4):
            wodf = self.df[self.df.line.str.contains(f'l{num}wo')]
            dodf = self.df[self.df.line.str.contains(f'l{num}do')]
            if len(wodf)==1 and len(dodf)==1:
                wo = wodf.iloc[0]
                do = dodf.iloc[0]
                for s in ['aspect', 'yshift', 'xshift']:
                    try:
                        addValue(aves, aveunits, f'delta_{s}', difference(do, wo, s), self.du[s])
                    except ValueError:
                        pass
                for s in ['h', 'w']:
                    try:
                        addValue(aves, aveunits, f'delta_{s}_n', difference(do, wo, s)/wo[s], '')
                    except ValueError:
                        pass
                for s in ['xc']:
                    try:
                        addValue(aves, aveunits, f'delta_{s}_n', difference(do, wo, s)/self.pxpmm/self.pv.dEst, 'dEst')
                    except ValueError:
                        pass


        ucombine = aveunits 
        lists = aves
        self.convertValuesAndExport(ucombine, lists)
        return self.summary, self.summaryUnits
    
    
def xsDisturbMeasureSummarize(topFolder:str, dire:str='+y', overwrite:bool=False, **kwargs) -> Tuple[dict, dict]:
    return xsDisturbMeasures(topFolder, overwrite=overwrite, **kwargs).summarize(dire)

def xsDisturbSummariesRecursive(topFolder:str, dire:str, overwrite:bool=False, **kwargs) -> None:
    '''recursively go through folders'''
    s = summaries(topFolder, xsDisturbMeasureSummarize, overwrite=overwrite, dire=dire, **kwargs)
    return s.out, s.units
    
def xsDisturbSummaries(topFolder:str, exportFolder:str, overwrite:bool=False, **kwargs) -> None:
    '''measure all cross-sections in the folder and export table'''
    for dire in ['+y', '+z']:
        s = summaries(topFolder, xsDisturbMeasureSummarize, overwrite=overwrite, dire=dire, **kwargs)
        s.export(os.path.join(exportFolder, f'xs{dire}DisturbSummaries.csv'))
    
 
