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
sys.path.append(os.path.dirname(currentdir))
import im.crop as vc
import im.morph as vm
from im.segment import *
from im.imshow import imshow
from tools.plainIm import *
from val.v_print import *
from vid.noz_detect import nozData
from m_tools import *
from m_file import *

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
        if self.diag<=0:
            return
        # show the image with annotated dimensions
        self.segmenter.display()
        if hasattr(self.segmenter, 'labelsBW'):
            im2 = cv.cvtColor(self.segmenter.labelsBW,cv.COLOR_GRAY2RGB)
        else:
            im2 = self.im.copy()
        imgi = self.im.copy()
        if hasattr(self, 'x0'):
            cv.rectangle(imgi, (self.x0,self.y0), (self.x0+self.w,self.y0+self.h), (0,0,255), 1)   # bounding box
            cv.circle(imgi, (int(self.xc), int(self.yc)), 2, (0,0,255), 2)     # centroid
        # cv.circle(imgi, (self.x0+int(self.w/2),self.y0+int(self.h/2)), 2, (0,255,255), 2) # center of bounding box
        if hasattr(self, 'idealspx'):
            io = {}
            for s in ['x0', 'xf', 'y0', 'yf']:
                io[s] = int(self.idealspx[s]/self.scale)
            cv.rectangle(imgi, (io['x0'],io['yf']), (io['xf'],io['y0']), (237, 227, 26), 1)   # bounding box of intended
        if hasattr(self, 'nozPx'):
            io = {}
            for s in ['x0', 'xf', 'y0', 'yf']:
                io[s] = int(self.nozPx[s]/self.scale)
            cv.rectangle(imgi, (io['x0'],io['yf']), (io['xf'],io['y0']), (0,0,0), 1)   # bounding box of nozzle
        if self.diag>1 and hasattr(self, 'componentMask'):
            # show the roughness as well
            cm = self.componentMask.copy()
            cm = cv.cvtColor(cm,cv.COLOR_GRAY2RGB)
            if hasattr(self, 'hull'):
                cv.drawContours(cm, [self.hull], -1, (110, 245, 209), 6)
            if hasattr(self, 'hull2'):
                cv.drawContours(cm, [self.hull2], -1, (252, 223, 3), 6)
            if hasattr(self, 'cnt'):
                cv.drawContours(cm, self.cnt, -1, (186, 6, 162), 6)
            imshow(imgi, im2, cm, self.statText())
        else:
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
        if len(contours)==0:
            return
        if len(contours[0])==0:
            return
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

        self.x0,self.y0,self.w,self.h = cv.boundingRect(self.hull)   # x0,y0 is top left
        aspect = self.h/self.w

        M = cv.moments(self.cnt)
        if M['m00']==0:
            self.xc=0
            self.yc=0
        else:
            self.xc = int(M['m10']/M['m00'])
            self.yc = int(M['m01']/M['m00'])
        boxcx = self.x0+self.w/2 # x center of bounding box
        boxcy = self.y0+self.h/2 # y center of bounding box
        xshift = (self.xc-boxcx)/self.w
        yshift = (self.yc-boxcy)/self.h

        self.units = {'line':'', 'segments':'', 
                      'aspect':'h/w', 'xshift':'w', 'yshift':'h', 'area':'px'
                      , 'x0':'px', 'y0':'px'
                      , 'xf':'px', 'yf':'px'
                        , 'xc':'px', 'yc':'px'
                         , 'w':'px', 'h':'px'
                      , 'porosity':'', 'excessPerimeter':''} # where pixels are in original scale
        self.stats = {'line':self.name, 'segments':len(self.segmenter.df)
                      , 'aspect':aspect, 'xshift':xshift, 'yshift':yshift, 'area':filledArea*self.scale**2
                      , 'x0':self.x0*self.scale, 'y0':self.y0*self.scale
                      , 'xf':self.x0*self.scale+self.w*self.scale, 'yf':self.y0*self.scale+self.h*self.scale
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
        self.adjustForCrop(self.stats, crop)
     
    
class xsSegmentSDT(xsSegment, segmentDisturb):
    '''for singledoubletriple lines'''
    
    def __init__(self, file:str, diag:int=0, acrit:int=100, **kwargs):
        self.numLines = int(re.split('_', os.path.basename(file))[1])
        super().__init__(file, diag=diag, acrit=acrit, **kwargs)
        
                
    def findIntendedCoords(self) -> None:
        '''find the intended x0,y0,xc,and yc of the assembly. assume that the center of the first filament should be at the center of the nozzle tip'''
        rc1, rc2, w1, w2, _ = self.intendedRC(fixList=['y','z'])
        for j in [['dx', 'w'], ['dy', 'h']]:
            coord = j[0][1]
            right = (rc2[j[0]]+w2/2)      # the left edge should be 1/2 diameter to the left of the first line center
            left = (rc1[j[0]]-w1/2)       # the right edge should be 1/2 diameter to the right of the last line center
            if coord=='y':
                y0 = left
                left = right
                right = y0
            self.ideals[f'{coord}0'] = left
            self.ideals[f'{coord}f'] = right
            self.ideals[j[1]] = abs(right - left)     # get the ideal width
            self.ideals[f'{coord}c'] = (right+left)/2  # get the ideal center
        self.ideals['area'] = self.progRows.a.sum()
        
    def findDisplacement(self) -> None:
        '''find the displacement of the center and dimensions relative to the intended dimensions'''
        for s in ['xc', 'yc', 'x0', 'y0', 'xf', 'yf']:
            # shift in position relative to intended, normalized by intended diameter
            if not s in self.ideals:
                raise ValueError(f'Missing value in ideals: {s}, {self.ideals}')
            if not s in self.stats:
                raise ValueError(f'Missing value in stats: {s}, {self.stats}')
            self.stats[s] = (self.stats[s]-self.ideals[s])/self.pv.dEst
            self.units[s] = 'dEst'
        for s in ['w', 'h']:
            # ratio of size to intended size
            self.stats[s] = (self.stats[s]/self.pv.pxpmm/self.ideals[s])
            self.units[s] = 'intended'
        self.stats['area'] = self.stats['area']/self.pv.pxpmm**2/self.ideals['area']
        self.units['area'] = 'intended'
        self.stats['aspectI'] = self.stats['h']/self.stats['w']   # ratio of aspect ratio to intended aspect ratio
        self.units['aspectI'] = 'intended'

    
    def measure(self) -> None:
        '''measure cross-section of single disturbed line'''
        # imshow(self.im)
        dilation = 4
        self.im = self.nd.subtractBackground(self.im, dilation)   # remove the background and the nozzle
        self.getProgDims()
        self.getProgRow()
        if '_1_' in os.path.basename(self.folder):
            rc = {'relative':True, 'w':200, 'h':250, 'wc':100, 'hc':180}
        else:
            if 'o' in self.tag:
                rc = {'relative':True, 'w':300, 'h':500, 'wc':100, 'hc':400}
            else:
                rc = {'relative':True, 'w':300, 'h':350, 'wc':100, 'hc':200}
        self.crop = vc.relativeCrop(self.pg, self.nd, self.tag, rc, fixList=['y', 'z'])  # get crop position based on the actual line position
        self.crop = vc.convertCrop(self.im, self.crop)    # make sure everything is in bounds
        if self.diag>0:
            self.findNozzlePx()

        self.im = vc.imcrop(self.im, self.crop)
        if 'water' in self.pv.ink.base:
            th = 140
        else:
            th = 110
        self.segmenter = segmenter(self.im, acrit=self.acrit, topthresh=th, nozData=self.nd, crops=self.crop, dilation=dilation, diag=max(0, self.diag-1))
        self.segmenter.eraseBorderComponents(10)
        if not self.segmenter.success:
            if self.diag>0:
                logging.warning(f'Segmenter failed on {self.file}')
            self.display()
            return
        self.multiMeasure()      # px, cropped
        if len(self.stats)==1:
            if self.diag>0:
                logging.warning(f'Measurement failed on {self.file}')
            self.display()
            return
        self.adjustForCrop(self.stats, self.crop)   # px, full image
        self.makeRelative()     # mm, relative to nozzle
        self.findIntendedCoords() 
        if self.diag>0:
            self.findIntendedPx()
        self.findDisplacement()  # dEst, relative to intended coords
        self.renameY()
        self.display()
        

        
        
        
#------------------------------------------------------------------------------

        
                    
def xsDisturbMeasure(file:str, **kwargs) -> Tuple[dict, dict]:
    return xsSegmentDisturb(file, **kwargs).values()  

def xsSDTMeasure(file:str, **kwargs) -> Tuple[dict, dict]:
    return xsSegmentSDT(file, **kwargs).values() 

def xsSDTTestFile(fstr:str, fistr:str, **kwargs) -> None:
    '''test a single file and print diagnostics'''
    testFile(fstr, fistr, xsSegmentSDT, ['w', 'h', 'xc', 'yc'], **kwargs)