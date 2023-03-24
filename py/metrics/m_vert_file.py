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




class vertSegment(metricSegment):
    '''collects data about vertical segments'''
    
    def __init__(self, file:str, diag:int=0, acrit:int=2500, **kwargs):
        super().__init__(file, diag=diag, acrit=acrit, **kwargs)
        
    def dims(self, numLines:int=1) -> None:
        '''get the dimensions of the segments'''
        print(numLines)
        df2 = self.segmenter.df.copy()
        
        if numLines==1:
            # take only components in line with the largest component
            filI = df2.a.idxmax() # index of filament label, largest remaining object
            component = df2.loc[filI]
            inline = df2.copy()
            inline = inline[(inline.x0>component['x0']-50)&(inline.x0<component['x0']+50)] # other objects inline with the biggest object
        else:
            # take all components
            inline = df2.copy()

        # measure overall dimensions
        inline['xf'] = inline.x0+inline.w
        inline['yf'] = inline.y0+inline.h
        self.x0 = int(inline.x0.min())  # unscaled
        self.y0 = int(inline.y0.min())  # unscaled
        self.xf = int(inline.xf.max()) # unscaled 
        self.yf = int(inline.yf.max()) # unscaled 
        self.w = self.xf - self.x0    # unscaled 
        if numLines==1:
            self.h = int(inline.h.sum())    # unscaled
        else:
            self.h = int(self.yf - self.y0)
        self.xc = int(sum(inline.a * inline.xc)/sum(inline.a))
        self.yc = int(sum(inline.a * inline.yc)/sum(inline.a))
        co = {'area':int(inline.a.sum())*self.scale**2
                     , 'x0':self.x0*self.scale, 'y0':self.y0*self.scale
                      , 'w':self.w*self.scale, 'h':self.h*self.scale
                      , 'xf':self.xf*self.scale, 'yf':self.yf*self.scale
                     , 'xc':self.xc*self.scale, 'yc':self.yc*self.scale, 'segments':len(inline)} 
        units = {'line':'', 'aspect':'h/w', 'area':'px'
                 ,'x0':'px', 'y0':'px', 'xf':'px', 'yf':'px', 'w':'px', 'h':'px'
                 , 'xc':'px', 'yc':'px', 'segments':''} # where pixels are in original scale
        
        # get combined mask of all objects in line
        self.componentMask = self.segmenter.reconstructMask(inline)
        # measure roughness, thickness, etc.
        componentMeasures, cmunits = self.measureComponent(horiz=False, reverse=True, diag=max(0,self.diag-1), combine=(numLines>1), emptiness=True, atot=inline.a.sum())
        if len(componentMeasures)==0:
            return
        
        # component measures and co are pre-scaled
        aspect = co['h']/componentMeasures['meanT'] # height/width
        ret = {**{'line':self.name, 'aspect':aspect}, **co}
        units = {**units}
        
        r = componentMeasures['meanT']/2
        ret['vest'] = calcVest(co['h'], r)
        units['vest'] = 'px^3'

        if numLines>1:
            ret['ldiff'] = self.getLDiff(horiz=False)/self.h
            units['ldiff'] = 'h'

        ret = {**ret, **componentMeasures}
        units = {**units, **cmunits}
        self.stats = {**self.stats, **ret}
        self.units = {**self.units, **units}
        
    def gaps(self, distancemm:float) -> None:
        if not hasattr(self, 'nd') or not hasattr(self, 'crop') or 'o' in self.name:
            return
        # get displacements
        disps = self.displacement('z', distancemm*self.nd.pxpmm)
        dispunits = dict([[ii, 'px'] for ii in disps])
        self.stats = {**self.stats, **disps}
        self.units = {**self.units, **dispunits}
        
   
    def display(self) -> None:
        '''display diagnostics'''
        if self.diag<=0:
            return
        self.segmenter.display()
        if hasattr(self, 'componentMask'):
            im2 = cv.cvtColor(self.componentMask,cv.COLOR_GRAY2RGB)
        else:
            im2 = self.im.copy()
        imgi = self.im.copy() 
        for im in [im2, imgi]:
            if hasattr(self, 'x0'):
                cv.rectangle(im, (self.x0,self.y0), (self.x0+self.w,self.y0+self.h), (0,0,255), 2)
                cv.circle(im, (self.xc, self.yc), 2, (0,0,255), 2)
            if hasattr(self, 'idealspx'):
                io = {}
                for s in ['x0', 'xf']:
                    io[s] = int(self.idealspx[s]/self.scale)
                cv.rectangle(im, (io['x0'],0), (io['xf'],600), (237, 227, 26), 2)   # bounding box of intended
            if hasattr(self, 'nozPx'):
                io = {}
                for s in ['x0', 'xf', 'y0', 'yf']:
                    io[s] = int(self.nozPx[s]/self.scale)
                cv.rectangle(im, (io['x0'],io['yf']), (io['xf'],io['y0']), (125, 125, 125), 3)   # bounding box of nozzle
        if hasattr(self, 'hull'):
            # show the roughness
            imshow(imgi, im2, self.roughnessIm(), self.statText(), title='vertFile')
        else:
            imshow(imgi, im2, self.statText(), title='vertFile')
        plt.title(os.path.basename(self.file))
        return 



    

class vertSegmentSingle(vertSegment, segmentSingle):
    '''for single vertical lines'''
    
    def __init__(self, file, progDims:pd.DataFrame, diag:int=0, acrit:int=2500, **kwargs):
        super().__init__(file, diag=diag, acrit=acrit, **kwargs)
        self.progDims = progDims
        self.measure()

    def measure(self) -> None:
        '''measure vertical lines'''
        self.name = int(self.lineName('vert'))
        self.maxlen = self.progDims[self.progDims.name==(f'vert{self.name}')].iloc[0]['l']
        self.maxlen = int(self.maxlen/self.scale)
        # label connected components

        for s in ['im', 'scale', 'maxlen']:
            if not hasattr(self, s):
                raise ValueError(f'{s} undefined for {self.file}')

        self.segmenter = segmenterSingle(self.im, acrit=self.acrit, diag=max(0,self.diag-1))
        if not self.segmenter.success:
            return 
 
        self.segmenter.eraseBorderComponents(10)
            # remove anything too close to the border
        if not self.segmenter.success:
            return
        
        self.dims(selectInline=True)
        self.display()


class vertSegmentDisturb(vertSegment, segmentDisturb):
    '''for disturbed lines'''
    
    def __init__(self, file:str, diag:int=0, acrit:int=2500, **kwargs):
        super().__init__(file, diag=diag, acrit=acrit, **kwargs)
        
    def prepareImage(self) -> None:
        '''clean and crop the image'''
        if 'water' in self.pv.ink.base:
            self.im = self.nd.subtractBackground(self.im, diag=self.diag-2)  # remove background and nozzle
            self.im = vm.removeBlack(self.im)   # remove bubbles and dirt from the image
            self.im = vm.removeChannel(self.im,0) # remove the blue channel
        if self.pv.ink.dye=='red':
            self.im = self.nd.maskNozzle(self.im)
            self.im = vm.removeChannel(self.im, 2)   # remove the red channel

        h,w,_ = self.im.shape
        self.maxlen = h
        self.scale = 1

        # segment components
        hc = 0
        if self.name[-1]=='o':
            # observing
            self.crop = {'y0':hc, 'yf':h-hc, 'x0':200, 'xf':self.nd.xL+20, 'w':w, 'h':h}
        else:
            # writing
            self.crop = {'y0':hc, 'yf':h-hc, 'x0':self.nd.xL-100, 'xf':self.nd.xR+100, 'w':w, 'h':h}
        self.im = vc.imcrop(self.im, self.crop)
    #     im = vm.removeDust(im)
        self.im = cv.cvtColor(self.im,cv.COLOR_BGR2GRAY)  # convert to grayscale
        self.im = vm.normalize(self.im)

        
    def measure(self) -> None:
        '''measure disturbed vertical lines'''
        self.nd.importNozzleDims()
        if not self.nd.nozDetected:
            raise ValueError(f'No nozzle dimensions in {nd.printFolder}')

        self.prepareImage()
        if 'water' in self.pv.ink.base:
            bt = 210
        else:
            bt = 90
            
        self.segmenter = segmenter(self.im, acrit=self.acrit, diag=max(0,self.diag-1), cutoffTop=0, topthresh=bt, removeBorder=False, nozData=self.nd, crops=self.crop)
        if not self.segmenter.success:
            return # nothing to measure here
        self.segmenter.eraseLeftRightBorder()
        if not self.segmenter.success:
            return  # nothing to measure here

        self.dims()
        self.adjustForCrop(self.crop)
        self.gaps(self.pv.dEst)
        self.display()
        
        
class vertSegmentSDT(vertSegment, segmentSDT):
    '''for singledoubletriple lines'''
    
    def __init__(self, file:str, diag:int=0, acrit:int=1000, **kwargs):
        super().__init__(file, diag=diag, acrit=acrit, **kwargs)

                
    def findIntendedCoords(self) -> None:
        '''find the intended x0,y0,xc,and yc of the assembly'''
        rc1, rc2, w1, w2, l = self.intendedRC()
        for j in [['dx', 'w']]:
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
        w = self.ideals['w']
        r = w/2
        self.ideals['area'] = l*w
        self.ideals['v'] = (l-w)*np.pi*r**2 + 4/3*np.pi*r**3
        self.ideals['h'] = l
        
    def findDisplacement(self) -> None:
        '''find the displacement of the center and dimensions relative to the intended dimensions'''
        for s,ival in {'xc':'xc', 'x0':'x0', 'xf':'xf', 'dxprint':'xc'}.items():
            # ratio of size to intended size
            if s in self.stats:
                self.stats[s] = ((self.stats[s]-self.ideals[ival])/self.pv.dEst)
                self.units[s] = 'dEst'
        for s in ['dxprint', 'dx0', 'dxf', 'space_a', 'space_at']:
            if s in self.stats:
                self.stats[s] = self.stats[s]/self.pv.dEst
                self.units[s] = 'dEst'
        for s,ival in {'w':'w', 'h':'h', 'meanT':'w', 'vest':'v', 'vintegral':'v'}.items():
            # ratio of size to intended size
            if s in self.stats:
                self.stats[s] = (self.stats[s]/self.ideals[ival])
                self.units[s] = 'intended'
        for s in ['area', 'vleak', 'aspect']:
            if s in self.stats:
                self.stats.pop(s)
                
        # remove length measurements for mid-print measurements 
        if not 'o' in self.tag:
            for s in ['h', 'vest', 'vintegral', 'w', 'xf', 'meanT', 'stdevT', 'minmaxT']:
                if s in self.stats:
                    self.stats.pop(s)

    
    def measure(self) -> None:
        '''measure vertical SDT line'''
        self.getProgRow()
        self.nd.maskPadRight=30  # there is nothing on the right side of the nozzle, so just mask generously
        h,w,_ = self.im.shape
        self.maxlen = h
        self.dilation = 3
        self.nd.maskPad = 0
        self.im = self.nd.subtractBackground(self.im, self.dilation)   # remove the background and the nozzle
        rc = {'relative':True, 'w':250, 'h':800, 'wc':80, 'hc':400}
        self.crop = vc.relativeCrop(self.pg, self.nd, self.tag, rc)  # get crop position based on the actual line position
        self.crop = vc.convertCrop(self.im, self.crop)    # make sure everything is in bounds
        self.im = vc.imcrop(self.im, self.crop)
        if 'water' in self.pv.ink.base:
            th = 140
        else:
            th = 180
        self.segmenter = segmenter(self.im, acrit=self.acrit, topthresh=th, diag=max(0, self.diag-1)
                                   , removeBorder=False, eraseMaskSpill=True, dilation=self.dilation
                                   , nozData=self.nd, crops=self.crop, adaptive=True, addNozzle=False
                                   , addNozzleBottom=True, fillTop=True, openBottom=True)
        # self.segmenter = segmenter(self.im, acrit=self.acrit, topthresh=200, diag=max(0, self.diag-1)
        #                            , removeBorder=False, eraseMaskSpill=True, dilation=self.dilation
        #                            , nozData=self.nd, crops=self.crop, adaptive=False, addNozzle=False
        #                            , addNozzleBottom=True, fillTop=True)
        self.segmenter.eraseFullWidthComponents() # remove glue or air
        # self.segmenter.emptyVertSpaces()
        if not self.segmenter.success:
            if self.diag>0:
                logging.warning(f'Segmenter failed on {self.file}')
            self.display()
            return
        if self.tag[2]=='d':
            ln = self.lnum-1
        else:
            ln = self.lnum
        self.dims(numLines=ln)
        for s in ['y0', 'yc', 'yf']:
            self.stats.pop(s)
            self.units.pop(s)
        self.gaps(self.pv.dEst)
        if len(self.stats)==1:
            if self.diag>0:
                logging.warning(f'Measurement failed on {self.file}')
            self.display()
            return
        self.adjustForCrop(self.stats, self.crop)  # px, full image
        self.stats, self.units = self.makeMM(self.stats, self.units)
        self.makeRelative()           # mm, relative to nozzle
        
        self.findIntendedCoords()
        if self.diag>0:
            self.findIntendedPx()
            self.findNozzlePx()
        self.findDisplacement()  # dEst, relative to intended coords
        self.display()

        
    
def vertDisturbMeasure(file:str, **kwargs) -> Tuple[dict, dict]:
    return vertSegmentDisturb(file, **kwargs).values()


def vertSDTMeasure(file:str, **kwargs) -> Tuple[dict, dict]:
    return vertSegmentSDT(file, **kwargs).values() 

def vertSDTTestFile(fstr:str, fistr:str, **kwargs) -> None:
    '''test a single file and print diagnostics'''
    testFile(fstr, fistr, vertSegmentSDT, ['w', 'h', 'xc', 'yc'], **kwargs)
    
def addVertToTestFile(csvstr:str, fstr:str, fistr:str, **kwargs) -> None:
    cdir = os.path.dirname(os.path.abspath(os.path.join('..')))
    file = os.path.join(cdir, 'tests', f'test_{csvstr}.csv')
    addToTestFile(file, fstr, fistr, vertSegmentSDT, ['emptiness', 'x0', 'segments'], **kwargs)

