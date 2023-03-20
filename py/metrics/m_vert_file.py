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
        
    def dims(self) -> None:
        '''get the dimensions of the segments'''
        df2 = self.segmenter.df
        filI = df2.a.idxmax() # index of filament label, largest remaining object
        component = df2.loc[filI]
        inline = df2[(df2.x0>component['x0']-50)&(df2.x0<component['x0']+50)] # other objects inline with the biggest object

        # get combined mask of all objects in line
        self.componentMask = self.segmenter.reconstructMask(inline)

        self.x0 = int(inline.x0.min())  # unscaled
        self.y0 = int(inline.y0.min())  # unscaled
        self.w = int(inline.w.max())    # unscaled
        self.h = int(inline.h.sum())    # unscaled
        self.xc = int(sum(inline.a * inline.xc)/sum(inline.a))
        self.yc = int(sum(inline.a * inline.yc)/sum(inline.a))
        co = {'area':int(inline.a.sum())*self.scale**2
                     , 'x0':self.x0*self.scale, 'y0':self.y0*self.scale, 'w':self.w*self.scale, 'h':self.h*self.scale
                     , 'xc':self.xc*self.scale, 'yc':self.yc*self.scale, 'segments':len(inline)}    
        componentMeasures, cmunits = self.measureComponent(self.componentMask, horiz=False, reverse=True, diag=max(0,self.diag-1))
        if len(componentMeasures)==0:
            return

        # component measures and co are pre-scaled
        aspect = co['h']/componentMeasures['meanT'] # height/width
        r = componentMeasures['meanT']/2
        vest = calcVest(co['h'], r)
        units = {'line':'', 'aspect':'h/w', 'area':'px'
                 ,'x0':'px', 'y0':'px', 'w':'px', 'h':'px'
                 , 'xc':'px', 'yc':'px', 'segments':'', 'vest':'px^3'} # where pixels are in original scale
        ret = {**{'line':self.name, 'aspect':aspect}, **co, **{'vest':vest}, **componentMeasures}
        units = {**units, **cmunits}
        self.stats = {**self.stats, **ret}
        self.units = {**self.units, **units}
        
    def gaps(self, distancemm:float) -> None:
        if not hasattr(self, 'nd') or not hasattr(self, 'crop') or 'o' in self.name:
            return
        # get displacements
        disps = self.displacement(self.componentMask, 'z', distancemm*self.nd.pxpmm)
        dispunits = dict([[ii, 'px'] for ii in disps])
        self.stats = {**self.stats, **disps}
        self.units = {**self.units, **dispunits}
        
   
    def display(self) -> None:
        '''display diagnostics'''
        if self.diag>0:
            self.segmenter.display()
            im2 = cv.cvtColor(self.componentMask,cv.COLOR_GRAY2RGB)
            im2 = cv.rectangle(im2, (self.x0,self.y0), (self.x0+self.w,self.y0+self.h), (0,0,255), 2)
            im2 = cv.circle(im2, (self.xc, self.yc), 2, (0,0,255), 2)
            imshow(self.im, im2, self.statText())
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
        
        self.dims()
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
        
        
class vertSegmentSDT(vertSegment, segmentDisturb):
    '''for singledoubletriple lines'''
    
    def __init__(self, file:str, diag:int=0, acrit:int=2500, **kwargs):
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

        
    
def vertDisturbMeasure(file:str, **kwargs) -> Tuple[dict, dict]:
    return vertSegmentDisturb(file, **kwargs).values()

