#!/usr/bin/env python
'''Functions for collecting data from stills of single vertical lines'''

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

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
from pic_stitch_bas import stitchSorter
from file_handling import isSubFolder, fileScale
import im_crop as vc
import im_morph as vm
from tools.imshow import imshow
from tools.plainIm import *
from val_print import *
from vid_noz_detect import nozData
from metrics_tools import *
from im_segment import *

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

        
    
def vertDisturbMeasure(file:str, **kwargs) -> Tuple[dict, dict]:
    return vertSegmentDisturb(file, **kwargs).values()


#---------------------------------------------------------------------------------------------------
    
    
class vertDisturbMeasures(disturbMeasures):
    '''for a vertDisturb measure, measure the disturbed lines'''
    
    def __init__(self, folder:str, overwrite:bool=False, **kwargs) -> None:
        super().__init__(folder, overwrite=overwrite, **kwargs)
    
    def measureFolder(self) -> None:
        '''measure all cross-sections in the folder and export table'''
        if not 'disturbVert' in os.path.basename(self.folder):
            return 1
        self.fn = self.pfd.newFileName('vertMeasure', '.csv')

        if 'lines' in self.kwargs:
            lines = self.kwargs['lines']
        else:
            lines = [f'l{i}{s}{s2}' for i in range(4) for s in ['w', 'd'] for s2 in ['', 'o']]
        self.measure(lines, vertDisturbMeasure)


    def summarize(self) -> Tuple[dict,dict]:
        '''summarize vertical measurements in the folder and export table'''
        errorRet = {},{}
        if not 'disturbVert' in os.path.basename(self.folder):
            return errorRet

        r = self.summaryHeader('vert')
        if r==0:
            return self.summary, self.summaryUnits
        elif r==2:
            return errorRet

        # find changes between observations
        aves = {}
        aveunits = {}
        for num in range(4):
            if num in [0,2]:
                ltype = 'bot'
            else:
                ltype = 'top'
            wodf = self.df[self.df.line==f'V_l{num}wo']
            dodf = self.df[self.df.line==f'V_l{num}do']
            if len(wodf)==1 and len(dodf)==1:
                wo = wodf.iloc[0]
                do = dodf.iloc[0]

                for s in ['segments', 'roughness']:
                    try:
                        addValue(aves, aveunits, f'{ltype}_delta_{s}', difference(do, wo, s), self.du[s])
                    except ValueError:
                        pass
                for s in ['h', 'meanT']:
                    try:
                        addValue(aves, aveunits, f'{ltype}_delta_{s}_n', difference(do, wo, s)/wo[s], '')
                    except ValueError:
                        pass
                for s in ['xc']:
                    try:
                        addValue(aves, aveunits, f'{ltype}_delta_{s}_n', difference(do, wo, s)/self.pxpmm/self.pv.dEst, 'dEst')
                    except ValueError:
                        pass

        # find displacements
        disps = {}
        dispunits = {}
        dlist = ['dxprint', 'dxf', 'space_at', 'space_a']
        for num in range(4):
            wdf = self.df[self.df.line==f'V_l{num}w']
            ddf = self.df[self.df.line==f'V_l{num}d']
            if num in [0,2]:
                ltype = 'bot'
            else:
                ltype = 'top'
            for s in dlist:
                for vdf in [wdf,ddf]:
                    if len(vdf)>0:
                        v = vdf.iloc[0]
                        if hasattr(v, s):
                            sii = str(v.line)[-1]
                            si = f'{sii}_{s}'
                            sifull = f'{ltype}_{si}'
                            if si not in ['w_dxf', 'w_space_a', 'w_space_at']:
                                val = v[s]/self.pxpmm/self.pv.dEst
                                addValue(disps, dispunits, sifull, val, 'dEst')

        ucombine = {**aveunits, **dispunits} 
        lists = {**aves, **disps}
        self.convertValuesAndExport(ucombine, lists)
        return self.summary, self.summaryUnits
    
def vertDisturbMeasureSummarize(topFolder:str, overwrite:bool=False, **kwargs) -> Tuple[dict, dict]:
    return vertDisturbMeasures(topFolder, overwrite=overwrite, **kwargs).summarize()

def vertDisturbSummariesRecursive(topFolder:str, overwrite:bool=False, **kwargs) -> None:
    '''recursively go through folders'''
    s = summaries(topFolder, vertDisturbMeasureSummarize, overwrite=overwrite, **kwargs)
    return s.out, s.units
    
def vertDisturbSummaries(topFolder:str, exportFolder:str, overwrite:bool=False, **kwargs) -> None:
    '''measure all cross-sections in the folder and export table'''
    s = summaries(topFolder, vertDisturbMeasureSummarize, overwrite=overwrite, **kwargs)
    s.export(os.path.join(exportFolder, 'vertDisturbSummaries.csv'))
    