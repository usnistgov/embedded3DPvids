#!/usr/bin/env python
'''Functions for collecting data from stills of single line horiz'''

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


class horizSegment(metricSegment):
    '''collects data about horizontal segments in a single horizontal line'''
    
    def __init__(self, file:str, diag:int=0, acrit:int=2500, **kwargs):
        super().__init__(file, diag=diag, acrit=acrit, **kwargs)
    

    def markHorizOnIm(self, row:pd.Series) -> np.array:
        '''mark horizontal element on the image'''
        self.annotated = cv.rectangle(self.annotated, (int(row['x0']),int(row['y0'])), (int(row['x0']+row['w']),int(row['y0']+row['h'])), (0,0,255), 2)
        self.annotated = cv.circle(self.annotated, (int(row['xc']), int(row['yc'])), 3, (0,0,255), 3)
    
    def display(self) -> None:
        self.annotated = self.segmenter.labelsBW.copy()
        self.annotated = cv.cvtColor(self.annotated,cv.COLOR_GRAY2RGB)
        if self.diag==0:
            return
        for i,row in self.segmenter.df.iterrows():
            self.markHorizOnIm(row)
        imshow(self.im, self.annotated, self.statText(cols=2))
        
    def selectLine(self, df:pd.DataFrame, maxlen:float, j:int) -> None:
        '''narrow down the selection to just the segments in the dataframe'''
        self.segmenter.selectComponents(df.w==df.w)
        self.maxlen = maxlen
        self.name = j
        self.title = f'{os.path.basename(self.file)}: horiz{j}'
        self.dims()
        self.display()

    def dims(self) -> None:
        '''measure one horizontal line. df has been filtered down from the full dataframe to only include one row 
        labeled is an image from connected component labeling
        im2 is the original image
        s is is the scaling of the stitched image compared to the raw images, e.g. 0.33
        j is the line number
        maxPossibleLen is the longest length in mm that should have been extruded
        '''
        
        df = self.segmenter.df
        maxlen0 = df.w.max()   # length of the longest segment
        totlen = df.w.sum()    # total length of all segments
        maxarea = df.a.max()   # area of largest segment
        totarea = df.a.sum()   # total area of all segments
        xc = int(sum(df.a * df.xc)/sum(df.a))   # weighted center of mass
        yc = int(sum(df.a * df.yc)/sum(df.a))

        co = {'line':self.name, 'segments':len(df)
            , 'maxlen':maxlen0*self.scale, 'totlen':df.w.sum()*self.scale
            , 'maxarea':df.a.max()*self.scale**2, 'totarea':int(df.a.sum())*self.scale**2
              , 'xc':xc*self.scale, 'yc':yc*self.scale
            }  
        counits = {'line':'', 'segments':''
            , 'maxlen':'px', 'totlen':'px'
            , 'maxarea':'px^2', 'totarea':'px^2'
                   , 'xc':'px', 'yc':'px'
            }  
        longest = df[df.w==maxlen0] # measurements of longest component
        self.componentMask = self.segmenter.reconstructMask(longest)   # get the image of just this component
        componentMeasures, cmunits = self.measureComponent(True, reverse=(self.name==1), diag=max(0,self.diag-1))
        # component measures and co are pre-scaled
        if 'totlen' in co and 'meanT' in componentMeasures:
            r = componentMeasures['meanT']/2
            h = co['totlen']
            aspect = h/(2*r) # height/width
            vest = calcVest(h,r)
        else:
            aspect = 0
            vest = 0
        units = {'line':'', 'aspect':'h/w'} # where pixels are in original scale
        ret = {**{'line':self.name, 'aspect':aspect}, **co, **{'vest':vest}, **componentMeasures}
        units = {**units, **counits, **{'vest':'px^3'}, **cmunits}
        self.stats = {**self.stats, **ret}
        self.units = {**self.units, **units}
        
    def gaps(self, distancemm:float) -> None:
        if not hasattr(self, 'nd') or not hasattr(self, 'crop') or 'o' in self.name:
            return
        # get displacements
        disps = self.displacement('y', distancemm*self.nd.pxpmm, diag=self.diag-1)
        dispunits = dict([[ii, 'px'] for ii in disps])
        self.stats = {**self.stats, **disps}
        self.units = {**self.units, **dispunits}


#----------------------------
# single line

class horizSingleMeasures:
    '''for measuring all 3 lines from a stitched singleLine image'''
    
    def __init__(self, file:str, diag:int=0, acrit:int=1000, overwrite:bool=False, **kwargs):
        self.segmentOrig = horizSegment(file, diag=diag, acrit=acrit, **kwargs)
        self.overwrite = overwrite
        self.folder = os.path.dirname(file)
        self.pfd = fh.printFileDict(self.folder)
        self.progDims = getProgDims(self.folder)

    def splitLines(self, margin:float=80, **kwargs) -> list:
        '''split the table of segments into the three horizontal lines. 
        margin is the max allowable vertical distance in px from the line location to be considered part of the line'''

        linelocs = [275, 514, 756] # expected positions of lines
        ylocs = [-1000,-1000,-1000] # actual positions of lines
        df0 = self.segmented.df
        # get position of largest segment
        if len(df0)==0:
            return df0
        largesty = float(df0[df0.a==df0.a.max()]['yc'])

        # take segments that are far away
        df = df0[(df0.yc<largesty-100)|(df0.yc>largesty+100)]
        if len(df)>0:
            secondy = float(df[df.a==df.a.max()]['yc'])
            df = df[(df.yc<secondy-100)|(df.yc>secondy+100)]
            if len(df)>0:
                thirdy = float(df[df.a==df.a.max()]['yc'])
                ylocs = ([largesty, secondy, thirdy])
                ylocs.sort()
            else:
                # only 2 lines
                largestI = closestIndex(largesty, linelocs)
                secondI = closestIndex(secondy, linelocs)
                if secondI==largestI:
                    if secondI==2:
                        if secondy>largesty:
                            largestI = largestI-1
                        else:
                            secondI = secondI-1
                    elif secondI==0:
                        if secondy>largesty:
                            secondI = secondI+1
                        else:
                            largestI = largestI+1
                    else:
                        if secondy>largesty:
                            secondI = secondI+1
                        else:
                            secondI = secondI-1
                ylocs[largestI] = largesty
                ylocs[secondI] = secondy
        else:
            # everything is in this line
            largestI = closestIndex(largesty, linelocs)
            ylocs[largestI] = largesty

        if diag>1:
            logging.info(f'ylocs: {ylocs}')
        self.dflist = [df0[(df0.yc>yloc-margin)&(df0.yc<yloc+margin)] for yloc in ylocs]


    def measure(self) -> None:
        '''segment the image and take measurements
        progDims holds timing info about the lines
        s is is the scaling of the stitched image compared to the raw images, e.g. 0.33
        acrit is the minimum segment size in px to be considered part of a line
        satelliteCrit is the min size of segment, as a fraction of the largest segment, to be considered part of a line
        '''
        self.fn = self.pfd.newFileName(f'horizSummary', '.csv')
        if os.path.exists(self.fn) and not self.overwrite:
            return
        self.segmentOrig.im = vm.removeBorders(self.segmentOrig.im)
        self.segmenter = segmenterSingle(self.im, acrit=self.acrit, diag=max(0, self.diag-1), removeVert=True)
        self.segmenter.eraseSmallestComponents(**self.kwargs)
        self.splitLines(**kwargs)
        self.units = {}
        self.out = []
        for j,df in enumerate(self.dflist):
            if len(df)>0:
                maxlen = self.progDims[self.progDims.name==f'horiz{j}'].iloc[0]['l']  # length of programmed line
                segment = copy.deepcopy(self.segmentOrig)
                segment.selectLine(df, maxlen, j)
                r,cmu = segment.values()
                self.out.append(r)
                if len(cmu)>len(self.units):
                    self.units = cmu
        self.df = pd.DataFrame(self.out)
        plainExp(self.fn, self.df, self.units)



#----------------------------
# disturb

class horizSegmentDisturb(horizSegment, segmentDisturb):
    '''for disturbed horizontal lines'''
    
    def __init__(self, file:str, diag:int=0, acrit:int=2500, f:float=0.4, f2:float=0.3, **kwargs):
        self.f = f
        self.f2 = f2
        super().__init__(file, diag=diag, acrit=acrit, **kwargs)
        

    def removeThreads(self, diag:int=0) -> np.array:
        '''remove zigzag threads from bottom left and top right part of binary image'''
        f = self.f
        f2 = self.f2
        thresh = self.segmenter.labelsBW.copy()
        if diag>0:
            thresh0 = thresh.copy()
            thresh0 = cv.cvtColor(thresh0, cv.COLOR_GRAY2BGR)
        h0,w0 = thresh.shape
        left = thresh[:, :int(w0*f)]
        right0 = int(w0*(1-f))
        right = thresh[:, right0:]
        for i,im in enumerate([left, right]):
            contours = cv.findContours(im, 1, 2)
            if int(cv.__version__[0])>=4:
                contours = contours[0]
            else:
                contours = contours[1]
            contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True) # select the largest contour
            if len(contours)>0:
                x,y,w,h = cv.boundingRect(contours[0])
                if i==0:
                    # mask the top if the right edge is tall
                    if thresh[:y, int(w0*(1-f2)):].sum(axis=0).sum(axis=0)>0:
                        thresh[:y-10, :] = 0
                        if diag>0:
                            thresh0 = cv.rectangle(thresh0, (0,0), (w0,y-10), (255,0,0), 2)
                            thresh0 = cv.rectangle(thresh0, (x,y), (x+w,y+h), (255,0,0), 2)
                else:
                    # mask the bottom on the left side if the left edge is tall
                    if thresh[h+y+10:, :int(w0*f2)].sum(axis=0).sum(axis=0)>0:
                        thresh[h+y+10:, :int(w0*f2)] = 0
                        if diag>0:
                            x = x+right0
                            thresh0 = cv.rectangle(thresh0, (0,h+y+10), (int(w0*f2),h0), (0,0,255), 2)
                            thresh0 = cv.rectangle(thresh0, (x,y), (x+w,y+h), (0,0,255), 2)                    
        if diag>0:
            imshow(thresh0, thresh)
        self.segmenter.filled = thresh
        self.segmenter.getConnectedComponents()
        
    def prepareImage(self):
        '''clean and crop the image'''
        if self.pv.ink.dye=='blue':
            self.im = self.nd.subtractBackground(self.im, diag=self.diag-2)  # remove background and nozzle
            self.im = vm.removeBlack(self.im)   # remove bubbles and dirt from the image
            self.im = vm.removeChannel(self.im,0) # remove the blue channel
        elif self.pv.ink.dye=='red':
            self.im = self.nd.maskNozzle(self.im)
            self.im = vm.removeChannel(self.im, 2)   # remove the red channel

        h,w,_ = self.im.shape
        self.scale = 1
        self.maxlen = w

        # segment components
        hc = 0
        if self.name[-1]=='o':
            # observing
            self.crop = {'y0':int(h/2), 'yf':int(h*6/6), 'x0':hc, 'xf':w-hc, 'w':w, 'h':h}   # crop the left and right edge to remove zigs
        else:
            # writing
            self.crop = {'y0':int(h/6), 'yf':int(h*5/6), 'x0':hc, 'xf':w-hc, 'w':w, 'h':h}
        self.im = ic.imcrop(self.im, self.crop)
    #     im = vm.removeDust(im)
        self.im = vm.normalize(self.im)


    def measure(self) -> None:
        '''measure disturbed horizontal lines'''
        self.nd.importNozzleDims()
        if not self.nd.nozDetected:
            raise ValueError(f'No nozzle dimensions in {self.nd.printFolder}')

        self.prepareImage()
        if 'water' in self.pv.ink.base:
            bt = 200
        else:
            bt = 90
            
        self.segmenter = segmenter(self.im, acrit=self.acrit, diag=max(0,self.diag-1), cutoffTop=0, topthresh=bt, removeBorder=False, nozData=self.nd, crops=self.crop, eraseMaskSpill=True)
        if not self.segmenter.success:
            return
        self.segmenter.eraseFullWidthComponents()
        self.segmenter.eraseTopBottomBorder()
        self.removeThreads(diag=self.diag-1)
        if self.diag>1:
            self.segmenter.display()
        self.dims()
        self.adjustForCrop(self.crop)
        self.gaps(self.pv.dEst)
        self.display()
        
def horizDisturbMeasure(file:str, **kwargs) -> Tuple[dict, dict]:
    return horizSegmentDisturb(file, **kwargs).values()
