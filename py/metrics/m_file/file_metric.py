#!/usr/bin/env python
'''Functions for collecting measurements from a single image'''

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
import time

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
sys.path.append(os.path.dirname(os.path.dirname(currentdir)))
import file.file_handling as fh
import im.crop as vc
from im.segment import *
from im.imshow import imshow
import im.contour as co
from tools.plainIm import *
from tools.timeCounter import *
from tools.config import cfg
from val.v_print import printVals
from progDim.prog_dim import getProgDims, getProgDimsPV
from vid.noz_detect import nozData
from m_tools import *
from file_unit import *
from file_ML import *
from crop_locs import *


# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 4)
pd.set_option('display.max_rows', 500)


#----------------------------------------------
    
def fileMetricFromTag(func, folder:str, tag:str, **kwargs):
    '''get the filehorizSDT from a string that is in the file name'''
    pfd = fh.printFileDict(folder)
    pfd.findVstill()
    i = 0
    while i<len(pfd.vstill):
        if tag in os.path.basename(pfd.vstill[i]):
            fhs = func(pfd.vstill[i], **kwargs)
            if len(tag)==6:
                return fhs
        i = i+1
    return fhs   

class fileMetric(timeObject):
    '''collects data about fluid segments in an image'''
    
    def __init__(self, file:str, diag:int=0, acrit:int=2500, exportDiag:int=2, normalize:bool=True, **kwargs):
        self.file = file
        self.folder = os.path.dirname(self.file)
        if not os.path.exists(self.file):
            raise ValueError(f'File {self.file} does not exist')
        self.acrit = acrit
        self.diag = diag
        self.exportDiag = exportDiag
        self.normalize = normalize
        self.hasIm = False
        self.stats = {'line':'', 'usedML':False}
        self.units = {'line':'', 'usedML':''}
        
    def __getattr__(self, s):
        if s=='im':
            self.importIm()
            return self.im
        else:
            raise AttributeError
        
    def importIm(self):
        '''import the image and determine if it was pre-scaled'''
        self.scale = 1/fh.fileScale(self.file)
        self.im = cv.imread(self.file)
        self.hasIm = True
        
    def openInPaint(self):
        '''open the image in paint'''
        openInPaint(self.file)
        
    def imDims(self):
        '''image dimensions'''
        if not self.hasIm:
            h = 590
            w = 790
        else:
            h,w = self.im.shape[:2]
        return h,w
        
    def values(self) -> Tuple[dict,dict]:
        '''measured parameters and units'''
        return self.stats, self.units
    
    def dropVariables(self, l:list):
        '''get rid of the variables listed'''
        for s in l:
            if s in self.stats:
                self.stats.pop(s)
                self.units.pop(s)
    
    def statText(self, cols:int=1) -> str:
        '''return measurements as readable text'''
        out = ''
        col = 0
        for key,val in self.stats.items():
            out = out + '{:13s}'.format(key)
            if type(val) is str:
                out = out + '{:8s}'.format(val)
            else:
                out = out + "{:8.2f}".format(val)
            out = out + '   {:5s}'.format(self.units[key])
            col = col+1
            if col==cols:
                col = 0
                out = out + '\n'
            else:
                out = out + '   '
        return out 
    
    def addToTestFile(self, csv:str, slist:list) -> None:
        '''add the values to the csv for unit testing'''
        fstr = self.folder.replace(cfg.path.server, '')[1:]
        fistr = os.path.basename(self.file)
        df, _ = plainIm(csv, ic=None)
        l = {'folder':fstr, 'file':fistr}
        for s in slist:
            if s in self.stats:
                l[s] = self.stats[s]
            else:
                l[s] = -1
        
        if len(df)>0 and fistr in df.file:
            for key,val in l.items:
                df.loc[df.file==fistr, key] = val
        else:
            if len(df)>0:
                df = pd.concat([df, pd.DataFrame([l])])
            else:
                df = pd.DataFrame([l])
        plainExp(csv, df, {}, index=False)
    
    def getProgDims(self):
        '''initialize the progDims object'''
        if not hasattr(self, 'pg'):
            self.pg  = getProgDimsPV(self.pv)
            self.pg.importProgDims()
            if self.pg.progDims.a.sum()==0:
                logging.warning(f'Empty area in {self.folder}. Redoing progDims')
                self.pg.exportAll(overwrite=True)
    
    #------------------------------           
    
    def importCrop(self):
        '''import the crop dimensions from the file'''
        if hasattr(self, 'crop'):
            return
        if not hasattr(self, 'cl'):
            self.getCropLocs()
        self.crop = self.cl.getCrop(self.file)
                
    def getCropLocs(self):
        '''get the crop locations'''
        self.cl = cropLocs(self.folder, pfd=self.pfd)
        
    def exportCropDims(self, **kwargs):
        '''calculate crop dimensions and export'''
        self.initialize()
        self.getCrop(**kwargs)
        
    def makeCrop(self, rc:dict, export:bool=True, overwrite:bool=False) -> None:
        '''get the crop dimensions and export'''
        self.importCrop()
        if len(self.crop)==0 or overwrite:
            # generate a new crop value
            h,w = self.imDims()
            self.crop = vc.relativeCrop(self.pg, self.nd, self.tag, rc)  # get crop position based on the actual line position
            self.crop = vc.convertCropHW(h,w, self.crop)    # make sure everything is in bounds
            self.cl.changeCrop(self.file, self.crop)
            if export:
                self.cl.export()
                
    #------------------------------
                
    def subFN(self, subFolder:str, title:str) -> str:
        '''get the filename of a file that goes into a subfolder'''
        cropfolder = os.path.join(self.folder, subFolder)
        if not os.path.exists(cropfolder):
            os.mkdir(cropfolder)
        fnorig = os.path.join(cropfolder, self.title.replace('vstill', title))
        return fnorig
        
    def exportImage(self, att:str, subFolder:str, title:str, overwrite:bool=False, diag:int=2, **kwargs) -> None:
        '''export an image stored as attribute att to a subfolder subFolder with the new title title'''
        if not hasattr(self, att):
            raise ValueError(f'No {att} found for {self.file}')
        fnorig = self.subFN(subFolder, title)
        if not os.path.exists(fnorig) or overwrite:
            im = getattr(self, att)
            out = cv.imwrite(fnorig, im)
            if diag>1 and self.exportDiag>1:
                if out:
                    logging.info(f'Exported {os.path.basename(fnorig)}')
                else:
                    folderExists = os.path.exists(os.path.dirname(fnorig))
                    writePermission = os.access(os.path.dirname(fnorig), os.W_OK)
                    logging.info(f'Failed to export {fnorig}. Folder exists: {folderExists}. Write permission: {writePermission}. Name length: {len(fnorig)}')
                    
            
    def generateIm0(self):
        '''generate the initial image'''
        self.im0 = self.im.copy()
        if hasattr(self, 'crop'):
            self.im0 = vc.imcrop(self.im0, self.crop) 
            
        
    def reconcileImportedSegment(self, func:str='vert', **kwargs):
        '''reconcile the difference between the imported segmentation files'''
        if func=='vert':
            f = segmentCombinerV
        elif func=='horiz':
            f = segmentCombinerH
            
        
        if hasattr(self, 'Usegment'):
            self.Usegment = self.nd.maskNozzle(self.Usegment, crops=self.crop, invert=True)
            if self.Usegment.sum().sum()==0:
                delattr(self, 'Usegment')
        else:
            self.generateSegment(overwrite=False)
            self.segmenter = segmenterDF(self.componentMask, acrit=self.acrit)
            return
        
        if hasattr(self, 'MLsegment') and self.useML:
            self.MLsegment = self.nd.maskNozzle(self.MLsegment, crops=self.crop, invert=True)
            if self.MLsegment.sum().sum()==0:
                delattr(self, 'MLsegment')
               
        if hasattr(self, 'MLsegment') and self.useML: 
            self.stats['usedML'] = True
            if hasattr(self, 'Usegment'):
                self.segmenter = f(self.MLsegment, self.Usegment, self.acrit, diag=self.diag-1, **kwargs).segmenter
            else:
                self.segmenter = segmenterDF(self.MLsegment, acrit=self.acrit)
        elif hasattr(self, 'Usegment'):
            self.segmenter = segmenterDF(self.Usegment, acrit=self.acrit)
        else:
            self.segmenter = segmenterFail()
        if self.segmenter.success:
            self.componentMask = self.segmenter.filled.copy()
            
    def importSegmentation(self) -> None:
        '''import any pre-segmented images'''
        self.importUsegment()
        if self.useML:
            self.importMLsegment()
        if self.diag>0:
            uu = hasattr(self, 'Usegment')
            mm = hasattr(self, 'MLsegment')
            print(f'Usegment: {uu}, MLsegment: {mm}')
        
    def importUsegment(self):
        '''import the image segmented using the unsupervised model'''
        s = self.segmentFN()
        if os.path.exists(s):
            self.Usegment = cv.imread(s, cv.IMREAD_GRAYSCALE)
            h,w = self.Usegment.shape
            if not h==self.crop['yf']-self.crop['y0'] or not w==self.crop['xf']-self.crop['x0']:
                raise ValueError(f'{self.file}: Usegment is wrong shape')
            self.importedImages = True
            
    def importMLsegment(self):
        '''import the image segmented using the ML model'''
        m = self.MLFN()
        if os.path.exists(m):
            self.MLsegment = cv.imread(m, cv.IMREAD_GRAYSCALE)
            h,w = self.MLsegment.shape
            if not h==self.crop['yf']-self.crop['y0'] or not w==self.crop['xf']-self.crop['x0']:
                raise ValueError(f'{self.file}: MLsegment is wrong shape')
            self.importedImages = True
        
        
    def exportCrop(self, **kwargs) -> None:
        '''add the cropped image to the folder for machine learning'''
        if not hasattr(self, 'im0'):
            self.generateIm0()
        self.exportImage('im0', 'crop', 'vcrop', **kwargs)
        
    def segmentFN(self) -> str:
        '''get the file name for the segmented image'''
        return self.subFN('Usegment', 'Usegment')
    
    def MLFN(self) -> str:
        '''get the file name for the ML segmented image'''
        fn = self.subFN('MLsegment2', 'MLsegment2')
        if not os.path.exists(fn):
            return self.subFN('MLsegment', 'MLsegment')
        else:
            return fn
            
    def exportSegment(self, **kwargs) -> None:
        '''add the cropped image to the folder for machine learning'''
        self.exportImage('componentMask', 'Usegment', 'Usegment', **kwargs)
    
    #------------------------------
        
    def disableFile(self):
        '''draw big black boxes over everything so you can't measure anything off this file'''
        self.importCrop()
        self.importSegmentation()
        if hasattr(self, 'Usegment'):
            self.Usegment[:,:] = 0
        else:
            self.Usegment = np.zeros((self.crop['yf']-self.crop['y0'], self.crop['xf']-self.crop['x0']), np.uint8)
        self.componentMask = self.Usegment
        self.exportImage('Usegment', 'Usegment', 'Usegment', overwrite=True)
        if hasattr(self, 'MLsegment'):
            self.MLsegment[:,:] = 0
        else:
            self.MLsegment = np.zeros((self.crop['yf']-self.crop['y0'], self.crop['xf']-self.crop['x0']), np.uint8)
        self.exportImage('MLsegment', 'MLsegment', 'MLsegment', overwrite=True)
        
    def acceptML(self):
        '''overwrite the Usegment file with the ML segmentation'''
        self.importCrop()
        self.importSegmentation()
        if not hasattr(self, 'MLsegment'):
            return
        self.Usegment = self.MLsegment
        self.exportImage('Usegment', 'Usegment', 'Usegment', overwrite=True)
      
    #------------------------------

    def renameY(self) -> None:
        '''rename y variables to clarify what is top and bottom'''
        
        replacement = {'yf':'yBot', 'y0':'yTop', 'x0':'xLeft', 'xf':'xRight'}
        for k, v in list(self.stats.items()):
            if k in self.stats and k in self.units:
                self.stats[replacement.get(k, k)] = self.stats.pop(k)
                self.units[replacement.get(k, k)] = self.units.pop(k)
            
    def makeRelative(self) -> None:
        '''convert the coords to relative coordinates'''
        for s in ['c', '0', 'f']:
            xs = f'x{s}'
            ys = f'y{s}'
            if xs in self.stats:
                x = self.stats[xs]
            else:
                x = 0
            if ys in self.stats:
                y = self.stats[ys]
            else:
                y = 0
            out = {}
            out[xs],out[ys] = self.nd.relativeCoords(x,y)
            for s2 in [xs, ys]:
                if s2 in self.stats:
                    self.stats[s2] = out[s2]
                    self.units[s2] = 'mm'
                
    def findNozzlePx(self) -> None:
        '''find the nozzle position on the cropped image'''
        self.nozPx = {}
        if hasattr(self.nd, 'xL'):
            self.nozPx['x0'] = self.nd.xL
            self.nozPx['xf'] = self.nd.xR
            self.nozPx['yf'] = self.nd.yB
            self.adjustForCrop(self.nozPx, self.crop, reverse=True)
            self.nozPx['y0'] = 0
        else:
            self.nozPx['x0'] = self.nd.xC-self.nd.r
            self.nozPx['xf'] = self.nd.xC+self.nd.r
            self.nozPx['yf'] = self.nd.yC+self.nd.r
            self.nozPx['y0'] = self.nd.yC-self.nd.r
            self.adjustForCrop(self.nozPx, self.crop, reverse=True)
            
        
    def findIntendedPx(self) -> None:
        '''convert the intended coordinates to pixels on the cropped image'''
        self.idealspx = {}
        for s in ['c', '0', 'f']:
            xs = f'x{s}'
            ys = f'y{s}'
            if xs in self.ideals:
                x = self.ideals[xs]
            else:
                x = 0
            if ys in self.ideals:
                y = self.ideals[ys]
            else:
                y = 0
            self.idealspx[xs], self.idealspx[ys] = self.nd.relativeCoords(x,y, reverse=True)
        self.adjustForCrop(self.idealspx, self.crop, reverse=True)
        
    def makeMM(self, d:dict, u:dict) -> Tuple[dict, dict]:
        '''convert measurements to mm'''
        for power, l in {1:['w', 'h', 'meanT', 'dxprint', 'dx0', 'dxf', 'space_a', 'space_at', 'dy0l', 'dy0lr', 'dyfr', 'space_l', 'space_b', 'dy0r', 'dyfl', 'dyflr', 'space_r'], 2:['area'], 3:['vest', 'vintegral', 'vleak']}.items():
            if power==1:
                u2 = 'mm'
            else:
                u2 = f'mm^{power}'
            for s in l:
                # ratio of size to intended size
                if s in d and not u[s]==u2:
                    d[s] = d[s]/self.pv.pxpmm**power
                    u[s] = u2
        return d,u

        
    def intendedRC(self, fixList:list=[]) -> tuple:
        '''get the intended center position and width of the whole structure written so far, as coordinates in mm relative to the nozzle'''
        rc1 = self.pg.relativeCoords(self.tag, fixList=fixList)   # position of line 1 in mm, relative to the nozzle. 
           # we know y and z stay the same during travel, so we can just use the endpoint
        w1 = self.progRows.iloc[0]['wmax']
        if w1==0 or self.progRows.a.sum()==0:
            raise ValueError(f'No flow anticipated in {self.folder} {self.name}')
        self.ideals = {}
        if self.numLines>1:
            rc2 = self.pg.relativeCoords(self.tag, self.lnum, fixList=fixList)  # position of last line in mm, relative to the nozzle
            w2 = self.progRows.iloc[self.lnum-1]['wmax']    # width of that written line
        else:
            rc2 = rc1
            w2 = w1
        l = self.progRows.l.max() # length of longest line written so far
        lprog = self.progRows.lprog.max()
        return rc1, rc2, w1, w2, l, lprog
        
    def checkWhite(self, val:int=254) -> bool:
        '''check if the image is all white'''
        if len(self.im.shape)==2:
            return self.im.sum().sum()>np.product(self.im.shape)*val
        else:
            return self.im.sum().sum().sum()>np.product(self.im.shape)*val
        
    #------------------------------------
    
    def getContour(self, combine:bool=False) -> int:
        '''get the contour of the mask, combining all objects if requested, otherwise using the largest object'''
        if hasattr(self, 'cnt'):
            return 0
        self.excessPerimeter = 0
        contours = co.getContours(self.componentMask)
        if len(contours)==0:
            return -1
        if len(contours)==1:
            cnt = contours[0]
        else:
            if combine:
                # combine all contours into one big contour
                cnt = [] 
                for ctr in contours:
                    if len(cnt)>0:
                        self.excessPerimeter = self.excessPerimeter + ppdist(list(ctr[0][0]), list(cnt[-1]))
                    cnt += [pt[0] for pt in ctr]
                self.excessPerimeter = self.excessPerimeter + ppdist(cnt[0], cnt[-1])
                cnt = np.array(cnt)
            else:  
                contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True) # select the largest contour
                cnt = contours[0]
        self.contours = contours
        self.cnt = cv.approxPolyDP(cnt, 1, True) # smooth the contour by 1 px
        self.hull = cv.convexHull(self.cnt)
        if hasattr(self, 'nd') and hasattr(self, 'crop'):
            # conform the hull to the nozzle to avoid extra emptiness and roughness
            self.hull = self.nd.dentHull(self.hull, self.crop)
        return 0
    
    
    def getRoughness(self, diag:int=0) -> float:
        '''measure roughness as perimeter of object / perimeter of convex hull. 
        componentMask is a binarized image of just one segment'''
        perimeter = cv.arcLength(self.cnt,True) - self.excessPerimeter
        if perimeter==0:
            return {}, {}
        hullperimeter = cv.arcLength(self.hull,True)
        roughness = perimeter/hullperimeter-1  # how much extra perimeter there is compared to the convex hull

        return roughness
    
    def getEmptiness(self, emptiness:bool=True) -> float:
        '''measure the empty space inside of the segment'''
        if emptiness and hasattr(self, 'hull'):
            ha = cv.contourArea(self.hull)
            atot = cv.contourArea(self.cnt)
            return 1-(atot/ha)     # how much of the middle of the component is empty
        else:
            return 0
    
    def getLDiffPoints(self, horiz:bool=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        '''get the left and right points in the simplified convex hull'''
        df1 = pd.DataFrame(self.hull2[:, 0, :], columns=['x', 'y'])
        if horiz:
            midx = (df1.x.max()+df1.x.min())/2
            left = df1[df1.x<=midx]
            right = df1[df1.x>midx]
            if len(left)<2 or len(right)<2:
                return [],[]
            bot = pd.concat([left[left.y==left.y.max()], right[right.y==right.y.max()]])
            top = pd.concat([left[left.y==left.y.min()], right[right.y==right.y.min()]])
            return bot, top
        else:
            midy = (df1.y.max()+df1.y.min())/2
            bot = df1[df1.y<=midy]
            top = df1[df1.y>midy]
            right = pd.concat([bot[bot.x==bot.x.max()], top[top.x==top.x.max()]])
            left = pd.concat([bot[bot.x==bot.x.min()], top[top.x==top.x.min()]])
            return left, right
        
    def getLURU(self, left:pd.DataFrame, right:pd.DataFrame, horiz:bool) -> Tuple[int, int]:
        '''get the length-wise distances of the two edges of the simplified convex hull for measuring ldiff'''
        if len(left)<2 or len(right)<2:
            return 0,0
        if horiz:
            lu = len(left.x.unique())
            ru = len(right.x.unique())
        else:
            lu = len(left.y.unique())
            ru = len(right.y.unique())
        return lu, ru
    
    def getLDiff(self, horiz:bool=False) -> float:
        '''get the difference in length between 
        the left and right lines if not horiz, 
        or the top and bottom lines if horiz'''
        
        # smooth the hull until it is a quadrilateral
        self.hull2 = [0,0,0,0,0]
        ii = 30
        while len(self.hull2)>4:
            ii = ii+5
            self.hull2 = cv.approxPolyDP(self.hull, ii, True)
        left,right = self.getLDiffPoints(horiz)
        lu, ru = self.getLURU(left, right, horiz)
        while (len(self.hull2)<4 or lu<2 or ru<2) and ii>0:
            ii = ii-1
            self.hull2 = cv.approxPolyDP(self.hull, ii, True)
            left,right = self.getLDiffPoints(horiz)
            lu, ru = self.getLURU(left, right, horiz)

        if horiz:
            wbottom = left.x.max()-left.x.min()
            wtop = right.x.max()-right.x.min()
            return wtop-wbottom
        else:
            hright = right.y.max()-right.y.min()  # height of points on the right
            hleft = left.y.max()-left.y.min()     # height of points to the left
            return hright-hleft
        
    def roughnessIm(self, hull2:bool=True, scalebar:bool=False, export:bool=False, display:bool=False) -> np.array:
        '''add annotations for roughness to the image'''
        cm = self.componentMask.copy()
        cm = cv.cvtColor(cm,cv.COLOR_GRAY2RGB)
        if hasattr(self, 'contours'):
            for i in range(len(self.contours)):
                cv.drawContours(cm, self.contours, i, (186, 6, 162), 2)
        if hasattr(self, 'hull'):
            cv.drawContours(cm, [self.hull], -1, (110, 245, 209), 2)
        if hasattr(self, 'hull2') and hull2:
            cv.drawContours(cm, [self.hull2], -1, (252, 223, 3), 2)
        if scalebar:
            cm[10:20, 10:10+self.pv.pxpmm, :] = 255  # 1 mm scale bar
        if display:
            imshow(cm)
        if export:
            self.roughnessIm = cm
            self.exportImage('roughnessIm', 'annotations', 'roughness', overwrite=True)
            self.exportImage('im0', 'annotations', 'orig', overwrite=True)
        return cm
    
    def ldiffIm(self, export:bool=False, display:bool=True, scalebar:bool=True) -> np.array:
        '''add annotations for length asymmetry to the initial image'''
        im = self.im0.copy()
        if hasattr(self, 'hull2'):
            cv.drawContours(im, [self.hull2], -1, (255,255,255), 2)
        if scalebar:
            im[10:20, 10:10+self.pv.pxpmm, :] = 255  # 1 mm scale bar
        if display:
            imshow(im)
        if export:
            self.ldiffIm = im
            self.exportImage('ldiffIm', 'annotations', 'ldiff', overwrite=True)
        return im

    
    def sumsAndWidths(self, horiz:bool) -> Tuple[list, list]:
        '''sum up the number of pixels per row and the width of the pixels in the row'''
        if horiz:
          # total number of pixels per row
            sums = self.componentMask.sum(axis=0)/255
        else:
            sums = self.componentMask.sum(axis=1)/255
        sums = list(filter(lambda i:i>0, sums)) # remove empty rows
        return sums
    
    def limitLen(self, sums:list, reverse:bool) -> Tuple[list,list]:
        '''limit the maximum length of the line and put the rest in leaks'''
        if self.maxlen>0:
            # limit the measurements to only the length where extrusion was on
            if reverse:
                ilast = max(len(sums)-self.maxlen, 0)
                sums = sums[ilast:]
                leaks = sums[0:ilast]
            else:
                ilast = min(self.maxlen+1, len(sums))
                leaks = sums[ilast:]
                sums = sums[0:ilast]
        else:
            leaks = []
        return sums, leaks
        
    def measureVolumes(self, sums:list, leaks:list) -> Tuple[float, float]:
        '''measure the volumes of the part and the leak'''
        vest = sum([np.pi*(d/2)**2 for d in sums])*self.scale**3
        if len(leaks)>0:
            vleak = sum([np.pi*(d/2)**2 for d in leaks])*self.scale**3
        else:
            vleak = 0
        return vest, vleak
    
    def measureWidths(self, sums:list) -> Tuple[float, float, float]:
        '''measure the width of the line and the variation in width along the line'''
        meant = np.mean(sums)                       # mean line thickness
        midrange = sums[int(meant/2):-int(meant/2)] # remove endcaps
        if len(midrange)>0:
            stdev = np.std(midrange)/meant*self.scale               # standard deviation of thickness normalized by mean
            minmax = (max(midrange)-min(midrange))/meant*self.scale # total variation in thickness normalized by mean
        else:
            stdev = np.nan
            minmax = np.nan
        meant = meant*self.scale
        return meant, stdev, minmax

        
    def measureComponent(self, horiz:bool=True, reverse:bool=False, emptiness:bool=True, combine:bool=False, diag:int=0) -> Tuple[dict,dict]:
        '''measure parts of a segmented fluid. 
        horiz = True to get variation along length of horiz line. False to get variation along length of vertical line.
        scale is the scaling of the stitched image compared to the raw images, e.g. 0.33 
        if maxlen>0, maxlen is the maximum length of the expected line in px. anything outside is leaks
        reverse=True to measure from top or right, false to measure from bottom or left'''
        for s in ['componentMask', 'scale', 'maxlen']:
            if not hasattr(self, s):
                raise ValueError(f'{s} undefined for {self.file}')
        
        errorRet = {}, {}
        out = self.getContour(combine=combine)
        if out<0:
            return errorRet
        roughness = self.getRoughness(diag=max(0,diag-1))
        sums = self.sumsAndWidths(horiz)
        if len(sums)==0:
            return errorRet
        sums, leaks = self.limitLen(sums, reverse)
        empty = self.getEmptiness(emptiness)
        vest, vleak = self.measureVolumes(sums, leaks)
        meant, stdev, minmax = self.measureWidths(sums)
        units = {'roughness':'', 'emptiness':'', 'meanT':'px', 'stdevT':'meanT', 'minmaxT':'meanT', 'vintegral':'px^3', 'vleak':'px^3'}
        retval = {'roughness':roughness, 'emptiness':empty, 'meanT':meant, 'stdevT':stdev, 'minmaxT':minmax, 'vintegral':vest, 'vleak':vleak}
        return retval, units
    
    #---------------------
    
    def zdisplacement(self, dd:dict, distance:int, size:int, diag:int=0):
        '''horizontal displacement between component and nozzle'''
        bot = dd['bot']
        out = {}
        abovey = bot-distance
        belowy = bot+distance
        dd['x0a'], dd['xfa'] = meanBounds(self.componentMask[int(abovey-size/2):int(abovey+size/2)], rows=True)  # above
        dd['x0b'], dd['xfb'] = meanBounds(self.componentMask[int(belowy-size/2):int(belowy+size/2)], rows=True)  # below
        dd['x0at'], dd['xfat'] = meanBounds(self.componentMask[int(bot-2):int(bot-1)], rows=True)   # at bottom of nozzle
        for tt in [['x0a', 'x0b', 'dx0'], ['xfa', 'xfb', 'dxf'], ['left', 'xfa', 'space_a'], ['left', 'xfat', 'space_at']]:
            if dd[tt[0]]>0 and dd[tt[1]]>0:
                out[tt[2]] = dd[tt[0]]-dd[tt[1]]-1    
                if 'space' in tt[2]:
                    out[tt[2]] = max(0, out[tt[2]])
                # displacement of left side of filament between above and below
        if dd['x0b']>0 and dd['xfb']>0:
            out['dxprint'] = (dd['x0b']+dd['xfb'])/2-dd['mid']   
            # distance between center of nozzle and center of filament below. positive means filament is right of nozzle
        if diag>0:
            im2 = self.componentMask.copy()
            im2 = cv.cvtColor(im2,cv.COLOR_GRAY2RGB)
            for pt in [[abovey, dd['x0a']], [abovey, dd['xfa']], [belowy, dd['x0b']], [belowy, dd['xfb']], [bot, dd['x0at']], [bot, dd['xfat']]]:
                im2 = cv.circle(im2, (int(pt[1]), int(pt[0])), 3, (0,0,255), 3)
            imshow(im2)
        return out
    
    def ydisplacement(self, dd:dict, distance:int, size:int, diag:int=0):
        '''vertical displacement between component and nozzle'''
        out = {}
        leftx = dd['left']-distance
        rightx = dd['right']+distance
        mid = dd['mid']
        # 0 is top, f is bottom
        dd['y0l'], dd['yfl'] = meanBounds(self.componentMask[:, int(leftx-size/2):int(leftx+size/2)], rows=False)
        dd['y0r'], dd['yfr'] = meanBounds(self.componentMask[:, int(rightx-size/2):int(rightx+size/2)], rows=False)
        dd['y0b'], dd['yfb'] = meanBounds(self.componentMask[:, int(mid-size/2):int(mid+size/2)], rows=False)
        for tt in [['y0b', 'y0l', 'dy0l'], ['y0b', 'y0r', 'dy0r'], ['y0l', 'y0r', 'dy0lr']
                   , ['yfb', 'yfl', 'dyfl'], ['yfb', 'yfr', 'dyfr'], ['yfl', 'yfr', 'dyflr']
                   , ['y0l', 'bot', 'space_l'], ['y0r', 'bot', 'space_r'], ['y0b', 'bot', 'space_b']]:
            if dd[tt[0]]>0 and dd[tt[1]]>0:
                out[tt[2]] = dd[tt[0]]-dd[tt[1]]    # displacement of top and bottom side of filament between left and right
        if diag>0:
            im2 = self.componentMask.copy()
            im2 = cv.cvtColor(im2,cv.COLOR_GRAY2RGB)
            im2 = cv.rectangle(im2, (int(dd['left']),0), (int(dd['right']),int(dd['bot'])), (255,255,0), 2)
            for pt in [[leftx, dd['y0l']], [leftx, dd['yfl']], [rightx, dd['y0r']], [rightx, dd['yfr']], [mid, dd['y0b']], [mid, dd['yfb']]]:
                im2 = cv.circle(im2, (int(pt[0]), int(pt[1])), 3, (0,0,255), 3)
            imshow(im2)
        return out
    
    def displacement(self, direc:str, distance:int, size:float=20, diag:int=0) -> Tuple[dict,dict]:
        '''determine displacement of the filament. 
        direc: if 'z', then find horizontal displacement above and below the bottom of the nozzle, e.g. for vertical lines. if 'y', then find vertical displacement under, left, and right of the nozzle, e.g. for horizontal lines. 
        crop indicates how this image has already been cropped
        distance is the distance from the nozzle in px to take the measurement
        size is the thickness of the slice in px to use for the measurement'''
        for s in ['nd', 'componentMask', 'crop']:
            if not hasattr(self, s):
                raise ValueError(f'{s} undefined for {self.file}')
        nd = self.nd
        crop = self.crop
        n0 = nd.nozBounds()
        dd = {}
        bot = n0['yf'] - crop['y0']
        dd['bot'] = bot
        dd['left'] = n0['x0'] - crop['x0']  
        if hasattr(nd, 'ndGlobal'):
            w = nd.ndGlobal.nozWidthPx()
            if w==0:
                w = nd.nozWidthPx()
        else:
            w = nd.nozWidthPx()
            
        dd['right'] = dd['left'] + w
        # dd['right'] = nd.xR - crop['x0']
        mid = (dd['left']+dd['right'])/2
        dd['mid'] = mid
        
        if direc=='z':
            return self.zdisplacement(dd, distance, size, diag=diag)
        elif direc=='y':
            return self.ydisplacement(dd, distance, size, diag=diag)
        else:
            raise ValueError(f'Unexpected direc {direc} given to displacement. Value should be y or z')
  
    def adjustForCrop(self, d:dict, crop:dict, reverse:bool=False) -> None:
        '''adjust the stats for cropping. reverse to go to the cropped '''
        for si in ['x', 'y']:
            for s in ['0', 'c', 'f']:
                ss = f'{si}{s}'
                if ss in d and f'{si}0' in crop:
                    if reverse:
                        d[ss] = d[ss]-crop[f'{si}0']*self.scale
                    else:
                        d[ss] = d[ss]+crop[f'{si}0']*self.scale
                        