#!/usr/bin/env python
'''Functions for collecting data from stills of single lines, for a single image'''

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
import file.file_handling as fh
from im.imshow import imshow
from tools.plainIm import *
from tools.config import cfg
from val.v_print import printVals
from progDim.prog_dim import getProgDims, getProgDimsPV
from vid.noz_detect import nozData
from m_tools import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 4)
pd.set_option('display.max_rows', 500)


#----------------------------------------------


class metricSegment:
    '''collects data about fluid segments in an image'''
    
    def __init__(self, file:str, diag:int=0, acrit:int=2500, **kwargs):
        self.file = file
        self.folder = os.path.dirname(self.file)
        if not os.path.exists(self.file):
            raise ValueError(f'File {self.file} does not exist')
        self.scale = 1/fh.fileScale(self.file)
        self.im = cv.imread(self.file)
        self.acrit = acrit
        self.diag = diag
        self.stats = {'line':''}
        self.units = {'line':''}
        
    def values(self) -> Tuple[dict,dict]:
        return self.stats, self.units
    
    def statText(self, cols:int=1) -> str:
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

    
    #------------------------------------
    
    def getContour(self, combine:bool=False) -> int:
        '''get the contour of the mask, combining all objects if requested, otherwise using the largest object'''
        if hasattr(self, 'cnt'):
            return 0
        self.excessPerimeter = 0
        contours = getContours(self.componentMask)
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
        self.cnt = cv.approxPolyDP(cnt, 1, True) # smooth the contour by 1 px
        self.hull = cv.convexHull(self.cnt)
        if hasattr(self, 'nd') and hasattr(self, 'crop'):
            # conform the hull to the nozzle to avoid extra emptiness and roughness
            self.hull = self.nd.dentHull(self.hull, self.crop)
        return 0
    
    
    def getRoughness(self, diag:int=0, combine:bool=False) -> float:
        '''measure roughness as perimeter of object / perimeter of convex hull. 
        componentMask is a binarized image of just one segment'''
        perimeter = cv.arcLength(self.cnt,True) - self.excessPerimeter
        if perimeter==0:
            return {}, {}
        hullperimeter = cv.arcLength(self.hull,True)
        roughness = perimeter/hullperimeter-1  # how much extra perimeter there is compared to the convex hull

        return roughness
    
    def getLDiff(self, horiz:bool=False) -> float:
        '''get the difference in length between 
        the left and right lines if not horiz, 
        or the top and bottom lines if horiz'''
        
        # smooth the hull until it is a quadrilateral
        self.hull2 = [0,0,0,0,0]
        ii = 20
        while len(self.hull2)>4:
            ii = ii+5
            self.hull2 = cv.approxPolyDP(self.hull, ii, True)
        while len(self.hull2)<4:
            ii = ii-1
            self.hull2 = cv.approxPolyDP(self.hull, ii, True)
            
        # filter the points
        df1 = pd.DataFrame(self.hull2[:, 0, :], columns=['x', 'y'])
        if horiz:
            mid = df1.y.nlargest(2).min()
            bottom = df1[df1.y>=mid]
            wbottom = bottom.x.max()-bottom.x.min()
            top = df1[df1.y<mid]
            wtop = top.x.max()-top.x.min()
            return wtop-wbottom
        else:
            mid = df1.x.nlargest(2).min()
            right = df1[df1.x>=mid]
            hright = right.y.max()-right.y.min()
            left = df1[df1.x<mid]
            hleft = left.y.max()-left.y.min()
            return hright-hleft
        
    def roughnessIm(self) -> np.array:
        '''add annotations for roughness to the image'''
        cm = self.componentMask.copy()
        cm = cv.cvtColor(cm,cv.COLOR_GRAY2RGB)
        if hasattr(self, 'cnt'):
            cv.drawContours(cm, [self.cnt], -1, (186, 6, 162), 2)
        if hasattr(self, 'hull'):
            cv.drawContours(cm, [self.hull], -1, (110, 245, 209), 2)
        if hasattr(self, 'hull2'):
            cv.drawContours(cm, [self.hull2], -1, (252, 223, 3), 2)
        return cm

    
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
    
    def getEmptiness(self, atot:float, emptiness:bool=True) -> float:
        '''measure the empty space inside of the segment'''
        if emptiness and hasattr(self, 'hull'):
            ha = cv.contourArea(self.hull)
            return 1-(atot/ha)     # how much of the middle of the component is empty
        else:
            return 0
        
    def measureVolumes(self, sums:list, leaks:list) -> Tuple[float, float]:
        '''measure the volumes of the part and the leak'''
        vest = sum([np.pi*(d/2)**2 for d in sums])
        if len(leaks)>0:
            vleak = sum([np.pi*(d/2)**2 for d in leaks])
        else:
            vleak = 0
        return vest, vleak
    
    def measureWidths(self, sums:list) -> Tuple[float, float, float]:
        '''measure the width of the line and the variation in width along the line'''
        meant = np.mean(sums)                       # mean line thickness
        midrange = sums[int(meant/2):-int(meant/2)] # remove endcaps
        if len(midrange)>0:
            stdev = np.std(midrange)/meant               # standard deviation of thickness normalized by mean
            minmax = (max(midrange)-min(midrange))/meant # total variation in thickness normalized by mean
        else:
            stdev = ''
            minmax = ''
        return meant, stdev, minmax

        
    def measureComponent(self, horiz:bool=True, reverse:bool=False, emptiness:bool=True, atot:float=0, combine:bool=False, diag:int=0) -> Tuple[dict,dict]:
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
        roughness = self.getRoughness(diag=max(0,diag-1), combine=combine)
        sums = self.sumsAndWidths(horiz)
        if len(sums)==0:
            return errorRet
        sums, leaks = self.limitLen(sums, reverse)
        empty = self.getEmptiness(atot, emptiness)
        vest, vleak = self.measureVolumes(sums, leaks)
        meant, stdev, minmax = self.measureWidths(sums)
        units = {'roughness':'', 'emptiness':'', 'meanT':'px', 'stdevT':'meanT', 'minmaxT':'meanT', 'vintegral':'px^3', 'vleak':'px^3'}
        retval = {'roughness':roughness, 'emptiness':empty, 'meanT':meant*self.scale, 'stdevT':stdev*self.scale, 'minmaxT':minmax*self.scale, 'vintegral':vest*self.scale**3, 'vleak':vleak*self.scale**3}
        return retval, units
    
    #---------------------
    
    def zdisplacement(self, dd:dict, distance:int, size:int, diag:int=0):
        '''vertical displacement between component and nozzle'''
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
                # displacement of left side of filament between above and below
        if dd['x0b']>0 and dd['xfb']>0:
            out['dxprint'] = (dd['x0b']+dd['xfb'])/2-dd['mid']   
            # distance between center of nozzle and center of filament below. positive means filament is right of nozzle
        if diag>0:
            im2 = self.componentMask.copy()
            for pt in [[abovey, dd['x0a']], [abovey, dd['xfa']], [belowy, dd['x0b']], [belowy, dd['xfb']], [bot, dd['x0at']], [bot, dd['xfat']]]:
                im2 = cv.circle(im2, (int(pt[0]), int(pt[1])), 3, (0,0,255), 3)
            imshow(im2)
        return out
    
    def ydisplacement(self, dd:dict, distance:int, size:int, diag:int=0):
        '''horizontal displacement between component and nozzle'''
        out = {}
        leftx = dd['left']-distance
        rightx = dd['right']+distance
        mid = dd['mid']
        # 0 is top, f is bottom
        dd['y0l'], dd['yfl'] = meanBounds(self.componentMask[:, int(leftx-size/2):int(leftx+size/2)], rows=False)
        dd['y0r'], dd['yfr'] = meanBounds(self.componentMask[:, int(rightx-size/2):int(rightx+size/2)], rows=False)
        dd['y0b'], dd['yfb'] = meanBounds(self.componentMask[:, int(mid-size/2):int(mid+size/2)], rows=False)
        for tt in [['y0b', 'y0l', 'dy0l'], ['y0b', 'y0r', 'dy0r'], ['y0l', 'y0r', 'dy0lr'], ['yfb', 'yfl', 'dyfl'], ['yfb', 'yfr', 'dyfr'], ['yfl', 'yfr', 'dyflr'], ['y0l', 'bot', 'space_l'], ['y0r', 'bot', 'space_r'], ['y0b', 'bot', 'space_b']]:
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
        
        dd = {}
        bot = nd.yB - crop['y0']   
        dd['bot'] = bot
        dd['left'] = nd.xL - crop['x0'] - self.nd.maskPad   # nozzle cover was 10 extra pixels to left and right
        dd['right'] = nd.xR - crop['x0'] + self.nd.maskPad
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
                    
    
#------------------------------------------------------------------------------------- 
        
class segmentSingle(metricSegment):
    '''collect measurements of segments in singleLine prints'''
    
    def __init__(self, file:str):
        super().__init__(file)
        
    def lineName(self, tag:str) -> float:
        '''for single lines, get the number of the line from the file name based on tag, e.g. 'vert', 'horiz', 'xs'. '''
        spl = re.split('_',os.path.basename(self.file))
        for st in spl:
            if tag in st:
                return float(st.replace(tag, ''))
        return -1
    
    
class segmentDisturb(metricSegment):
    '''collect measurements of segments in disturbed prints'''
    
    def __init__(self, file:str, diag:int=0, acrit:int=2500, measure:bool=True, **kwargs):
        super().__init__(file, diag=diag, acrit=acrit, **kwargs)
        self.scale = 1
        if 'nd' in kwargs:
            self.nd = kwargs['nd']
        else:
            self.nd = nozData(os.path.dirname(file))   # detect nozzle
        self.pfd = self.nd.pfd
        if 'pv' in kwargs:
            self.pv =  kwargs['pv']
        else:
            self.pv = printVals(os.path.dirname(file), pfd = self.pfd, fluidProperties=False)
        if 'pg' in kwargs:
            self.pg = kwargs['pg']
        else:
            self.getProgDims()
        self.title = os.path.basename(self.file)
        self.lineName()
        if measure:
            self.measure()
            
    def initializeTimeCounter(self):
        self.timeCount = time.time()
        
    def timeCounter(self, s:str):
        tt = time.time()
        print(f'segmentDisturb {s} {(tt-self.timeCount):0.4f} seconds')
        self.timeCount = tt
        
    def lineName(self) -> None:
        '''get the name of a singleDisturb, or tripleLine line'''
        if not 'vstill' in self.file:
            raise ValueError(f'Cannot determine line name for {self.file}')
        spl = re.split('_', re.split('vstill_', os.path.basename(self.file))[1])
        self.name = f'{spl[0]}_{spl[1]}'  # this is the full name of the pic, e.g. HOx2_l3w2o1
        self.tag = spl[1]                 # this is the name of the pic, e.g. l1wo
        self.gname = self.tag[:2]     # group name, e.g. l3
        
    def getProgDims(self):
        '''initialize the progDims object'''
        if not hasattr(self, 'pg'):
            self.pg  = getProgDimsPV(self.pv)
            self.pg.importProgDims()
            if self.pg.progDims.a.sum()==0:
                logging.warning(f'Empty area in {self.folder}. Redoing progDims')
                self.pg.exportAll(overwrite=True)
            
    def getProgRow(self):
        '''get the progDims row for this line'''
        progRows = pd.concat([self.pg.progLine(i+1, self.gname) for i in range(self.lnum)])
        self.progRows = progRows
        
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
        self.nozPx['x0'] = self.nd.xL
        self.nozPx['xf'] = self.nd.xR
        self.nozPx['yf'] = self.nd.yB
        self.adjustForCrop(self.nozPx, self.crop, reverse=True)
        self.nozPx['y0'] = 0
        
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
        for power, l in {1:['w', 'h', 'meanT', 'dxprint', 'dx0', 'dxf', 'space_a', 'space_at'], 2:['area'], 3:['vest', 'vintegral', 'vleak']}.items():
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

        
    def intendedRC(self, fixList:list=[]) -> Tuple[dict, list]:
        '''get the intended center position and width of the whole structure written so far, as coordinates in mm relative to the nozzle'''
        rc1 = self.pg.relativeCoords(self.tag, fixList=fixList)   # position of line 1 in mm, relative to the nozzle. 
           # we know y and z stay the same during travel, so we can just use the endpoint
        w1 = self.progRows.iloc[0]['w']
        if w1==0 or self.progRows.a.sum()==0:
            raise ValueError(f'No flow anticipated in {self.folder} {self.name}')
        self.ideals = {}
        if self.numLines>1:
            rc2 = self.pg.relativeCoords(self.tag, self.lnum, fixList=fixList)  # position of last line in mm, relative to the nozzle
            w2 = self.progRows.iloc[self.lnum-1]['w']    # width of that written line
        else:
            rc2 = rc1
            w2 = w1
        l = self.progRows.l.max() # length of longest line written so far
        return rc1, rc2, w1, w2, l
    
    
    
class segmentSDT(segmentDisturb):
    '''singleDoubleTriple single files'''
    
    def __init__(self, file:str, diag:int=0, acrit:int=2500, **kwargs):
        super().__init__(file, diag=diag, acrit=acrit, **kwargs)
        
    
    def lineName(self) -> None:
        '''get the name of a singleDisturb, or tripleLine line'''
        if not 'vstill' in self.file:
            raise ValueError(f'Cannot determine line name for {self.file}')
        self.numLines = int(re.split('_', os.path.basename(self.file))[1])
        spl = re.split('_', re.split('vstill_', os.path.basename(self.file))[1])
        self.name = f'{spl[0]}_{spl[1]}'  # this is the full name of the pic, e.g. HOx2_l3w2o1
        lt = re.split('o', re.split('_', self.name)[1][2:])[0]
        if lt=='d' or lt=='w':
            # get the last line
            self.lnum = self.numLines
        else:
            self.lnum = int(lt[1])
        self.tag = spl[1]                 # this is the name of the pic, e.g. l3w2o1
        self.gname = self.tag[:2]     # group name, e.g. l3
        
    def subFN(self, subFolder:str, title:str) -> str:
        '''get the filename of a file that goes into a subfolder'''
        cropfolder = os.path.join(self.folder, subFolder)
        if not os.path.exists(cropfolder):
            os.mkdir(cropfolder)
        fnorig = os.path.join(cropfolder, self.title.replace('vstill', title))
        return fnorig
        
    def exportImage(self, att:str, subFolder:str, title:str, overwrite:bool=False) -> None:
        '''export an image stored as attribute att to a subfolder subFolder with the new title title'''
        if not hasattr(self, att):
            raise ValueError(f'No {att} found for {self.file}')
        fnorig = self.subFN(subFolder, title)
        if not os.path.exists(fnorig) or overwrite:
            im = getattr(self, att)
            cv.imwrite(fnorig, im)
            print(f'Exported {fnorig}')
        
        
    def exportCrop(self, overwrite:bool=False) -> None:
        '''add the cropped image to the folder for machine learning'''
        if not hasattr(self, 'im0'):
            self.generateIm0()
        self.exportImage('im0', 'crop', 'vcrop')
        
    def segmentFN(self) -> str:
        return self.subFN('Usegment', 'Usegment')
            
    def exportSegment(self, overwrite:bool=False) -> None:
        '''add the cropped image to the folder for machine learning'''
        self.exportImage('componentMask', 'Usegment', 'Usegment')
        
    def addToTraining(self, trainFolder:str=r'singleDoubleTriple\trainingVert', s:str='componentMask', openPaint:bool=False):
        '''add the original image and the segmented image to the training dataset'''
        folder = os.path.join(cfg.path.fig, trainFolder)
        fnorig = os.path.join(folder, 'orig', self.title)
        cv.imwrite(fnorig, self.im0)
        fnseg = os.path.join(folder, 'segmented', self.title)
        if s=='thresh':
            img = self.segmenter.thresh
            ccl = segmenterDF(img)   # remove little segments
            ccl.eraseFullWidthComponents()
            img = ccl.labelsBW
        else:
            img = getattr(self, s)
        cv.imwrite(fnseg, img)
        if s=='thresh' or openPaint:
            openInPaint(fnseg)
        print(f'Exported {self.title} to training data')
  
        
#-------------------------------------------------------------------
    
def testFile(fstr:str, fistr:str, func, slist:list, diag:int=4, **kwargs) -> dict:
    '''test a single file, for any print type given a metricSegment class '''
    folder = os.path.join(cfg.path.server, fstr)
    file = os.path.join(folder, fistr)
    d,u = func(file, diag=diag, **kwargs).values()
    out = f'{fstr},{fistr}'
    olist = {'folder': fstr, 'file':fistr}
    for s in slist:
        if s in d:
            v = d[s]
        else:
            v = -1
        out = f'{out},{v}'
        olist[s] =v
    return olist

def addToTestFile(csv:str, fstr:str, fistr:str, func, slist:list, diag:int=4, **kwargs) -> None:
    l = testFile(fstr, fistr, func, slist, diag=diag, **kwargs)
    df, _ = plainIm(csv, ic=None)
    if len(df)==0:
        df = pd.DataFrame([l])
    else:
        if l['file'] in df.file:
            for key, val in l:
                df.loc[df.file==l['file'], key] = val
        else:
            df = pd.concat([df, pd.DataFrame([l])])
    plainExp(csv, df, {}, index=False)
    
class unitTester:
    '''this class lets you run unit tests and evaluate functions later. fn is the test name, e.g. SDTXS or disturbHoriz, func is the function that you run on a file to get values'''
    
    def __init__(self, fn:str, func):
        cdir = os.path.dirname(os.path.abspath(os.path.join('..')))
        self.testcsv = os.path.join(cdir, 'tests', f'test_{fn}.csv')  # the csv file for the test
        self.testpy = f'test_{fn}'   # the python file for the test
        self.func = func
        
    def run(self):
        currentdir = os.path.dirname(os.path.realpath(__file__))
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(currentdir)), 'tests'))
        print(self.testpy)
        tp = __import__(self.testpy)
        runner = tp.unittest.TextTestRunner()
        result = runner.run(tp.suite())
        self.failedFiles = [int(re.split(': ', str(s))[-1][:-4]) for s in result.failures]  # indices of failed files
        
    def importList(self):
        if not hasattr(self, 'testList'):
            self.testList = pd.read_csv(self.testcsv)
            
    def runTest(self, i:int, diag:int=0, **kwargs) -> Tuple[pd.Series, dict, list]:
        self.importList()
        if i>=len(self.testList):
            print(f'UnitTester has {len(self.testList)} entries. {i} not in list')
            return [],{},[]
        row = self.testList.loc[i]
        folder = row['folder']
        file = row['file']
        if diag>0:
            print(f'TEST {i} (excel row {i+2})\nFolder: {folder}\nFile: {file}')
        folder = os.path.join(cfg.path.server, folder)
        file = os.path.join(folder, file)
        d,u = self.func(file, diag=diag, **kwargs)
        cols = list(self.testList.keys())
        cols.remove('folder')
        cols.remove('file')
        return row, d, cols
        
    def compareTest(self, i:int, diag:int=1, **kwargs) -> None:
        '''print diagnostics on why a test failed. fn is the basename of the test csv, e.g. test_SDTXS.csv. i is the row number'''
        row, d, cols = self.runTest(i, diag=diag, **kwargs)
        if len(row)==0:
            return
        df = pd.DataFrame({})
        for c in cols:
            df.loc['expected', c] = row[c]
            if c in d:
                df.loc['actual', c] = d[c]
        pd.set_option("display.precision", 8)
        display(df)
        
    def compareAll(self):
        '''check diagnostics for all failed files'''
        c = 0
        if len(self.failedFiles)>5:
            print(f'{len(self.failedFiles)} failed files, showing 5')
        for i in self.failedFiles:
            self.compareTest(i)
            c = c+1
            if c==5:
                return
            
    def openCSV(self):
        subprocess.Popen([cfg.path.excel, self.testcsv]);
        
    def exportTestList(self):
        '''overwrite the list of tests with current values'''
        self.testList.to_csv(self.testcsv, index=False)
        logging.info(f'Exported {self.testcsv}')
        
    def keepTest(self, i:int, export:bool=True) -> None:
        '''overwrite the value in the csv file with the current values'''
        row, d, cols = self.runTest(i, diag=1)
        for c in cols:
            self.testList.loc[i, c] = d[c]
        if export:
            self.exportTestList()
        
    def keepAllTests(self) -> None:
        '''overwrite all failed values with the values found now'''
        for i in self.failedFiles:
            self.keepTest(i, export=False)
        self.exportTestList()
            
def runUnitTest(testName:str, func):
    ut = unitTester(testName, func)
    ut.run()
    ut.compareAll()
    
    
def convertFilesToBW(folder:str, diag:bool=False) -> None:
    '''convert all the files in the folder to black and white'''
    for f in os.listdir(folder):
        ffull = os.path.join(folder, f)
        im = cv.imread(ffull)
        im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        _,im = cv.threshold(im, 120, 255, cv.THRESH_BINARY)
        im[im==255]=1
        if diag:
            print(im.shape, np.unique(im), im[0,0])
        else:
            cv.imwrite(ffull, im)
            
def removeOme(folder:str):
    '''remove .ome from all of the file names'''
    for f in os.listdir(folder):
        ffull = os.path.join(folder, f)
        new = ffull.replace('.ome', '')
        os.rename(ffull, new)
            
class segmentCompare:
    '''for comparing pre-segmented images to freshly segmented images. func should be a metricSegment class definition '''
    
    def __init__(self, segFolder:str, serverFolder:str, origFolder:str, func):
        self.segFolder = segFolder
        self.serverFolder = serverFolder
        self.origFolder = origFolder
        self.func = func
        self.images = {}
        self.df = pd.DataFrame({'bn':os.listdir(self.segFolder)})
        
    def compare(self):
        '''iterate through all images in the segmentation folder and compare them to the current code'''
        for i in self.df.index:
            self.compareFile(i)
            
    def getSegIdeal(self, i:int, bn:str) -> np.array:
        '''get the manually segmented image'''
        segfile = os.path.join(self.segFolder, bn)
        self.df.loc[i, 'segfile'] = segfile
        segIdeal = cv.imread(segfile, cv.IMREAD_GRAYSCALE)
        return segIdeal
    
    def getSegReal(self, i:int, bn:str, diag:int=0, **kwargs) -> np.array:
        '''segment the image using the current algorithm'''
        try:
            fn = fh.findFullFN(bn, self.serverFolder)
        except FileNotFoundError:
            self.df.loc[i, 'result'] = 'FileNotFound'
            return []
        ob = self.func(fn, diag=diag, **kwargs) 
        if not hasattr(ob, 'componentMask'):
            self.df.loc[i, 'result'] = 'NoMask'
            return []
        segReal = ob.componentMask
        return segReal
    
    def getSegMachine(self, i:int, bn:str) -> np.array:
        '''get the ML segmented image from a folder'''
        fn = os.path.join(self.serverFolder, bn)
        if not os.path.exists(fn):
            self.df.loc[i, 'result'] = 'MLNotFound'
            return []
        segReal = cv.imread(fn, cv.IMREAD_GRAYSCALE)
        return segReal
    
    def getOrig(self, bn:str) -> np.array:
        '''get the original image'''
        fn = os.path.join(self.origFolder, bn)
        if not os.path.exists(fn):
            return []
        orig = cv.imread(fn)
        return orig
            
    def compareFile(self, i:int, diag:int=0, **kwargs) -> dict:
        '''segfile is the pre-segmented file'''
        bn = self.df.loc[i, 'bn']
        self.df.loc[i, 'difference'] = 1
        
        # find original file
        segIdeal = self.getSegIdeal(i, bn)
        
        if type(self.func) is str:
            # get the file from the folder
            segReal = self.getSegMachine(i, bn)
        else:
            # segment the file
            segReal = self.getSegReal(i, bn, diag=diag, **kwargs)
        if len(segReal)==0:
            return
        
        # compare segmentation to ideal
        if not segIdeal.shape==segReal.shape:
            self.df.loc[i, 'result'] = 'ShapeMismatch'
            return
        diff = self.measureDifference(bn, i, segIdeal, segReal)
        self.createImDiag(bn, segIdeal, segReal, diff, diag=diag)
        
        
    def measureDifference(self, bn:str, i:int, segIdeal:np.array, segReal:np.array) -> None:
        '''measure the number of pixels that are different between the two images'''
        diff = cv.bitwise_xor(segIdeal, segReal)
        self.df.loc[i, 'result'] = 'Found'
        shape = segIdeal.shape
        a = shape[0]*shape[1]
        self.df.loc[i, 'difference'] = diff.sum(axis=0).sum(axis=0)/255/a  # number of different pixels
        return diff
        
    def createImDiag(self, bn:str, segIdeal:np.array, segReal:np.array, diff:np.array, diag:int=0) -> None:
        '''create a diagnostic image that compares the found image to the manual segmented image and shows differences in color'''
        imdiag = cv.cvtColor(segIdeal, cv.COLOR_GRAY2BGR)
        r0 = cv.bitwise_and(segIdeal, segReal)
        imdiag[(r0==255)] = [100,100,100]
        r1 = cv.subtract(segIdeal, segReal)
        imdiag[(r1==255)] = [0,255,0]
        r2 = cv.subtract(segReal, segIdeal)
        imdiag[(r2==255)] = [0,0,255]
        if diag>0:
            orig = self.getOrig(bn)
            if len(orig)>0:
                imshow(imdiag, segIdeal, segReal, orig, titles=['diff', 'ideal', 'real', 'orig'])
            else:
                imshow(imdiag, segIdeal, segReal, diff, titles=['diff', 'ideal', 'real', 'difference'])
        self.images[bn] = imdiag 
        
    def showWorstSegmentation(self, n:int=6) -> None:
        '''show the segmented images with the worst diff values'''
        print(f'Success rate: {self.successRate():0.3f}, Average diff: {self.averageDiff():0.3f}')
        df2 = self.df.sort_values(by='difference', ascending=False)
        ims = []
        print(df2.iloc[:n*2][['result', 'difference']])
        print('Red = algorithm, green = manual')
        i = 0
        j = 0
        while j<n and i<len(df2):
            bn = df2.iloc[i]['bn']
            if bn in self.images:
                ims.append(self.images[bn])
                j = j+1
            i = i+1
        imshow(*ims)
        
    def successRate(self, dcrit:float=0.01) -> float:
        '''get the percentage of files that have a diff less than dcrit'''
        return (len(self.df[self.df.difference<dcrit])/len(self.df))
    
    def averageDiff(self) -> float:
        '''get the average difference'''
        return self.df.difference.mean()

    
class trainingGenerator:
    '''a class for generating training data'''
    
    def __init__(self, topFolder:str, excludeFolders:list=[], mustMatch:list=[], someIn:list=[]):
        self.topFolder = topFolder
        self.excludeFolders = excludeFolders
        self.printFolders = fh.printFolders(topFolder, tags=mustMatch, someIn=someIn)
        self.numFolders = len(self.printFolders)
        
    def randomFolder(self):
        '''get a random folder'''
        i = np.random.randint(self.numFolders)
        return self.printFolders[i]
    
    def excluded(self, bn:str) -> bool:
        '''check if the file basename is already in one of the excluded folders. return True if it should be excluded'''
        for f in self.excludeFolders:
            if os.path.exists(os.path.join(f, bn)):
                return True
        return False
    
    def randomFile(self) -> str:
        '''get a random vert vstill from the topFolder, but get a new one if it's already in one of the excludeFolders '''
        folder = self.randomFolder()
        pfd = fh.printFileDict(folder)
        pfd.sort()
        numstills = len(pfd.vstill)
        if numstills==0:
            return self.randomFile()
        excluded = True
        guessed = []
        while excluded:
            i = np.random.randint(numstills)
            if not i in guessed:
                guessed.append(i)
                file = pfd.vstill[i]
                excluded = self.excluded(os.path.basename(file))
            if len(guessed)==numstills:
                # we've guessed all of the files. remove this folder from contention and pick a different one
                self.printFolders.remove(folder)
                return self.randomFile()
        return file
    
def moveMLResult(file:str, serverFolder:str) -> None:
    '''move the segmented machine learning file into the folder'''
    origname = file.replace('vcrop', 'vstill')
    origname = origname.replace('.ome','')
    bn = os.path.basename(origname)
    fn = fh.findFullFN(bn, serverFolder)
    if not os.path.exists(fn):
        raise FileNotFoundError(f'Cannot find folder for {file}')
    folder = os.path.dirname(fn)
    segfolder = os.path.join(folder, 'MLsegment')
    if not os.path.exists(segfolder):
        os.mkdir(segfolder)
    newname = os.path.join(segfolder, bn.replace('vstill', 'MLsegment'))
    shutil.copyfile(file, newname)
    print(f'copied file to {newname}')
    
def moveMLResults(segFolder:str, serverFolder:str) -> list:
    '''move all of the segmented images from segFolder into the appropriate folders in serverFolder'''
    error = []
    for file in os.listdir(segFolder):
        try:
            moveMLResult(os.path.join(segFolder, file), serverFolder)
        except FileNotFoundError:
            error.append(file)
    return error
    