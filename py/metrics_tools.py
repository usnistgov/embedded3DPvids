#!/usr/bin/env python
'''Functions for collecting data from stills of single lines'''

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
from pic_stitch_bas import stitchSorter
from file_handling import isSubFolder, fileScale
import im_crop as vc
import im_morph as vm
from tools.imshow import imshow
from tools.plainIm import *
from val_print import *
from val_progDim import *
from vid_noz_detect import nozData

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)


#----------------------------------------------

def openImageInPaint(folder:str, st:str, i:int) -> None:
    '''open the image in paint. this is useful for erasing smudges or debris that are erroneously detected by the algorithm as filaments'''
    file = stitchFile(folder, st, i)
    if not os.path.exists(file):
        return
    subprocess.Popen([r'C:\Windows\System32\mspaint.exe', file]);



def sem(l:list) -> float:
    l = np.array(l)
    l = l[~np.isnan(l)]
    if len(l)==0:
        return np.nan
    return np.std(l)/np.sqrt(len(l))
    

def getRoughness(componentMask:np.array, diag:int=0) -> float:
    '''measure roughness as perimeter of object / perimeter of convex hull. 
    componentMask is a binarized image of just one segment'''
    contours = cv.findContours(componentMask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    if int(cv.__version__[0])>=4:
        contours = contours[0]
    else:
        contours = contours[1]
    if len(contours)==0:
        return -1
    contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True) # select the largest contour
    cnt = contours[0]
    cnt = cv.approxPolyDP(cnt, 1, True) # smooth the contour by 1 px
    perimeter = cv.arcLength(cnt,True)
    if perimeter==0:
        return {}, {}
    hull = cv.convexHull(cnt)
    hullperimeter = cv.arcLength(hull,True)
    roughness = perimeter/hullperimeter-1  # how much extra perimeter there is compared to the convex hull
    if diag:
        # show annotated image
        cm = componentMask.copy()
        cm = cv.cvtColor(cm,cv.COLOR_GRAY2RGB)
        cv.drawContours(cm, [hull], -1, (110, 245, 209), 6)
        cv.drawContours(cm, cnt, -1, (186, 6, 162), 6)
        
        x,y,w,h = cv.boundingRect(cnt)
        cm = cm[max(y-5,0):min(y+h+5, cm.shape[0]),max(x-5, 0):min(x+w+5, cm.shape[1])]
        imshow(cm)
    return roughness

def widthInRow(row:list) -> int:
    '''distance between first and last 255 value of row'''
    first,last = bounds(row)
    return last-first

def boundsInArray(arr:np.array) -> np.array:
    '''left and right bounds in the array'''
    if arr.sum()==0:
        return []
    a2 = np.stack(np.where(arr)).transpose()
    idx = np.where(np.diff(a2[:,0])!=0)[0]+1
    a3 = np.split(a2,list(idx))
    
    return np.array([[i[0,1],i[-1,1]] for i in a3])
    

def widthsInArray(arr:np.array) -> list:
    '''get the distance between first and last nonzero value of each row'''
    if arr.sum()==0:
        return []
    a2 = np.stack(np.where(arr)).transpose()  # get positions of 255
    idx = np.where(np.diff(a2[:,0])!=0)[0]+1  # find changes in row
    a3 = np.split(a2,list(idx))               # split into rows
    return [i[-1,1]-i[0,1] for i in a3]              # get distance between first and last
    

def bounds(row:list) -> Tuple[int,int]:
    '''get position of first and last 255 value in row'''
    if not type(row) is list:
        row = list(row)
    if not 255 in row:
        return -1, -1
    last = len(row) - row[::-1].index(255) 
    first = row.index(255)
    return first,last



def meanBounds(chunk:np.array, rows:bool=True) -> Tuple[float,float]:
    '''get average bounds across rows or columns'''
    if not rows:
        chunk = chunk.transpose()
    b = boundsInArray(chunk)
    if len(b)==0:
        return -1,-1
    x0 = np.mean(b[:,0])
    xf = np.mean(b[:,1])
    return x0,xf



        
def closestIndex(val:float, l1:list) -> int:
    '''index of closest value in list l1 to value val'''
    l2 = [abs(x-val) for x in l1]
    return l2.index(min(l2))

def addValue(results:dict, units:dict, name:str, value:Any, unit:str) -> None:
    '''add the result to the results dict list and units dict'''
    if name in results:
        results[name].append(value)
    else:
        results[name] = [value]
        units[name] = unit
        
def difference(do:pd.Series, wo:pd.Series, s:str) -> float:
    '''get difference between values'''
    if hasattr(do, s) and hasattr(wo, s) and not pd.isna(do[s]) and not pd.isna(wo[s]):
        return do[s]-wo[s]
    else:
        raise ValueError('No value detected')
        
def convertValue(key:str, val:list, units_in:dict, pxpmm:float, units_out:dict, vals_out:dict) -> Tuple:
    '''convert the values from px to mm'''
    uke = units_in[key]
    if uke=='px':
        c = 1/pxpmm
        u2 = 'mm'
    elif uke=='px^2':
        c = 1/pxpmm**2
        u2 = 'mm^2'
    elif uke=='px^3':
        c = 1/pxpmm**3
        u2 = 'mm^3'
    else:
        c = 1
        u2 = uke
    units_out[key] = u2
    units_out[f'{key}_SE'] = u2
    vals_out[key] = np.mean(val)*c
    vals_out[f'{key}_SE'] = sem(val)*c
    
def calcVest(h:float, r:float) -> float:
    '''estimate the volume of an object, assuming it is a cylinder with spherical end caps'''
    if h>2*r:
        vest = (h - 2*r)*np.pi*(r)**2 + 4/3*np.pi*r**3 # cylinder + hemisphere endcaps
    else:
        vest = 4/3*np.pi*r**2*(h/2) # ellipsoid
    return vest
    
    
class metricSegment:
    '''collects data about fluid segments in an image'''
    
    def __init__(self, file:str, diag:int=0, acrit:int=2500, **kwargs):
        self.file = file
        if not os.path.exists(self.file):
            raise ValueError(f'File {self.file} does not exist')
        self.scale = 1/fileScale(self.file)
        self.im = cv.imread(self.file)
        self.acrit = acrit
        self.diag = diag
        self.stats = {}
        self.units = {}
        
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
            col = col+1
            if col==cols:
                col = 0
                out = out + '\n'
            else:
                out = out + '   '
        return out  
    

        
    def measureComponent(self, horiz:bool=True, reverse:bool=False, emptiness:bool=True, diag:int=0) -> Tuple[dict,dict]:
        '''measure parts of a segmented fluid. 
        horiz = True to get variation along length of horiz line. False to get variation along length of vertical line.
        scale is the scaling of the stitched image compared to the raw images, e.g. 0.33 
        if maxlen>0, maxlen is the maximum length of the expected line in px. anything outside is leaks
        reverse=True to measure from top or right, false to measure from bottom or left'''
        for s in ['componentMask', 'scale', 'maxlen']:
            if not hasattr(self, s):
                raise ValueError(f'{s} undefined for {self.file}')
        
        errorRet = {}, {}
        roughness = getRoughness(self.componentMask, diag=max(0,diag-1))
        if horiz:
    #         sums = [sum(i)/255 for i in componentMask.transpose()]            # total number of pixels per row
            sums = self.componentMask.sum(axis=0)/255
            widths = widthsInArray(self.componentMask.transpose())
        else:
    #         sums = [sum(i)/255 for i in self.componentMask]
            sums = self.componentMask.sum(axis=1)/255
            widths = widthsInArray(self.componentMask)
        sums = list(filter(lambda i:i>0, sums)) # remove empty rows
        if len(sums)==0:
            return errorRet
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
        if emptiness:
            emptiness = 1-sum(sums)/sum(widths)     # how much of the middle of the component is empty
        vest = sum([np.pi*(r/2)**2 for r in sums])
        if len(leaks)>0:
            vleak = sum([np.pi*(r/2)**2 for r in leaks])
        else:
            vleak = 0
        meant = np.mean(sums)                       # mean line thickness
        midrange = sums[int(meant/2):-int(meant/2)] # remove endcaps
        if len(midrange)>0:
            stdev = np.std(midrange)/meant               # standard deviation of thickness normalized by mean
            minmax = (max(midrange)-min(midrange))/meant # total variation in thickness normalized by mean
        else:
            stdev = ''
            minmax = ''
        units = {'roughness':'', 'emptiness':'', 'meanT':'px', 'stdevT':'meanT', 'minmaxT':'meanT', 'vintegral':'px^3', 'vleak':'px^3'}
        retval = {'roughness':roughness, 'emptiness':emptiness, 'meanT':meant*self.scale, 'stdevT':stdev, 'minmaxT':minmax, 'vintegral':vest*self.scale**3, 'vleak':vleak*self.scale**3}
        return retval, units
    
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
                out[tt[2]] = dd[tt[0]]-dd[tt[1]]-1    # displacement of left side of filament between above and below
        if dd['x0b']>0 and dd['xfb']>0:
            out['dxprint'] = (dd['x0b']+dd['xfb'])/2-dd['mid']   # distance between center of nozzle and center of filament below. positive means filament is right of nozzle
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
        dd['left'] = nd.xL - crop['x0'] - 10   # nozzle cover was 10 extra pixels to left and right
        dd['right'] = nd.xR - crop['x0'] + 10
        mid = (dd['left']+dd['right'])/2
        dd['mid'] = mid
        
        if direc=='z':
            return self.zdisplacement(dd, distance, size, diag=diag)
        elif direc=='y':
            return self.ydisplacement(dd, distance, size, diag=diag)
        else:
            raise ValueError(f'Unexpected direc {direc} given to displacement. Value should be y or z')
    
    def lineName(self) -> None:
        '''get the name of a singleDisturb, or tripleLine line'''
        if not 'vstill' in self.file:
            raise ValueError(f'Cannot determine line name for {self.file}')
        spl = re.split('_', re.split('vstill_', os.path.basename(self.file))[1])
        self.name = f'{spl[0]}_{spl[1]}'
        self.tag = spl[1]
        
    def adjustForCrop(self, crop:dict) -> None:
        '''adjust the stats for cropping'''
        for si in ['x', 'y']:
            for s in ['0', 'c']:
                ss = f'{si}{s}'
                if ss in self.stats and f'{si}0' in crop:
                    self.stats[ss] = self.stats[ss]+crop[f'{si}0']*self.scale
                    
    
    
        
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
    
    def __init__(self, file:str, diag:int=0, acrit:int=2500, **kwargs):
        super().__init__(file, diag=diag, acrit=acrit, **kwargs)
        self.scale = 1
        self.nd = nozData(os.path.dirname(file))   # detect nozzle
        self.pfd = self.nd.pfd
        self.pv = printVals(os.path.dirname(file), levels=self.nd.levels, pfd = self.pfd, fluidProperties=False)
        self.title = os.path.basename(self.file)
        self.lineName()
        self.measure()
        
    def getProgDims(self):
        '''initialize the progDims object'''
        if not hasattr(self, 'pg'):
            self.pg  = getProgDimsPV(self.pv)


#----------------------------------
    
    
class metricSummary:
    '''holds data and functions for handling metric summary tables'''
    
    def __init__(self, file:str):
        self.file = file
        
    def importStillsSummary(self, diag:bool=False) -> pd.DataFrame:
        self.ss, self.u = plainIm(self.file)
        
        
    def addRatios(self, ss:pd.DataFrame, startName:str, varlist = ['Ca', 'dPR', 'dnorm', 'We', 'Oh', 'Bm'], operator:str='Prod') -> pd.DataFrame:
        '''add products and ratios of nondimensional variables. operator could be Prod or Ratio'''
        k = ss.keys()
        idx = int(np.argmax(k==startName))
        for j, s2 in enumerate(varlist):
            xvar =  f'{s2}{operator}'
            if not xvar in ss:
                if not f'ink_{s2}' in ss or not  'sup_{s2}' in ss:
                    ss = flipInv(ss)
                if operator=='Prod':
                    ss.insert(idx, xvar, ss[f'ink_{s2}']*ss[f'sup_{s2}'])
                elif operator=='Ratio':
                    ss.insert(idx, xvar, ss[f'ink_{s2}']/ss[f'sup_{s2}'])
                idx+=1
        return ss

    def addLogs(self, ss:pd.DataFrame, startName:str, varlist:List[str]) -> pd.DataFrame:
        '''add log values for the list of variables to the dataframe'''
        k = ss.keys()
        idx = int(np.argmax(k==startName))
        for j, s2 in enumerate(varlist):
            xvar = f'{s2}_log'
            if not xvar in s2:
                ss.insert(idx, xvar, np.log10(ss[s2]))
                idx+=1
        return ss
    
    def printStillsKeys(self, ss:pd.DataFrame) -> None:
        '''sort the keys into dependent and independent variables and print them out'''
        k = ss.keys()
        k = k[~(k.str.endswith('_SE'))]
        k = k[~(k.str.endswith('_N'))]
        idx = self.idx0(k)
        controls = k[:idx]
        deps = k[idx:]
        print(f'Independents: {list(controls)}')
        print()
        print(f'Dependents: {list(deps)}')
        
    def idx(self, k:list, name:str) -> int:
        if name in k:
            return int(np.argmax(k==name))
        else:
            return 1
        
    def idx0(self, k:list) -> int:
        '''get the index of the first dependent variable'''
        return self.idx(k, self.firstDepCol())
    
#----------------------------------------------

class disturbMeasures:
    '''for a folder, measure the disturbed lines'''
    
    def __init__(self, folder:str, overwrite:bool=False, **kwargs) -> None:
        self.folder = folder
        self.overwrite = overwrite
        self.pfd = fh.printFileDict(folder)
        self.kwargs = kwargs
        
        
    def measure(self, lines:List[str], func) -> int:
        '''after initialize file name, measure all lines'''
        if not hasattr(self, 'fn'):
            raise ValueError(f'Missing export file name in {self.folder}')
        if os.path.exists(self.fn) and not self.overwrite:
            return 1
        
        files = {}
        for f in os.listdir(self.folder):
            if 'vstill' in f:
                files[re.split('_', re.split('vstill_', f)[1])[1]] = f

        self.du = {}
        out = []
        
        for l in lines:
            m, u = func(file, pfd=self.pfd, **kwargs)
            if len(u)>len(self.du):
                self.du = u
            out.append(m)
        self.df = pd.DataFrame(out)

        plainExp(self.fn, self.df, self.du)
        return 0
        
    def summaryHeader(self, lineType:str) -> int:
        '''top matter for summaries. return 0 if we already have a summary, return 1 if we have measurements, return 2 if we cannot proceed'''
        retval = 2
        self.summaryFn = self.pfd.newFileName(f'{lineType}Summary', '.csv')
        if os.path.exists(self.summaryFn) and not self.overwrite:
            self.summary, self.summaryUnits = plainImDict(self.summaryFn, unitCol=1, valCol=2)
            return 0
        if not hasattr(self, 'df'):
            self.pfd = fh.printFileDict(self.folder)
            measureFile = f'{lineType}Measure'
            if hasattr(self.pfd, measureFile):
                self.df, self.du = plainIm(getattr(self.pfd, measureFile), ic=0)
                retval = 1
            else:
                self.measureFolder()
                if hasattr(self, 'df'):
                    retval = 1
                else:
                    return 2
        self.pv = printVals(self.folder)
        self.pxpmm = self.pv.pxpmm
        self.mr, self.mu = self.pv.metarow()
        return retval
        
        
    def convertValuesAndExport(self, ucombine:dict, lists:dict):
        '''find changes between ovservations'''
        out = {}
        units = {}
        for key,val in lists.items():
            convertValue(key, val, ucombine, self.pxpmm, units, out)
        self.summary = {**self.mr, **out}
        self.summaryUnits = {**self.mu, **units}
        plainExpDict(self.summaryFn, self.summary, units=self.summaryUnits)
        

        
        
class summaries:
    '''recursively create summaries'''
    
    def __init__(self, topFolder:str, measureFunc, overwrite:bool=False, **kwargs):
        self.topFolder = topFolder
        self.overwrite = overwrite
        self.measureFunc = measureFunc
        self.out = []
        self.units = {}
        self.recurse(topFolder, **kwargs)
        
    def recurse(self, topFolder:str, **kwargs):
        if not fh.isPrintFolder(topFolder):
            for f in os.listdir(topFolder):
                self.recurse(os.path.join(topFolder, f), **kwargs)
            return
        try:
            summary, units = self.measureFunc(topFolder, overwrite=self.overwrite, **kwargs)
        except Exception as e:
            print(f'Error in {topFolder}: {e}')
            traceback.print_exc()
        else:
            if len(summary)>0:
                if len(units)>len(self.units):
                    self.units = units
                self.out.append(summary)

    def export(self, fn:str) -> None:
        df = pd.DataFrame(self.out)
        plainExp(fn, df, self.units, index=False)