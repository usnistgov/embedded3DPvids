#!/usr/bin/env python
'''Functions for collecting data from stills of single horizontal lines'''

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
from vid_noz_detect import nozData
from metrics_tools import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)



#----------------------------------------------

#-------------------------------------------------

def markHorizOnIm(im2:np.array, row:pd.Series) -> np.array:
    '''mark horizontal element on the image'''
    im2 = cv.rectangle(im2, (int(row['x0']),int(row['y0'])), (int(row['x0']+row['w']),int(row['y0']+row['h'])), (0,0,255), 2)
    im2 = cv.circle(im2, (int(row['xc']), int(row['yc'])), 3, (0,0,255), 3)
    return im2
    

def horizLineMeasure(df:pd.DataFrame, labeled:np.array, im2:np.array, s:float, name:Union[int, str], maxPossibleLen:float, diag:bool=0, distancemm:float=0.603, **kwargs) -> Tuple[dict, dict]:
    '''measure one horizontal line. 
    labeled is an image from connected component labeling
    im2 is the original image
    s is is the scaling of the stitched image compared to the raw images, e.g. 0.33
    j is the line number
    maxPossibleLen is the longest length in mm that should have been extruded
    '''
    tic = time.perf_counter()
    numlines = len(df)
    measures = []
    cmunits = {}
    
    if diag:
        for i,row in df.iterrows():
            im2 = markHorizOnIm(im2, row)

    maxlen0 = df.w.max()
    maxlen = maxlen0*s
    totlen = df.w.sum()*s
    maxarea = df.a.max()*s**2
    totarea = df.a.sum()*s**2
    
    co = {'line':name, 'segments':len(df)
        , 'maxlen':maxlen0*s, 'totlen':df.w.sum()*s
        , 'maxarea':df.a.max()*s**2, 'totarea':int(df.a.sum())*s**2
        }  
    counits = {'line':'', 'segments':''
        , 'maxlen':'px', 'totlen':'px'
        , 'maxarea':'px^2', 'totarea':'px^2'
        }  
    longest = df[df.w==maxlen0] # measurements of longest component
    longest = longest.iloc[0]
    componentMask = (labeled == longest.name).astype("uint8") * 255   # image with line in it
    componentMeasures, cmunits = measureComponent(componentMask, True, s, maxPossibleLen, reverse=(name==1), diag=max(0,diag-1))
    # component measures and co are pre-scaled
    aspect = co['totlen']/componentMeasures['meanT'] # height/width
    r = componentMeasures['meanT']/2
    if co['totlen']>2*r:
        h = co['totlen']
        vest = (h - 2*r)*np.pi*(r)**2 + 4/3*np.pi*r**3 # cylinder + hemisphere endcaps
    else:
        vest = 4/3*np.pi*r**3 # sphere
    units = {'line':'', 'aspect':'h/w'} # where pixels are in original scale
    ret = {**{'line':name, 'aspect':aspect}, **co, **{'vest':vest}, **componentMeasures}
    units = {**units, **counits, **{'vest':'px^3'}, **cmunits}
    if 'nozData' in kwargs and not name[-1]=='o':
        # get displacements
        disps = displacement(componentMask, kwargs['nozData'], 'y', kwargs['crop'], distancemm*kwargs['nozData'].pxpmm, diag=diag-1)
        dispunits = dict([[ii, 'px'] for ii in disps])
        ret = {**ret, **disps}
        units = {**units, **dispunits}
    if diag:
        imshow(im2, labeled, '\n'.join([key+'    '+(val if type(val) is str else "{:.2f}".format(val)) for key,val in ret.items()]))
    return ret, units

#----------------------------
# single line

def splitLines(df0:pd.DataFrame, diag:int=0, margin:float=80, **kwargs) -> list:
    '''split the table of segments into the three horizontal lines. 
    margin is the max allowable vertical distance in px from the line location to be considered part of the line'''
    
    linelocs = [275, 514, 756] # expected positions of lines
    ylocs = [-1000,-1000,-1000] # actual positions of lines
    
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
    dflist = [df0[(df0.yc>yloc-margin)&(df0.yc<yloc+margin)] for yloc in ylocs]
    return dflist


def horizSegment(im0:np.array, progDims:pd.DataFrame, s:float, acrit:float=1000, satelliteCrit:float=0.2, diag:int=0, **kwargs) -> Tuple[pd.DataFrame, dict]:
    '''segment the image and take measurements
    progDims holds timing info about the lines
    s is is the scaling of the stitched image compared to the raw images, e.g. 0.33
    acrit is the minimum segment size in px to be considered part of a line
    satelliteCrit is the min size of segment, as a fraction of the largest segment, to be considered part of a line
    '''
    im2, markers, attempt = vm.segmentInterfaces(im0, diag=max(0,diag-1), removeVert=True, acrit=acrit, **kwargs)
    if len(markers)==0 or markers[0]==1:
        return [], {}, attempt, im2
    labeled = markers[1]
    df = vm.markers2df(markers)
    df = df[df.a>acrit]
    df = df[(df.a>satelliteCrit*df.a.max())]  # eliminate tiny satellite droplets
    if diag:
        im2 = cv.cvtColor(im2,cv.COLOR_GRAY2RGB)
    ret = []
    cmunits = {}
    if len(df)==0:
        return [],{},attempt,im2
    dfsplit = splitLines(df, diag=diag) # split segments into lines
    for j,df in enumerate(dfsplit):
        if len(df)>0:
            maxlen = progDims[progDims.name==f'horiz{j}'].iloc[0]['l']  # length of programmed line
            r,cmu = horizLineMeasure(df, labeled, im2, s, j, maxlen, diag=diag)
            if len(r)>0:
                ret.append(r)
                cmunits = cmu
    return ret, cmunits, attempt, im2
    

def horizMeasure(file:str, progDims:pd.DataFrame, diag:int=0, **kwargs) -> Tuple[pd.DataFrame, dict]:
    '''measure horizontal lines. 
    progDims holds timing info about the lines
    diag=1 to print diagnostics for this function, diag=2 to print this function and the functions it calls'''
    s = 1/fileScale(file)
    im = cv.imread(file)
    im0 = im
    im0 = vm.removeBorders(im0)
    ret, cmunits, attempt, im2 = horizSegment(im0, progDims, s, diag=diag, **kwargs)
    
    if len(ret)==0:
        return [], {}
    if diag:
        imshow(im, im2)
        plt.title(os.path.basename(file))
    units = {'line':'', 'segments':'', 'maxlen':'px', 'totlen':'px', 'maxarea':'px', 'totarea':'px', 'roughness':cmunits['roughness'], 'meanT':cmunits['meanT'], 'stdevT':cmunits['stdevT'], 'minmaxT':cmunits['minmaxT'], 'vest':'px^3'}
    if diag>=2:
        display(pd.DataFrame(ret))
    return pd.DataFrame(ret), units


#----------------------------
# disturb

def removeThreads(thresh:np.array, f:float=0.4, f2:float=0.3, diag:int=0) -> np.array:
    '''remove zigzag threads from bottom left and top right part of binary image'''
    if diag>0:
        thresh0 = thresh.copy()
        thresh0 = cv.cvtColor(thresh0, cv.COLOR_GRAY2BGR)
    h,w0 = thresh.shape
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
            else:
                # mask the bottom on the left side if the left edge is tall
                if thresh[:y, :int(w0*f2)].sum(axis=0).sum(axis=0)>0:
                    thresh[h+y+10:, :int(w0*f)] = 0
            if diag>0:
                if i==1:
                    x = x+right0
                thresh0 = cv.rectangle(thresh0, (x,y), (x+w,y+h), (0,0,255), 2)
    if diag>0:
        imshow(thresh0, thresh)
    return thresh

def horizDisturbMeasure(file:str, acrit:int=2500, diag:int=0, **kwargs) -> Tuple[dict,dict]:
    '''measure disturbed horizontal lines'''
    tic = time.perf_counter()
    errorRet = {},{}
    if not os.path.exists(file):
        raise ValueError(f'File {file} does not exist')
    spl = re.split('_', re.split('vstill_', os.path.basename(file))[1])
    name = f'{spl[0]}_{spl[1]}'        # e.g. V_l0do
    im = cv.imread(file)
    nd = nozData(os.path.dirname(file))   # detect nozzle
    nd.importNozzleDims()
    pv = printVals(os.path.dirname(file), levels=nd.levels, pfd=nd.pfd, fluidProperties=False)
    if not nd.nozDetected:
        raise ValueError(f'No nozzle dimensions in {nd.printFolder}')

    if 'water' in pv.ink.base:
        im = nd.subtractBackground(im, diag=diag-2)  # remove background and nozzle
        im = vm.removeBlack(im)   # remove bubbles and dirt from the image
        im = vm.removeChannel(im,0) # remove the blue channel
    if pv.ink.dye=='red':
        im = nd.maskNozzle(im)
        im = vm.removeChannel(im, 2)   # remove the red channel
    
    h,w,_ = im.shape
    s = 1
    title = os.path.basename(file)

    # segment components
    hc = 0
    if name[-1]=='o':
        # observing
        crop = {'y0':int(h/2), 'yf':int(h*6/6), 'x0':hc, 'xf':w-hc, 'w':w, 'h':h}   # crop the left and right edge to remove zigs
    else:
        # writing
        crop = {'y0':int(h/6), 'yf':int(h*5/6), 'x0':hc, 'xf':w-hc, 'w':w, 'h':h}
    im = vc.imcrop(im, crop)
#     im = vm.removeDust(im)
    im = vm.normalize(im)
    
    if 'water' in pv.ink.base:
        bt = 190
    else:
        bt = 80
        
    im2, markers, attempt = vm.segmentInterfaces(im, acrit=acrit, diag=max(0,diag-1), cutoffTop=0, botthresh=bt, topthresh=bt, removeBorder=False, nozData=nd, crops=crop, eraseMaskSpill=True)
    if attempt<6:
        return errorRet

    if len(markers)==0 or markers[0]==1:
        return errorRet # nothing to measure here
    df = vm.markers2df(markers)
    df = df[(df.a>acrit)]   # remove small segments
    df = df[df.w<crop['xf']-crop['x0']]  # remove segments that are the width of the whole image
    df = df[(df.y0>0)&(df.y0+df.h<(crop['yf']-crop['y0']))]  # remove segments that are on top and bottom border
    im2 = vm.reconstructMask(markers, df)

    im2 = removeThreads(im2, diag=diag-1)       # trim off end threads from zigzags
    im2, markers, _ = vm.filterMarkers(im2, acrit=acrit)  # relabel connected components

    if len(markers)==0 or markers[0]==1:
        return errorRet # nothing to measure here
    df = vm.markers2df(markers)
    df = df[(df.y0>1)&(df.y0+df.h<(crop['yf']-crop['y0']))]  # remove segments that are on top and bottom border
   
    if len(df)==0:
        return errorRet
    retval, units = horizLineMeasure(df, markers[1], im, s, name, w, diag=diag, nozData=nd, crop=crop, distancemm=pv.dEst)
    for s in ['x0', 'xc']:
        if s in retval:
            retval[s] = retval[s]+crop['x0']
    for s in ['y0', 'yc']:
        if s in retval:
            retval[s] = retval[s] + hc

    return retval, units


def horizDisturbMeasures(folder:str, overwrite:bool=False, **kwargs) -> None:
    '''measure all cross-sections in the folder and export table'''
    if not 'disturbHoriz' in os.path.basename(folder):
        return
    pfd = fh.printFileDict(folder)
    fn = pfd.newFileName('horizMeasure', '.csv')
    if os.path.exists(fn) and not overwrite:
        return
    files = {}
    for f in os.listdir(folder):
        if 'vstill' in f:
            files[re.split('_', re.split('vstill_', f)[1])[1]] = f
    
    units = {}
    out = []
    for i in range(4):
        for s in ['w', 'd']:
            for s2 in ['', 'o']:
                m,u = horizDisturbMeasure(os.path.join(folder, files[f'l{i}{s}{s2}']), **kwargs)
                if len(u)>len(units):
                    units = u
                out.append(m)
    df = pd.DataFrame(out)
    
    plainExp(fn, df, units)