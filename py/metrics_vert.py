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

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)



#----------------------------------------------

#------------------------------------


def vertSegmentMeasure(df2:pd.DataFrame, markers:Tuple, im:np.array, im2:np.array, s:float, name:str, file:str, maxlen:float=20000, diag:int=0, distancemm:float=0.603, **kwargs) -> Tuple[dict,dict]:
    '''given a dataframe and marker map, measure filament segments'''
    filI = df2.a.idxmax() # index of filament label, largest remaining object
    component = df2.loc[filI]
    inline = df2[(df2.x0>component['x0']-50)&(df2.x0<component['x0']+50)] # other objects inline with the biggest object

    # get combined mask of all objects in line
    componentMask = vm.reconstructMask(markers, inline)

    x0 = int(inline.x0.min())  # unscaled
    y0 = int(inline.y0.min())  # unscaled
    w = int(inline.w.max())    # unscaled
    h = int(inline.h.sum())    # unscaled
    xc = int(sum(inline.a * inline.xc)/sum(inline.a))
    yc = int(sum(inline.a * inline.yc)/sum(inline.a))
    co = {'area':int(inline.a.sum())*s**2
                 , 'x0':x0*s, 'y0':y0*s, 'w':w*s, 'h':h*s
                 , 'xc':xc*s, 'yc':yc*s, 'segments':len(inline)}    
    componentMeasures, cmunits = measureComponent(componentMask, False, s, maxlen=maxlen, reverse=True, diag=max(0,diag-1))
    
    if len(componentMeasures)==0:
        return {}, {}
    
    # component measures and co are pre-scaled
    aspect = co['h']/componentMeasures['meanT'] # height/width
    r = componentMeasures['meanT']/2
    if co['h']>2*r:
        vest = (co['h'] - 2*r)*np.pi*(r)**2 + 4/3*np.pi*r**3 # cylinder + hemisphere endcaps
    else:
        vest = 4/3*np.pi*r**3 # sphere
    units = {'line':'', 'aspect':'h/w', 'area':'px'
             ,'x0':'px', 'y0':'px', 'w':'px', 'h':'px'
             , 'xc':'px', 'yc':'px', 'segments':'', 'vest':'px^3'} # where pixels are in original scale
    ret = {**{'line':name, 'aspect':aspect}, **co, **{'vest':vest}, **componentMeasures}
    units = {**units, **cmunits}
    if 'nozData' in kwargs and not name[-1]=='o':
        # get displacements
        disps = displacement(componentMask, kwargs['nozData'], 'z', kwargs['crop'], distancemm*kwargs['nozData'].pxpmm)
        dispunits = dict([[ii, 'px'] for ii in disps])
        ret = {**ret, **disps}
        units = {**units, **dispunits}
    if diag:
        im2 = cv.cvtColor(componentMask,cv.COLOR_GRAY2RGB)
        im2 = cv.rectangle(im2, (x0,y0), (x0+w,y0+h), (0,0,255), 2)
        im2 = cv.circle(im2, (xc, yc), 2, (0,0,255), 2)
        imshow(im, im2, '\n'.join([key+'    '+(val if type(val) is str else "{:.2f}".format(val)) for key,val in ret.items()]))
        plt.title(os.path.basename(file))
    return ret, units


def vertSegment(im:np.array, s:float, maxlen:float, name:str, file:str, diag:int, acrit:int=2500, **kwargs) -> Tuple[pd.DataFrame, dict, dict, float, pd.Series, np.array]:
    '''segment out the filament and measure it
    s is is the scaling of the stitched image compared to the raw images, e.g. 0.33
    if maxlen>0, maxlen is the maximum length of the expected line. anything outside is leaks
    acrit is the minimum segment size to be considered a part of a line
    '''
    im2, markers, attempt = vm.segmentInterfaces(im, acrit=acrit, diag=max(0,diag-1))
    errorRet = {}, {}, {}, im2
    if len(markers)==0 or markers[0]==1:
        return errorRet # nothing to measure here
    df = vm.markers2df(markers)
    df = df[(df.a>acrit)]
        # remove anything too small
    df2 = df[(df.x0>10)&(df.y0>10)&(df.x0+df.w<im.shape[1]-10)&(df.y0+df.h<im.shape[0]-10)] 
        # remove anything too close to the border
    if len(df2)==0:
        return errorRet
    return vertSegmentMeasure(df2, markers, im, im2, s, name, file, maxlen=maxlen, diag=diag)
    
    
def vertMeasure(file:str, progDims:pd.DataFrame, diag:int=0, **kwargs) -> Tuple[dict,dict]:
    '''measure vertical lines. progDims holds timing info about the lines'''
    s = 1/fileScale(file)
    name = lineName(file, 'vert')
    im = cv.imread(file)
    maxlen = progDims[progDims.name==(f'vert{int(name)}')].iloc[0]['l']
    maxlen = int(maxlen/s)
    # label connected components
    
    return vertSegment(im, s, maxlen, name, file, diag, **kwargs)



def vertDisturbMeasure(file:str, acrit:int=2500, diag:int=0, **kwargs) -> Tuple[dict,dict]:
    '''measure disturbed vertical lines'''
    errorRet = {},{}
    if not os.path.exists(file):
        raise ValueError(f'File {file} does not exist')
    spl = re.split('_', re.split('vstill_', os.path.basename(file))[1])
    name = f'{spl[0]}_{spl[1]}'        # e.g. V_l0do
    im = cv.imread(file)
    nd = nozData(os.path.dirname(file))   # detect nozzle
    nd.importNozzleDims()
    pv = printVals(os.path.dirname(file))
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
        crop = {'y0':hc, 'yf':h-hc, 'x0':200, 'xf':nd.xL+20, 'w':w, 'h':h}
    else:
        # writing
        crop = {'y0':hc, 'yf':h-hc, 'x0':nd.xL-100, 'xf':nd.xR+100, 'w':w, 'h':h}
    im = vc.imcrop(im, crop)
#     im = vm.removeDust(im)
    im = vm.normalize(im)
    
    if 'water' in pv.ink.base:
        bt = 190
    else:
        bt = 80
    im2, markers, attempt = vm.segmentInterfaces(im, acrit=acrit, diag=max(0,diag-1), cutoffTop=0, botthresh=bt, topthresh=bt, removeBorder=False, nozData=nd, crops=crop)
    if len(markers)==0 or markers[0]==1:
        return errorRet # nothing to measure here
    df = vm.markers2df(markers)
    df = df[(df.a>acrit)]   # remove small segments
    df = df[df.w<crop['xf']-crop['x0']]  # remove segments that are the width of the whole image
    df = df[(df.x0>0)&(df.x0+df.w<(crop['xf']-crop['x0']))]  # remove segments that are on left and right border
    if len(df)==0:
        return errorRet
    
    retval, units = vertSegmentMeasure(df, markers, im, im2, s, name, file, maxlen=h, diag=diag, nozData=nd, crop=crop, distancemm=pv.dEst)
    for s in ['x0', 'xc']:
        retval[s] = retval[s]+crop['x0']
    for s in ['y0', 'yc']:
        retval[s] = retval[s] + hc
    return retval, units

def vertDisturbMeasures(folder:str, overwrite:bool=False, **kwargs) -> None:
    '''measure all cross-sections in the folder and export table'''
    if not 'disturbVert' in os.path.basename(folder):
        return
    pfd = fh.printFileDict(folder)
    fn = pfd.newFileName('vertMeasure', '.csv')
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
                m,u = vertDisturbMeasure(os.path.join(folder, files[f'l{i}{s}{s2}']), **kwargs)
                if len(u)>len(units):
                    units = u
                out.append(m)
    df = pd.DataFrame(out)
    
    plainExp(fn, df, units)
    
    

    
    
def vertDisturbSummary(folder:str, overwrite:bool=False, **kwargs) -> None:
    '''summarize vertical measurements in the folder and export table'''
    if not 'disturbVert' in os.path.basename(folder):
        return {},{}
    pfd = fh.printFileDict(folder)
    fn = pfd.newFileName('vertSummary', '.csv')
    if os.path.exists(fn) and not overwrite:
        out,u = plainImDict(fn, unitCol=1, valCol=2)
        return out,u
    if not hasattr(pfd, 'vertMeasure'):
        vertDisturbMeasures(folder, **kwargs)
    if not hasattr(pfd, 'vertMeasure'):
        return {},{}
    
    df, du = plainIm(pfd.vertMeasure, ic=0)
    pv = printVals(folder)
    pxpmm = pv.pxpmm
    mr, mu = pv.metarow()
    
    # find changes between observations
    aves = {}
    aveunits = {}
    for num in range(4):
        if num in [0,2]:
            ltype = 'bot'
        else:
            ltype = 'top'
        wodf = df[df.line==f'V_l{num}wo']
        dodf = df[df.line==f'V_l{num}do']
        if len(wodf)==1 and len(dodf)==1:
            wo = wodf.iloc[0]
            do = dodf.iloc[0]
            
            for s in ['segments', 'roughness']:
                try:
                    addValue(aves, aveunits, f'{ltype}_delta_{s}', difference(do, wo, s), du[s])
                except ValueError:
                    pass
            for s in ['h', 'meanT']:
                try:
                    addValue(aves, aveunits, f'{ltype}_delta_{s}_n', difference(do, wo, s)/wo[s], '')
                except ValueError:
                    pass
            for s in ['xc']:
                try:
                    addValue(aves, aveunits, f'{ltype}_delta_{s}_n', difference(do, wo, s)/pxpmm/pv.dEst, 'dEst')
                except ValueError:
                    pass

    # find displacements
    disps = {}
    dispunits = {}
    dlist = ['dxprint', 'dxf', 'space_at', 'space_a']
    for num in range(4):
        wdf = df[df.line==f'V_l{num}w']
        ddf = df[df.line==f'V_l{num}d']
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
                        si = f'{ltype}_{sii}_{s}'
                        if si not in ['w_dxf', 'w_space_a', 'w_space_at']:
                            val = v[s]/pxpmm/pv.dEst
                            addValue(disps, dispunits, si, val, 'dEst')
                        
    ucombine = {**aveunits, **dispunits} 
    out = {}
    units = {}
    lists = {**aves, **disps}
    for key,val in lists.items():
        convertValue(key, val, ucombine, pxpmm, units, out)

    out = {**mr, **out}
    units = {**mu, **units}

    plainExpDict(fn, out, units=units)
    
    return out,units


def vertDisturbSummariesRecursive(topFolder:str, overwrite:bool=False, **kwargs) -> None:
    '''recursively go through folders'''
    out = []
    units = {}
    if not fh.isPrintFolder(topFolder):
        for f in os.listdir(topFolder):
            summaries, u = vertDisturbSummariesRecursive(os.path.join(topFolder, f), overwrite=overwrite, **kwargs)
            if len(u)>len(units):
                units = u
            out = out + summaries
        return out, units
    try:
        summary, units = vertDisturbSummary(topFolder, overwrite=overwrite, **kwargs)
    except Exception as e:
        print(f'Error in {topFolder}: {e}')
    else:
        if len(summary)>0:
            return [summary], units
        else:
            return [], {}
    

def vertDisturbSummaries(folder:str, exportFolder:str, overwrite:bool=False, **kwargs) -> None:
    '''measure all cross-sections in the folder and export table'''
    out, units  = vertDisturbSummariesRecursive(folder, overwrite=overwrite, **kwargs)
    df = pd.DataFrame(out)
    fn = os.path.join(exportFolder, 'vertDisturbSummaries.csv')
    plainExp(fn, df, units, index=False)