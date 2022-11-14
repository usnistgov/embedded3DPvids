#!/usr/bin/env python
'''Functions for collecting data from stills of single line XS'''

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

#-------------------------------

def filterXSComponents(markers:Tuple, im2:np.ndarray) -> pd.DataFrame:
    '''filter out cross-section components'''
    errorRet = [],[]
    df = vm.markers2df(markers)
    xest = im2.shape[1]/2 # estimated x
    if im2.shape[0]>600:
        yest = im2.shape[0]-300
        dycrit = 200
    else:
        yest = im2.shape[0]/2
        dycrit = im2.shape[0]/2
    df2 = df.copy()
    df2 = df2[(df2.x0>10)&(df2.y0>10)&(df2.x0+df2.w<im2.shape[1]-10)&(df2.y0+df2.h<im2.shape[0]-10)] 
        # remove anything too close to the border
    df2 = df[(abs(df.xc-xest)<100)&(abs(df.yc-yest)<dycrit)] 
        # filter by location relative to expectation and area
    if len(df2)==0:
        df2 = df[(df.a>1000)] # if everything was filtered out, only filter by area
        if len(df2)==0:
            return errorRet
    return df, df2


def singleXSMeasure(im:np.array, im2:np.array, markers:Tuple, attempt:int, s:float, title:str, name:str, diag:bool=False, **kwargs) -> dict:
    '''measure a single xs'''
    errorRet = {}, {}
    if markers[0]==1:
        return errorRet
    roughness = getRoughness(im2, diag=max(0,diag-1))
    df, df2 = filterXSComponents(markers, im2)
    if len(df2)>1 and df2.a.max() < 2*list(df.a.nlargest(2))[1]:
        # largest object not much larger than 2nd largest
        return errorRet
    if len(df2)==0:
        return errorRet
    m = (df2[df2.a==df2.a.max()]).iloc[0] # select largest object
    x0 = int(m['x0'])
    y0 = int(m['y0'])
    w = int(m['w'])
    h = int(m['h'])
    area = int(m['a'])
    xc = m['xc']
    yc = m['yc']
    aspect = h/w # height/width
    boxcx = x0+w/2 # x center of bounding box
    boxcy = y0+h/2 # y center of bounding box
    xshift = (xc-boxcx)/w
    yshift = (yc-boxcy)/h

    if diag:
        # show the image with annotated dimensions
        im2 = cv.cvtColor(im2,cv.COLOR_GRAY2RGB)
        for j, imgi in enumerate([im]):
            cv.rectangle(imgi, (x0,y0), (x0+w,y0+h), (0,0,255), 1)   # bounding box
            cv.circle(imgi, (int(xc), int(yc)), 2, (0,0,255), 2)     # centroid
            cv.circle(imgi, (x0+int(w/2),y0+int(h/2)), 2, (0,255,255), 2) # center of bounding box
        imshow(im, im2)
        plt.title(title)
    units = {'line':'', 'aspect':'h/w', 'xshift':'w', 'yshift':'h', 'area':'px','x0':'px', 'y0':'px', 'w':'px', 'h':'px', 'xc':'px', 'yc':'px', 'roughness':''} # where pixels are in original scale
    retval = {'line':name, 'aspect':aspect, 'xshift':xshift, 'yshift':yshift, 'area':area*s**2, 'x0':x0*s, 'y0':y0*s, 'w':w*s, 'h':h*s, 'xc':xc*s, 'yc':yc*s, 'roughness':roughness}
    return retval, units
    

def xsMeasureIm(im:np.ndarray, s:float, title:str, name:str, acrit:int=100, diag:bool=False, **kwargs) -> Tuple[dict,dict]:
    '''im is imported image. 
    s is is the scaling of the stitched image compared to the raw images, e.g. 0.33 
    title is the title to put on the plot
    name is the name of the line, e.g. xs1
    acrit is the minimum segment size to be considered a cross-section
    '''
    
    im2, markers, attempt = vm.segmentInterfaces(im, acrit=acrit, diag=max(0,diag-1))
    return singleXSMeasure(im, im2, markers, attempt, s, title, name, diag=diag)

def xsMeasure(file:str, diag:bool=False) -> Tuple[dict,dict]:
    '''measure cross-section'''
    name = lineName(file, 'xs')
    im = cv.imread(file)
    if 'I_M' in file or 'I_PD' in file:
        im = vc.imcrop(im, 10)
    # label connected components
    s = 1/fileScale(file)
    title = os.path.basename(file)
    im = vm.normalize(im)
    return xsMeasureIm(im, s, title, name, diag=diag)

#--------



def xs3Measure(file:str, acrit:int=100, diag:int=0, **kwargs) -> Tuple[dict,dict]:
    '''measure cross-section of 3 lines'''
    errorRet = {},{}
    spl = re.split('xs', os.path.basename(file))
    name = re.split('_', spl[0])[-1] + 'xs' + re.split('_', spl[1])[1]
    im = cv.imread(file)
    s = 1/float(fileScale(file))
    title = os.path.basename(file)
    im = vm.normalize(im)
    
    # segment components
    if 'LapRD_LapRD' in file:
        # use more aggressive segmentation to remove leaks
        im2, markers, attempt = vm.segmentInterfaces(im, acrit=acrit, botthresh=75, topthresh=75, diag=max(0,diag-1))
    else:
        im2, markers, attempt = vm.segmentInterfaces(im, acrit=acrit, diag=max(0,diag-1))
    df, df2 = filterXSComponents(markers, im2)
    if len(df2)==0:
        return errorRet
    labels = markers[1]
    for i in list(df.index):
        if i in df2.index:
            labels[labels==i] = 255
        else:
            labels[labels==i] = 0
    labels = labels.astype(np.uint8)

    # find contours
    contours = cv.findContours(labels,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    cnt = np.vstack(contours[0])
    hull = cv.convexHull(cnt)
    
    if diag>0:
        cm = im2.copy()
        cm = cv.cvtColor(cm,cv.COLOR_GRAY2RGB)
        cv.drawContours(cm, [hull], -1, (110, 245, 209), 6)
        cv.drawContours(cm, cnt, -1, (186, 6, 162), 6)
        imshow(cm)
        
    # measure components
    hullArea = cv.contourArea(hull)
    filledArea = df2.a.sum()
    porosity = 1-(filledArea/hullArea)
    
    perimeter = 0
    for cnti in contours[0]:
        perimeter+=cv.arcLength(cnti, True)
    hullPerimeter = cv.arcLength(hull, True)
    excessPerimeter = perimeter/hullPerimeter - 1
    
    x0,y0,w,h = cv.boundingRect(hull)
    aspect = h/w
    
    M = cv.moments(cnt)
    xc = int(M['m10']/M['m00'])
    yc = int(M['m01']/M['m00'])
    boxcx = x0+w/2 # x center of bounding box
    boxcy = y0+h/2 # y center of bounding box
    xshift = (xc-boxcx)/w
    yshift = (yc-boxcy)/h
    
    units = {'line':'', 'aspect':'h/w', 'xshift':'w', 'yshift':'h', 'area':'px','x0':'px', 'y0':'px', 'w':'px', 'h':'px', 'porosity':'', 'excessPerimeter':''} # where pixels are in original scale
    retval = {'line':name, 'aspect':aspect, 'xshift':xshift, 'yshift':yshift, 'area':filledArea*s**2, 'w':w*s, 'h':h*s, 'porosity':porosity, 'excessPerimeter':excessPerimeter}
    return retval, units

    
    
def xsDisturbMeasure(file:str, acrit:int=100, diag:int=0, **kwargs) -> Tuple[dict,dict]:
    '''measure cross-section of single disturbed line'''
    errorRet = {},{}
    spl = re.split('_', re.split('vstill_', os.path.basename(file))[1])
    name = f'{spl[0]}_{spl[1]}'
    im = cv.imread(file)
    h,w,_ = im.shape
    s = 1
    title = os.path.basename(file)
    im = vm.normalize(im)
    pv = printVals(os.path.dirname(file))
    
    # segment components
    hc = 150
    crop = {'y0':hc, 'yf':h-hc, 'x0':170, 'xf':300}
    im = vc.imcrop(im, crop)
    
    if 'water' in pv.ink.base:
        th = 130
    else:
        th = 80
    im2, markers, attempt = vm.segmentInterfaces(im, acrit=acrit, botthresh=th, topthresh=th, diag=max(0,diag-1))  
    retval, units = singleXSMeasure(im, im2, markers, attempt, s, title, name, diag=diag)
    if len(retval)==0:
        return retval, units
    for s in ['x0', 'xc']:
        retval[s] = retval[s]+crop['x0']
    for s in ['y0', 'yc']:
        retval[s] = retval[s] + hc
    return retval, units

def xsDisturbMeasures(folder:str, overwrite:bool=False, **kwargs) -> None:
    '''measure all cross-sections in the folder and export table'''
    if not 'disturbXS' in os.path.basename(folder):
        return
    pfd = fh.printFileDict(folder)
    fn = pfd.newFileName('xsMeasure', '.csv')
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
            m,u = xsDisturbMeasure(os.path.join(folder, files[f'l{i}{s}o']), **kwargs)
            if len(u)>0:
                units = u
                out.append(m)
    df = pd.DataFrame(out)
    
    plainExp(fn, df, units)
