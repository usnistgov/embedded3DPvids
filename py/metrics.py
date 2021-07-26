#!/usr/bin/env python
'''Functions for plotting video data. Adapted from https://github.com/usnistgov/openfoamEmbedded3DP'''

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

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
from stitchBas import fileList
from fileHandling import isSubFolder
import vidCrop as vc
import vidMorph as vm
from imshow import imshow

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)

# info
__author__ = "Leanne Friedrich"
__copyright__ = "This data is publicly available according to the NIST statements of copyright, fair use and licensing; see https://www.nist.gov/director/copyright-fair-use-and-licensing-statements-srd-data-and-software"
__credits__ = ["Leanne Friedrich"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Leanne Friedrich"
__email__ = "Leanne.Friedrich@nist.gov"
__status__ = "Development"

#----------------------------------------------

def fileScale(file:str) -> str:
    '''get the scale from the file name'''
    try:
        scale = float(re.split('_', os.path.basename(file))[-2])
    except:
        return 1
    return scale

def lineName(file:str, tag:str) -> float:
    '''get the number of the line from the file name based on tag, e.g. 'vert', 'horiz', 'xs'. '''
    spl = re.split('_',os.path.basename(file))
    for st in spl:
        if tag in st:
            return float(st.replace(tag, ''))
    return -1

def xsMeasure(file:str, diag:bool=False) -> Tuple[dict,dict]:
    '''measure cross-section'''
    name = lineName(file, 'xs')
    im = cv.imread(file)
    im = vc.imcrop(im, 10)
    im2, markers = vm.segmentInterfaces(im)
    if markers[0]==1:
        return {}, {}
    filI = np.argmax(markers[2][1:,4])+1 # index of filament label, 2nd largest object
    dims = markers[2][filI]
    centroid = markers[3][filI]
    w = dims[2]
    h = dims[3]
    x0 = dims[0]
    y0 = dims[1]
    aspect = h/w # height/width
    boxcx = x0+w/2 # x center of bounding box
    boxcy = y0+h/2 # y center of bounding box
    xshift = (centroid[0]-boxcx)/w
    yshift = (centroid[1]-boxcy)/h
    area = dims[4]
    if diag:
        im2 = cv.cvtColor(im2,cv.COLOR_GRAY2RGB)
        im2 = cv.rectangle(im2, (x0,y0), (x0+w,y0+h), (0,0,255), 2)
        im2 = cv.circle(im2, (int(centroid[0]), int(centroid[1])), 2, (0,0,255), 2)
        imshow(im, im2)
    units = {'line':'', 'aspect':'h/w', 'xshift':'w', 'yshift':'h', 'area':'px','x0':'px', 'y0':'px', 'w':'px', 'h':'px', 'xc':'px', 'yc':'px'} # where pixels are in original scale
    s = 1/fileScale(file)
    retval = {'line':name, 'aspect':aspect, 'xshift':xshift, 'yshift':yshift, 'area':area*s, 'x0':x0*s, 'y0':y0*s, 'w':w*s, 'h':h*s, 'xc':centroid[0]*s, 'yc':centroid[1]*s}
    return retval, units


def measureComponent(componentMask:np.array, horiz:bool, scale:float) -> Tuple[dict,dict]:
    '''measure parts of a segmented fluid. horiz = True to get variation along length of horiz line. False to get variation along length of vertical line.'''
    contours = cv.findContours(componentMask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    cnt = contours[1][0]
    perimeter = cv.arcLength(cnt,True)
    if perimeter==0:
        return {}, {}
    hull = cv.convexHull(cnt)
    hullperimeter = cv.arcLength(hull,True)
    
    roughness = perimeter/hullperimeter
    if horiz:
        sums = [sum(i)/255 for i in componentMask.transpose()]
    else:
        sums = [sum(i)/255 for i in componentMask]
    sums = list(filter(lambda i:i>0, sums))
    meant = np.mean(sums) # mean line thickness
    midrange = sums[int(meant/2):-int(meant/2)] # remove endcaps
    if len(midrange)>0:
        stdev = np.std(midrange)/meant # standard deviation of thickness normalized by mean
        minmax = (max(midrange)-min(midrange))/meant # total variation in thickness normalized by mean
    else:
        stdev = ''
        minmax = ''
    units = {'roughness':'', 'meanT':'px', 'stdevT':'meanT', 'minmaxT':'meanT'}
    retval = {'roughness':roughness, 'meanT':meant*scale, 'stdevT':stdev, 'minmaxT':minmax}
    return retval, units
    
def vertMeasure(file:str, diag:bool=False) -> Tuple[dict,dict]:
    '''measure vertical lines'''
    name = lineName(file, 'vert')
    s = 1/fileScale(file)
    im = cv.imread(file)
    
    # white out the top 1% of the image
    im = vc.imcrop(im, 10)
    imtop = int(im.shape[0]*0.01)
    im[0:imtop, :] = np.ones(im[0:imtop, :].shape)*255 
    
    # label connected copmonents
    im2, markers = vm.segmentInterfaces(im)
    if markers[0]==1:
        return {}, {} # nothing to measure here
    filI = np.argmax(markers[2][1:,4])+1 # index of filament label, 2nd largest object
    componentMask = (markers[1] == filI).astype("uint8") * 255
    componentMeasures, cmunits = measureComponent(componentMask, False, s)
    dims = markers[2][filI]
    centroid = markers[3][filI]
    w = dims[2]
    h = dims[3]
    x0 = dims[0]
    y0 = dims[1]
    aspect = h/w # height/width
    area = dims[4]
    units = {'line':'', 'aspect':'h/w', 'area':'px','x0':'px', 'y0':'px', 'w':'px', 'h':'px', 'xc':'px', 'yc':'px'} # where pixels are in original scale
    ret = {'line':name, 'aspect':aspect, 'area':area*s, 'x0':x0*s, 'y0':y0*s, 'w':w*s, 'h':h*s, 'xc':centroid[0]*s, 'yc':centroid[1]*s}
    ret = {**ret, **componentMeasures}
    units = {**units, **cmunits}
    if diag:
        im2 = cv.cvtColor(im2,cv.COLOR_GRAY2RGB)
        im2 = cv.rectangle(im2, (x0,y0), (x0+w,y0+h), (0,0,255), 2)
        im2 = cv.circle(im2, (int(centroid[0]), int(centroid[1])), 2, (0,0,255), 2)
        imshow(im, im2)
    return ret, units

def horizMeasure(file:str, diag:bool=False) -> Tuple[pd.DataFrame, dict]:
    '''measure horizontal lines'''
    s = 1/fileScale(file)
    im = cv.imread(file)
    im2 = im
    im2[0, 0] = np.zeros(im2[0, 0].shape)
    im2, markers = vm.segmentInterfaces(im2)
    if markers[0]==1:
        return [], {}
    labeled = markers[1]
    boxes = pd.DataFrame(markers[2], columns=['x0', 'y0', 'w', 'h', 'area'])
    centroids = pd.DataFrame(markers[3], columns=['xc', 'yc'])
    linelocs = [275, 514, 756] # expected positions of lines
    margin = 50
    ret = []
    if diag:
        im2 = cv.cvtColor(im2,cv.COLOR_GRAY2RGB)
    for j,y in enumerate(linelocs):
        lines = centroids[(centroids.yc>y-margin)&(centroids.yc<y+margin)&(boxes.area>10)]
        numlines = len(lines)
        measures = []
        for i,line in lines.iterrows():
            componentMask = (labeled == i).astype("uint8") * 255
            w = boxes.loc[i,'w']
            h = boxes.loc[i,'h']
            area = boxes.loc[i,'area']
            box = {'i':i, 'w':w, 'h':h, 'area':area}
            componentMeasures, cmunits = measureComponent(componentMask, True, s)
            if len(componentMeasures)>0:
                row = {**box, **componentMeasures}
                measures.append(row)
                if diag:
                    im2 = cv.rectangle(im2, (boxes.loc[i,'x0'],boxes.loc[i,'y0']), (boxes.loc[i,'x0']+boxes.loc[i,'w'],boxes.loc[i,'y0']+boxes.loc[i,'h']), (0,0,255), 2)
                    im2 = cv.circle(im2, (int(line['xc']), int(line['yc'])), 3, (0,0,255), 3)
        if len(measures)>0:
            measures = pd.DataFrame(measures)
            maxlen0 = measures.w.max()
            maxlen = maxlen0*s
            totlen = measures.w.sum()*s
            maxarea = measures.area.max()*s
            totarea = measures.area.sum()*s
            longest = measures[measures.w==maxlen0] # measurements of longest component
            longest = longest.iloc[0]
            roughness = longest['roughness']
            meanT = longest['meanT']
            stdevT = longest['stdevT']
            minmaxT = longest['minmaxT']
            ret.append({'line':j, 'segments':numlines, 'maxlen':maxlen, 'totlen':totlen, 'maxarea':maxarea, 'totarea':totarea, 'roughness':roughness, 'meanT':meanT, 'stdevT':stdevT, 'minmaxT':minmaxT})
    if len(ret)==0:
        return [], {}
    if diag:
        imshow(im, im2)
    units = {'line':'', 'segments':'', 'maxlen':'px', 'totlen':'px', 'maxarea':'px', 'totarea':'px', 'roughness':cmunits['roughness'], 'meanT':cmunits['meanT'], 'stdevT':cmunits['stdevT'], 'minmaxT':cmunits['minmaxT']}
    return pd.DataFrame(ret), units

def stitchMeasure(file:str, st:str) -> Union[Tuple[dict,dict], Tuple[pd.DataFrame,dict]]:
    if st=='xs':
        return xsMeasure(file)
    elif st=='vert':
        return vertMeasure(file)
    elif st=='horiz':
        return horizMeasure(file)
    
def fnMeasures(folder:str, st:str) -> str:
    return os.path.join(folder, os.path.basename(folder)+'_'+st+'Summary.csv')
    
def exportMeasures(t:pd.DataFrame, st:str, folder:str, units:dict) -> None:
    '''export measured values'''
    fn = fnMeasures(folder, st)
    col = pd.MultiIndex.from_tuples([(k,v) for k, v in units.items()])
    data = np.array(t)
    df = pd.DataFrame(data, columns=col)       
    df.to_csv(fn)
    logging.info(f'Exported {fn}')
    
def exportGeneric(self, title:str, table:pd.DataFrame, units:dict, overwrite:bool=False) -> None:
    fmethod = getattr(self, title)
    fn = fmethod()
    if os.path.exists(fn) and not overwrite:
        return
    col = pd.MultiIndex.from_tuples([(k,v) for k, v in units.items()])
    data = np.array(table)
    df = pd.DataFrame(data, columns=col)
    df.to_csv(fn)
    logging.info(f'Exported {fn}')

def measureStills(folder:str, overwrite:bool=False) -> None:
    try:
        fl = fileList(folder)
    except:
        return
    logging.info(f'Measuring {os.path.basename(folder)}')
    for st in ['xs', 'vert']:
        fn = fnMeasures(folder, st)
        if overwrite or not os.path.exists(fn):
            xs = []
            for i in range(getattr(fl, st+'Cols')):
                file = getattr(fl, st+str(i+1)+'Stitch')
                if len(file)>0:
                    ret = stitchMeasure(file[0], st)
                    if len(ret[0])>0:
                        sm, units = ret
                        xs.append(sm)
            if len(xs)>0:
                xs = pd.DataFrame(xs)
                exportMeasures(xs, st, folder, units)
    fn = fnMeasures(folder, 'horiz')
    if overwrite or not os.path.exists(fn):
        file = fl.horizfullStitch
        if len(file)>0:
            hm, units = horizMeasure(file[0])
            if len(hm)>0:
                exportMeasures(hm, 'horiz', folder, units)
            
def measureStillsRecursive(topfolder:str, overwrite:bool=False) -> None:
    '''measure stills recursively in all folders'''
    for f1 in os.listdir(topfolder):
        f1f = os.path.join(topfolder, f1)
        if isSubFolder(f1f):
            try:
                measureStills(f1f, overwrite=overwrite)
            except:
                traceback.print_exc()
                pass
        elif os.path.isdir(f1f):
            measureStillsRecursive(f1f, overwrite=overwrite)
