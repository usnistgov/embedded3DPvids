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

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)



#----------------------------------------------


def lineName(file:str, tag:str) -> float:
    '''get the number of the line from the file name based on tag, e.g. 'vert', 'horiz', 'xs'. '''
    spl = re.split('_',os.path.basename(file))
    for st in spl:
        if tag in st:
            return float(st.replace(tag, ''))
    return -1



def getRoughness(componentMask:np.array, diag:int=0) -> float:
    '''measure roughness as perimeter of object / perimeter of convex hull. 
    componentMask is a binarized image of just one segment'''
    contours = cv.findContours(componentMask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    if int(cv.__version__[0])>=4:
        contours = contours[0]
    else:
        contours = contours[1]
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
        cm = cm[y-5:y+h+5,x-5:x+w+5]
        imshow(cm)
    return roughness

def widthInRow(row:list) -> int:
    '''distance between first and last 255 value of row'''
    if not 255 in row:
        return 0
    last = len(row) - row[::-1].index(255) 
    first = row.index(255)
    return last-first

def measureComponent(componentMask:np.array, horiz:bool, scale:float, maxlen:int=0, reverse:bool=False, diag:int=0) -> Tuple[dict,dict]:
    '''measure parts of a segmented fluid. 
    horiz = True to get variation along length of horiz line. False to get variation along length of vertical line.
    scale is the scaling of the stitched image compared to the raw images, e.g. 0.33 
    if maxlen>0, maxlen is the maximum length of the expected line. anything outside is leaks
    reverse=True to measure from top or right, false to measure from bottom or left'''
    roughness = getRoughness(componentMask, diag=max(0,diag-1))
    
    if horiz:
        sums = [sum(i)/255 for i in componentMask.transpose()]            # total number of pixels per row
        widths = [widthInRow(list(i)) for i in componentMask.transpose()] # overall width per row
    else:
        sums = [sum(i)/255 for i in componentMask]
        widths = [widthInRow(list(i)) for i in componentMask]
    sums = list(filter(lambda i:i>0, sums)) # remove empty rows
    if maxlen>0:
        # limit the measurements to only the length where extrusion was on
        
        if reverse:
            ilast = max(len(sums)-maxlen, 0)
            sums = sums[ilast:]
            leaks = sums[0:ilast]
        else:
            ilast = min(maxlen+1, len(sums))
            leaks = sums[ilast:]
            sums = sums[0:ilast]
    else:
        leaks = []

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
    retval = {'roughness':roughness, 'emptiness':emptiness, 'meanT':meant*scale, 'stdevT':stdev, 'minmaxT':minmax, 'vintegral':vest, 'vleak':vleak}
    return retval, units


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

def xsMeasureIm(im:np.ndarray, s:float, title:str, name:str, acrit:int=100, diag:bool=False, **kwargs) -> Tuple[dict,dict]:
    '''im is imported image. 
    s is is the scaling of the stitched image compared to the raw images, e.g. 0.33 
    title is the title to put on the plot
    name is the name of the line, e.g. xs1
    acrit is the minimum segment size to be considered a cross-section
    '''
    errorRet = {}, {}
    im2, markers, attempt = vm.segmentInterfaces(im, acrit=acrit, diag=max(0,diag-1))
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
        for j, imgi in enumerate([im, im2]):
            cv.rectangle(imgi, (x0,y0), (x0+w,y0+h), (0,0,255), 2)   # bounding box
            cv.circle(imgi, (int(xc), int(yc)), 2, (0,0,255), 2)     # centroid
            cv.circle(imgi, (x0+int(w/2),y0+int(h/2)), 2, (0,255,255), 2) # center of bounding box
        imshow(im, im2)
        plt.title(title)
    units = {'line':'', 'aspect':'h/w', 'xshift':'w', 'yshift':'h', 'area':'px','x0':'px', 'y0':'px', 'w':'px', 'h':'px', 'xc':'px', 'yc':'px', 'roughness':''} # where pixels are in original scale
    retval = {'line':name, 'aspect':aspect, 'xshift':xshift, 'yshift':yshift, 'area':area*s**2, 'x0':x0*s, 'y0':y0*s, 'w':w*s, 'h':h*s, 'xc':xc*s, 'yc':yc*s, 'roughness':roughness}
    return retval, units

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

    
    
        
    
        
    


#------------------------------------


def vertSegment(im:np.array, s:float, maxlen:float, diag:int, acrit:int=2500, **kwargs) -> Tuple[pd.DataFrame, dict, dict, float, pd.Series, np.array]:
    '''segment out the filament and measure it
    s is is the scaling of the stitched image compared to the raw images, e.g. 0.33
    if maxlen>0, maxlen is the maximum length of the expected line. anything outside is leaks
    acrit is the minimum segment size to be considered a part of a line
    '''
    im2, markers, attempt = vm.segmentInterfaces(im, acrit=acrit, diag=max(0,diag-1))
    if len(markers)==0 or markers[0]==1:
        return {}, {}, {}, attempt, [], im2 # nothing to measure here
    df = vm.markers2df(markers)
    df = df[(df.a>acrit)]
        # remove anything too small
    df2 = df[(df.x0>10)&(df.y0>10)&(df.x0+df.w<im.shape[1]-10)&(df.y0+df.h<im.shape[0]-10)] 
        # remove anything too close to the border
    if len(df2)==0:
        return {}, {}, {}, attempt, [], im2
    filI = df2.a.idxmax() # index of filament label, largest remaining object
    component = df2.loc[filI]
    inline = df2[(df2.x0>component['x0']-50)&(df2.x0<component['x0']+50)] # other objects inline with the biggest object

    # get combined mask of all objects in line
    masks = [(markers[1] == i).astype("uint8") * 255 for i,row in inline.iterrows()]
    componentMask = masks[0]
    if len(masks)>1:
        for mask in masks[1:]:
            componentMask = cv.add(componentMask, mask)
       
    component = pd.Series({'h':inline.h.sum(), 'w':inline.w.max(), \
                           'x0':inline.x0.min(), 'y0':inline.y0.min(), \
                           'a':inline.a.sum(),\
                           'yc':sum(inline.a * inline.yc)/sum(inline.a),\
                           'xc':sum(inline.a * inline.xc)/sum(inline.a)})       
    componentMeasures, cmunits = measureComponent(componentMask, False, s, maxlen=maxlen, reverse=True, diag=max(0,diag-1))
    
    return df2, componentMeasures, cmunits, attempt, component, im2
    
def vertMeasure(file:str, progDims:pd.DataFrame, diag:int=0, **kwargs) -> Tuple[dict,dict]:
    '''measure vertical lines. progDims holds timing info about the lines'''
    name = lineName(file, 'vert')
    s = 1/fileScale(file)
    im = cv.imread(file)
    maxlen = progDims[progDims.name==('vert'+str(int(name)))].iloc[0]['l']
    maxlen = int(maxlen/s)
    # label connected components
    df2, componentMeasures, cmunits, attempt, co, im2 = vertSegment(im, s, maxlen, diag, **kwargs)
    if len(df2)==0:
        return {}, {}
    w = int(co['w'])
    h = int(co['h'])
    x0 = int(co['x0'])
    y0 = int(co['y0'])
    area = int(co['a'])
    xc = int(co['xc'])
    yc = int(co['yc'])
    componentMeasures['vintegral'] = componentMeasures['vintegral']*s**3
    aspect = h/componentMeasures['meanT'] # height/width
    r = componentMeasures['meanT']/2
    if h*s>2*r:
        vest = (h*s - 2*r)*np.pi*(r)**2 + 4/3*np.pi*r**3 # cylinder + hemisphere endcaps
    else:
        vest = 4/3*np.pi*r**3 # sphere
    units = {'line':'', 'aspect':'h/w', 'area':'px','x0':'px', 'y0':'px', 'w':'px', 'h':'px', 'xc':'px', 'yc':'px', 'vest':'px^3'} # where pixels are in original scale
    ret = {'line':name, 'aspect':aspect, 'area':area*s**2, 'x0':x0*s, 'y0':y0*s, 'w':w*s, 'h':h*s, 'xc':xc*s, 'yc':yc*s, 'vest':vest}
    ret = {**ret, **componentMeasures}
    units = {**units, **cmunits}
    if diag:
        im2 = cv.cvtColor(im2,cv.COLOR_GRAY2RGB)
        im2 = cv.rectangle(im2, (x0,y0), (x0+w,y0+h), (0,0,255), 2)
        im2 = cv.circle(im2, (int(xc), int(yc)), 2, (0,0,255), 2)
        imshow(im, im2)
        plt.title(os.path.basename(file))
    return ret, units


#-------------------------------------------------

def markHorizOnIm(im2:np.array, row:pd.Series) -> np.array:
    '''mark horizontal element on the image'''
    im2 = cv.rectangle(im2, (int(row['x0']),int(row['y0'])), (int(row['x0']+row['w']),int(row['y0']+row['h'])), (0,0,255), 2)
    im2 = cv.circle(im2, (int(row['xc']), int(row['yc'])), 3, (0,0,255), 3)
    return im2
    

def horizLineMeasure(df:pd.DataFrame, labeled:np.array, im2:np.array, diag:bool, s:float, j:int, progDims:pd.DataFrame) -> Tuple[dict, dict]:
    '''measure one horizontal line. 
    labeled is an image from connected component labeling
    im2 is the original image
    s is is the scaling of the stitched image compared to the raw images, e.g. 0.33
    j is the line number
    progDims holds timing info about the lines
    '''
    numlines = len(df)
    measures = []
    cmunits = {}
    maxlen = progDims[progDims.name==f'horiz{j}'].iloc[0]['l']  # length of programmed line
    for i,row in df.iterrows():
        componentMask = (labeled == i).astype("uint8") * 255   # image with line in it
        if row['w']>row['h']*0.7: # horiz lines must be circular or wide and not empty
            box = {'i':i, 'w':row['w'], 'h':row['h'], 'area':row['a']}
            componentMeasures, cmunits = measureComponent(componentMask, True, s, maxlen, reverse=(j==1), diag=max(0,diag-1))
            if len(componentMeasures)>0 and componentMeasures['emptiness']<0.5:
                row0 = {**box, **componentMeasures}
                measures.append(row0)
                if diag:
                    im2 = markHorizOnIm(im2, row)
                    
    if len(measures)==0:
        return {}, {}
    measures = pd.DataFrame(measures)
    maxlen0 = measures.w.max()
    maxlen = maxlen0*s
    totlen = measures.w.sum()*s
    maxarea = measures.area.max()*s**2
    totarea = measures.area.sum()*s**2
    longest = measures[measures.w==maxlen0] # measurements of longest component
    longest = longest.iloc[0]
    roughness = longest['roughness']
    meanT = longest['meanT']
    stdevT = longest['stdevT']
    minmaxT = longest['minmaxT']
    r = meanT/2
    if totlen>2*r:
        vest = (totlen - 2*r)*np.pi*(r)**2 + 4/3*np.pi*r**3 # cylinder + hemisphere endcaps
    else:
        vest = 4/3*np.pi*r**3
    ret = {'line':j, 'segments':numlines, 'maxlen':maxlen, 'totlen':totlen, 'maxarea':maxarea, 'totarea':totarea, 'roughness':roughness, 'meanT':meanT, 'stdevT':stdevT, 'minmaxT':minmaxT, 'vest':vest}
    return ret, cmunits

def closestIndex(val:float, l1:list) -> int:
    '''index of closest value in list l1 to value val'''
    l2 = [abs(x-val) for x in l1]
    return l2.index(min(l2))

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


def horizSegment(im0:np.array, progDims, diag:int, s:float, acrit:float=1000, satelliteCrit:float=0.2, **kwargs) -> Tuple[pd.DataFrame, dict]:
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
            r,cmu = horizLineMeasure(df, labeled, im2, diag, s, j, progDims)
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
    ret, cmunits, attempt, im2 = horizSegment(im0, progDims, diag, s, **kwargs)
    
    if len(ret)==0:
        return [], {}
    if diag:
        imshow(im, im2)
        plt.title(os.path.basename(file))
    units = {'line':'', 'segments':'', 'maxlen':'px', 'totlen':'px', 'maxarea':'px', 'totarea':'px', 'roughness':cmunits['roughness'], 'meanT':cmunits['meanT'], 'stdevT':cmunits['stdevT'], 'minmaxT':cmunits['minmaxT'], 'vest':'px^3'}
    if diag>=2:
        display(pd.DataFrame(ret))
    return pd.DataFrame(ret), units

#--------------------------------

def stitchMeasure(file:str, st:str, progDims:pd.DataFrame, diag:int=0, **kwargs) -> Union[Tuple[dict,dict], Tuple[pd.DataFrame,dict]]:
    '''measure one stitched image
    st is a line type, e.g. xs, vert, or horiz
    progDims holds timing info about the print
    '''
    if st=='xs':
        return xsMeasure(file, diag=diag)
    elif st=='vert':
        return vertMeasure(file, progDims, diag=diag, **kwargs)
    elif st=='horiz':
        return horizMeasure(file, progDims, diag=diag, **kwargs)
    
def fnMeasures(folder:str, st:str) -> str:
    '''get a filename for summary table. st is xs, vert, or horiz'''
    return os.path.join(folder, f'{os.path.basename(folder)}_{st}Summary.csv')

    
def importProgDims(folder:str) -> Tuple[pd.DataFrame, dict]:
    '''import the programmed dimensions table to a dataframe, and get units'''
    pv = printVals(folder)
    progDims, units = pv.importProgDims()
    for s in ['l', 'w']:
        progDims[s] = progDims[s]*cfg.const.pxpmm # convert to mm
        units[s] = 'px'
    return progDims, units  


def stitchFile(folder:str, st:str, i:int) -> str:
    '''get the name of the stitch file, where st is vert, horiz, or xs, and i is a line number'''
    try:
        fl = stitchSorter(folder)
    except:
        return
    if st=='horiz':
        sval = 'horizfullStitch'
    else:
        sval = f'{st}{i+1}Stitch'
    file = getattr(fl, sval)
    if len(file)>0:
        return file[0]
    else:
        return ''

def measure1Line(folder:str, st:str, i:int, diag:int=0, **kwargs) -> Union[Tuple[dict,dict], Tuple[pd.DataFrame,dict]]:
    '''measure just one line. st is vert, horiz, or xs. i is the line number'''
    progDims, units = importProgDims(folder)
    file = stitchFile(folder, st, i)
    if os.path.exists(file):
        return stitchMeasure(file, st, progDims, diag=diag, **kwargs)
    else:
        return {},{}
    
def copyImage(folder:str, st:str, i:int) -> None:
    '''make a copy of the image. st is vert, horiz, or xs. i is the line number'''
    file = stitchFile(folder, st, i)
    if not os.path.exists(file):
        return
    if not '00' in file:
        return
    newfile = file.replace('_00', '_01')
    if os.path.exists(newfile):
        return
    shutil.copyfile(file, newfile)
    logging.info(f'Created new file {newfile}')
    
def openImageInPaint(folder:str, st:str, i:int) -> None:
    '''open the image in paint. this is useful for erasing smudges or debris that are erroneously detected by the algorithm as filaments'''
    file = stitchFile(folder, st, i)
    if not os.path.exists(file):
        return
    subprocess.Popen([r'C:\Windows\System32\mspaint.exe', file]);
    



def measureStills(folder:str, overwrite:bool=False, diag:int=0, overwriteList:List[str]=['xs', 'vert', 'horiz'], **kwargs) -> None:
    '''measure the stills in folder. 
    overwrite=True to overwrite files. 
    overwriteList is the list of files that should be overwritten'''
    if not isSubFolder(folder):
        return
    try:
        fl = stitchSorter(folder)
    except Exception as e:
        return
    if fl.date<210500:
        # no stitches to measure
        return
    if 'dates' in kwargs and not fl.date in kwargs['dates']:
        return
    progDims, units = importProgDims(folder)
    
    # measure xs and vert
    logging.info(f'Measuring {os.path.basename(folder)}')
    for st in ['xs', 'vert']:
        fn = fnMeasures(folder, st)
        if overwrite and st in overwriteList and os.path.exists(fn):
            os.remove(fn)
        if not os.path.exists(fn):
            xs = []
            for i in range(getattr(fl, f'{st}Cols')):
                file = getattr(fl, f'{st}{i+1}Stitch')
                if len(file)>0:
                    ret = stitchMeasure(file[0], st, progDims, diag=diag, **kwargs)
                    if len(ret[0])>0:
                        sm, units = ret
                        xs.append(sm)
            if len(xs)>0:
                xs = pd.DataFrame(xs)
#                 exportMeasures(xs, st, folder, units)
                plainExp(fnMeasures(folder, st), xs, units)
    
    # measure horiz
    fn = fnMeasures(folder, 'horiz')
    if overwrite and 'horiz' in overwriteList and os.path.exists(fn):
        os.remove(fn)
    if not os.path.exists(fn):
        file = fl.horizfullStitch
        if len(file)>0:
            hm, units = horizMeasure(file[0], progDims,  diag=diag, **kwargs)
            if len(hm)>0:
                plainExp(fnMeasures(folder, 'horiz'), hm, units)
#                 exportMeasures(hm, 'horiz', folder, units)
            
def measureStillsRecursive(topfolder:str, overwrite:bool=False, diag:int=0, **kwargs) -> None:
    '''measure stills recursively in all folders'''
    
    if isSubFolder(topfolder):
        try:
            measureStills(topfolder, overwrite=overwrite, diag=diag, **kwargs)
        except:
            traceback.print_exc()
            pass
    else:
        for f1 in os.listdir(topfolder):
            f1f = os.path.join(topfolder, f1)
            if os.path.isdir(f1f):
                measureStillsRecursive(f1f, overwrite=overwrite, diag=diag, **kwargs)

#-------------------------------------

class metricList:
    '''holds info about measured lines'''
    
    def __init__(self, folder:str):
        if not os.path.isdir(folder):
            raise NameError(f'Cannot create metricList: {folder} is not directory')
        self.folder = folder
        self.bn = os.path.basename(folder)
        self.findSummaries()
        
    def validS(self) -> List[str]:
        return ['vert', 'horiz', 'xs']
        
    def findSummaries(self) -> None:
        '''import summary data'''
        for s in self.validS():
            fn = os.path.join(self.folder, f'{self.bn}_{s}Summary.csv')
            if os.path.exists(fn):
                t,u = plainIm(fn,0)
                setattr(self, f'{s}Sum', t)
                setattr(self, f'{s}SumUnits', u)
            else:
                setattr(self, f'{s}Sum', [])
                setattr(self, f'{s}SumUnits', {})
                
    def findRhe(self, vink:float=5, vsup:float=5, di:float=0.603, do:float=0.907) -> None:
        '''find viscosity for ink and support at flow speed vink and translation speed vsup for nozzle of inner diameter di and outer diameter do'''
        pv = printVals(folder)
        inkrate = vink/di # 1/s
        suprate = vsup/do # 1/s
        inknu = pv.ink.visc(inkrate)
        supnu = pv.sup.visc(suprate)
        return {'ink':pv.ink.shortname, 'sup':pv.sup.shortname, 'nuink':inknu, 'nusup':supnu}
        
        
                
    #-----------------------------
        
    def checkS(self, s:str) -> None:
        '''check if s is valid. s is a line type, i.e. vert, horiz, or xs'''
        if not s in self.validS():
            raise NameError(f'Line name must be in {self.validS()}')
        
    def numLines(self, s:str) -> int:
        '''number of lines where s is vert, horiz, or xs'''
        self.checkS(s)
        return len(getattr(self, f'{s}Sum'))

    def missingLines(self, s:str) -> list:
        '''indices of missing lines where s is vert, horiz, or xs'''
        self.checkS(s)
        if s=='xs':
            allL = [1,2,3,4,5]
        elif s=='horiz':
            allL = [0,1,2]
        elif s=='vert':
            allL = [1,2,3,4]
        tab = getattr(self, f'{s}Sum')
        if len(tab)==0:
            return allL
        else:
            return set(allL) - set(tab.line)
    
    def inconsistent(self, s:str, vlist:List[str], tolerance:float=0.25) -> list:
        '''get a list of variables in which the cross-sections are inconsistent, i.e. range is greater than median*tolerance
         where s is vert, horiz, or xs
         vlist is a list of variable names
         tolerance is a fraction of median value
         '''
        self.checkS(s)
        out = []
        for v in vlist:
            t = getattr(self, f'{s}Sum')
            if len(t)>0:
                ma = t[v].max()
                mi = t[v].min()
                me = t[v].median()
                if ma-mi > me*tolerance:
                    out.append(v)
        return out  
    
    def summarize(self) -> dict:
        '''summarize all of the summary data into one line'''
        rhe = self.findRhe()

#-----------------------------------------

def checkMeasurements(folder:str) -> None:
    '''check the measurements in the folder and identify missing or unusual data'''
    problems = []
    ml = metricList(folder)
    for i,s in enumerate(ml.validS()):
        missing = ml.missingLines(s)
        if len(missing)>0:
            problems.append({'code':i+1, 'description':f'Missing # {s} lines', 'value':missing, 'st':s})
    verti = ml.inconsistent('vert', ['xc', 'yc', 'area'], tolerance=0.5)
    if len(verti)>0:
        problems.append({'code':4, 'description':f'Inconsistent vert values', 'value':verti, 'st':'vert'})
    horizi = ml.inconsistent('horiz', ['totarea'], tolerance=0.5)
    if len(horizi)>0:
        problems.append({'code':5, 'description':f'Inconsistent horiz values', 'value':horizi, 'st':'horiz'})
    xsi = ml.inconsistent('xs', ['xc', 'yc'])
    if len(xsi)>0:
        problems.append({'code':6, 'description':f'Inconsistent xs values', 'value':xsi, 'st':'xs'})
    return pd.DataFrame(problems)

def checkAndDiagnose(folder:str, redo:bool=False) -> None:
    '''check the folder and show images to diagnose problems
    redo=True if you want to re-measure any bad values
    '''
    problems = checkMeasurements(folder)
    if len(problems)==0:
        logging.info(f'No problems detected in {folder}')
    else:  
        logging.info(f'Problems detected in {folder}')
        display(problems)
        
    relist = []
    if redo:
        # get a list of images to re-analyze
        for i, row in problems.iterrows():
            if row['code']<4:
                for m in row['value']:
                    relist.append([row['st'], m-1])
            else:
                if row['st']=='horiz':
                    mlist = [0]
                elif row['st']=='vert':
                    mlist = [0,1,2,3]
                elif row['st']=='xs':
                    mlist = [0,1,2,3,4]
                for m in mlist:
                    val = [row['st'], m]
                    if not val in relist:
                        relist.append(val)
        relist.sort()
        for r in relist:
    #         logging.info(f'Measuring {r[0]}{r[1]}')
            measure1Line(folder, r[0], r[1], diag=1)
    
    
def checkAndDiagnoseRecursive(topfolder:str, redo:bool=False) -> None:
    '''go through all folders recursively and check and diagnose measurements. redo=True to redo any measurements that are bad'''
    if isSubFolder(topfolder):
        try:
            checkAndDiagnose(topfolder, redo=redo)
        except:
            traceback.print_exc()
            pass
    elif os.path.isdir(topfolder):
        for f in os.listdir(topfolder):
            f1f = os.path.join(topfolder, f)
            if os.path.isdir(f1f):
                checkAndDiagnoseRecursive(f1f, redo=redo) 
                
def returnNewSummary(pv) -> Tuple[pd.DataFrame, dict]:
    '''get summary data from a printVals object'''
    t,u = pv.summary()
    return pd.DataFrame([t]),u
                

def stillsSummaryRecursive(topfolder:str) -> Tuple[pd.DataFrame, dict]:
    '''go through all of the folders and summarize the stills'''
    if isSubFolder(topfolder):
        try:
            pv = printVals(topfolder)
            t,u = pv.summary()
            return pd.DataFrame([t]),u
        except:
            traceback.print_exc()
            logging.warning(f'failed to summarize {topfolder}')
            return {}, {}
    elif os.path.isdir(topfolder):
        tt = []
        u = {}
        logging.info(topfolder)
        for f in os.listdir(topfolder):
            f1f = os.path.join(topfolder, f)
            if os.path.isdir(f1f):
                # recurse into next folder level
                t,u0=stillsSummaryRecursive(f1f)
                if len(t)>0:
                    if len(tt)>0:
                        # combine dataframes
                        tt = pd.concat([tt,t], ignore_index=True)
                    else:
                        # adopt this dataframe
                        tt = t
                    if len(u0)>len(u):
                        u = dict(u, **u0)
        return tt, u
    
def stillsSummary(topfolder:str, exportFolder:str, newfolders:list=[], filename:str='stillsSummary.csv') -> pd.DataFrame:
    '''go through all of the folders and summarize the stills'''
    outfn = os.path.join(exportFolder, filename) # file to export to
    if os.path.exists(outfn) and len(newfolders)>0:
        # import existing table and only get values from new folders
        ss,u = plainIm(outfn, ic=0)
        for f in newfolders:
            tt,units = stillsSummaryRecursive(f)
            newrows = []
            for i,row in tt.iterrows():
                if row['folder'] in list(ss.folder):
                    # overwrite existing row
                    ss.loc[ss.folder==row['folder'], ss.keys()] = row[ss.keys()]
                else:
                    # add new row
                    newrows.append(i)
            if len(newrows)>0:
                # add new rows to table
                ss = pd.concat([ss, tt.loc[newrows]])
    else:
        # create an entirely new table
        ss,units = stillsSummaryRecursive(topfolder)
        
    # export results
    if os.path.exists(exportFolder):
        plainExp(outfn, ss, units)
    return ss,units

def idx0(k:list) -> int:
    '''get the index of the first dependent variable'''
    if 'xs_aspect' in k:
        idx = int(np.argmax(k=='xs_aspect'))
    elif 'projectionN' in k:
        idx = int(np.argmax(k=='projectionN'))
    elif 'horiz_segments' in k:
        idx = int(np.argmax(k=='horiz_segments'))
    else:
        idx = 1
    return idx

def printStillsKeys(ss:pd.DataFrame) -> None:
    '''sort the keys into dependent and independent variables and print them out'''
    k = ss.keys()
    k = k[~(k.str.endswith('_SE'))]
    k = k[~(k.str.endswith('_N'))]
    idx = idx0(k)
    controls = k[:idx]
    deps = k[idx:]
    print(f'Independents: {list(controls)}')
    print()
    print(f'Dependents: {list(deps)}')
    
def fluidAbbrev(row:pd.Series) -> str:
    '''get a short abbreviation to represent fluid name'''
    it = row['ink_type']
    if it=='water':
        return 'W'
    elif it=='mineral oil':
        return 'M'
    elif it=='mineral oil_Span 20':
        return 'MS'
    elif it=='PDMS_3_mineral_25':
        return 'PM'
    elif it=='PDMS_3_silicone_25':
        return 'PS'
    elif it=='PEGDA_40':
        return 'PEG'
    
def indVarSymbol(var:str, fluid:str, commas:bool=True) -> str:
    '''get the symbol for an independent variable, eg. dnorm, and its fluid, e.g ink
    commas = True to use commas, otherwise use periods'''
    if commas:
        com = ','
    else:
        com = '.'
    if var=='visc' or var=='visc0':
        return '$\eta_{'+fluid+'}$'
    elif var=='tau0':
        return '$\tau_{y'+com+fluid+'}$'
    elif var=='dPR':
        return '$d_{PR'+com+fluid+'}$'
    elif var=='dnorm':
        return '$\overline{d_{PR'+com+fluid+'}}$'
    elif var=='dnormInv':
        return '$1/\overline{d_{PR'+com+fluid+'}}$'
    elif var=='rate':
        return '$\dot{\gamma}_{'+fluid+'}$'
    else:
        if var.endswith('Inv'):
            varsymbol = '1/'+var[:-3]
        else:
            varsymbol = var
        return '$'+varsymbol+'_{'+fluid+'}$'
    
def varSymbol(s:str, lineType:bool=True, commas:bool=True, **kwargs) -> str:
    '''get a symbolic representation of the variable
    lineType=True to include the name of the line type in the symbol
    commas = True to use commas, otherwise use periods'''
    if s.startswith('xs_'):
        varlist = {'xs_aspect':'XS height/width'
                   , 'xs_xshift':'XS horiz shift/width'
                   , 'xs_yshift':'XS vertical shift/height'
                   , 'xs_area':'XS area'
                   , 'xs_areaN':'XS area/intended'
                   , 'xs_wN':'XS width/intended'
                   , 'xs_hN':'XS height/intended'
                   , 'xs_roughness':'XS roughness'}
    elif s.startswith('vert_'):
        varlist = {'vert_wN':'vert bounding box width/intended'
                , 'vert_hN':'vert length/intended'
                   , 'vert_vN':'vert bounding box volume/intended'
               , 'vert_vintegral':'vert integrated volume'
               , 'vert_viN':'vert integrated volume/intended'
               , 'vert_vleak':'vert leak volume'
               , 'vert_vleakN':'vert leak volume/line volume'
               , 'vert_roughness':'vert roughness'
               , 'vert_meanTN':'vert diameter/intended'
                   , 'vert_stdevTN':'vert stdev(diameter)/diameter'
               , 'vert_minmaxTN':'vert diameter variation/diameter'}
    elif s.startswith('horiz_') or s=='vHorizEst':
        varlist = {'horiz_segments':'horiz segments'
               , 'horiz_segments_manual':'horiz segments'
               , 'horiz_maxlenN':'horiz droplet length/intended'
               , 'horiz_totlenN':'horiz total length/intended'
               , 'horiz_vN':'horiz volume/intended'
               , 'horiz_roughness':'horiz roughness'
               , 'horiz_meanTN':'horiz height/intended'
               , 'horiz_stdevTN':'horiz stdev(height)/intended'
               , 'horiz_minmaxTN':'horiz height variation/diameter'
               , 'vHorizEst':'horiz volume'}
    elif s.startswith('proj'):
        varlist = {'projectionN':'projection into bath/intended'
                   , 'projShiftN':'$x$ shift of lowest point/$d_{est}$'}
    elif s.startswith('vertDisp'):
        varlist = {'vertDispBotN':'downstream $z_{bottom}/d_{est}$'
                  ,'vertDispBotN':'downstream $z_{middle}/d_{est}$'
                  ,'vertDispBotN':'downstream $z_{top}/d_{est}$'}
    elif s.endswith('Ratio') or s.endswith('Prod'):
        if s.endswith('Ratio'):
            symb = '/'
            var1 = s[:-5]
        else:
            symb = r'\times '
            var1 = s[:-4]
        return indVarSymbol(var1, 'ink', commas=commas)[:-1]+symb+indVarSymbol(var1, 'sup', commas=commas)[1:]
    elif s=='int_Ca':
        return r'$Ca=v_{ink}\eta_{sup}/\sigma$'
    elif s.startswith('ink_') or s.startswith('sup_'):
        fluid = s[:3]
        var = s[4:]
        return indVarSymbol(var, fluid, commas=commas)
    else:
        if s=='pressureCh0':
            return 'Extrusion pressure (Pa)'
        else:
            return s
    
    if lineType:
        return varlist[s]
    else:
        s1 = varlist[s]
        typ = re.split('_', s)[0]
        s1 = s1[len(typ)+1:]
        return s1

def importStillsSummary(file:str='stillsSummary.csv', diag:bool=False) -> pd.DataFrame:
    '''import the stills summary and convert sweep types, capillary numbers'''
    ss,u = plainIm(os.path.join(cfg.path.fig, file), ic=0)
    
    ss = ss[ss.date>210500]       # no good data before that date
    ss = ss[ss.ink_days==1]       # remove 3 day data
    ss.date = ss.date.replace(210728, 210727)   # put these dates together for sweep labeling
    k = ss.keys()
    k = k[~(k.str.contains('_SE'))]
    k = k[~(k.str.endswith('_N'))]
    idx = idx0(k)
    controls = k[:idx]
    deps = k[idx:]
    ss = flipInv(ss)
    ss.insert(idx+2, 'sweepType', ['visc_'+fluidAbbrev(row) for j,row in ss.iterrows()])
    ss.loc[ss.bn.str.contains('I_3.50_S_2.50_VI'),'sweepType'] = 'speed_W_high_visc_ratio'
    ss.loc[ss.bn.str.contains('I_2.75_S_2.75_VI'),'sweepType'] = 'speed_W_low_visc_ratio'
    ss.loc[ss.bn.str.contains('I_3.00_S_3.00_VI'),'sweepType'] = 'speed_W_int_visc_ratio'
    ss.loc[ss.bn.str.contains('VI_10_VS_5_210921'), 'sweepType'] = 'visc_W_high_v_ratio'
    ss.loc[ss.bn.str.contains('I_M5_S_3.00_VI'), 'sweepType'] = 'speed_M_low_visc_ratio'
    ss.loc[ss.bn.str.contains('I_M6_S_3.00_VI'), 'sweepType'] = 'speed_M_high_visc_ratio'
#     ss.loc[ss.ink_type=='PEGDA_40', 'sweepType'] = 'visc_PEG'
    
    # remove vertical data for speed sweeps with inaccurate vertical speeds
    
    for key in k[k.str.startswith('vert_')]:
        ss.loc[(ss.sweepType.str.startswith('speed'))&(ss.date<211000), key] = np.nan
    
    if diag:
        printStillsKeys(ss)
    return ss,u

def plainTypes(sslap:pd.DataFrame, incSweep:int=1, abbrev:bool=True) -> pd.DataFrame:
    '''convert types to cleaner form for plot legends
    incSweep=2 for a long sweep name, 1 for a short type of sweep, 0 for no sweep type label
    abbrev=True to use short names, False to use long names
    '''
    if incSweep==2:
        vsweep = '$v$ sweep, '
        etasweep = '$\eta$ sweep, '
    elif incSweep==1:
        vsweep = '$v$, '
        etasweep = '$\eta$, '
    else:
        vsweep = ''
        etasweep = ''
    if not abbrev:
        waterlap = 'water/Lap'
        mo = 'mineral oil'
        peg = 'PEGDA'
    else:
        waterlap = 'water'
        mo = 'MO'
        peg = 'PEG'
        
    
    sslap.loc[sslap.sweepType=='speed_M', 'sweepType'] = vsweep + mo
    sslap.loc[sslap.sweepType=='visc_W', 'sweepType'] = etasweep + waterlap
    sslap.loc[sslap.sweepType=='visc_W_high_v_ratio', 'sweepType'] = etasweep + waterlap + ', high $v_i/v_s$'
    sslap.loc[sslap.sweepType=='visc_M', 'sweepType'] = etasweep + mo
    sslap.loc[sslap.sweepType=='visc_PEG', 'sweepType'] = etasweep + peg
    sslap.loc[sslap.sweepType=='speed_W', 'sweepType'] = vsweep + waterlap
    sslap.loc[sslap.sweepType=='speed_M_low_visc_ratio', 'sweepType'] = vsweep + mo+ ', low $\eta_i/\eta_s$'
    sslap.loc[sslap.sweepType=='speed_M_high_visc_ratio', 'sweepType'] = vsweep + mo+', high $\eta_i/\eta_s$'
    sslap.loc[sslap.sweepType=='speed_W_high_visc_ratio', 'sweepType'] = vsweep + waterlap + ', high $\eta_i/\eta_s$'
    sslap.loc[sslap.sweepType=='speed_W_low_visc_ratio', 'sweepType'] = vsweep + waterlap + ', low $\eta_i/\eta_s$'
    sslap.loc[sslap.sweepType=='speed_W_int_visc_ratio', 'sweepType'] = vsweep + waterlap + ', med $\eta_i/\eta_s$'
    for s in ['sweepType', 'ink_type']:
        if s=='sweepType':
            sap = etasweep + ''
        else:
            sap = ''
        if not abbrev:
            sslap.loc[sslap.ink_type=='PDMS_3_mineral_25',s] = sap+'PDMS/mineral oil'
            sslap.loc[sslap.ink_type=='PDMS_3_silicone_25', s] = sap+'PDMS/silicone oil'
            sslap.loc[sslap.ink_type=='mineral oil_Span 20', s] = sap+'mineral oil/Span'
            sslap.loc[sslap.ink_type=='PEGDA_40', s] = sap+'PEGDA'
        else:
            sslap.loc[sslap.ink_type=='PDMS_3_mineral_25',s] = sap+'PDMS/MO'
            sslap.loc[sslap.ink_type=='PDMS_3_silicone_25', s] = sap+'PDMS/SO'
            sslap.loc[sslap.ink_type=='mineral oil_Span 20', s] = sap+'MO/Span'
            sslap.loc[sslap.ink_type=='PEGDA_40', s] = sap+'PEG'



def flipInv(ss:pd.DataFrame, varlist = ['Ca', 'dPR', 'dnorm', 'We', 'Oh']) -> pd.DataFrame:
    '''find inverse values and invert them (e.g. WeInv)'''
    k = ss.keys()
    idx = idx0(k)
    for j, s2 in enumerate(varlist):
        for i,s1 in enumerate(['sup', 'ink']):
            xvar = f'{s1}_{s2}'
            if f'{s1}_{s2}Inv' in ss and not xvar in ss:
                ss.insert(idx, xvar, 1/ss[f'{s1}_{s2}Inv'])
                idx+=1
    if 'int_Ca' not in ss:
        ss.insert(idx, 'int_Ca', 1/ss['int_CaInv'])
    return ss

def addRatios(ss:pd.DataFrame, varlist = ['Ca', 'dPR', 'dnorm', 'We', 'Oh', 'Bm'], operator:str='Prod') -> pd.DataFrame:
    '''add products and ratios of nondimensional variables. operator could be Prod or Ratio'''
    k = ss.keys()
    idx = int(np.argmax(k=='xs_aspect'))
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

def addLogs(ss:pd.DataFrame, varlist:List[str]) -> pd.DataFrame:
    '''add log values for the list of variables to the dataframe'''
    k = ss.keys()
    idx = int(np.argmax(k=='xs_aspect'))
    for j, s2 in enumerate(varlist):
        xvar = f'{s2}_log'
        if not xvar in s2:
            ss.insert(idx, xvar, np.log10(ss[s2]))
            idx+=1
    return ss
    

def speedTableRecursive(topfolder:str) -> pd.DataFrame:
    '''go through all of the folders and summarize the stills'''
    if isSubFolder(topfolder):
        try:
            pv = printVals(topfolder)
            t = {'bn':pv.bn, 'vink':pv.vink, 'vsup':pv.vsup, 'pressure':pv.targetPressures[0]}
            u = {'bn':'','vink':'mm/s', 'vsup':'mm/s','pressure':'mbar'}
        except:
            traceback.print_exc()
            logging.warning(f'failed to get speed from {topfolder}')
            return {}, {}
        return [t],u
    elif os.path.isdir(topfolder):
        tt = []
        u = {}
        for f in os.listdir(topfolder):
            f1f = os.path.join(topfolder, f)
            if os.path.isdir(f1f):
                t,u0=speedTableRecursive(f1f)
                if len(t)>0:
                    tt = tt+t
                    if len(u)==0:
                        u = u0
        return tt, u

def speedTable(topfolder:str, exportFolder:str, filename:str) -> pd.DataFrame:
    '''go through all the folders, get a table of the speeds and pressures, and export to fn'''
    tt,units = speedTableRecursive(topfolder)
    tt = pd.DataFrame(tt)
    if os.path.exists(exportFolder):
        plainExp(os.path.join(exportFolder, filename), tt, units)
    return tt,units


def progTableRecursive(topfolder:str, useDefault:bool=False, overwrite:bool=False, **kwargs) -> pd.DataFrame:
    '''go through all of the folders and summarize the programmed timings'''
    if isSubFolder(topfolder):
        try:
            pv = printVals(topfolder)
            if (not 'dates' in kwargs or pv.date in kwargs['dates']) and overwrite:
                pv.redoSpeedFile()
                pv.fluigent()
                pv.exportProgDims() # redo programmed dimensions
            if useDefault:
                pv.useDefaultTimings()
            t,u = pv.progDimsSummary()
        except:
            traceback.print_exc()
            logging.warning(f'failed to get programmed timings from {topfolder}')
            return {}, {}
        return t,u
    elif os.path.isdir(topfolder):
        tt = []
        u = {}
        for f in os.listdir(topfolder):
            f1f = os.path.join(topfolder, f)
            if os.path.isdir(f1f):
                t,u0=progTableRecursive(f1f, useDefault=useDefault, overwrite=overwrite, **kwargs)
                if len(t)>0:
                    if len(tt)>0:
                        tt = pd.concat([tt,t])
                    else:
                        tt = t
                    if len(u)==0:
                        u = u0
        return tt, u
    
def checkProgTableRecursive(topfolder:str, **kwargs) -> None:
    '''go through the folder recursively and check if the pressure calibration curves are correct, and overwrite if they're wrong'''
    if isSubFolder(topfolder):
        try:
            pv = printVals(topfolder)
            pv.importProgDims()
            if 0 in list(pv.progDims.a):
                pv.fluigent()
                pv.exportProgDims() # redo programmed dimensions
        except:
            traceback.print_exc()
            logging.warning(f'failed to get programmed timings from {topfolder}')
            return
        return 
    elif os.path.isdir(topfolder):
        for f in os.listdir(topfolder):
            f1f = os.path.join(topfolder, f)
            if os.path.isdir(f1f):
                checkProgTableRecursive(f1f, **kwargs)
        return

def progTable(topfolder:str, exportFolder:str, filename:str, **kwargs) -> pd.DataFrame:
    '''go through all the folders, get a table of the speeds and pressures, and export to filename'''
    tt,units = progTableRecursive(topfolder, **kwargs)
    tt = pd.DataFrame(tt)
    if os.path.exists(exportFolder):
        plainExp(os.path.join(exportFolder, filename), tt, units)
    return tt,units
                
    