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

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
from stitchBas import fileList
from fileHandling import isSubFolder
import vidCrop as vc
import vidMorph as vm
from imshow import imshow
from plainIm import *
from printVals import *

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


def getRoughness(componentMask:np.array, diag:int=0) -> float:
    '''measure roughness as perimeter of object / perimeter of convex hull'''
    contours = cv.findContours(componentMask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
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
    if diag and perimeter>1000:
        cm = componentMask.copy()
        cm = cv.cvtColor(cm,cv.COLOR_GRAY2RGB)
        cv.drawContours(cm, cnt, -1, (0,255,0), 2)
        cv.drawContours(cm, [hull], -1, (0,0,255), 1)
#         ellipse = cv.fitEllipse(cnt) # fit the ellipse to the contour for that droplet
#         cv.ellipse(cm, ellipse,(255,0,0), 2)
#         contours = cv.findContours(componentMask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
#         cnt = contours[0]
        x,y,w,h = cv.boundingRect(cnt)
        cm = cm[y-5:y+h+5,x-5:x+w+5]
        imshow(cm)
    return roughness

def widthInRow(row:list) -> int:
    '''difference between first and last 255 value of row'''
    if not 255 in row:
        return 0
    last = len(row) - row[::-1].index(255) 
    first = row.index(255)
    return last-first

def measureComponent(componentMask:np.array, horiz:bool, scale:float, maxlen:int=0, reverse:bool=False, diag:int=0) -> Tuple[dict,dict]:
    '''measure parts of a segmented fluid. horiz = True to get variation along length of horiz line. False to get variation along length of vertical line.'''
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

    emptiness = 1-sum(sums)/sum(widths)     # how much of the middle of the component is empty
    vest = sum([np.pi*(r/2)**2 for r in sums])
    vleak = sum([np.pi*(r/2)**2 for r in leaks])
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

def xsMeasureIm(im:np.ndarray, s:float, attempt0:int, title:str, name:str, acrit:int=100, diag:bool=False, **kwargs) -> Tuple[dict,dict]:
    '''im is imported image. s is scale as fraction of initial image size. attempt0 indicates type of segmentation to start with in vm.segmentInterfaces'''
    im2, markers, attempt = vm.segmentInterfaces(im, attempt0=attempt0, acrit=acrit, diag=max(0,diag-1))
    if markers[0]==1:
        return {}, {}
    df = vm.markers2df(markers)
    roughness = getRoughness(im2, diag=max(0,diag-1))
    xest = im2.shape[1]/2 # estimated x
    if im2.shape[0]>600:
        yest = im2.shape[0]-300
        dycrit = 200
    else:
        yest = im2.shape[0]/2
        dycrit = im2.shape[0]/2
    df = df[(df.x0>10)&(df.y0>10)&(df.x0+df.w<im.shape[1]-10)&(df.y0+df.h<im.shape[0]-10)] 
        # remove anything too close to the border
    df = df[(df.a>acrit)]
    df2 = df[(abs(df.xc-xest)<100)&(abs(df.yc-yest)<dycrit)] 
        # filter by location relative to expectation and area
    if len(df2)==0:
        df2 = df[(df.a>1000)] # if everything was filtered out, only filter by area
        if len(df2)==0:
            return {},{}
    if len(df2)>1 and df2.a.max() < 2*list(df.a.nlargest(2))[1]:
        # largest object not much larger than 2nd largest
        return {},{}
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
        im2 = cv.cvtColor(im2,cv.COLOR_GRAY2RGB)
        for j, imgi in enumerate([im, im2]):
            cv.rectangle(imgi, (x0,y0), (x0+w,y0+h), (0,0,255), 2)   # bounding box
            cv.circle(imgi, (int(xc), int(yc)), 2, (0,0,255), 2)     # centroid
            cv.circle(imgi, (x0+int(w/2),y0+int(h/2)), 2, (0,255,255), 2) # center of bounding box
        imshow(im, im2)
        plt.title(title)
        cv.imwrite(r'C:\Users\lmf1\OneDrive - NIST\NIST\data\shopbot\results\figures\yshift_example.png', im[600:900, 300:450])
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
    if 'PEG' in file:
        attempt0 = 4
    else:
        attempt0 = 0
    s = 1/fileScale(file)
    title = os.path.basename(file)
    im = vm.normalize(im)
    return xsMeasureIm(im, s, attempt0, title, name, diag=diag)
    

#------------------------------------


def vertSegment(im:np.array, attempt0:int, s:float, maxlen:float, diag:int) -> Tuple[pd.DataFrame, dict, dict, float, pd.Series, np.array]:
    '''segment out the filament and measure it'''
    acrit=2500
    im2, markers, attempt = vm.segmentInterfaces(im, attempt0=attempt0, acrit=acrit, diag=max(0,diag-1))
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
    componentMask = (markers[1] == filI).astype("uint8") * 255
    componentMeasures, cmunits = measureComponent(componentMask, False, s, maxlen=maxlen, reverse=True, diag=max(0,diag-1))
    component = df2.loc[filI]
    return df2, componentMeasures, cmunits, attempt, component, im2
    
def vertMeasure(file:str, progDims:pd.DataFrame, diag:int=0) -> Tuple[dict,dict]:
    '''measure vertical lines'''
    name = lineName(file, 'vert')
    s = 1/fileScale(file)
    im = cv.imread(file)
    maxlen = progDims[progDims.name==('vert'+str(int(name)))].iloc[0]['l']
    maxlen = int(maxlen/s)
    # label connected copmonents
    if 'LapRD LapRD' in file:
        attempt0 = 1
    elif 'PEG' in file:
        attempt0 = 0
    else:
        attempt0 = 0
    df2, componentMeasures, cmunits, attempt, co, im2 = vertSegment(im, attempt0, s, maxlen, diag)
#     if len(componentMeasures)==0 or componentMeasures['emptiness']>0.5:
    if len(componentMeasures)==0:
        df2, componentMeasures, cmunits, attempt, co, im2 = vertSegment(im, attempt+1, s, maxlen, diag)
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
    

def horizLineMeasure(df:pd.DataFrame, y:float, margin:float, labeled:np.array, im2:np.array, diag:bool, s:float, j:int, progDims:pd.DataFrame) -> Tuple[dict, dict]:
    '''measure one horizontal line'''
    df = df[(df.yc>y-margin)&(df.yc<y+margin)]
    df = df[(df.a>0.2*df.a.max())]  # eliminate tiny satellite droplets
    numlines = len(df)
    measures = []
    cmunits = {}
    maxlen = progDims[progDims.name=='horiz'+str(j)].iloc[0]['l']  # length of programmed line
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

def horizSegment(im0:np.array, attempt0:int, progDims, diag:int, s:float) -> Tuple[pd.DataFrame, dict]:
    '''segment the image and take measurements'''
    im2, markers, attempt = vm.segmentInterfaces(im0, attempt0=attempt0, diag=max(0,diag-1), removeVert=True, acrit=2000)
    if markers[0]==1:
        return [], {}, attempt, im2
    labeled = markers[1]
    df = vm.markers2df(markers)
    linelocs = [275, 514, 756] # expected positions of lines
    margin = 150
    if diag:
        im2 = cv.cvtColor(im2,cv.COLOR_GRAY2RGB)
    ret = []
    cmunits = {}
    for j,y in enumerate(linelocs):
        r,cmu = horizLineMeasure(df, y, margin,labeled, im2, diag, s, j, progDims)
        if len(r)>0:
            ret.append(r)
            cmunits = cmu
    return ret, cmunits, attempt, im2
    

def horizMeasure(file:str, progDims:pd.DataFrame, diag:int=0, critHorizLines:int=3, **kwargs) -> Tuple[pd.DataFrame, dict]:
    '''measure horizontal lines. diag=1 to print diagnostics for this function, diag=2 to print this function and the functions it calls'''
    s = 1/fileScale(file)
    im = cv.imread(file)
    im0 = im
#     im0[0, 0] = np.zeros(im0[0, 0].shape)
#     im0 = vc.imcrop(im0, {'dx':0, 'dy':80})
    im0 = vm.removeBorders(im0)
    if 'LapRD LapRD' in file:
        attempt0 = 0
    elif 'PEG' in file:
        attempt0 = 3
    else:
        attempt0 = 0
    ret, cmunits, attempt, im2 = horizSegment(im0, attempt0, progDims, diag, s)
    if len(ret)<critHorizLines:
        ret2, cmunits2, attempt, im3 = horizSegment(im0, attempt+1, progDims, diag, s)
        if len(ret2)>len(ret):
            ret = ret2
            cmunits = cmunits2
            im2 = im3
    
    if len(ret)==0:
        return [], {}
    if diag:
        imshow(im, im2)
        plt.title(os.path.basename(file))
    units = {'line':'', 'segments':'', 'maxlen':'px', 'totlen':'px', 'maxarea':'px', 'totarea':'px', 'roughness':cmunits['roughness'], 'meanT':cmunits['meanT'], 'stdevT':cmunits['stdevT'], 'minmaxT':cmunits['minmaxT'], 'vest':'px^3'}
    return pd.DataFrame(ret), units

#--------------------------------

def stitchMeasure(file:str, st:str, progDims:pd.DataFrame, diag:int=0, **kwargs) -> Union[Tuple[dict,dict], Tuple[pd.DataFrame,dict]]:
    '''measure one stitched image'''
    if st=='xs':
        return xsMeasure(file, diag=diag)
    elif st=='vert':
        return vertMeasure(file, progDims, diag=diag)
    elif st=='horiz':
        return horizMeasure(file, progDims, diag=diag, **kwargs)
    
def fnMeasures(folder:str, st:str) -> str:
    '''get a filename for summary table'''
    return os.path.join(folder, os.path.basename(folder)+'_'+st+'Summary.csv')

    
def importProgDims(folder:str) -> Tuple[pd.DataFrame, dict]:
    pv = printVals(folder)
    progDims, units = pv.importProgDims()
    for s in ['l', 'w']:
        progDims[s] = progDims[s]*cfg.const.pxpmm # convert to mm
        units[s] = 'px'
    return progDims, units  


def measure1Line(folder:str, st:str, i:int, diag:int=0, **kwargs) -> Union[Tuple[dict,dict], Tuple[pd.DataFrame,dict]]:
    '''measure just one line'''
    try:
        fl = fileList(folder)
    except:
        return
    progDims, units = importProgDims(folder)
    if st=='horiz':
        sval = 'horizfullStitch'
    else:
        sval = st+str(i+1)+'Stitch'
    file = getattr(fl, sval)
    if len(file)>0:
        return stitchMeasure(file[0], st, progDims, diag=diag, **kwargs)


def measureStills(folder:str, overwrite:bool=False, diag:int=0, overwriteList:List[str]=['xs', 'vert', 'horiz'], **kwargs) -> None:
    '''measure the stills in folder'''
    if not isSubFolder(folder):
        return
    try:
        fl = fileList(folder)
    except:
        return
    if fl.date<210500:
        return
    if 'dates' in kwargs and not fl.date in kwargs['dates']:
        return
    progDims, units = importProgDims(folder)
    logging.info(f'Measuring {os.path.basename(folder)}')
    for st in ['xs', 'vert']:
        fn = fnMeasures(folder, st)
        if overwrite and st in overwriteList and os.path.exists(fn):
            os.remove(fn)
        if not os.path.exists(fn):
            xs = []
            for i in range(getattr(fl, st+'Cols')):
                file = getattr(fl, st+str(i+1)+'Stitch')
                if len(file)>0:
                    ret = stitchMeasure(file[0], st, progDims, diag=diag, **kwargs)
                    if len(ret[0])>0:
                        sm, units = ret
                        xs.append(sm)
            if len(xs)>0:
                xs = pd.DataFrame(xs)
#                 exportMeasures(xs, st, folder, units)
                plainExp(fnMeasures(folder, st), xs, units)
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
            fn = os.path.join(self.folder, self.bn+'_'+s+'Summary.csv')
            if os.path.exists(fn):
                t,u = plainIm(fn,0)
                setattr(self, s+'Sum', t)
                setattr(self, s+'SumUnits', u)
            else:
                setattr(self, s+'Sum', [])
                setattr(self, s+'SumUnits', {})
                
    def findRhe(self, vink:float=5, vsup:float=5, di:float=0.603, do:float=0.907) -> None:
        '''find rheology for ink and support at flow speed vink and translation speed vsup for nozzle of inner diameter di and outer diameter do'''
        pv = printVals(folder)
        inkrate = vink/di # 1/s
        suprate = vsup/do # 1/s
        inknu = pv.ink.visc(inkrate)
        supnu = pv.sup.visc(suprate)
        return {'ink':pv.ink.shortname, 'sup':pv.sup.shortname, 'nuink':inknu, 'nusup':supnu}
        
        
                
    #-----------------------------
        
    def checkS(self, s:str) -> None:
        '''check if s is valid'''
        if not s in self.validS():
            raise NameError(f'Line name must be in {self.validS()}')
        
    def numLines(self, s:str) -> int:
        '''number of lines where s is vert, horiz, or xs'''
        self.checkS(s)
        return len(getattr(self, s+'Sum'))

    def missingLines(self, s:str) -> list:
        '''indices of missing lines'''
        self.checkS(s)
        if s=='xs':
            allL = [1,2,3,4,5]
        elif s=='horiz':
            allL = [0,1,2]
        elif s=='vert':
            allL = [1,2,3,4]
        tab = getattr(self, s+'Sum')
        if len(tab)==0:
            return allL
        else:
            return set(allL) - set(tab.line)
    
    def inconsistent(self, s:str, vlist:List[str], tolerance:float=0.25) -> list:
        '''get a list of variables in which the cross-sections are inconsistent, i.e. range is greater than median*tolerance'''
        self.checkS(s)
        out = []
        for v in vlist:
            t = getattr(self, s+'Sum')
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
    '''check the folder and show images to diagnose problems'''
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
    '''go through all folders recursively and check and diagnose measurements'''
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
                
                

def stillsSummaryRecursive(topfolder:str) -> pd.DataFrame:
    '''go through all of the folders and summarize the stills'''
    if isSubFolder(topfolder):
        try:
            pv = printVals(topfolder)
            t,u = pv.summary()
        except:
            traceback.print_exc()
            logging.warning(f'failed to summarize {topfolder}')
            return {}, {}
        return [t],u
    elif os.path.isdir(topfolder):
        tt = []
        u = {}
        logging.info(topfolder)
        for f in os.listdir(topfolder):
            f1f = os.path.join(topfolder, f)
            if os.path.isdir(f1f):
                t,u0=stillsSummaryRecursive(f1f)
                if len(t)>0:
                    tt = tt+t
                    if len(u0)>len(u):
                        u = dict(u, **u0)
        return tt, u
    
def stillsSummary(topfolder:str, exportFolder:str, filename:str='stillsSummary.csv') -> pd.DataFrame:
    '''go through all of the folders and summarize the stills'''
    tt,units = stillsSummaryRecursive(topfolder)
    tt = pd.DataFrame(tt)
    if os.path.exists(exportFolder):
        plainExp(os.path.join(exportFolder, filename), tt, units)
    return tt,units

def printStillsKeys(ss:pd.DataFrame) -> None:
    k = ss.keys()
    k = k[~(k.str.contains('_SE'))]
    idx = int(np.argmax(k=='xs_aspect'))
    controls = k[:idx]
    deps = k[idx:]
    k = ss.keys()
    k = k[~(k.str.contains('_SE'))]
    controls = k[:idx]
    deps = k[idx:]
    print(f'Independents: {list(controls)}')
    print()
    print(f'Dependents: {list(deps)}')

def importStillsSummary(diag:bool=False) -> pd.DataFrame:
    '''import the stills summary and convert sweep types, capillary numbers'''
    ss,u = plainIm(os.path.join(cfg.path.fig, 'stillsSummary.csv'), ic=0)
    
    ss = ss[ss.date>210500]
    ss = ss[ss.ink_days==1]
    ss.date = ss.date.replace(210728, 210727)
    
    k = ss.keys()
    k = k[~(k.str.contains('_SE'))]
    idx = int(np.argmax(k=='xs_aspect'))
    controls = k[:idx]
    deps = k[idx:]
    ss = flipInv(ss)
    ss.insert(idx+2, 'sweepType', ['visc_'+str(i['sigma']) for j,i in ss.iterrows()])
    ss.loc[ss.bn.str.contains('I_3.50_S_2.50_VI'),'sweepType'] = 'speed_0_high_visc_ratio'
    ss.loc[ss.bn.str.contains('I_2.75_S_2.75_VI'),'sweepType'] = 'speed_0_low_visc_ratio'
    ss.loc[ss.bn.str.contains('VI_10_VS_5_210921'), 'sweepType'] = 'visc_0_high_v_ratio'
    ss.loc[ss.bn.str.contains('I_M5_S_3.00_VI'), 'sweepType'] = 'speed_20_low_visc_ratio'
    ss.loc[ss.bn.str.contains('I_M6_S_3.00_VI'), 'sweepType'] = 'speed_20_high_visc_ratio'
    ss.loc[ss.ink_type=='PEGDA_40', 'sweepType'] = 'visc_PEG'
    
    if diag:
        printStillsKeys(ss)
    return ss,u

def flipInv(ss:pd.DataFrame, varlist = ['Ca', 'dPR', 'dnorm', 'We', 'Oh']) -> pd.DataFrame:
    '''find inverse values and invert them (e.g. WeInv)'''
    k = ss.keys()
    idx = int(np.argmax(k=='xs_aspect'))
    for j, s2 in enumerate(varlist):
        for i,s1 in enumerate(['sup', 'ink']):
            xvar = s1+'_'+s2
            if f'{s1}_{s2}Inv' in ss and not xvar in ss:
                ss.insert(idx, xvar, 1/ss[f'{s1}_{s2}Inv'])
                idx+=1
    return ss

def addRatios(ss:pd.DataFrame, varlist = ['Ca', 'dPR', 'dnorm', 'We', 'Oh', 'Bm'], operator:str='Prod') -> pd.DataFrame:
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

def addLogs(ss:pd.DataFrame, varlist) -> pd.DataFrame:
    k = ss.keys()
    idx = int(np.argmax(k=='xs_aspect'))
    for j, s2 in enumerate(varlist):
        xvar = s2+'_log'
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
    '''go through all the folders, get a table of the speeds and pressures, and export to fn'''
    tt,units = progTableRecursive(topfolder, **kwargs)
    tt = pd.DataFrame(tt)
    if os.path.exists(exportFolder):
        plainExp(os.path.join(exportFolder, filename), tt, units)
    return tt,units
                
    