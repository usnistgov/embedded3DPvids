#!/usr/bin/env python
'''Functions for taking measurements from videos of single line extrusion'''

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
import csv

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
from val.v_print import printVals
from im.imshow import imshow
import im.morph as vm
from tools.config import cfg
from tools.plainIm import *
from file_handling import isSubFolder
import metrics.m_single as me

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)


#----------------------------------------------


class singleVidData:
    '''holds metadata and tables about video'''
    
    def __init__(self, folder:str, pxpmm:float=cfg.const.pxpmm):
        self.folder = folder
        self.nozData = nozData(folder)
        self.pv = printVals(folder) # object that holds metadata about folder
        self.file = self.pv.vidFile()    # video file
        self.measures = []
        self.measuresUnits = []
        self.nozMask = []
        self.prog = []
        self.streamOpen = False
        self.pxpmm = pxpmm
        self.importNozzleDims() # if the pxpmm was defined in file, this will adopt the pxpmm from file
        if not os.path.exists(self.file):
            # file does not exist
            return   
        pg = self.getProgDims()
        if pg>0:
            return
        self.defineCritVals()  
        
        
    def getVertFrame(self, time:float) -> np.ndarray:
        '''obtain frame and crop to nozzle width'''
        frame = self.getFrameAtTime(time)
        frame = frame[:, self.xL-100:self.xR+100]        # crop to nozzle width
        return frame
    
    def getXSFrame(self, time:float) -> np.ndarray:
        '''obtain and mask a frame for cross-sections'''
        frame = self.getFrameAtTime(time)
        frame = self.maskNozzle(frame, dilate=30)        # cover up the nozzle
        frame = frame[0:int(self.yB)+150, 0:self.xL+30]  # crop to region to left of nozzle outlet
        return frame
    
    
    
    def getLineTime(self, s:str, dt0:float, **kwargs) -> float:
        '''get the time of the end of the line. s is the line name, e.g. xs1. dt0 is the time to add to the end time'''
        if 'time' in kwargs:
            time = kwargs['time']
        else:
            i = self.prog.index[self.prog.name==(s[:-1])+'1'].tolist()[0]  # index of line 1
            dt = self.prog.loc[i+1,'t0']-self.prog.loc[i,'t0']             # time between line starts
            row = (self.prog[self.prog.name==s])
            if not 'xs' in s:
                dt = dt/2
            time = row.iloc[0]['t0']+dt          # go to the end of the line
        return time+dt0
    
    def getLineFrame(self, s:str, dt0:float, **kwargs) -> np.ndarray:
        '''get one frame from a given line, e.g. s='xs2'. dt0 is the time to add to the end time'''
        time = self.getLineTime(s, dt0, **kwargs)
        if 'xs' in s:
            frame = self.getXSFrame(time)
        elif 'vert' in s:
            frame = self.getVertFrame(time)
        else:
            frame = self.getFrameAtTime(time)
        return frame
    
    
    def overwriteFrame(self, s:str, dt0:float, diag:bool=False,  **kwargs) -> None:
        '''overwrite the cross-section image file. s is the line name, e.g. xs1. dt0 is time to add to the end time '''
        frame = self.getLineFrame(s, dt0, **kwargs)
        fn = os.path.join(self.folder, f'{os.path.basename(self.folder)}_vid_{s}.png')
        cv.imwrite(fn, frame)
        logging.info(f'Exported {fn}')
        if diag:
            imshow(frame)
            
            
    def measureHorizFrame(self, time:float, s:str, f:float, behindX:float=2, diag:int=0, **kwargs) -> Tuple[dict, dict]:
        '''get measurements from the frame. 
        time is the time to collect the frame
        s is the name of the line, e.g. horiz0
        f is the fraction representing how far within the line we are
        behindX is distance behind nozzle at which to get vertical displacement.
        increase diag to print more diagnostics
        '''
        
        out = {'name':s, 'time':time, 'frac':f, 'behindX':behindX} # initialize row w/ metadata
        
        if len(self.nozMask)==0:
            self.detectNozzle()
            
        # get the frame
        self.openStream()
        frame = self.getFrameAtTime(time)
        self.closeStream()
        
        # mask the nozzle
        frame2 = self.maskNozzle(frame, dilate=20, ave=True, **kwargs)
        m = 10
        my0 = (self.yB-200)
        myf = (self.yB+20)
        white = frame2.max(axis=0).max(axis=0)
        black = frame2.min(axis=0).min(axis=0)
        if s[-1]=='1':
            # 2nd horizontal line: look to the right of nozzle
            frame2[m:-m, -m:] = white                        # empty out  right edges so the filaments don't get removed during segmentation
            frame2[my0:myf, -2*m:-m] = black                 # fill right edges so the filaments get filled in
            frame2[my0:myf, self.xR+20:self.xR+m+20] = black # fill right edge of nozzle so the filaments get filled in
        else:
            # 1st or 3rd horizontal line: look to the left of nozzle
            frame2[m:-m, :m] = white                         # empty out left edges 
            frame2[my0:myf, m:2*m] = black                   # fill left edges so the filaments get filled in
            frame2[my0:myf, self.xL-m-20:self.xL-20] = black # fill left edge of nozzle so the filaments get filled in
        frame2[:m, m:-m] = white                             # empty out top so filaments don't get removed
        frame2[self.yB:myf, self.xL-20:self.xR+20] = black   # fill bottom edge of nozzle
        
        # segment the filament out
        acrit = 1000    # minimum area for segmentation
        filled, markers, finalAt = vm.segmentInterfaces(frame2, acrit=acrit, diag=(diag>1), **kwargs)
        df = vm.markers2df(markers) # convert to dataframe
        df = df[df.a>acrit]
        if len(df)==0:
            # nothing detected
            return {},{}
        
        filI = df.a.idxmax()                                       # index of filament label, largest remaining object
        componentMask = (markers[1] == filI).astype("uint8") * 255 # get largest object
        componentMask = vm.openMorph(componentMask, 5)             # remove burrs
        contours = cv.findContours(componentMask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)  # get filament outline
        if int(cv.__version__[0])>=4:
            contours = contours[0] # turn into an array
        else:
            contours = np.array(contours[1])
        contours = np.concatenate(contours)                  # turn into a list of points
        contours = contours.reshape((contours.shape[0],2))   # reshape
        contours = pd.DataFrame(contours, columns=['x','y']) # list of points on contour

        # find how far the ink projects into bath under nozzle
        underNozzle = contours[(contours.x<self.xR)&(contours.x>self.xL)]
        if len(underNozzle)>0:
            bottomPeak = underNozzle[underNozzle.y==underNozzle.y.max()]       # y value of bottommost point
            projection = -(bottomPeak.iloc[0]['y'] - self.yB)/self.pxpmm  # positive shift is upward
            projShift = (self.xM - bottomPeak.iloc[0]['x'])/self.pxpmm    # positive shift is downstream
            if s[-1]=='1':
                projShift = -projShift
            out['projection'] = projection
            out['projShift'] = projShift
        
        # find vertical displacement behind nozzle
        dist = -2*self.pxpmm # 2 mm
        if s[-1]=='1':
            dist = -dist
        dx = 0.25*self.pxpmm
        behind = contours[(contours.x>self.xM+dist-dx)&(contours.x<self.xM+dist+dx)] # span near 2mm behind nozzle
        if len(behind)>0:
            behindBot = behind[behind.y>behind.y.mean()]                         # bottom edge
            out['vertDispBot'] = -(behindBot.y.mean() - self.yB)/self.pxpmm # positive shift is upward
            behindTop = behind[behind.y<behind.y.mean()]                         # top edge
            out['vertDispTop'] = -(behindTop.y.mean() - self.yB)/self.pxpmm # positive shift is upward
            out['vertDispMid'] = (out['vertDispTop']+out['vertDispBot'])/2       # positive shift is upward
        else:
            vertDispBot = ''
            vertDispMid = ''
            vertDispTop = ''
            
        # plot results
        if diag>0:
            componentMask = frame.copy()
            self.initLineImage()
            self.drawNozzleOnFrame(colors=False)
            if len(behind)>0:
                cv.circle(componentMask,(int(behindBot.x.mean()),int(behindBot.y.mean())),5,(0,0,255),5)
                cv.circle(componentMask,(int(behind.x.mean()),int((behindBot.y.mean()+behindTop.y.mean())/2)),5,(255,0,255),5)
                cv.circle(componentMask,(int(behindTop.x.mean()),int(behindTop.y.mean())),5,(255,0,0),5)
            imshow(componentMask)
            if len(underNozzle)>0:
                cv.circle(componentMask,(bottomPeak.iloc[0]['x'],bottomPeak.iloc[0]['y']),5,(0,255,0),5)
                plt.plot(underNozzle['x'], underNozzle['y'], color='g')
            if len(behind)>0:
                plt.plot(behindBot['x'], behindBot['y'], color='r')
                plt.plot(behindTop['x'], behindTop['y'], color='b')
        units = {'name':'','time':'s', 'frac':'','behindX':'mm','projection':'mm', 'projShift':'mm', 'vertDispBot':'mm', 'vertDispMid':'mm', 'vertDispTop':'mm'}
        return out, units
    
    def measureFrameFromLine(self, s:str, f:float, diag:int=0, **kwargs) -> Tuple[dict,dict]:
        '''get measurements from a frame, where you designate the line name (s, e.g. xs1) and how far within the line to collect (f, e.g. 0.5)'''
        row = (self.prog[self.prog.name==s])   # get the line timing
        t0 = row.iloc[0]['t0']                 # start time
        tf = row.iloc[0]['tf']                 # end time
        dt = tf-t0                             # time spent writing
        t = t0+dt*f                            # get the time f fraction within the line
        return self.measureHorizFrame(t, s, f, diag=diag, **kwargs)
    
    def vidMeasuresFN(self, tag) -> str:
        '''file name for video measurement table'''
        return os.path.join(self.pv.folder, f'{os.path.basename(self.pv.folder)}_vid{tag}Measures.csv')
        
        
    def measureVideoHoriz(self, diag:int=0, overwrite:int=0, **kwargs) -> Tuple[pd.DataFrame, dict]:
        '''get info about the ellipses. Returns 1 when video is done. Returns 0 to continue grabbing.'''
        logging.info(f'measuring horiz frames in {self.folder}')
        fn  = self.vidMeasuresFN('Horiz')
        if os.path.exists(fn) and overwrite==0:
            return
        if len(self.prog)==0:
            return
        if not 'name' in self.prog.keys():
            # no programmed timings found
            return
        out = []
        units = []
        for s in ['horiz0', 'horiz1', 'horiz2']:
            row = (self.prog[self.prog.name==s])
            t0 = row.iloc[0]['t0']
            tf = row.iloc[0]['tf']
            dt = tf-t0
            # iterate through times inside of this line
            for f in np.linspace(0.3,0.9,num=20):
                t = t0+dt*f
                if f==0.9:
                    d = diag
                else:
                    d = diag-1
                try:
                    framerow, u = self.measureHorizFrame(t, s, f, diag=d, **kwargs)  # measure the frame
                except:
                    traceback.print_exc()
                if len(framerow)>0:
                    out.append(framerow)
                    units = u
                    
        # store table in object
        self.measures = pd.DataFrame(out)
        self.measuresUnits = units
        
        # export
        plainExp(fn, self.measures, self.measuresUnits)
        return self.measures, self.measuresUnits

    
    def measureVideoXS(self, diag:int=0, overwrite:int=0, **kwargs) -> Tuple[pd.DataFrame, dict]:
        '''get stills from right after the XS lines are printed, and analyze them. overwrite=2 to overwrite images, overwrite=1 to overwrite file'''
        filefn  = self.vidMeasuresFN('XS')
        if os.path.exists(filefn) and overwrite==0:
            return
        if len(self.prog)==0:
            return
        if not 'name' in self.prog.keys():
            return
        out = []
        units = []
        dt = self.prog.loc[2,'t0']-self.prog.loc[1,'t0']       # time between starts of lines
        for s in ['xs2', 'xs3', 'xs4', 'xs5']:                 # throw out last line because the last move is up, not right
            fn = os.path.join(self.folder, f'{os.path.basename(self.folder)}_vid_{s}.png')
            if overwrite<2 and os.path.exists(fn):
                frame = cv.imread(fn)
                framerow, u = me.xsMeasureIm(frame, 1, 0, os.path.basename(self.folder), s, diag=diag, acrit=300, **kwargs)
            else:
                framerow = []
            if len(framerow)==0:
                row = (self.prog[self.prog.name==s])
                time = row.iloc[0]['t0']+dt
                # iterate through times inside of this line
                dtlist = list(np.arange(-0.2, 0.8, 0.05))
                while len(dtlist)>0:
                    dt0 = dtlist.pop(0)
                    frame = self.getXSFrame(time+dt0)
                    framerow, u = me.xsMeasureIm(frame, 1, 0, f'{os.path.basename(self.folder)}_{s}_{dt0}', s, diag=diag, acrit=300, **kwargs)
                    if len(framerow)>0:
                        if framerow['x0']<40 or framerow['x0']>self.xL-100 or framerow['y0']>self.yB+100:
                            # detected particle is too far left or too far right or too low
                            framerow = []
                        else:
                            # success
                            dtlist = []
            if len(framerow)>0:
                out.append(framerow)
                units = u
                if overwrite>1 or not os.path.exists(fn):
                    cv.imwrite(fn, frame)
                    logging.info(f'Exported {fn}')
                    
        # store table in object
        self.measures = pd.DataFrame(out)
        self.measuresUnits = units
        
        # export
        plainExp(filefn, self.measures, self.measuresUnits)
        return self.measures, self.measuresUnits
    
    def exportStillVerts(self, diag:int=0, overwrite:int=0, tfrac:float=0.4, **kwargs) -> None:
        '''get stills from right after the XS lines are printed, and analyze them. 
        overwrite=2 to overwrite images, overwrite=1 to overwrite file'''
        if len(self.prog)==0:
            return
        if not 'name' in self.prog.keys():
            return
        dt = self.prog.loc[6,'t0']-self.prog.loc[5,'t0'] # time between lines
        for s in ['vert1', 'vert2', 'vert3', 'vert4']:
            # there was a run that used 30%, but didn't save that number in the file name. correct that
            fnorig = os.path.join(self.folder, f'{os.path.basename(self.folder)}_vid_{s}.png')
            if os.path.exists(fnorig):
                os.rename(fnorig, os.path.join(self.folder, f'{os.path.basename(self.folder)}_vid_{s}_30.png'))  
                
            # get the correct file name
            fn = os.path.join(self.folder, f'{os.path.basename(self.folder)}_vid_{s}_{round(100*tfrac)}.png')
            if overwrite>=2 or not os.path.exists(fn):
                row = (self.prog[self.prog.name==s])
                time = row.iloc[0]['t0']                   # start time for this line
                frame = self.getVertFrame(time+dt*tfrac)   # 30% through the line
                cv.imwrite(fn, frame)
                logging.info(f'Exported {fn}')
        return 
    
    def exportStillHoriz(self, diag:int=0, overwrite:int=0, **kwargs) -> None:
        '''get stills from right after the XS lines are printed, and analyze them. overwrite=2 to overwrite images, overwrite=1 to overwrite file'''
        if len(self.prog)==0:
            return
        if not 'name' in self.prog.keys():
            return
        dt = self.prog.loc[10,'t0']-self.prog.loc[9,'t0'] # time between lines
        for s in ['horiz0', 'horiz1', 'horiz2']:
            fn = os.path.join(self.folder, f'{os.path.basename(self.folder)}_vid_{s}.png')
            if overwrite>=2 or not os.path.exists(fn):
                row = (self.prog[self.prog.name==s])
                time = row.iloc[0]['t0']    # start time for this line
                frame = self.getFrameAtTime(time+dt/2) # halfway through the line
                cv.imwrite(fn, frame)
                logging.info(f'Exported {fn}')
        return 
    
    def measureSingle(self, diag:int=0, measureHoriz=True, measureXS=True, exportVert=True, exportHoriz=True, **kwargs) -> str:
        '''initialize data and measure the video. return name of folder if failed'''
        try:
            self.detectNozzle(diag=diag)
        except:
            logging.error(f'Error detecting nozzle in {self.folder}')
#             traceback.print_exc()
            return self.folder
        if measureHoriz:
            # measure horizontal lines
            self.measureVideoHoriz(diag=diag, **kwargs)
        if measureXS:
            # measure xs
            self.measureVideoXS(diag=diag, **kwargs)
        if exportVert:
            # export stills of vertical lines
            self.exportStillVerts(diag=diag, **kwargs)
        if exportHoriz:
            # export stills of horizontal lines
            self.exportStillHoriz(diag=diag, **kwargs)
        return ''
        
        
    def importVidMeasures(self, tag:str) -> Tuple[pd.DataFrame, dict]:
        '''import measurements from file, where tag is Horiz or XS'''
        self.measures, self.measuresUnits = plainIm(self.vidMeasuresFN(tag), ic=0)
        
    def summarySingle(self) -> Tuple[dict,dict]:
        '''summarize measurement table into a single row'''
        if len(self.measures)==0:
            self.importVidMeasures('horiz')
        if len(self.measures)==0:
            return {}, {}
        data = {}
        units = {}
        meta,metaunits = self.pv.metarow()
        for s in ['projection', 'projShift', 'vertDispBot', 'vertDispTop', 'vertDispMid']:
            # mean, with 3 sigma outliers removed, normalized by estimated filament diameter
            if s in self.measures:
                if s=='projection':
                    norm = meta['dEst']   # normalize by intended filament diameter
                else:
                    norm = meta['dEst']
                ro = removeOutliers(self.measures, s)[s]   # remove outliers at 3 sigma
        
                # get number of measurements and standard error
                if len(ro.unique())>1:
                    data[s+'N'] = ro.mean()/norm
                    data[s+'N_SE'] = ro.sem()/norm
                    data[s+'N_N'] = len(ro)
                    units[s+'N'] = ''
                    units[s+'N_SE'] = ''
                    units[s+'N_N'] = ''
        
        data = {**meta,**data}
        units = {**metaunits,**units}
        return data, units
    
    
    
        
        
#-----------------------------------------
        
        
def measureVideosRecursiveSingle(topfolder:str, **kwargs) -> List[str]:
    '''go through all of the videos and measure them and export values into individual tables
    specifical for singleLines videos
    compile a list of folders with errors'''
    errorFolders = []
    if isSubFolder(topfolder):
        try:
            vd = vidData(topfolder, pxpmm=139)
            s = vd.measureSingle(**kwargs)
        except:
            traceback.print_exc()
            errorFolders = errorFolders+[topfolder]
            pass
        if len(s)>0:
            errorFolders = errorFolders+[topfolder]
    else:
        for f1 in os.listdir(topfolder):
            f1f = os.path.join(topfolder, f1)
            if os.path.isdir(f1f):
                s = measureVideosRecursiveSingle(f1f, **kwargs)
                errorFolders = errorFolders+s
    return errorFolders


def summarizeVideosRecursiveSingle(topfolder:str) -> None:
    '''go through all of the videos and summarize them into a single table'''
    data = []
    units = []
    if isSubFolder(topfolder):
        # summarize this folder
        try:
            vd = vidData(topfolder)
            d,u = vd.summarySingle()
            if len(d)>0:
                data = [d]
                units = u
        except:
            traceback.print_exc()
            pass
    else:
        # go through subfolders and summarize
        logging.info(topfolder)
        data = []
        units = []
        for f1 in os.listdir(topfolder):
            f1f = os.path.join(topfolder, f1)
            if os.path.isdir(f1f):
                d,u = summarizeVideosRecursiveSingle(f1f)
                if len(d)>0:
                    data = data + d
                    if len(u)>len(units):
                        units = u
    return data, units

def videoSummarySingle(topfolder:str, exportFolder:str, filename:str='videoSummary.csv') -> pd.DataFrame:
    '''go through all of the folders and summarize the stills'''
    tt,units = summarizeVideosRecursiveSingle(topfolder)
    tt = pd.DataFrame(tt)
    if os.path.exists(exportFolder):
        plainExp(os.path.join(exportFolder, filename), tt, units)
    return tt,units