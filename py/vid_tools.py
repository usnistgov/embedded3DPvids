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
from val_print import *
from val_progDim import *
from tools.imshow import imshow
import im_morph as vm
from config import cfg
from tools.plainIm import *
from file_handling import isSubFolder
import metrics as me

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)


#----------------------------------------------


def streamInfo(stream) -> Tuple:
    '''get information about the stream object'''
    time = stream.get(cv.CAP_PROP_POS_MSEC)/1000
    frame = stream.get(cv.CAP_PROP_POS_FRAMES)
    return time, frame


class vidData:
    '''holds metadata and tables about video'''
    
    def __init__(self, folder:str, pxpmm:float=cfg.const.pxpmm):
        self.folder = folder
        
        self.pv = printVals(folder) # object that holds metadata about folder
        self.pfd = self.pv.pfd
        if len(self.pfd.vid)>0:
            self.file = self.pfd.vid[0]    # video file
        else:
            logging.warning(f'No video file found in {self.folder}')
            self.file = ''
        self.measures = []
        self.measuresUnits = []
        self.streamOpen = False
        self.getPxpmm()
        self.getProgDims()

    def getMeta(self):
        if len(self.pfd.meta)==0:
            self.meta = {}
        else:
            df = pd.read_csv(self.pfd.meta[0], header=0, names=['var', 'units', 'val'])
            self.meta = dict(zip(df['var'], df['val']))
        
    def getPxpmm(self):
        '''get pixels per mm'''
        self.getMeta()
        if 'camera_magnification' in self.meta:
            cm = float(self.meta['camera_magnification'])
            d = {0.5:71, 1:139}
            if cm in d:
                self.pxpmm = d[cm]
            else:
                self.pxpmm = cfg.const.pxpmm
        else:
            self.pxpmm = cfg.const.pxpmm
                
        
    def findNozzle(self):
        self.nozData = nozData(self.folder)
        
    def getProgDims(self) -> int:
        '''get line starts and stops'''
        try:
            self.progDims = getProgDims(self.folder)
        except ValueError:
            # no print type known
            return 1
        self.progDims.importProgDims()
        self.prog = self.progDims.progDims
        if len(self.prog)==0:
            self.closeStream()
            # no timing file
            return 2
        
        if self.pfd.date>220929:
            self.progDims.importProgPos()
            pp = self.progDims.progPos
            self.maxT = pp[pp.zt<0].tf.max()
        else:
            self.progDims.importTimeFile()
            self.maxT = self.progDims.ftable.time.max() # final time in programmed run
        return 0
        
    def openStream(self) -> None:
        '''open the video stream and get metadata'''
        if not self.streamOpen:
            self.stream = cv.VideoCapture(self.file)
            self.frames = int(self.stream.get(cv.CAP_PROP_FRAME_COUNT)) # total number of frames
            self.fps = self.stream.get(cv.CAP_PROP_FPS)
            self.duration = self.frames/self.fps
            self.streamOpen = True
            if self.pfd.date>220901:
                # timing rate should be correct, but vid started earlier than timing
                self.dstart = max(self.duration-self.maxT,0)+1
            
        
    def setTime(self, t:float) -> None:
        '''go to the time in seconds, scaling by video length to fluigent length'''
        if self.pfd.date>220901:
            # offset start time
            self.stream.set(cv.CAP_PROP_POS_FRAMES,int((t+self.dstart)*self.fps))
        else:
            # rescale time
            self.stream.set(cv.CAP_PROP_POS_FRAMES,int(t/self.maxT*self.frames))
        
    def getFrameAtTime(self, t:float) -> None:
        '''get the frame at a specific time'''
        self.openStream()
        self.setTime(t)
        grabbed, frame = self.stream.read() # read frame
        if not grabbed:
            logging.info(f'Frame not collected at time {t}: (t,frame) = {streamInfo(self.stream)}')
            return 1
        else:
            return frame[5:-5,5:-5] # crop
        self.closeStream()
 
    def closeStream(self) -> None:
        '''close the stream'''
        if self.streamOpen:
            self.stream.release()
            self.streamOpen = False
            
            
    def exportStills(self, overwrite:bool=False) -> None:
        '''export stills for all times in the progdims table'''
        if not 'tpic' in self.prog:
            raise ValueError('No pic time noted')
        for i,row in self.prog.iterrows():
            name = row['name']
            prefix = fh.singleDisturbName(os.path.basename(self.folder))
            fn = self.pfd.newFileName(f'still_{prefix}_{name}', 'png')
            elif not os.path.exists(fn) or overwrite:
                frame = self.getFrameAtTime(row['tpic'])
                cv.imwrite(fn, frame)
                logging.info(f'Exported {fn}')
            
#----------------------------------------------

def exportStillsRecursive(folder:str, overwrite:bool=False, overwriteDims:bool=False) -> None:
    '''export stills of key lines from videos'''
    errorList = []
    if not os.path.isdir(folder):
        return errorList
    if not fh.isPrintFolder(folder):
        for f1 in os.listdir(folder):
            errorList = errorList + exportStillsRecursive(os.path.join(folder, f1), overwrite=overwrite, overwriteDims=overwriteDims)
        return errorList

    try:
        pdim = getProgDims(folder)
        pdim.exportAll(overwrite=overwriteDims)
        vd = vidData(folder)
        vd.exportStills(overwrite=overwrite)
    except Exception as e:
        errorList.append(folder)
        print(e)
        return errorList
    else:
        return errorList