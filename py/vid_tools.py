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
    
    def __init__(self, folder:str):
        self.folder = folder
        self.pfd = fh.printFileDict(folder)
        if len(self.pfd.vid)>0:
            self.file = self.pfd.vid[0]    # video file
        else:
            logging.warning(f'No video file found in {self.folder}')
            self.file = ''
        self.measures = []
        self.measuresUnits = []
        self.streamOpen = False
        self.pxpmm = self.pfd.pxpmm()
        self.getProgDims()

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
#             self.maxT = pp.tf.max()
        else:
            self.progDims.importTimeFile()
            self.maxT = self.progDims.ftable.time.max() # final time in programmed run
        return 0
    
    def vidStatsFN(self) -> str:
        '''name of vidstats csv'''
        return self.pfd.newFileName('vidStats','csv')
    
    def importVidStats(self) -> int:
        '''import video stats from a csv file. return 0 if successful'''
        fn = self.vidStatsFN()
        if not os.path.exists(fn):
            return 1
        else:
            d, _ = plainImDict(fn, unitCol=-1, valCol=1)
            tlist = ['frames', 'fps', 'duration', 'dstart']
            for st,val in d.items():
                setattr(self, st, val)
            if len(set(tlist)-set(d))>0:
                return 1
            else:
                return 0
                        
    def exportVidStats(self, overwrite:bool=False) -> None:
        '''export the video stats'''
        fn = self.vidStatsFN()  # nozzle dimensions file name
        if os.path.exists(fn) and not overwrite:
            return
        l = ['frames', 'fps', 'duration', 'dstart']
        for st in l:
            if not hasattr(self, st):
                self.openStream()
                return
        plainExpDict(fn, dict([[li, getattr(self, li)] for li in l]))
        
    def getVidStats(self) -> None:
        '''get the video stats from the stream'''
        self.frames = int(self.stream.get(cv.CAP_PROP_FRAME_COUNT)) # total number of frames
        self.fps = self.stream.get(cv.CAP_PROP_FPS)
        self.duration = self.frames/self.fps
        if self.pfd.date>220901:
            if not hasattr(self, 'maxT'):
                self.getProgDims()
            # timing rate should be correct, but vid started earlier than timing
            self.dstart = max(self.duration-self.maxT,0)+1.25
        
    def openStream(self) -> None:
        '''open the video stream and get metadata'''
        if not self.streamOpen:
            self.stream = cv.VideoCapture(self.file)
            self.streamOpen = True
            result = self.importVidStats()
            if result>0:
                self.getVidStats()
                self.exportVidStats()
            
        
    def setTime(self, t:float) -> None:
        '''go to the time in seconds, scaling by video length to fluigent length'''
        if self.pfd.date>220901:
            # offset start time
            f = int((t+self.dstart)*self.fps)
        else:
            # rescale time
            f = int(t/self.maxT*self.frames)
        if f>=self.frames:
            f = self.frames-1
        self.stream.set(cv.CAP_PROP_POS_FRAMES,f)
        
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
            
            
    def exportStills(self, prefixes:list=[], overwrite:bool=False, **kwargs) -> None:
        '''export stills for all times in the progdims table'''
        self.getProgDims()
        if not 'tpic' in self.prog:
            raise ValueError('No pic time noted')
        if self.pfd.printType=='singleLine':
            prefix = ''
        elif self.pfd.printType=='singleDisturb':
            prefix = fh.singleDisturbName(os.path.basename(self.folder))
        elif self.pfd.printType=='tripleLine':
            prefix = fh.tripleLineName(os.path.basename(self.folder))
        else:
            raise ValueError(f'Unknown print type in {self.folder}')
        if len(prefixes)>0 and not prefix in prefixes:
            return
        for i,row in self.prog.iterrows():
            name = row['name']
            if len(prefix)>0:
                fn = self.pfd.newFileName(f'vstill_{prefix}_{name}', 'png')
            else:
                fn = self.pfd.newFileName(f'vstill_{name}', 'png')
            if not os.path.exists(fn) or overwrite:
                frame = self.getFrameAtTime(row['tpic'])
                cv.imwrite(fn, frame)
                logging.info(f'Exported {fn}')
            
#----------------------------------------------

def exportStillsRecursive(folder:str, overwrite:bool=False, overwriteDims:bool=False, **kwargs) -> None:
    '''export stills of key lines from videos'''
    errorList = []
    if not os.path.isdir(folder):
        return errorList
    if not fh.isPrintFolder(folder):
        for f1 in os.listdir(folder):
            errorList = errorList + exportStillsRecursive(os.path.join(folder, f1), overwrite=overwrite, overwriteDims=overwriteDims, **kwargs)
        return errorList

    try:
        pdim = getProgDims(folder)
        pdim.exportAll(overwrite=overwriteDims)
        vd = vidData(folder)
        vd.exportStills(overwrite=overwrite, **kwargs)
    except Exception as e:
        errorList.append(folder)
        print(e)
        return errorList
    else:
        return errorList