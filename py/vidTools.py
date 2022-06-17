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
from printVals import printVals
from imshow import imshow
import vidMorph as vm
from config import cfg
from plainIm import *
from fileHandling import isSubFolder
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

def combineLines(df:pd.DataFrame) -> dict:
    '''combine groups of similar Hough transform lines into one line'''
    top = (df[df.y0==df.y0.min()]).iloc[0]
    bot = (df[df.yf==df.yf.max()]).iloc[0]
    return {'x0':top['x0'],'y0':top['y0'],'xf':bot['xf'],'yf':bot['yf']}

def removeOutliers(df:pd.DataFrame, col:str, sigma:float=3) -> pd.DataFrame:
    '''remove outliers in column by # of standard deviation sigma'''
    return df[np.abs(df[col]-df[col].mean()) <= (sigma*df[col].std())]


def lineIntersect(line1:pd.Series, line2:pd.Series) -> Tuple[float,float]:
    '''find intersection between two lines'''
    if line1['xf']-line1['x0']==0:
        # line 1 is vertical
        x = line1['xf']
        m1 = (line2['yf']-line2['y0'])/(line2['xf']-line2['x0'])
        line1['y0']=line2['y0']
        line1['x0']=line2['x0']
    elif line2['xf']-line2['x0']==0:
        # line 2 is vertical
        x = line2['xf']
        m1 = (line1['yf']-line1['y0'])/(line1['xf']-line1['x0'])
    else:
        m1 = (line1['yf']-line1['y0'])/(line1['xf']-line1['x0'])
        m2 = (line2['yf']-line2['y0'])/(line2['xf']-line2['x0'])
        x = (line2['y0']-line1['y0']-m2*line2['x0']-m1*line1['x0'])/(m1-m2)
    y = line1['y0']+m1*(x-line1['x0'])
    return int(x),int(y)



        
        


class vidData:
    '''holds metadata and tables about video'''
    
    def __init__(self, folder:str, pxpmm:float=cfg.const.pxpmm):
        self.folder = folder
#         self.folder = folder
#         self.nozData = nozData(folder)
#         self.pv = printVals(folder) # object that holds metadata about folder
#         self.file = self.pv.vidFile()    # video file
#         self.measures = []
#         self.measuresUnits = []
#         self.nozMask = []
#         self.prog = []
#         self.streamOpen = False
#         self.pxpmm = pxpmm
#         self.importNozzleDims() # if the pxpmm was defined in file, this will adopt the pxpmm from file
#         if not os.path.exists(self.file):
#             # file does not exist
#             return   
#         pg = self.getProgDims()
#         if pg>0:
#             return
#         self.defineCritVals()       
        
    def getProgDims(self) -> int:
        '''get line starts and stops'''
        self.pv.importProgDims() # generate programmed timings
        self.prog = self.pv.progDims   # programmed timings
        if len(self.prog)==0:
            self.stream.release()
            # no timing file
            return 2
        self.maxT = self.prog.tf.max() # final time in programmed run
        return 0
        
    def openStream(self) -> None:
        '''open the video stream and get metadata'''
        if not self.streamOpen:
            self.stream = cv.VideoCapture(self.file)
            self.frames = int(self.stream.get(cv.CAP_PROP_FRAME_COUNT)) # total number of frames
            self.fps = self.stream.get(cv.CAP_PROP_FPS)
            self.duration = self.frames/self.fps
            self.streamOpen = True
        
    def setTime(self, t:float) -> None:
        '''go to the time in seconds, scaling by video length to fluigent length'''
        self.stream.set(cv.CAP_PROP_POS_FRAMES,int(t/self.maxT*self.frames))
        
    def getFrameAtTime(self, t:float) -> None:
        '''get the frame at a specific time'''
        self.openStream()
        self.setTime(t)
        grabbed, frame = self.stream.read() # read frame
        if not grabbed:
            logging.info(f'Frame not collected at time {t}: {streamInfo(self.stream)}')
            return 1
        else:
            return frame[5:-5,5:-5] # crop
        self.closeStream()
    
   

    
        
    def closeStream(self) -> None:
        '''close the stream'''
        if self.streamOpen:
            self.stream.release()
            self.streamOpen = False
    
    
    
        
    
    



