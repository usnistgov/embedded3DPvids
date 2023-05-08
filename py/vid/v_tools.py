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
from progDim.prog_dim import getProgDims
from tools.plainIm import *
import file.file_handling as fh

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

def frameError(pfd) -> str:
    '''determine if the timing of the video is mismatched from the table'''
    if pfd.date>220901:
        if pfd.date<230117:
            return 'sc_sh'  # rescale and shift the video
        else:
            return 'sh'  # just shift the video
    else:
        return 'sc'   # rescale the video


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
        self.frameError = frameError(self.pfd)
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
        

        if self.frameError[-2:]=='sh':
            self.progDims.importProgPos()
            pp = self.progDims.progPos
            if self.frameError=='sc_sh':
                self.maxT = pp[pp.zt<0].tf.max()
            elif self.frameError=='sh':
                self.maxT = pp.tf.max()
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
            if 'dstart_manual' in d:
                self.dstart = d['dstart_manual']
                self.dstart_manual = d['dstart_manual']
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
            if not hasattr(self, st) or overwrite:
                self.openStream(overwrite=overwrite)
                return
        self.exportVidStats0()

    def exportVidStats0(self):
        '''export the video stats'''
        l = ['frames', 'fps', 'duration', 'dstart', 'dstart_manual']
        fn = self.vidStatsFN()  # nozzle dimensions file name
        d = {}
        for li in l:
            if hasattr(self, li):
                d[li] = getattr(self, li)
        plainExpDict(fn, d)
        
    def importMetaData(self):
        '''import the frame rate from the metadata file'''
        self.collectionFrameRate = 120
        if not hasattr(self.pfd, 'meta') or len(self.pfd.meta)==0:
            return 1
        file = self.pfd.meta[0]
        d,u = plainImDict(file, unitCol=1, valCol=2)
        if 'Basler_camera_collection_frame_rate' in d:
            self.collectionFrameRate = d['Basler_camera_collection_frame_rate']
            

        
    def getVidStats(self) -> None:
        '''get the video stats from the stream'''
        self.importMetaData()
        self.frames = int(self.stream.get(cv.CAP_PROP_FRAME_COUNT)) # total number of frames
        self.fps = self.stream.get(cv.CAP_PROP_FPS)
        self.duration = self.frames/self.fps
        if hasattr(self, 'dstart_manual'):
            self.dstart = self.dstart_manual
            return
        if self.frameError=='sc_sh':
            if not hasattr(self, 'maxT'):
                self.getProgDims()
            # timing rate should be correct, but vid started earlier than timing
            self.dstart = max(self.duration-self.maxT,0)+1.25
        elif self.frameError=='sh':
            self.dstart = 0
            shift = max(0, ((300/self.collectionFrameRate)-1)/10)
            self.dstart = self.dstart-shift
            self.rawPicTimes()   # determine if the raw pic times match the progDims times and shift dstart if they are off
        else:
            self.dstart = 0
        
        
    def openStream(self, overwrite:bool=False) -> None:
        '''open the video stream and get metadata'''
        if not self.streamOpen:
            self.stream = cv.VideoCapture(self.file)
            self.streamOpen = True
        result = self.importVidStats()
        if result>0 or overwrite:
            self.getVidStats()
            self.exportVidStats0()
            
        
    def setTime(self, t:float) -> None:
        '''go to the time in seconds, scaling by video length to fluigent length'''
        if self.frameError[-2:]=='sh':
            # offset start time
            f = int((t+self.dstart)*self.fps)
        else:
            # convert time to frames
            f = max(1, int(t/self.duration*self.frames))
        if f>=self.frames:
            f = self.frames-1
        self.stream.set(cv.CAP_PROP_POS_FRAMES,f)
        
    def getFrameAtTime(self, t:float, overwrite:bool=False) -> None:
        '''get the frame at a specific time'''
        self.openStream(overwrite=overwrite)
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
            
    def picTime(self, file:str) -> None:
        '''get the time of the picture in minutes'''
        if 'Thumbs' in file:
            return -1
        bn = os.path.splitext(os.path.basename(file))[0]
        spl = re.split('_', bn)
        return float(spl[-2][-6:-4])*60*60+float(spl[-2][-4:-2])*60+float(spl[-2][-2:])+float(spl[-1])/10
            
    def rawPicTimes(self) -> pd.DataFrame:
        '''get the times when the raw pictures were taken and compare to progDims'''
        vidStart = self.picTime(self.file)
        rawfolder = os.path.join(self.folder, 'raw')
        if not os.path.exists(rawfolder):
            return []
#         print([self.picTime(file) for file in os.listdir(rawfolder)])
#         print(vidStart)
        times = [self.picTime(file)-vidStart for file in os.listdir(rawfolder)]
        times.sort()
#         print(times)
        times = list(filter(lambda x:x>0, times))
        if not hasattr(self, 'prog'):
            self.getProgDims()
        calc = self.prog[self.prog.name.str.contains('o')].tpic
        if not len(calc)==len(times):
            logging.warning(f'Mismatch in {self.folder}: {len(times)} pics and {len(calc)} progDims')
            return {'raw':[round(x,2) for x in times], 'calc':[round(x,2) for x in calc]}
        diffe = times-calc
        if self.frameError=='sh':
            self.dstart = self.dstart + round(diffe.mean(), 2)
        df = pd.DataFrame({'raw':[round(x,2) for x in times], 'calc':[round(x,2) for x in calc], 'diffo':[round(x,2) for x in diffe]})
        return df
            
            
    def exportStills(self, prefixes:list=[], overwrite:bool=False, diag:int=1, **kwargs) -> None:
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
        elif self.pfd.printType=='SDT':
            prefix = fh.SDTName(os.path.basename(self.folder))
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
                frame = self.getFrameAtTime(row['tpic'], (overwrite and i==0))   # this also exports video stats on the first loop if overwriting
                cv.imwrite(fn, frame)
                if diag>0:
                    logging.info(f'Exported {fn}')
            
#----------------------------------------------

class stillsExporter(fh.folderLoop):
    '''for exporting stills from a video. folders is either a list of folders or the top folder. fileMetricFunc is a class definition for a fileMetric object. overwrite=True to overwrite images.'''
    
    def __init__(self, folders:Union[str, list], overwritePics:bool=False, overwriteDims:bool=False, exportDiag:int=0, **kwargs):
        super().__init__(folders, self.folderFunc, **kwargs)
        self.overwritePics = overwritePics
        self.overwriteDims = overwriteDims
        self.exportDiag = exportDiag

    def folderFunc(self, folder, **kwargs) -> None:
        '''the function to run on a single folder'''
        pdim = getProgDims(folder)
        pdim.exportAll(overwrite=self.overwriteDims)
        vd = vidData(folder)
        vd.exportStills(overwrite=self.overwritePics, diag=self.exportDiag, **kwargs)
        
        
class setManualStillTime(fh.folderLoop):
    '''for exporting stills from a video. folders is either a list of folders or the top folder. fileMetricFunc is a class definition for a fileMetric object. overwrite=True to overwrite images.'''
    
    def __init__(self, folders:Union[str, list], **kwargs):
        super().__init__(folders, self.folderFunc, **kwargs)

    def folderFunc(self, folder:str, dstart:float=0, **kwargs) -> None:
        '''manually adjust the start time for collecting stills for this folder'''
        vd = vidData(folder)
        vd.dstart_manual = dstart
        vd.dstart = dstart
        vd.exportStills(overwrite=True, **kwargs)
