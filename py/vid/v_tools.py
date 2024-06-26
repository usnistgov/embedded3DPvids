#!/usr/bin/env python
'''Functions for holding metadata and data from a video'''

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
import imageio
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
    
    def __init__(self, folder:str, **kwargs):
        self.folder = folder
        if 'pfd' in kwargs:
            self.pfd = kwargs['pfd']
        else:
            self.pfd = fh.printFileDict(folder)
        if len(self.pfd.vid)>0:
            self.file = self.pfd.vid[0]    # video file
        else:
            logging.warning(f'No video file found in {self.folder}')
            self.file = ''
        if 'dstart_manual' in kwargs:
            self.dstart_manual = kwargs['dstart_manual']
        self.measures = []
        self.measuresUnits = []
        self.streamOpen = False
        self.pxpmm = self.pfd.pxpmm()
        self.frameError = frameError(self.pfd)

    def getProgDims(self) -> int:
        '''get line starts and stops'''
        try:
            self.progDims = getProgDims(self.folder)
        except ValueError:
            # no print type known
            return 1
        self.progDims.importProgDims()
        if len(self.progDims.progDims.tpic.dropna())<len(self.progDims.progDims):
            self.progDims.getProgDims()
            self.progDims.exportProgDims(overwrite=True)
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
            
            # adopt frame num, fps, and duration from file, and dstart if we haven't defined a manual dstart
            tlist = ['frames', 'fps', 'duration']
            if not hasattr(self, 'dstart_manual'):
                tlist.append('dstart')
            for st,val in d.items():
                if st in tlist:
                    setattr(self, st, val)
                    
            # adopt dstart
            if hasattr(self, 'dstart_manual'):
                print(self.dstart_manual)
                self.dstart = self.dstart_manual
            elif 'dstart_manual' in d:
                self.dstart = d['dstart_manual']
                self.dstart_manual = d['dstart_manual']
                
            # missing values
            if len(set(tlist)-set(d))>0:
                return 1
            else:
                return 0
                        
    def exportVidStats(self, overwrite:bool=False) -> None:
        '''export the video stats'''
        fn = self.vidStatsFN()  # nozzle dimensions file name
        if os.path.exists(fn) and not overwrite:
            return
        l = ['frames', 'fps', 'duration', 'dstart', 'dstart_manual']
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
            # self.dstart = 0.55
            # shift = max(0, ((300/self.collectionFrameRate)-1)/10)
            # self.dstart = self.dstart+shift
            shift = 1.2-0.002*self.collectionFrameRate
            self.dstart = shift
            self.rawPicTimes()   # determine if the raw pic times match the progDims times and shift dstart if they are off
        else:
            self.dstart = 0
        
        
    def openStream(self, overwrite:bool=False) -> None:
        '''open the video stream and get metadata'''
        if self.streamOpen:
            return

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
        if len(calc)==4*len(times) or len(calc)==4*(len(times)-2):
            # extra calcs
            calc = self.prog[(self.prog.name.str.contains('o1'))|self.prog.name.str.contains('o8')].tpic
        if len(calc)==4*(len(times)-2) or len(calc)==len(times)-2:
            times = times[2:]
        if not (len(calc)==len(times) or len(calc)+2==len(times)):
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
                    logging.info(f'Exported {os.path.basename(fn)}')
                    
    def exportGIF(self, line:str, compression:int=1, speedScale:float=1, color:bool=True, crop:dict={}, sizeCompression:int=1, prestart:float=0, postend:float=0) -> None:
        '''export a gif of just the writing and observing of one line. line is the line name, e.g. l1w1.
        compression is the factor of how many frames to drop. e.g take on frame per compression frames
        speedScale is how much to speed up the video'''
        self.openStream()
        dt = compression/self.fps  # time step size to take frames
        giffps = int(speedScale/dt)   # frames per second of the gif
        newName = self.pfd.newFileName(f'clip_{line}', 'gif')
        result = imageio.get_writer(newName, fps=giffps)
        if not hasattr(self, 'prog'):
            self.getProgDims()
        pline = self.prog[self.prog.name.str.contains(f'{line}p')]
        if len(pline)==0:
            raise ValueError(f'Cannot find line {line}')
        t0 = pline.iloc[0]['t0']+prestart
        oline = self.prog[self.prog.name.str.contains(f'{line}o')]
        tf = oline.iloc[-1]['tpic']+postend

        for t in np.arange(t0, tf+dt, dt):
            frame = self.getFrameAtTime(t, False)
            if color:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            else:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            if 'y0' in crop and 'yf' in crop and 'x0' in crop and 'xf' in crop:
                frame = frame[crop['y0']:crop['yf'], crop['x0']:crop['xf']]
            if not sizeCompression==1:
                dim = frame.shape
                h = int(dim[0]/sizeCompression)
                w = int(dim[1]/sizeCompression)
                dim = (w,h)
                frame = cv.resize(frame, dim, interpolation = cv.INTER_AREA)
            result.append_data(frame)

          # When everything done, release 
        # the video capture and video 
        # write objects
        result.close()
        file_stats = os.stat(newName)
        logging.info(f'Exported {newName}: {file_stats.st_size / (1024 * 1024)} MB')
        return
            
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
        vd.exportVidStats0()
        