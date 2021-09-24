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
import imutils

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
from printVals import printVals
from imshow import imshow
import vidMorph as vm
from config import cfg
from plainIm import *
from fileHandling import isSubFolder

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


def streamInfo(stream) -> Tuple:
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
    
    def __init__(self, folder:str):
        self.folder = folder
        self.pv = printVals(folder) # object that holds metadata about folder
        self.file = self.pv.vidFile()    # video file
        self.measures = []
        self.measuresUnits = []
        self.nozMask = []
        self.prog = []
        self.streamOpen = False
        if not os.path.exists(self.file):
            # file does not exist
            return   
        pg = self.getProgDims()
        if pg>0:
            return
        self.defineCritVals()
        
        
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
        
        
    def closeStream(self) -> None:
        '''close the stream'''
        if self.streamOpen:
            self.stream.release()
            self.streamOpen = False
            
            
    
    def defineCritVals(self):
        '''critical values to trigger an error'''
        self.xLmin = 200 # px
        self.xLmax = 400
        self.xRmin = 300
        self.xRmax = 600
        self.yBmin = 250
        self.yBmax = 400
        self.nozwidthMin = 0.75 # mm
        self.nozWidthMax = 1.05 # mm
        
    #-----------------------------
    
    def nozzleFrame(self, mode:int=0) -> np.array:
        '''get an averaged frame from several points in the stream to blur out all fluid and leave just the nozzle. mode=0 to use median frame, mode=1 to use mean frame'''
        if len(self.prog)==0:
            raise ValueError('No programmed timings in folder')
        l0 = list(self.prog.loc[:10, 'tf'])
        l1 = list(self.prog.loc[1:, 't0'])
        ar = np.asarray([l0,l1]).transpose()
        tlist = np.mean(ar, axis=1) # list of times in gaps between prints
        tlist = np.insert(tlist, 0, 0)
        frames = [self.getFrameAtTime(t) for t in tlist]
        if mode==0:
            out = np.median(frames, axis=0).astype(dtype=np.uint8) 
        elif mode==1:
            out = np.mean(frames, axis=0).astype(dtype=np.uint8) 
        elif mode==2:
            out = np.max(frames, axis=0).astype(dtype=np.uint8) 
        return out
    
    def drawNozzleOnFrame(self, colors:bool=True) -> None:
        '''draw the nozzle on the original frame'''
        # draw left and right edge
        for i,line in self.lines.iterrows():
            if colors:
                c = (0,0,255)
            else:
                c = (255,255,255)
            cv.line(self.line_image,(int(line['x0']),int(line['y0'])),(int(line['xf']),int(line['yf'])),c,2)
            
        # draw bottom edge
        if colors:
            c = (255,255,0)
        else:
            c = (255,255,255)
        cv.line(self.line_image,(int(self.lines.loc[0,'xf']),int(self.yB)),(int(self.lines.loc[1,'xf']),int(self.yB)),c,2)
        
        # draw corner points
        for l in [self.leftCorner, self.rightCorner]:
            if colors:
                c = (0,255,0)
            else:
                c = (255,255,255)
            cv.circle(self.line_image,(int(l['xf']),int(l['yf'])),10,c,3)
        
    def drawLinesOnFrame(self) -> None:
        '''draw the list of all lines on the thresholded frame'''
        for df in [self.lines0, self.lines0h]:
            for i,line in df.iterrows():
                color = list(np.random.random(size=3) * 256)
                cv.line(self.lines_image,(int(line['x0']),int(line['y0'])),(int(line['xf']),int(line['yf'])),color,4)
            
    def drawDiagnostics(self, diag:int) -> None:
        '''create an image with diagnostics'''
        if diag==0:
            return
        try:
            self.drawNozzleOnFrame()   # draw nozzle on original frame
        except Exception as e:
            pass
        if diag>1:
            try:
                self.drawLinesOnFrame() 
            except:
                pass
        try:
            imshow(self.line_image, self.lines_image, self.edgeImage, scale=4, title=os.path.basename(self.folder))
        except:
            pass
        
    def thresholdNozzle(self, mode) -> None:
        '''get a nozzle frame and convert it into an edge image'''
        self.openStream()
        frame = self.nozzleFrame(mode=mode)           # get median or averaged frame
        self.closeStream()
        self.line_image = np.copy(frame)              # copy of original frame to draw nozzle lines on
        
        # convert to gray, blurred, normalized
        gray2 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # convert to gray
        gray2 = vm.normalize(gray2)                   # normalize frame
        gray2 = cv.GaussianBlur(gray2,(5,5),0)        # blur edges
        self.lines_image = cv.cvtColor(gray2, cv.COLOR_GRAY2BGR) # blank thresholded image to draw all lines on
        
        # take edge
        thres2 = cv.Canny(gray2, 5, 80)             # edge detect
        thres2 = vm.dilate(thres2,3)                  # thicken edges
        
        # only include points above a certain threshold (nozzle is black, so this should get rid of most ink)
        _,threshmask = cv.threshold(gray2, 120,255,cv.THRESH_BINARY_INV)
        threshmask = vm.dilate(threshmask, 15)
        thres2 = cv.bitwise_and(thres2, thres2, mask=threshmask)

        self.edgeImage = thres2.copy()                # store edge image for displaying diagnostics
        
    def nozzleLines(self) -> pd.DataFrame:
        '''get lines from the stored edge image'''
        rho = 3             # distance resolution in pixels of the Hough grid
        theta = np.pi/180   # angular resolution in radians of the Hough grid
        threshold = 30      # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 50  # minimum number of pixels making up a line
        max_line_gap = 300    # maximum gap in pixels between connectable line segments

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv.HoughLinesP(self.edgeImage, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
        
        if len(lines)==0:
            raise ValueError('Failed to detect any lines in nozzle')

        # convert to dataframe
        lines = pd.DataFrame(lines.reshape(len(lines),4), columns=['x0', 'y0', 'xf', 'yf'], dtype='int32')
        lines['slope'] = abs(lines['x0']-lines['xf'])/abs(lines['y0']-lines['yf'])

        # find horizontal lines
        horizLines = lines[(lines['slope']>20)&(lines['y0']>self.yBmin)&(lines['y0']<self.yBmax)]
        self.lines0h = horizLines.copy()
        
        # only take nearly vertical lines, where slope = dx/dy
        critslope = 0.1
        lines = lines[(lines['slope']<critslope)&(lines['x0']>self.xLmin)&(lines['x0']<self.xRmax)]
        self.lines0 = lines.copy()
        # sort each line by y
        for i,row in self.lines0.iterrows():
            if row['yf']<row['y0']:
                self.lines0.loc[i,['x0','y0','xf','yf']] = list(row[['xf','yf','x0','y0']])
        self.lines0 = self.lines0.convert_dtypes() # convert back to int
        self.lines0 = self.lines0[self.lines0.yf<400]
        
        
    def useHoriz(self) -> None:
        '''use horizontal line to find corners'''
        horizLine = self.lines0h.iloc[0] # dominant line
        xL, yL = lineIntersect(horizLine, self.lines.loc[0])
        self.leftCorner = pd.Series({'xf':xL, 'yf':yL})
        xR, yR = lineIntersect(horizLine, self.lines.loc[1]) 
        self.rightCorner = pd.Series({'xf':xR, 'yf':yR})
        
    def useVerticals(self) -> None:
        '''use vertical lines to find corners'''
        # corners are bottom points of each line
        self.leftCorner = self.lines.loc[0,['xf','yf']]
        self.rightCorner = self.lines.loc[1,['xf', 'yf']]
        
    def findNozzlePoints(self) -> None:
        '''find lines and corners of nozzle from list of lines'''
        # based on line with most votes, group lines into groups on the right edge and groups on the left edge
        best = self.lines0.iloc[0]
        dx = max(10,2*abs(best['xf']-best['x0'])) # margin of error for inclusion in the group
        nearbest = self.lines0[(self.lines0.x0<best['x0']+dx)&(self.lines0.x0>best['x0']-dx)]
        
        # group lines between 0.5 and 1.5 nozzles away on left and right side of best line
        margin = 0.45*cfg.const.pxpmm # half a nozzle
        right = self.lines0[(self.lines0.x0>best['x0']+margin)&(self.lines0.x0<best['x0']+3*margin)] 
        left = self.lines0[(self.lines0.x0<best['x0']-margin)&(self.lines0.x0>best['x0']-3*margin)]
        
        if len(right)>len(left):
            left = nearbest
        else:
            right = nearbest
            
        if len(left)==0 or len(right)==0:
            raise ValueError('Failed to detect left and right edges of nozzle')

        # combine all left lines into one line and combine all right lines into one line
        self.lines = pd.DataFrame([combineLines(left), combineLines(right)])
        
        if len(self.lines0h)>0:
            # use horizontal lines to find corners
            self.useHoriz()
            if min([self.leftCorner['yf'],self.rightCorner['yf']]) > self.lines0.yf.max():
                # horiz line is far below verticals
                # throw out the horiz line and use the verticals
                self.useVerticals()
        else:
            self.useVerticals()
        
        # store left edge x, right edge x, and bottom edge y
        self.xL=self.leftCorner['xf']
        self.xR=self.rightCorner['xf']
        self.xM=(self.xL+self.xR)/2
        
        if abs(self.leftCorner['yf']-self.rightCorner['yf'])>20:
            # if bottom edge is not horizontal, use bottommost point
            self.yB = max([self.leftCorner['yf'],self.rightCorner['yf']])
        else:
            # otherwise, use mean of bottom point of left and right edges
            self.yB = (self.leftCorner['yf']+self.rightCorner['yf'])/2
            
    def nozWidth(self):
        '''nozzle width in mm'''
        return (self.xR-self.xL)/cfg.const.pxpmm
            
    def checkNozzleValues(self) -> None:
        '''check that nozzle values are within expected boudns'''
        # check values
        if self.xL<self.xLmin or self.xL>self.xLmax:
            raise ValueError(f'Detected left corner is outside of expected bounds: {self.xL}')
        if self.xR<self.xRmin or self.xR>self.xRmax:
            raise ValueError(f'Detected right corner is outside of expected bounds: {self.xR}')
        if self.yB<self.yBmin or self.yB>self.yBmax:
            raise ValueError(f'Detected bottom edge is outside of expected bounds: {self.yB}')
        nozwidth = self.nozWidth()
        if nozwidth<self.nozwidthMin:
            raise ValueError(f'Detected nozzle width is too small: {nozwidth} mm')
        if nozwidth>self.nozWidthMax:
            raise ValueError(f'Detected nozzle width is too large: {nozwidth} mm')
            
    def createNozzleMask(self) -> None:
        '''create a nozzle mask, so we can erase nozzle from images'''
        # create mask
        frame = self.line_image
        self.nozMask = 255*np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        self.nozCover = np.copy(frame)*0 # create mask
        average = frame.mean(axis=0).mean(axis=0)
        self.nozCover[0:int(self.yB), self.xL-10:self.xR+10]=255
        m = cv.cvtColor(self.nozCover,cv.COLOR_BGR2GRAY)
        _,self.nozMask = cv.threshold(m,0,255,cv.THRESH_BINARY_INV)
        
    
    def detectNozzle0(self, diag:int=0, suppressSuccess:bool=False, mode:int=0) -> None:
        '''find the bottom corners of the nozzle. suppressSuccess=True to only print diagnostics if the run fails'''
        self.thresholdNozzle(mode)
        self.nozzleLines()
        if len(self.lines0)==0:
            raise ValueError('Failed to detect any vertical lines in nozzle')
        self.findNozzlePoints()
        self.checkNozzleValues()
        self.createNozzleMask()
        if diag>0 and not suppressSuccess:
            self.drawDiagnostics(diag) # show diagnostics
        

    def detectNozzle(self, diag:int=0, suppressSuccess:bool=False, mode:int=0) -> None:
        '''find the bottom corners of the nozzle, trying different images. suppressSuccess=True to only print diagnostics if the run fails'''
        if len(self.prog)==0:
            # nothing detected
            return 1
        for mode in [0,1,2]: # min, then median, then mean
            try:
                self.detectNozzle0(diag=diag, suppressSuccess=suppressSuccess, mode=mode)
            except:
                pass
            else:
                return 0
        # if both failed
        self.drawDiagnostics(diag) # show diagnostics
        raise ValueError('Failed to detect nozzle after 3 iterations')
        
    #------------------------------------------------------------------------------------
        
        
    def maskNozzle(self, frame:np.array) -> np.array:
        '''block the nozzle out of the image'''
        frameMasked = cv.bitwise_and(frame,frame,mask = self.nozMask)
        out = cv.add(frameMasked, self.nozCover) 
        norm = np.zeros(out.shape)
        out = cv.normalize(out,  norm, 0, 255, cv.NORM_MINMAX) # normalize the image
        return out
    
    def measureHorizFrame(self, time:float, s:str, f:float, behindX:float=2, diag:int=0, **kwargs) -> Tuple[dict, dict]:
        '''get measurements from the frame. 
        behindX is distance behind nozzle at which to get vertical displacement.
        s is the name of the line, e.g. horiz0
        f is the fraction representing how far within the line we are'''
        if len(self.nozMask)==0:
            self.detectNozzle()
        self.openStream()
        frame = self.getFrameAtTime(time)
        frame2 = self.maskNozzle(frame)
        acrit=1000
        m = 10
        my = int(frame2.shape[0]*0.25)
        if s[-1]=='1':
            frame2[m:-m, -m:]=255 # empty out  right edges so the filaments don't get removed during segmentation
            frame2[my:-my, -2*m:-m]=0 # fill  right edges so the filaments gets filled
        else:
            frame2[m:-m, :m]=255
            frame2[my:-my, m:2*m]=0
        frame2[:m, m:-m]=255 # empty out top so filaments don't get removed
        
        
        filled, markers, finalAt = vm.segmentInterfaces(frame2, acrit=acrit, diag=(diag>1))
        df = vm.markers2df(markers) # convert to dataframe
        df = df[df.a>acrit]
        out = {'name':s, 'time':time, 'frac':f, 'behindX':behindX}
        if len(df)==0:
            return {},{}
        filI = df.a.idxmax() # index of filament label, largest remaining object
        componentMask = (markers[1] == filI).astype("uint8") * 255 # get largest object
        componentMask = vm.openMorph(componentMask, 5) # remove burrs
        contours = cv.findContours(componentMask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        contours = np.array(contours[1]) # turn into an array
        contours = np.concatenate(contours) # turn into a list of points
        contours = contours.reshape((contours.shape[0],2)) # reshape
        contours = pd.DataFrame(contours, columns=['x','y']) # list of points on contour
        
        # find how far the ink projects into bath under nozzle
        underNozzle = contours[(contours.x<self.xR)&(contours.x>self.xL)]
        if len(underNozzle)>0:
            bottomPeak = underNozzle[underNozzle.y==underNozzle.y.max()]
            projection = -(bottomPeak.iloc[0]['y'] - self.yB)/cfg.const.pxpmm  # positive shift is upward
            projShift = (self.xM - bottomPeak.iloc[0]['x'])/cfg.const.pxpmm    # positive shift is downstream
            if s[-1]=='1':
                projShift = -projShift
            out['projection'] = projection
            out['projShift'] = projShift
        
        # find vertical displacement behind nozzle
        dist = -2*cfg.const.pxpmm # 2 mm
        if s[-1]=='1':
            dist = -dist
        dx = 0.25*cfg.const.pxpmm
        behind = contours[(contours.x>self.xM+dist-dx)&(contours.x<self.xM+dist+dx)] # span near 2mm behind nozzle
        if len(behind)>0:
            behindBot = behind[behind.y>behind.y.mean()] # bottom edge
            out['vertDispBot'] = -(behindBot.y.mean() - self.yB)/cfg.const.pxpmm # positive shift is upward
            behindTop = behind[behind.y<behind.y.mean()] # top edge
            out['vertDispTop'] = -(behindTop.y.mean() - self.yB)/cfg.const.pxpmm # positive shift is upward
            out['vertDispMid'] = (out['vertDispTop']+out['vertDispBot'])/2       # positive shift is upward
        else:
            vertDispBot = ''
            vertDispMid = ''
            vertDispTop = ''
            
        # plot results
        if diag>0:
#             componentMask = cv.cvtColor(componentMask, cv.COLOR_GRAY2BGR)
            componentMask = frame.copy()
            self.drawNozzleOnFrame(componentMask, colors=False)
            if len(behind)>0:
                cv.circle(componentMask,(bottomPeak.iloc[0]['x'],bottomPeak.iloc[0]['y']),5,(0,255,0),5)
                cv.circle(componentMask,(int(behindBot.x.mean()),int(behindBot.y.mean())),5,(0,0,255),5)
                cv.circle(componentMask,(int(behind.x.mean()),int((behindBot.y.mean()+behindTop.y.mean())/2)),5,(255,0,255),5)
                cv.circle(componentMask,(int(behindTop.x.mean()),int(behindTop.y.mean())),5,(255,0,0),5)
            imshow(componentMask)
            if len(underNozzle)>0:
                plt.plot(underNozzle['x'], underNozzle['y'], color='g')
            if len(behind)>0:
                plt.plot(behindBot['x'], behindBot['y'], color='r')
                plt.plot(behindTop['x'], behindTop['y'], color='b')
#             fig = plt.gcf()
#             fig.savefig(os.path.join(cfg.path.fig, 'figures', 'horizVid_detection.svg'), bbox_inches='tight', dpi=300)
        units = {'name':'','time':'s', 'frac':'','behindX':'mm','projection':'mm', 'projShift':'mm', 'vertDispBot':'mm', 'vertDispMid':'mm', 'vertDispTop':'mm'}
        self.closeStream()
        return out, units
    
    def measureFrameFromLine(self, s:str, f:float, diag:int=0, **kwargs) -> Tuple[dict,dict]:
        '''get measurements from a frame, where you designate the line name (s) and how far within the line to collect (f)'''
        row = (self.prog[self.prog.name==s])
        t0 = row.iloc[0]['t0']
        tf = row.iloc[0]['tf']
        dt = tf-t0
        t = t0+dt*f
        return self.measureHorizFrame(t, s, f, diag=diag, **kwargs)
    
    def vidMeasuresFN(self) -> str:
        '''file name for video measurement table'''
        return os.path.join(self.pv.folder, os.path.basename(self.pv.folder)+'_vidHorizMeasures.csv')
        
        
    def measureVideo(self, diag:int=0, overwrite:bool=False, **kwargs) -> Tuple[pd.DataFrame, dict]:
        '''get info about the ellipses. Returns 1 when video is done. Returns 0 to continue grabbing.'''
        fn  = self.vidMeasuresFN()
        if os.path.exists(fn):
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
                    framerow, u = self.measureHorizFrame(t, s, f, diag=d, **kwargs)
                except:
                    print(s, t)
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
    
    def measureAll(self, diag:int=0, **kwargs) -> Tuple[pd.DataFrame, dict]:
        '''initialize data and measure the video'''
        try:
            self.detectNozzle(diag=diag)
        except:
            logging.error(f'Error detecting nozzle in {self.folder}')
            traceback.print_exc()
            return
        self.measureVideo(diag=diag, **kwargs)
        
        
    def importVidMeasures(self) -> Tuple[pd.DataFrame, dict]:
        '''import measurements from file'''
        self.measures, self.measuresUnits = plainIm(self.vidMeasuresFN(), ic=0)
        
    def summary(self) -> Tuple[dict,dict]:
        '''summarize measurement table into a single row'''
        if len(self.measures)==0:
            self.importVidMeasures()
        if len(self.measures)==0:
            return {}, {}
        data = {}
        units = {}
        for s in ['projection', 'projShift', 'vertDispBot', 'vertDispTop', 'vertDispMid']:
            # mean, with 3 sigma outliers removed, normalized by nozzle inner diameter
            if s in self.measures:
                data[s+'N'] = removeOutliers(self.measures, s)[s].mean()/cfg.const.di 
                data[s+'N_SE'] = removeOutliers(self.measures, s)[s].sem()/cfg.const.di 
                units[s+'N'] = ''
                units[s+'N_SE'] = ''
        meta,metaunits = self.pv.metarow()
        data = {**meta,**data}
        units = {**metaunits,**units}
        return data, units
        
#-----------------------------------------
        
        
def measureVideosRecursive(topfolder:str, **kwargs) -> None:
    '''go through all of the videos and measure them'''
    if isSubFolder(topfolder):
        try:
            vd = vidData(topfolder)
            vd.measureAll(**kwargs)
        except:
            traceback.print_exc()
            pass
    else:
        for f1 in os.listdir(topfolder):
            f1f = os.path.join(topfolder, f1)
            if os.path.isdir(f1f):
                measureVideosRecursive(f1f, **kwargs)
                
                
#----------------------------------------


def summarizeVideosRecursive(topfolder:str) -> None:
    '''go through all of the videos and measure them'''
    data = []
    units = []
    if isSubFolder(topfolder):
        try:
            vd = vidData(topfolder)
            d,u = vd.summary()
            if len(d)>0:
                data = [d]
                units = u
        except:
            traceback.print_exc()
            pass
    else:
        data = []
        units = []
        for f1 in os.listdir(topfolder):
            f1f = os.path.join(topfolder, f1)
            if os.path.isdir(f1f):
                d,u = summarizeVideosRecursive(f1f)
                if len(d)>0:
                    data = data + d
                    if len(u)>len(units):
                        units = u
    return data, units

def videoSummary(topfolder:str, exportFolder:str, filename:str='videoSummary.csv') -> pd.DataFrame:
    '''go through all of the folders and summarize the stills'''
    tt,units = summarizeVideosRecursive(topfolder)
    tt = pd.DataFrame(tt)
    if os.path.exists(exportFolder):
        plainExp(os.path.join(exportFolder, filename), tt, units)
    return tt,units


