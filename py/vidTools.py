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
        
        
    def closeStream(self) -> None:
        '''close the stream'''
        if self.streamOpen:
            self.stream.release()
            self.streamOpen = False
    
    
    def defineCritVals(self):
        '''critical nozzle position values that indicate nozzle detection may have failed, so trigger an error'''
        self.xLmin = 200 # px
        self.xLmax = 500
        self.xRmin = 300
        self.xRmax = 600
        self.yBmin = 250
        self.yBmax = 430
        self.nozwidthMin = 0.75 # mm
        self.nozWidthMax = 1.05 # mm
        
    #-----------------------------
    
    def nozzleFrame(self, mode:int=0) -> np.array:
        '''get an averaged frame from several points in the stream to blur out all fluid and leave just the nozzle. 
        mode=0 to use median frame, mode=1 to use mean frame, mode=2 to use lightest frame'''
        if len(self.prog)==0:
            raise ValueError('No programmed timings in folder')
        l0 = list(self.prog.loc[:10, 'tf'])     # list of end times
        l1 = list(self.prog.loc[1:, 't0'])      # list of start times
        ar = np.asarray([l0,l1]).transpose()    # put start times and end times together
        tlist = np.mean(ar, axis=1)             # list of times in gaps between prints
        tlist = np.insert(tlist, 0, 0)          # put a 0 at the beginning, so take a still at t=0 before the print starts
        frames = [self.getFrameAtTime(t) for t in tlist]  # get frames in gaps between prints
        if mode==0:
            out = np.median(frames, axis=0).astype(dtype=np.uint8) # median frame
        elif mode==1:
            out = np.mean(frames, axis=0).astype(dtype=np.uint8)  # average all the frames
        elif mode==2:
            out = np.max(frames, axis=0).astype(dtype=np.uint8)   # lightest frame
        return out
    
    def drawNozzleOnFrame(self, colors:bool=True) -> None:
        '''draw the nozzle outline on the original frame. colors=True to draw annotations in color, otherwise in white'''
        # draw left and right edge
        try:
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
        except:
            pass
        
        # draw corner points
        for l in [self.xL, self.xR]:
            if colors:
                c = (0,255,0)
            else:
                c = (255,255,255)
            cv.circle(self.line_image,(int(l),int(self.yB)),10,c,3)
        
    def drawLinesOnFrame(self) -> None:
        '''draw the list of all detected nozzle edge lines on the thresholded frame'''
        for df in [self.lines0, self.lines0h]:
            for i,line in df.iterrows():
                color = list(np.random.random(size=3) * 256)            # assign a random color to each line
                cv.line(self.lines_image,(int(line['x0']),int(line['y0'])),(int(line['xf']),int(line['yf'])),color,4)
                
    def initLineImage(self) -> None:
        '''initialize line_image0 and line_image, which are diagnostic images that help to visualize nozzle detection metrics'''
        try:
            self.line_image = self.line_image0.copy()
        except:
            self.openStream()
            frame = self.nozzleFrame()           # get median or averaged frame
            self.closeStream()
            self.line_image0 = np.copy(frame) 
            self.line_image = self.line_image0.copy()
            
    def drawDiagnostics(self, diag:int) -> None:
        '''create an image with diagnostics.
        diag=0 to only get the nozzle image. 1 to draw only final nozzle edges. 2 to draw all detected edges. '''
        self.initLineImage()      # create images to annotate
        if diag==0:
            return
        try:
            self.drawNozzleOnFrame()   # draw nozzle on original frame
        except Exception as e:
            traceback.print_exc()
            pass
        if diag>1:
            try:
                self.drawLinesOnFrame()   # draw detected lines on the frame
            except:
                traceback.print_exc()
                pass
        try:
            imshow(self.line_image, self.lines_image, self.edgeImage, scale=4, title=os.path.basename(self.folder))
        except Exception as e:
            if 'lines_image' in str(e):
                imshow(self.line_image, scale=4, title=os.path.basename(self.folder))
            pass
        
    def showFrames(self, tlist:List[float], crop:dict={'x0':0, 'xf':-1, 'y0':0, 'yf':-1}, figw:float=6.5) -> None:
        '''show the list of frames at times in tlist. 
        crop is the boundaries of the image, from the top left. 
        figw is the figure width in inches'''
        n = len(tlist)
        
        frames = [vm.white_balance(self.getFrameAtTime(t)[crop['y0']:crop['yf'],crop['x0']:crop['xf']]) for t in tlist]  # apply white balance to each frame
        
        h = frames[0].shape[0]    # size of the images
        w = frames[0].shape[1]
        fig,axs = plt.subplots(1,n, figsize=(figw, figw*h/w))  # create one subplot for each time
        if n==1:
            axs = [axs]
        for i,im in enumerate(frames):
            # show the image in the plot
            if len(im.shape)>2:
                axs[i].imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
            else:
                axs[i].imshow(im, cmap='Greys')
                
            # show the relative time in the title of the axis
            axs[i].set_title('{:.3} s'.format(tlist[i]-tlist[0]), fontsize=8)
            axs[i].axis('off')
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.05, hspace=0)
        plt.close()
        return fig
        
        
    def thresholdNozzle(self, mode) -> None:
        '''get a nozzle frame and convert it into an edge image.
        mode=0 to use median frame, mode=1 to use mean frame, mode=2 to use lightest frame
        '''
        self.openStream()
        frame = self.nozzleFrame(mode=mode)           # get median or averaged frame
        self.closeStream()
        self.line_image0 = np.copy(frame)              # copy of original frame to draw nozzle lines on
        
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
        self.lines0 = self.lines0.convert_dtypes()         # convert back to int
        self.lines0 = self.lines0[self.lines0.yf<400]      # only take lines that extend close to the top of the frame
        
        
    def useHoriz(self) -> None:
        '''use horizontal line to find nozzle corners'''
        horizLine = self.lines0h.iloc[0]                        # dominant line
        xL, yL = lineIntersect(horizLine, self.lines.loc[0])    # find the point where the horizontal line and left vertical line intersect
        self.leftCorner = pd.Series({'xf':xL, 'yf':yL})
        xR, yR = lineIntersect(horizLine, self.lines.loc[1])    # find the point where the horizontal line and right vertical line intersect
        self.rightCorner = pd.Series({'xf':xR, 'yf':yR})
        
    def useVerticals(self) -> None:
        '''use vertical lines to find corners'''
        # corners are bottom points of each line
        self.leftCorner = self.lines.loc[0,['xf','yf']]         # take bottom point of left vertical line
        self.rightCorner = self.lines.loc[1,['xf', 'yf']]       # take bottom point of right vertical line
        
    def findNozzlePoints(self) -> None:
        '''find lines and corners of nozzle from list of lines'''
        
        # based on line with most votes, group lines into groups on the right edge and groups on the left edge
        
        best = self.lines0.iloc[0]                # line with most votes
        dx = max(10,2*abs(best['xf']-best['x0'])) # margin of error for inclusion in the group
        nearbest = self.lines0[(self.lines0.x0<best['x0']+dx)&(self.lines0.x0>best['x0']-dx)]  # lines that are near the best line
        
        # group lines between 0.5 and 1.5 nozzles away on left and right side of best line
        margin = 0.45*cfg.const.pxpmm # half a nozzle
        right = self.lines0[(self.lines0.x0>best['x0']+margin)&(self.lines0.x0<best['x0']+3*margin)] # lines that are 1-3 margins to the left of the best line
        left = self.lines0[(self.lines0.x0<best['x0']-margin)&(self.lines0.x0>best['x0']-3*margin)]  # lines that are 1-3 margins to the right of the best line
        
        if len(right)>len(left):
            left = nearbest     # best was the left side, use nearbest as the left lines
        else:
            right = nearbest    # best was the right side, use nearbest as the right lines
            
        if len(left)==0 or len(right)==0:
            raise ValueError('Failed to detect left and right edges of nozzle')

        # combine all left lines into one line and combine all right lines into one line
        self.lines = pd.DataFrame([combineLines(left), combineLines(right)])
        
        if len(self.lines0h)>0:
            # we have already defined horizontal lines. use horizontal lines to find corners
            self.useHoriz()
            if min([self.leftCorner['yf'],self.rightCorner['yf']]) > self.lines0.yf.max():
                # horiz line is far below verticals
                # throw out the horiz line and use the verticals
                self.useVerticals()
        else:
            # we do not have horizontal lines. Use the bottom of the verticals to defined the bottom edge
            self.useVerticals()
        
        # store left edge x, right edge x, and bottom edge midpoint
        self.xL = self.leftCorner['xf']   # left corner x
        self.xR = self.rightCorner['xf']  # right corner x
        self.xM = (self.xL+self.xR)/2     # middle of bottom edge
        
        if abs(self.leftCorner['yf']-self.rightCorner['yf'])>20:
            # if bottom edge is not horizontal, use bottommost point for y position of nozzle bottom
            self.yB = max([self.leftCorner['yf'],self.rightCorner['yf']])
        else:
            # otherwise, use mean of bottom point of left and right edges
            self.yB = (self.leftCorner['yf']+self.rightCorner['yf'])/2
            
    def nozWidth(self):
        '''nozzle width in mm'''
        return (self.xR-self.xL)/cfg.const.pxpmm
            
    def checkNozzleValues(self) -> None:
        '''check that nozzle values are within expected bounds'''
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
        try:
            frame = self.line_image
        except:
            frame = self.getFrameAtTime(1)
        
        self.nozMask = 255*np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        self.nozCover = np.copy(frame)*0                          # create empty mask
#         average = frame.mean(axis=0).mean(axis=0)                 # background color
        self.nozCover[0:int(self.yB), self.xL-10:self.xR+10]=255      # set the nozzle region to white
        m = cv.cvtColor(self.nozCover,cv.COLOR_BGR2GRAY)              # convert to gray
        _,self.nozMask = cv.threshold(m,0,255,cv.THRESH_BINARY_INV)   # binary mask of nozzle
        
    def nozDimsFN(self) -> str:
        '''file name of nozzle dimensions table'''
        return os.path.join(self.pv.folder, f'{os.path.basename(self.pv.folder)}_nozDims.csv')
        
    def exportNozzleDims(self) -> None:
        '''export the nozzle location to file'''
        fn = self.nozDimsFN()  # nozzle dimensions file name
        with open(fn, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['yB', str(self.yB)])
            writer.writerow(['xL', str(self.xL)])
            writer.writerow(['xR', str(self.xR)])
        logging.info(f'Exported {fn}')
        
    def importNozzleDims(self) -> None:
        '''find the target pressure from the calibration file. returns 0 if successful, 1 if not'''
        fn = self.nozDimsFN()      # nozzle dimensions file name
        if not os.path.exists(fn):
            return 1
        tlist = ['yB', 'xL', 'xR']
        with open(fn, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                # save all rows as class attributes
                if row[0] in tlist:
                    tlist.remove(row[0])
                    setattr(self, row[0], int(float(row[1])))
        if len(tlist)==0:
            self.xM = (self.xL+self.xR)/2
            self.initLineImage()
            return 0
        else:
            return 1
    
    def detectNozzle0(self, diag:int=0, suppressSuccess:bool=False, mode:int=0) -> None:
        '''find the bottom corners of the nozzle. suppressSuccess=True to only print diagnostics if the run fails'''
        self.thresholdNozzle(mode)    # threshold the nozzle
        self.nozzleLines()            # edge detect and Hough transform to get nozzle edges as lines
        if len(self.lines0)==0:
            raise ValueError('Failed to detect any vertical lines in nozzle')
        self.findNozzlePoints()       # filter the lines to get nozzle coords
        self.checkNozzleValues()      # make sure coords make sense
        self.exportNozzleDims()       # save coords
        self.createNozzleMask()       # create mask that we can use to remove the nozzle from images
        if diag>0 and not suppressSuccess and im==1:
            self.drawDiagnostics(diag) # show diagnostics
        

    def detectNozzle(self, diag:int=0, suppressSuccess:bool=False, mode:int=0, overwrite:bool=False) -> None:
        '''find the bottom corners of the nozzle, trying different images. suppressSuccess=True to only print diagnostics if the run fails'''
        logging.info(f'detecting nozzle in {self.folder}')
        if len(self.prog)==0:
            # no programmed timings detected
            return 1
        if not overwrite:
            im = self.importNozzleDims()
        if im==0:
            self.createNozzleMask()
            # we already detected the nozzle
            return 0
        
        # no existing file: detect nozzle
        for mode in [0,1,2]: # min, then median, then mean
            try:
                self.detectNozzle0(diag=diag, suppressSuccess=suppressSuccess, mode=mode)
            except:
                if diag>1:
                    traceback.print_exc()
                pass
            else:
                return 0
            
        # if all modes failed:
        self.drawDiagnostics(diag) # show diagnostics
        raise ValueError('Failed to detect nozzle after 3 iterations')
        
    #------------------------------------------------------------------------------------
        
        
    def maskNozzle(self, frame:np.array, dilate:int=0, ave:bool=False, **kwargs) -> np.array:
        '''block the nozzle out of the image. 
        dilate is number of pixels to expand the mask. 
        ave=True to use the background color, otherwise use white'''
        frameMasked = cv.bitwise_and(frame,frame,mask = vm.erode(self.nozMask, dilate))
        if ave:
            # mask nozzle with whitest value
            nc = np.copy(frame)*0 # create mask
            average = frame.max(axis=0).mean(axis=0)
            nc[0:int(self.yB), self.xL-10:self.xR+10]=average
        else:
            nc = self.nozCover
        out = cv.add(frameMasked, vm.dilate(nc, dilate)) 
        norm = np.zeros(out.shape)
        out = cv.normalize(out,  norm, 0, 255, cv.NORM_MINMAX) # normalize the image
        return out
    
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
            behindBot = behind[behind.y>behind.y.mean()]                         # bottom edge
            out['vertDispBot'] = -(behindBot.y.mean() - self.yB)/cfg.const.pxpmm # positive shift is upward
            behindTop = behind[behind.y<behind.y.mean()]                         # top edge
            out['vertDispTop'] = -(behindTop.y.mean() - self.yB)/cfg.const.pxpmm # positive shift is upward
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
    
    def measureAll(self, diag:int=0, measureHoriz=True, measureXS=True, exportVert=True, exportHoriz=True, **kwargs) -> str:
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
        
    def summary(self) -> Tuple[dict,dict]:
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
        
        
def measureVideosRecursive(topfolder:str, **kwargs) -> List[str]:
    '''go through all of the videos and measure them. compile a list of folders with errors'''
    errorFolders = []
    if isSubFolder(topfolder):
        try:
            vd = vidData(topfolder)
            s = vd.measureAll(**kwargs)
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
                s = measureVideosRecursive(f1f, **kwargs)
                errorFolders = errorFolders+s
    return errorFolders
                
                
#----------------------------------------


def summarizeVideosRecursive(topfolder:str) -> None:
    '''go through all of the videos and measure them'''
    data = []
    units = []
    if isSubFolder(topfolder):
        # summarize this folder
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
        # go through subfolders and summarize
        logging.info(topfolder)
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


