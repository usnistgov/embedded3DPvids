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
import random
import time

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
from im.imshow import imshow
import im.morph as vm
import im.crop as vc
from tools.config import cfg
from tools.plainIm import *
import file.file_handling as fh
from v_tools import vidData


# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)


#----------------------------------------------

def combineLines(df:pd.DataFrame) -> dict:
    '''combine groups of similar Hough transform lines into one line'''
    top = (df[df.y0==df.y0.min()]).iloc[0]
    bot = (df[df.yf==df.yf.max()]).iloc[0]
    return {'x0':top['x0'],'y0':top['y0'],'xf':bot['xf'],'yf':bot['yf']}

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

class nozData:
    '''holds metadata about the nozzle'''
    
    def __init__(self, folder:str, maskPad:int=0, **kwargs):
        # self.timeCount = time.time()
        # self.timeCounter('start')
        self.printFolder = fh.getPrintFolder(folder, **kwargs)
        # self.timeCounter('printFolder')
        
        if 'pfd' in kwargs:
            self.pfd = kwargs['pfd']
        else:
            self.pfd = fh.printFileDict(self.printFolder)
        # self.timeCounter('pfd')
        self.vidFile = self.pfd.vidFile()
        self.sampleName = fh.sampleName(self.printFolder)
        self.nozMask = []                   # mask that blocks nozzle
        self.prog = []                      # programmed timings
        self.maskPadLeft = maskPad
        self.maskPadRight = maskPad
        self.streamOpen = False
        self.pxpmm = self.pfd.pxpmm()
        self.importNozzleDims()
        # self.timeCounter('import')
        self.defineCritVals()
        # self.timeCounter('critvals')
        
    def timeCounter(self, s:str):
        tt = time.time()
        print(f'nozData {s} {(tt-self.timeCount):0.4f} seconds')
        self.timeCount = tt

        
    #-----------------------------
    
    def nozDimsFN(self) -> str:
        '''file name of nozzle dimensions table'''
        # store noz dimensions in the subfolder
        return self.pfd.newFileName('nozDims', 'csv')
    
    def nozDims(self) -> dict:
        '''get the nozzle dimensions
        yB is from top, xL and xR are from left'''
        return {'xL':self.xL, 'xR':self.xR, 'yB':self.yB}
    
    
    def exportNozzleDims(self, overwrite:bool=False) -> None:
        '''export the nozzle location to file'''
        fn = self.nozDimsFN()  # nozzle dimensions file name
        if os.path.exists(fn) and not overwrite:
            return
        plainExpDict(fn, {'yB':self.yB, 'xL':self.xL, 'xR':self.xR, 'pxpmm':self.pxpmm})
        
        
    def importNozzleDims(self) -> None:
        '''find the target pressure from the calibration file. returns 0 if successful, 1 if not'''
        fn = self.nozDimsFN()      # nozzle dimensions file name
        if not os.path.exists(fn):
            self.nozDetected = False
            return 
        tlist = ['yB', 'xL', 'xR', 'pxpmm']
        d,_ = plainImDict(fn, unitCol=-1, valCol=1)
        for st,val in d.items():
            setattr(self, st, int(val))
        if len(set(tlist)-set(d))==0:
            # we have all values
            self.xM = (self.xL+self.xR)/2
#             self.initLineImage()
            self.nozDetected = True
            return
        else:
            self.nozDetected = False
            return
        
    #---------------------------------

    def defineCritVals(self):
        '''critical nozzle position values that indicate nozzle detection may have failed, so trigger an error'''
        
        # bounds in px for a 600x800 image
        self.xLmin = 200 # px
        self.xLmax = 500
        self.xRmin = 300
        self.xRmax = 600
        self.yBmin = 200
        self.yBmax = 430
        
        # bounds of size of nozzle in mm. for 20 gauge nozzle, diam should be 0.908 mm
        self.nozwidthMin = 0.75 # mm
        self.nozWidthMax = 1.05 # mm
        
    def randTime(self, row:pd.Series) -> float:
        '''get a random time between these two times'''
        f = random.random()
        return row['t0']*f + row['tf']*(1-f)
    
    def nozzleFrame(self, mode:int=0, diag:int=0, numpics:int=6, ymin:int=5, ymax:int=70, zmin:int=-20, overwrite:bool=False, **kwargs) -> np.array:
        '''get an averaged frame from several points in the stream to blur out all fluid and leave just the nozzle. 
        mode=0 to use median frame, mode=1 to use mean frame, mode=2 to use lightest frame'''
        if not hasattr(self, 'frames') or overwrite:
            if len(self.pfd.progPos)>0:
                prog,units = plainIm(self.pfd.progPos, ic=0)
                prog = prog[(prog.l==0)&(prog.zt<0)&(prog.yt>ymin)&(prog.yt<ymax)&(prog.zt>zmin)]        # select moves with no extrusion that aren't close to the edge
                prog.reset_index(inplace=True, drop=True)
                tlist = list((prog['tf']+prog['t0'])/2)
                indices = random.sample(range(0, len(prog)), numpics)
                tlist = [self.randTime(prog.loc[i]) for i in indices]
            else:
                if len(self.pfd.progDims)>0:
                    self.prog, units = plainIm(self.pfd.progDims, ic=0)
                    if len(self.prog)==0:
                        raise ValueError('No programmed timings in folder')
                    else:
                        l0 = list(self.prog.loc[:10, 'tf'])     # list of end times
                        l1 = list(self.prog.loc[1:, 't0'])      # list of start times
                        ar = np.asarray([l0,l1]).transpose()    # put start times and end times together
                        tlist = np.mean(ar, axis=1)             # list of times in gaps between prints
                        tlist = np.insert(tlist, 0, 0)          # put a 0 at the beginning, so take a still at t=0 before the print starts
                else:
                    raise ValueError('No programmed dimensions in folder')
            self.vd = vidData(self.printFolder)
            self.frames = [vm.blackenRed(self.vd.getFrameAtTime(t)) for t in tlist]  # get frames in gaps between prints
            if diag>0:
                imshow(*self.frames, numbers=True, perRow=10)
        if mode==0:
            out = np.median(self.frames, axis=0).astype(dtype=np.uint8) # median frame
        elif mode==1:
            out = np.mean(self.frames, axis=0).astype(dtype=np.uint8)  # average all the frames
        elif mode==2:
            out = np.max(self.frames, axis=0).astype(dtype=np.uint8)   # lightest frame (do not do this for accurate nozzle dimensions)
        return out
    
    def subtractBackground(self, im:np.array, dilate:int=0, diag:int=0) -> np.array:
        '''subtract the nozzle frame from the color image'''
        self.importBackground()
        bg = self.background
        bg = cv.medianBlur(bg, 5)
        subtracted = 255-cv.absdiff(im, bg)
        subtracted = self.maskNozzle(subtracted, dilate=dilate)
        if diag>0:
            imshow(im, bg, subtracted)
        return subtracted
    
    def importBackground(self, overwrite:bool=False) -> None:
        '''import the background from file or create one and export it'''
        if hasattr(self, 'background') and not overwrite:
            # already have a background
            return
        fn = self.pfd.newFileName('background', 'png')
        if not os.path.exists(fn):
            # create a background
            self.exportBackground()
            return
        
        # import background from file
        self.background = cv.imread(fn)
        return
    
    def backgroundFN(self):
        return self.pfd.newFileName('background', 'png')
    
    def exportBackground0(self, diag:int=0):
        fn = self.backgroundFN()
        cv.imwrite(fn, self.background)
        logging.info(f'Exported {fn}')
        if diag>0:
            imshow(self.background)
            
    def exportBackground(self, overwrite:bool=False, diag:int=0, **kwargs) -> None:
        '''create a background file'''
        fn = self.backgroundFN()
        if not os.path.exists(fn) or overwrite:
            self.background = self.nozzleFrame(mode=2, diag=diag-1, overwrite=True, **kwargs)
            self.background = cv.medianBlur(self.background, 5)
            self.exportBackground0(diag=diag)
                
    def stealBackground(self, diag:int=0) -> None:
        '''steal a background from another folder in this series'''
        spacing = re.split('_', os.path.basename(self.printFolder))[-1]
        for n in ['0.625', '0.750', '0.875', '1.000']:
            newfolder = self.printFolder.replace(spacing, n)
            if os.path.exists(newfolder) and not newfolder==self.printFolder:
                nd = nozData(newfolder)
                nd.importBackground()
                if hasattr(nd, 'background'):
                    print(f'Stealing background from {newfolder}')
                    self.background = nd.background
                    self.exportBackground0(diag=diag)
                    return

    
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
            frame = self.nozzleFrame()           # get median or averaged frame
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
            imshow(self.line_image, self.lines_image, self.edgeImage, scale=4, title=os.path.basename(self.printFolder))
        except Exception as e:
            if 'lines_image' in str(e):
                imshow(self.line_image, scale=4, title=os.path.basename(self.printFolder))
            pass
        
    def showFrames(self, tlist:List[float], crop:dict={'x0':0, 'xf':-1, 'y0':0, 'yf':-1}, figw:float=6.5) -> None:
        '''show the list of frames at times in tlist. 
        crop is the boundaries of the image, from the top left. 
        figw is the figure width in inches'''
        n = len(tlist)
        
        frames = [vm.white_balance(self.getFrameAtTime(t)[crop['y0']:crop['yf'],crop['x0']:crop['xf']]) for t in tlist]  # apply white balance to each frame
        
        self.h = frames[0].shape[0]    # size of the images
        self.w = frames[0].shape[1]
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
        frame = self.nozzleFrame(mode=mode)           # get median or averaged frame
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
        
        
    def nozzleLines0(self, im:np.array, yBmin:int, yBmax:int, xLmin:int, xRmax:int, hmax:int
                     , min_line_length:int=50, max_line_gap:int=300, threshold:int=30, rho:int=0, critslope:float=0.1):
        if rho==0:
            rho=int(3*self.pxpmm/139)
        theta = np.pi/180   # angular resolution in radians of the Hough grid
      # threshold is minimum number of votes (intersections in Hough grid cell)
        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv.HoughLinesP(im, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
        
        if lines is None or len(lines)==0:
            return [], []

        # convert to dataframe
        lines = pd.DataFrame(lines.reshape(len(lines),4), columns=['x0', 'y0', 'xf', 'yf'], dtype='int32')
        lines['slope'] = abs(lines['x0']-lines['xf'])/abs(lines['y0']-lines['yf'])

        # find horizontal lines
        horizLines = lines[(lines['slope']>20)&(lines['y0']>yBmin)&(lines['y0']<yBmax)]
        lines0h = horizLines.copy()
        
        # only take nearly vertical lines, where slope = dx/dy
        lines = lines[(lines['slope']<critslope)&(lines['x0']>xLmin)&(lines['x0']<xRmax)]
        lines0 = lines.copy()
        # sort each line by y
        for i,row in lines0.iterrows():
            if row['yf']<row['y0']:
                lines0.loc[i,['x0','y0','xf','yf']] = list(row[['xf','yf','x0','y0']])
        lines0 = lines0.convert_dtypes()         # convert back to int
        lines0 = lines0[lines0.yf<hmax]      # only take lines that extend close to the top of the frame
        return lines0h, lines0
        
        
    def nozzleLines(self) -> pd.DataFrame:
        '''get lines from the stored edge image'''
        lines0h, lines0 = self.nozzleLines0(self.edgeImage, self.yBmin, self.yBmax, self.xLmin, self.xRmax, 400)
        if len(lines0)==0:
            raise ValueError('Failed to detect any lines in nozzle')
        if len(lines0h)>0:
            self.lines0h = lines0h
        if len(lines0)>0:
            self.lines0 = lines0
        
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
        margin = 0.45*self.pxpmm # half a nozzle
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
        return (self.xR-self.xL)/self.pxpmm
            
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
            
    def createNozzleMask(self, **kwargs) -> None:
        '''create a nozzle mask, so we can erase nozzle from images. nozMask is binary, and nozCover is grayscale'''
        # create mask
        
        if 'frame' in kwargs:
            frame = kwargs['frame']
        elif hasattr(self, 'line_image'):
            frame = self.line_image
        else:
            frame = self.vd.getFrameAtTime(1)
        
        self.nozMask = 255*np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        self.nozCover = np.copy(frame)*0                          # create empty mask
#         average = frame.mean(axis=0).mean(axis=0)                 # background color
        self.nozCover[0:int(self.yB), int(self.xL)-self.maskPadLeft:int(self.xR)+self.maskPadRight]=255      # set the nozzle region to white
        m = cv.cvtColor(self.nozCover,cv.COLOR_BGR2GRAY)              # convert to gray
        _,self.nozMask = cv.threshold(m,0,255,cv.THRESH_BINARY_INV)   # binary mask of nozzle
    
    def detectNozzle0(self, diag:int=0, suppressSuccess:bool=False, mode:int=0, overwrite:bool=False) -> None:
        '''find the bottom corners of the nozzle. suppressSuccess=True to only print diagnostics if the run fails'''
        self.thresholdNozzle(mode)    # threshold the nozzle
        self.nozzleLines()            # edge detect and Hough transform to get nozzle edges as lines
        if len(self.lines0)==0:
            raise ValueError('Failed to detect any vertical lines in nozzle')
        self.findNozzlePoints()       # filter the lines to get nozzle coords
        self.checkNozzleValues()      # make sure coords make sense
        self.exportNozzleDims(overwrite=overwrite)       # save coords
        self.createNozzleMask()       # create mask that we can use to remove the nozzle from images
        
        if diag>0 and not suppressSuccess:
            self.drawDiagnostics(diag) # show diagnostics
        

    def detectNozzle(self, diag:int=0, suppressSuccess:bool=False, mode:int=0, overwrite:bool=False, **kwargs) -> None:
        '''find the bottom corners of the nozzle, trying different images. suppressSuccess=True to only print diagnostics if the run fails'''
#         logging.info(f'Detecting nozzle in {self.printFolder}')
        if not overwrite:
            im = self.importNozzleDims()
            if im==0:
                self.createNozzleMask()
                # we already detected the nozzle
                return 0
        
        # no existing file: detect nozzle
        for i in range(3):
            for mode in [0,1,2]: # min, then median, then mean
                try:
                    self.detectNozzle0(diag=diag, suppressSuccess=suppressSuccess, mode=mode, overwrite=overwrite)
                except ValueError:
                    if diag>1:
                        traceback.print_exc()
                        print('Looping to next mode')
                    pass
                else:
                    return 0
            
        # if all modes failed:
        self.drawDiagnostics(diag) # show diagnostics
        raise ValueError('Failed to detect nozzle after 9 iterations')
        
    #------------------------------------------------------------------------------------
        
        
    def maskNozzle(self, frame:np.array, dilate:int=0, ave:bool=False, invert:bool=False, normalize:bool=True, bottomOnly:bool=False, **kwargs) -> np.array:
        '''block the nozzle out of the image. 
        dilate is number of pixels to expand the mask. 
        ave=True to use the background color, otherwise use white
        invert=False to draw the nozzle back on
        '''
        s = frame.shape
        if not hasattr(self, 'nozMask') or not hasattr(self, 'nozCover'):
            self.createNozzleMask(frame=frame)
        if 'crops' in kwargs:
            crops = kwargs['crops']
        frameMasked = frame.copy()
        
        if 'crops' in kwargs and 'y0' in kwargs['crops'] and 'x0' in kwargs['crops']:
            yB = int(self.yB-crops['y0'])
            xL = int(self.xL-crops['x0'])
            xR = int(self.xR-crops['x0'])
        else:
            yB = int(self.yB)
            xL = int(self.xL)
            xR = int(self.xR)
        
        # cover nozzle with new color
        if ave:
            # mask nozzle with whitest value
            nc = np.copy(frame)*0 # create mask
            average = int(frame.max(axis=0).mean(axis=0))
            if bottomOnly:
                y0 = yB-10
            else:
                y0 = 0
            nc[y0:yB, xL-self.maskPadLeft:xR+self.maskPadRight]=average
        else:
            # mask nozzle with white
            nc = self.nozCover
            if 'crops' in kwargs:
                nc = vc.imcrop(nc, crops)
            if bottomOnly:
                nc[0:yB-10, xL-self.maskPadLeft:xR+self.maskPadRight]=0
        
        # put the new mask on top
        nc = vm.dilate(nc, dilate)  
        if len(s)<3 and len(nc.shape)==3:
            nc = cv.cvtColor(nc, cv.COLOR_BGR2GRAY)
        if invert:
            out = cv.subtract(frameMasked, nc)
        else:
            out = cv.add(frameMasked, nc) 

        norm = np.zeros(out.shape)
        if normalize:
            out = cv.normalize(out,  norm, 0, 255, cv.NORM_MINMAX) # normalize the image
        return out
    
    def displayNearLines(self, lines0:pd.DataFrame, im:np.array, edges:np.array) -> None:
        print('Lines near nozzle: ', lines0)
        if len(lines0)>0:
            im2 = cv.cvtColor(im, cv.COLOR_GRAY2BGR)
            im3 = cv.cvtColor(im, cv.COLOR_GRAY2BGR)
            h,w = im.shape[:2]
            for i,line in lines0.iterrows():
                color = list(np.random.random(size=3) * 256)            # assign a random color to each line
                cv.line(im2,(int(line['x0']),int(line['y0'])),(int(line['xf']),int(line['yf'])),color,1)
            for i,line in lines0.iterrows():
                cv.line(im3,(int(line['x0']),int(0)),(int(line['x0']),int(h)),color,1)
                cv.line(im3,(int(line['xf']),int(0)),(int(line['xf']),int(h)),color,1)

            imshow(im3, im2, edges, title='eraseSpillover', maxwidth=8)
        
    
    def nearLines(self, im:np.array, l0:int, lf:int, diag:int=0, **kwargs):
        '''get lines near the nozzle'''
        h,w = im.shape[:2]
        xmin = l0-2
        hcrit = 0.5*255
        while xmin>0 and xmin>l0-20 and xmin<w and im[:, xmin].mean()>hcrit:
            xmin = xmin-1
        xmax = lf+1
        while xmax<w and xmax<lf+20 and xmax>0 and im[:, xmax].mean()>hcrit:
            xmax = xmax+1
        if diag>0:
            im2 = cv.cvtColor(im, cv.COLOR_GRAY2BGR)
            for x in [xmin, xmax]:
                color = (0,0,255)
                cv.line(im2,(int(x),int(0)),(int(x),int(h)),color,1)
            imshow(im, im2, title='eraseSpillover', maxwidth=6)
        return xmin, xmax
    
    def eraseSpillover(self, thresh:np.array, **kwargs) -> np.array:
        '''erase any extra nozzle on left and right of thresholded and cropped image'''
        if 'crops' in kwargs:
            crops = kwargs['crops']
            yB = int(self.yB-crops['y0'])
            xL = int(self.xL-crops['x0'])
            xR = int(self.xR-crops['x0'])
        else:
            yB = int(self.yB)
            xL = int(self.xL)
            xR = int(self.xR)
        xL2, xR2 = self.nearLines(thresh[0:yB], xL, xR, **kwargs)
        xL2 = int(min(xL, xL2))
        xR2 = int(max(xR, xR2))
        thresh[0:yB, xL2:xR2] = 0
        self.xL = self.xL + (xL2-xL)
        self.xR = self.xR + (xR2-xR)
        self.nozCover[0:yB, self.xL:self.xR, :] = 255   # update the nozzle cover
        
        return thresh
    
    def absoluteCoords(self, d:dict) -> dict:
        '''convert the relative coordinates in mm to absolute coordinates on the image in px. y is from the bottom, x is from the left'''
        if not hasattr(self, 'yB') or not hasattr(self, 'xM') or not hasattr(self, 'pxpmm'):
            self.importNozzleDims()
        nc = [self.xM, self.yB]    # convert y to from the bottom
        out = {'x':nc[0]+d['dx']*self.pxpmm, 'y':nc[1]-d['dy']*self.pxpmm}
        return out
    
    def relativeCoords(self, x:float, y:float, reverse:bool=False) -> dict:
        '''convert the absolute coordinates in px to relative coordinates in px, where y is from the top and x is from the left. reverse=True to go from mm to px'''
        if not hasattr(self, 'yB') or not hasattr(self, 'xM') or not hasattr(self, 'pxpmm'):
            self.importNozzleDims()
        nx = self.xM
        ny = self.yB
        if reverse:
            return x*self.pxpmm+nx, ny-y*self.pxpmm
        else:
            return (x-nx)/self.pxpmm, (ny-y)/self.pxpmm


        
#--------------------------------------------

def exportNozDims(folder:str, overwrite:bool=False, **kwargs) -> None:
    pfd = fh.printFileDict(folder)
    if not overwrite and hasattr(pfd, 'nozDims') and hasattr(pfd, 'background'):
        return
    nv = nozData(folder)
    nv.detectNozzle(overwrite=overwrite, **kwargs)
    nv.exportBackground(overwrite=overwrite)

def exportNozDimsRecursive(folder:str, overwrite:bool=False, **kwargs) -> list:
    '''export stills of key lines from videos'''
    fl = fh.folderLoop(folder, exportNozDims, overwrite=overwrite, **kwargs)
    return fl.run()
    
    
def checkBackground(folder:str, diag:bool=False) -> float:
    '''check if the background is good or bad'''
    nd = nozData(folder)
    nd.importBackground()
    empty = nd.maskNozzle(nd.background, dilate=10, ave=True, invert=False, normalize=False)
    mm = empty.min(axis=0).min(axis=0).min(axis=0)
    if diag:
        imshow(empty)
    return mm
    
def findBadBackgrounds(topFolder:str, exportFolder:str) -> pd.DataFrame:
    '''find the bad backgrounds in the folder'''
    out = []
    for folder in fh.printFolders(topFolder):
        out.append({'folder':folder, 'mm':checkBackground(folder)})
    df = pd.DataFrame(out)
    df.sort_values(by='mm', inplace=True)
    df.reset_index(inplace=True, drop=True)
    fn = os.path.join(exportFolder, 'badBackgrounds.csv')
    df.to_csv(fn)
    logging.info(f'Exported {fn}')
    return df

def fixBackground(folder:str, diag:int=0) -> int:
    '''try to fix the background image in this folder'''
    nv = nozData(folder)
    mm = 1
    count = 0
    mcrit =120
    while mm<mcrit and count<3:
        nv.exportBackground(diag=diag, overwrite=True)
        mm = checkBackground(folder)
        count+=1
        if diag>0:
            print(f'Count {count} mm {mm}')
    if mm<mcrit:
        nv.stealBackground(diag=diag)