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
from imshow import imshow
import vidMorph as vm
from config import cfg
from plainIm import *
import fileHandling as fh


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
    
    def __init__(self, vidfile:str):
        self.vidFile = vidfile
        self.levels = fh.labelLevels(vidfile)
        self.printFolder = self.levels.printFolder()
        self.pfd = fh.printFileDict(self.printFolder)
        self.sampleName = os.path.basename(self.levels.subFolder)
        self.nozMask = []                   # mask that blocks nozzle
        self.prog = []                      # programmed timings
        self.streamOpen = False
        self.gv = 
        self.pxpmm = pxpmm
        self.importNozzleDims()

        
    #-----------------------------
    
    def nozDimsFN(self) -> str:
        '''file name of nozzle dimensions table'''
        # store noz dimensions in the subfolder
        return os.path.join(self.printFolder, f'{self.sampleName}_nozDims.csv')
    
    
    def exportNozzleDims(self) -> None:
        '''export the nozzle location to file'''
        fn = self.nozDimsFN()  # nozzle dimensions file name
        with open(fn, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for st in ['yB', 'xL', 'xR', 'pxpmm']:
                try:
                    # write values to file
                    writer.writerow([st, str(getattr(self, st))])
                except:
                    pass
        logging.info(f'Exported {fn}')
        
    def importNozzleDims(self) -> None:
        '''find the target pressure from the calibration file. returns 0 if successful, 1 if not'''
        fn = self.nozDimsFN()      # nozzle dimensions file name
        if not os.path.exists(fn):
            self.nozDetected = False
            return 
        tlist = ['yB', 'xL', 'xR', 'pxpmm']
        with open(fn, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                # save all rows as class attributes
                if row[0] in tlist:
                    tlist.remove(row[0])                          # don't write this value again
                    setattr(self, row[0], int(float(row[1])))     # save to object
        if len(tlist)==0:
            # we have all values
#             self.xM = (self.xL+self.xR)/2
#             self.initLineImage()
            self.nozDetected = True
            return
        else:
            self.nozDetect = False
            return
        
    #---------------------------------

    def defineCritVals(self):
        '''critical nozzle position values that indicate nozzle detection may have failed, so trigger an error'''
        
        # bounds in px for a 600x800 image
        self.xLmin = 200 # px
        self.xLmax = 500
        self.xRmin = 300
        self.xRmax = 600
        self.yBmin = 250
        self.yBmax = 430
        
        # bounds of size of nozzle in mm. for 20 gauge nozzle, diam should be 0.908 mm
        self.nozwidthMin = 0.75 # mm
        self.nozWidthMax = 1.05 # mm
    
    def nozzleFrame(self, mode:int=0) -> np.array:
        '''get an averaged frame from several points in the stream to blur out all fluid and leave just the nozzle. 
        mode=0 to use median frame, mode=1 to use mean frame, mode=2 to use lightest frame'''
        if len(self.prog)==0:
            raise ValueError('No programmed timings in folder')
        else:
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
        logging.info(f'detecting nozzle in {self.printFolder}')
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