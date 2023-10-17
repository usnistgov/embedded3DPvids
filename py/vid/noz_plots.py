#!/usr/bin/env python
'''Functions for plotting nozzle detection lines on side views of the nozzle'''

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

class nozPlotter:
    '''for showing diagnostics of nozzle detection'''
    
    def __init__(self, printFolder:str):
        self.printFolder = printFolder
        if 'Under' in self.printFolder:
            self.under = True
        else:
            self.under = False
        
    def drawNozzleOnFrame(self, colors:bool=True) -> None:
        if self.under:
            self.drawNozzleOnFrameUnder(colors)
        else:
            self.drawNozzleOnFrameSide(colors)
            
    def drawLinesOnFrame(self) -> None:
        if self.under:
            self.drawLinesOnFrameUnder()
        else:
            self.drawLinesOnFrameSide()
     
    #-----------
    
    def drawNozzleOnFrameSide(self, colors:bool=True) -> None:
        '''draw the nozzle outline on the original frame. colors=True to draw annotations in color, otherwise in white'''
        # draw left and right edge
        if hasattr(self, 'lines'):
            lines = self.lines
            try:
                for i,line in lines.iterrows():
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
                cv.line(self.line_image,(int(lines.loc[0,'xf']),int(nd.yB)),(int(lines.loc[1,'xf']),int(nd.yB)),c,2)
            except:
                pass
        
        if hasattr(self, 'nd'):
            nd = self.nd
            # draw corner points
            for l in [nd.xL, nd.xR]:
                if colors:
                    c = (0,255,0)
                else:
                    c = (255,255,255)
                for im in [self.line_image, self.edgeImage2]:
                    cv.circle(im,(int(l),int(nd.yB)),10,c,3)
        
    def drawLinesOnFrameSide(self) -> None:
        '''draw the list of all detected nozzle edge lines on the thresholded frame'''
        if hasattr(self, 'lines0') and hasattr(self, 'lines0h'):
            for df in [self.lines0, self.lines0h]:
                for i,line in df.iterrows():
                    color = list(np.random.random(size=3) * 256)            # assign a random color to each line
                    cv.line(self.lines_image,(int(line['x0']),int(line['y0'])),(int(line['xf']),int(line['yf'])),color,4)
                    
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
            
    #-----------
                    
    def drawNozzleOnFrameUnder(self, colors:bool=True) -> None:
        '''draw the nozzle outline on the original frame. colors=True to draw annotations in color, otherwise in white'''
        # draw left and right edge       
        if hasattr(self, 'nd'):
            nd = self.nd
            # draw corner points
            if colors:
                c = (0,255,0)
            else:
                c = (255,255,255)
            for im in [self.line_image, self.edgeImage2]:
                cv.circle(im,(int(nd.xC), int(nd.yC)),int(nd.r),c,3)
        
    def drawLinesOnFrameUnder(self) -> None:
        '''draw the list of all detected nozzle edge lines on the thresholded frame'''
        if hasattr(self, 'circles'):
            circles = self.circles
            try:
                for i,circle in circles.iterrows():
                    color = list(np.random.random(size=3) * 256)
                    cv.circle(self.line_image,(int(circle['xC']),int(circle['yC'])),int(circle['r']),color,2)
            except Exception as e:
                pass
            
    #--------------------------------------------------------------------------

    def initLineImage(self) -> None:
        '''initialize line_image0 and line_image, which are diagnostic images that help to visualize nozzle detection metrics'''
        self.line_image = self.line_image0.copy()
        self.edgeImage2 = cv.cvtColor(self.edgeImage.copy(), cv.COLOR_GRAY2BGR)
            
    def drawDiagnostics(self, diag:int) -> None:
        '''create an image with diagnostics.
        diag=0 to only get the nozzle image. 1 to draw only final nozzle edges. 2 to draw all detected edges. '''
        if diag<1:
            return
        
        self.initLineImage()      # create images to annotate
        try:
            self.drawNozzleOnFrame()   # draw nozzle on original frame
        except Exception as e:
            traceback.print_exc()
            pass
        if diag>1:
            self.drawLinesOnFrame()   # draw detected lines on the frame
        if hasattr(self, 'lines_image'):
            imshow(self.line_image, self.lines_image, self.edgeImage2, scale=4, title=os.path.basename(self.printFolder))
        else:
            imshow(self.line_image, self.edgeImage2, scale=4, title=os.path.basename(self.printFolder))
        
        
    def showFrames(self, tlist:List[float], crop:dict={'x0':0, 'xf':-1, 'y0':0, 'yf':-1}, figw:float=6.5) -> None:
        '''show the list of frames at times in tlist. 
        crop is the boundaries of the image, from the top left. 
        figw is the figure width in inches'''
        n = len(tlist)
        
        frames = [vm.white_balance(self.vd.getFrameAtTime(t)[crop['y0']:crop['yf'],crop['x0']:crop['xf']]) for t in tlist]  # apply white balance to each frame
        
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
    

        
