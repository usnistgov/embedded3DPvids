#!/usr/bin/env python
'''Functions for collecting data from stills of single lines'''

# external packages
import os, sys
import magic
import traceback
import logging
import pandas as pd
from matplotlib import pyplot as plt
from typing import List, Dict, Tuple, Union, Any, TextIO
import re
import numpy as np
import cv2 as cv
import shutil
import subprocess
import time
import pyautogui

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
from im.imshow import imshow
from tools.plainIm import *
from tools.config import cfg
from m_stats import *
import file_handling as fh

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)
pd.set_option('display.max_rows', 500)

#----------------------------------------------

class paintObject:
    
    def __init__(self, file:str, fullScreen:bool=True, scrollDown:float=1, scrollRight:float=1, dropper:bool=False, thickness:int=3, zoom:int=5, onlyOpen:bool=False, white:bool=False, pause:float=3, **kwargs):
        self.file = file
        pyautogui.PAUSE = 0.1
        self.w = pyautogui.size().width
        self.h = pyautogui.size().height
        self.openPaint(fullScreen=fullScreen)
        if onlyOpen:
            return
        time.sleep(pause)
        self.selectPencil()
        self.selectDropper()
        self.selectTopRight()
        if zoom>1:
            self.magnify(zoom)
        self.selectPencil()
        if thickness>1:
            self.selectThickness(thickness)
        if scrollRight>0:
            self.scrollRight(scrollRight)
        if scrollDown>0:
            self.scrollDown(scrollDown)
        if dropper:
            self.selectDropper()
        if white:
            self.selectWhite()
        self.moveFrac(3/4, 1/2)
        
    def openPaint(self, fullScreen:bool=True):
        if fullScreen:
            subprocess.Popen(["cmd", "/c", "start", "/max", cfg.path.paint, self.file]);
        else:
            subprocess.Popen(["cmd", "/c",cfg.path.paint, self.file]);
        
    def selectPencil(self):
        pyautogui.moveTo(347, 115) # Move the mouse to the pencil button
        pyautogui.click() # Click the mouse at its current location.
        
    def selectDropper(self):
        pyautogui.moveTo(386, 159) # Move the mouse to the dropper button
        pyautogui.click() # Click the mouse at its current location.  
        
    def selectTopRight(self):
        pyautogui.moveTo(700, 376) # select from top right
        pyautogui.click() # Click the mouse at its current location.  
        
    def selectMagnifier(self):
        pyautogui.moveTo(426, 159) # Move the mouse to the magnifier button
        pyautogui.click() # Click the mouse at its current location. 
        
    def magnify(self, n:int):
        self.selectMagnifier()
        pyautogui.moveTo(548, 376) # select from top right
        for i in range(n):
            pyautogui.click() # zoom
            
    def selectThickness(self, t:int):  
        pyautogui.moveTo(347, 115) # Move the mouse to the pencil button
        pyautogui.click() # Click the mouse at its current location.
        pyautogui.moveTo(908, 114) # Move the mouse to the line thickness button
        pyautogui.click() # Click the mouse at its current location.
        time.sleep(0.25)
        y = {1:223, 2:276, 3:345, 4:391}[t]
        pyautogui.moveTo(908, y) # Move the mouse to the select 2nd line thickness
        pyautogui.click() # Click the mouse at its current location.
        
    def scrollRight(self, f):
        pyautogui.moveTo(300, self.h-120) # Move the mouse to the scroll button on bottom
        pyautogui.mouseDown()
        pyautogui.moveTo(300+(self.w-600)*f, self.h-120) # Move the mouse to the scroll button on bottom
        pyautogui.mouseUp()
        
    def scrollDown(self, f:float):
        pyautogui.moveTo(3821, 355) # Move the mouse to the scroll button on bottom
        pyautogui.mouseDown()
        pyautogui.moveTo(3821, (1380-355)*f+355) # Move the mouse to the scroll button on bottom
        pyautogui.mouseUp()
        
    def moveFrac(self, wfrac:float, hfrac:float):
        pyautogui.moveTo(int(self.w*wfrac), int(self.h*hfrac)) # Move the mouse to the bottom right
        
    def selectWhite(self) -> None:
        '''select the white color in paint'''
        pyautogui.moveTo(1090, 137) # Move the mouse to the pencil button
        pyautogui.click() # Click the mouse at its current location.

    
def openInPaint(file, **kwargs):
    '''open the file in MS paint'''
    paintObject(file, **kwargs)

    
def cursorFinder():
    while True:
        x, y = pyautogui.position()
        positionStr = 'X: ' + str(x).rjust(4) + ' Y: ' + str(y).rjust(4) + '\n'
        print(positionStr, end='')
        print('\b' * len(positionStr), end='', flush=True)
        time.sleep(0.01)
    
    
def openInExcel(file):
    '''open the file in MS excel'''
    subprocess.Popen([cfg.path.excel, file]);  