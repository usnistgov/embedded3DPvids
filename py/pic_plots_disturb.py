#!/usr/bin/env python
'''Functions for plotting video and image data for tripleLines'''

# external packages
import os, sys
import traceback
import logging
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
from typing import List, Dict, Tuple, Union, Any, TextIO
import re
import numpy as np
import cv2 as cv
import matplotlib.ticker as mticker

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
from pic_plots import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
# plotting
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rc('font', family='Arial')
matplotlib.rc('font', size='10.0')


#-----------------------------------------------

class multiPlots:
    '''given a sample type folder, plot values'''
    
    def __init__(self, folder:str, exportFolder:str, dates:List[str], **kwargs):
        self.folder = folder
        self.exportFolder = exportFolder
        self.dates = dates
        self.kwargs = kwargs
        self.inkvList = []
        self.supvList = []
        self.inkList = []
        self.supList = []
        self.spacingList = ['0.500', '0.625', '0.750', '0.875', '1.000', '1.250']
        for subfolder in os.listdir(self.folder):
            spl = re.split('_', subfolder)
            for i,s in enumerate(spl):
                if s=='I' and not spl[i+1] in self.inkList:
                    self.inkList.append(spl[i+1])
                elif s=='S' and not spl[i+1] in self.supList:
                    self.supList.append(spl[i+1])
                elif s=='VI' and not spl[i+1] in self.inkvList:
                    self.inkvList.append(spl[i+1])
                elif s=='VS' and not spl[i+1] in self.supvList:
                    self.supvList.append(spl[i+1])
                    
        # determine how many variables must be defined for a 2d plot
        self.freevars = 1
        self.freevarList = ['spacing']
        for s in ['ink', 'sup', 'inkv', 'supv']:
            l = getattr(self, f'{s}List')
            if len(l)>1:
                self.freevars+=1
                self.freevarList.append(s)

        if 'visc' in os.path.basename(folder):
            self.xvar = 'ink.var'
            self.yvar = 'sup.var'
        elif 'vels' in os.path.basename(folder):
            self.xvar = 'ink.v'
            self.yvar = 'sup.v'
            
    def keyPlots(self, **kwargs):
        '''most important plots'''
        for s in ['HIx', 'HOx', 'HOh', 'V']:
            self.plot(s, spacing=0.875, **kwargs)
            self.plot(s, ink=self.inkList[-1], **kwargs)     
            
        
    def spacingPlots(self, name:str, showFig:bool=False, export:bool=True):
        '''run all plots for object name (e.g. HOB, HIPxs)'''
        for spacing in self.spacingList:
            self.plot(spacing=spacing, showFig=showFig, export=export)
            
    def plot(self, name:str, showFig:bool=False, export:bool=True,  **kwargs):
        '''plot the values for name (e.g. horiz, xs_+y, xs_+z, or vert)'''
        yvar = self.yvar
        xvar = 'self.spacing'
        kwargs2 = {**self.kwargs.copy(), **kwargs.copy()}
        obj2file = fh.singleDisturb2FileDict()
        if not name in obj2file:
            raise ValueError(f'Unknown object requested: {name}')
        file = obj2file[name]
        allIn = [file]
        dates = self.dates
        tag = ['l1wo', 'l1do']
        freevars = 0
        if 'spacing' in kwargs:
            spacing = kwargs['spacing']
            if not type(spacing) is str:
                spacing = '{:.3f}'.format(spacing)
            allIn.append(f'{file}_{spacing}')
            xvar = self.xvar
            freevars+=1
        if 'ink' in kwargs:
            ink = kwargs['ink']
            allIn.append(f'I_{ink}')
            kwargs2['I'] = ink
            freevars+=1
        if 'sup' in kwargs:
            sup = kwargs['sup']
            allIn.append(f'S_{sup}')
            kwargs2['S'] = sup
            xvar = self.yvar
            freevars+=1
        if 'inkv' in kwargs:
            inkv = kwargs['inkv']
            allIn.append(f'VI_{inkv}')
            kwargs2['VI']=inkv
            freevars+=1
        if 'supv' in kwargs:
            supv = kwargs['supv']
            allIn.append(f'VS_{supv}')
            kwargs2['VS']=supv
            freevars+=1
        if freevars+2<self.freevars:
            raise ValueError(f'Too many variables to plot. Designate {self.freevarList} where needed')
            
        if 'crops' in kwargs:
            kwargs2['crops'] = kwargs['crops']
        else:
            crops = {'HIx':{'y0':150, 'yf':350, 'x0':200, 'xf':300},
                    'HOx':{'y0':150, 'yf':350, 'x0':200, 'xf':300},
                    'HOh':{'y0':0, 'yf':300, 'x0':10, 'xf':790},
                    'V':{'y0':0, 'yf':600, 'x0':200, 'xf':400}}
            if name in crops:
                kwargs2['crops'] = crops[name]
        if name in ['HOh']:
            concat = 'v'
        else:
            concat = 'h'
            
        exportFolder =  os.path.join(self.exportFolder, name)
        if not os.path.exists(exportFolder):
            os.mkdir(exportFolder)

        fig = picPlots0(self.folder, exportFolder
                        , allIn, dates, tag, showFig=showFig, export=export
                        , overlay={'shape':'2circles', 'dx':-0.8, 'dy':-0.8}
                        , xvar=xvar, yvar=yvar, concat=concat
                        , **kwargs2)
   