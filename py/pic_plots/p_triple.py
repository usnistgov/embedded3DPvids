#!/usr/bin/env python
'''Functions for plotting multiple stills on one plot for tripleLines'''

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
sys.path.append(os.path.dirname(currentdir))
from p_plots import *

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

class multiPlotsTriple(multiPlots):
    '''given a sample type folder, plot multiple images on one plot'''
    
    def __init__(self, folder:str, exportFolder:str, dates:List[str], **kwargs):
        super().__init__(self, folder, exportFolder, dates, **kwargs)

        if 'visc' in os.path.basename(folder):
            self.xvar = 'ink.var'
            self.yvar = 'sup.var'
        elif 'vels' in os.path.basename(folder):
            self.xvar = 'ink.v'
            self.yvar = 'sup.v'
            
    def keyPlots(self, **kwargs):
        '''most important plots'''
        for s in ['HIPxs', 'HOPxs', 'HOPh']:
            self.plot(s, spacing=0.875, index=[0,1,2,3], **kwargs)
            self.plot(s, ink=self.inkList[-1], index=[1], **kwargs)
        for s in ['VP']:
            self.plot(s, spacing=0.875, index=[0,1,2,3], **kwargs)
            self.plot(s, ink=self.inkList[-1], index=[2], **kwargs)
        for s in ['HOB', 'HOC', 'VB', 'VC']:
            self.plot(s, spacing=0.875, index=[0], **kwargs)
            self.plot(s, ink=self.inkList[-1], index=[0], **kwargs)

            
    def plot(self, name:str, showFig:bool=False, export:bool=True, index:List[int]=[0], **kwargs):
        '''plot the values for object name (e.g. HOB, HIPxs)'''
        yvar = self.yvar
        xvar = 'self.spacing'
        kwargs2 = {**self.kwargs.copy(), **kwargs.copy()}
        obj2file = fh.tripleLine2FileDict()
        if not name in obj2file:
            raise ValueError(f'Unknown object requested: {name}')
        file = obj2file[name]
        allIn = [file]
        dates = self.dates
        tag = [f'{name}_{i}' for i in index]
        freevars = 0
        if 'spacing' in kwargs:
            spacing = kwargs['spacing']
            if not type(spacing) is str:
                spacing = '{:.3f}'.format(spacing)
            allIn.append(f'{file}_{spacing}')
            tag = [f'{spacing}_{name}_{i}' for i in index]
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
            crops = {'HOC':{'y0':0, 'yf':600, 'x0':0, 'xf':750},
                    'HOB':{'y0':0, 'yf':750, 'x0':0, 'xf':750},
                    'HIPxs':{'y0':0, 'yf':250, 'x0':50, 'xf':300},
                    'HOPxs':{'y0':0, 'yf':300, 'x0':50, 'xf':200},
                    'VB':{'y0':0, 'yf':800, 'x0':100, 'xf':700},
                    'VC':{'y0':0, 'yf':800, 'x0':100, 'xf':700},
                    'HOPh':{'y0':50, 'yf':350, 'x0':50, 'xf':750},
                    'VP':{'y0':0, 'yf':800, 'x0':0, 'xf':284}}
            if name in crops:
                kwargs2['crops'] = crops[name]
        if name in ['HIPh', 'HOPh', 'HOPxs']:
            concat = 'v'
        else:
            concat = 'h'
            
        exportFolder =  os.path.join(self.exportFolder, name)
        if not os.path.exists(exportFolder):
            os.mkdir(exportFolder)

        fig = picPlots0(self.folder, exportFolder
                        , allIn, dates, tag, showFig=showFig, export=export
                        , overlay={'shape':'3circles', 'dx':-0.8, 'dy':-0.8}
                        , xvar=xvar, yvar=yvar, concat=concat
                        , **kwargs2)
   