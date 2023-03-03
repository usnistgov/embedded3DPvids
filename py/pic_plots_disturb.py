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

class multiPlotsSingleDisturb(multiPlots):
    '''given a sample type folder, plot values'''
    
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
        for s in ['HIx', 'HOx', 'HOh', 'V']:
            self.plot(s, spacing=0.875, **kwargs)
            self.plot(s, ink=self.inkList[-1], **kwargs)     
  
            
    def plot(self, name:str, showFig:bool=False, export:bool=True,  **kwargs):
        '''plot the values for name (e.g. horiz, xs_+y, xs_+z, or vert)'''
        yvar = self.yvar
        xvar = 'self.spacing'
        kwargs2 = {**self.kwargs.copy(), **kwargs.copy()}
        obj2file = fh.singleDisturb2FileDict()
        if not name in obj2file and not obj2file in name:
            raise ValueError(f'Unknown object requested: {name}')
        file = obj2file[name]
        allIn = [file]
        dates = self.dates
        for ss in ['o', '']:
            tag = [f'l2w{ss}', f'l2d{ss}']
            if 'crops' in kwargs:
                kwargs2['crops'] = kwargs['crops']
            else:
                if ss=='o':
                    crops = {'HIx':{'y0':200, 'yf':400, 'x0':200, 'xf':300},
                            'HOx':{'y0':175, 'yf':375, 'x0':200, 'xf':300},
                            'HOh':{'y0':0, 'yf':300, 'x0':10, 'xf':790},
                            'V':{'y0':0, 'yf':600, 'x0':200, 'xf':400}}
                else:
                    crops = {'HIx':{'y0':200, 'yf':400, 'x0':225, 'xf':425},
                            'HOx':{'y0':175, 'yf':375, 'x0':325, 'xf':450},
                            'HOh':{'y0':100, 'yf':400, 'x0':0, 'xf':800},
                            'V':{'y0':0, 'yf':600, 'x0':250, 'xf':450}}
                if name in crops:
                    kwargs2['crops'] = crops[name]
            freevars = 0
            if 'spacing' in kwargs:
                spacing = kwargs['spacing']
                if not type(spacing) is str:
                    spacing = '{:.3f}'.format(spacing)
                allIn.append(spacing)
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


            if name in ['HOh']:
                concat = 'v'
            else:
                concat = 'h'

            exportFolder =  os.path.join(self.exportFolder, name)
            if not os.path.exists(exportFolder):
                os.mkdir(exportFolder)
                
            if name in ['HIx', 'V']:
                shiftdir = 'x'
            else:
                shiftdir = 'y'

            fig = picPlots0(self.folder, exportFolder
                            , allIn, dates, tag, showFig=showFig, export=export
                            , overlay={'shape':'2circles', 'dx':-0.8, 'dy':-0.8, 'shiftdir':shiftdir}
                            , xvar=xvar, yvar=yvar, concat=concat
                            , **kwargs2)
   