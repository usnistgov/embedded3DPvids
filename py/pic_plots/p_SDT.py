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

class multiPlotsSDT(multiPlots):
    '''given a sample type folder, plot values'''
    
    def __init__(self, folder:str, exportFolder:str, **kwargs):
        super().__init__(folder, exportFolder, ['*'], **kwargs)
      

        if 'vels' in os.path.basename(folder):
            self.xvar = 'ink.v'
            self.yvar = 'sup.v'
        else:
            self.xvar = 'ink.var'
            self.yvar = 'sup.var'
            
            
    def keyPlots(self, **kwargs):
        '''most important plots'''
        for i in range(3):
            for s in ['HIx', 'HOx', 'HOh', 'V']:
                self.plot(f'{s}{i+1}', spacing=0.875, **kwargs)
                self.plot(f'{s}{i+1}', ink=self.inkList[-1], **kwargs)  
                
    def getTag(self, name:str, overlay:dict, ss:str) -> list:
        '''get the lines to search for and the overlay shape'''
        if name.endswith('1'):
            if 'x' in name:
                tag = [f'l2w{ss}', f'l2d{ss}']
            else:
                tag = [f'l2w1{ss}', f'l2d1{ss}']
            overlay['shape'] = '2circles'
        elif name.endswith('2'):
            if 'x' in name:
                tag = [f'l2w1{ss}', f'l2w2{ss}', f'l2d{ss}']
            else:
                tag = [f'l2w1{ss}', f'l2w2{ss}', f'l2d2{ss}']
            overlay['shape'] = '2circles'
        elif name.endswith('3'):
            tag = [f'l2w1{ss}', f'l2w2{ss}', f'l2w3{ss}']
            overlay['shape'] = '3circles'
        else:
            raise ValueError(f'Unknown object requested: {name}')
        if 'o' in ss:
            overlay['color'] = 'black'
        elif name[:-1] in ['HOx']:
            overlay['color'] = 'white'
        return tag
    
    def getCrops(self, name:str, kwargs2:dict, kwargs:dict) -> None:
        '''get the crop dimensions'''
        if 'crops' in kwargs:
            kwargs2['crops'] = kwargs['crops']
        else:
            crops = {'relative':True}
            if name[:-1]=='HIx':
                crops = {**crops, 'w':200, 'h':150, 'wc':50, 'hc':110}
            elif name[:-1]=='HOx':
                crops = {**crops, 'w':100, 'h':250, 'wc':50, 'hc':200}
            elif name[:-1]=='HOh':
                crops = {**crops, 'w':800, 'h':275, 'wc':400, 'hc':200}
            elif name[:-1]=='V':
                crops = {**crops, 'w':200, 'h':700, 'wc':60, 'hc':350}
            kwargs2['crops'] = crops
                
    def checkFreeVars(self, allIn:list, kwargs2:dict, kwargs:dict) ->Tuple[list, str, str]:
        '''identify how many free variables are available, and create filters (allIn) and kwargs (kwargs2)'''
        yvar = self.yvar
        xvar = 'self.spacing'
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
        return allIn, xvar, yvar

    def getConcat(self, name:str) -> str:
        '''get the direction to concatenate images'''
        if name[:-1] in ['HOh', 'HIx']:
            concat = 'v'
        else:
            concat = 'h'
        return concat
    
    def getOverlay(self, name:str, overlay:dict) -> None:
        '''get the direction and position of the overlay scale'''
        if name[:-1] in ['HIx', 'V']:
            overlay['shiftdir'] = 'x'
            if name[:-1]=='HIx':
                overlay['dx'] = 0.1
        elif name[:-1] in ['HOx', 'HOh']:
            overlay['shiftdir'] = 'y'
            if name[:-1]=='HOx':
                overlay['dy'] = 0.1
            
    def getExportFolder(self, name:str) -> str:
        '''get the export folder and create subfolder if needed'''
        exportFolder =  os.path.join(self.exportFolder, name)
        if not os.path.exists(exportFolder):
            os.mkdir(exportFolder)
        return exportFolder

            
    def plot(self, name:str, showFig:bool=False, export:bool=True,  **kwargs):
        '''plot the values for name (e.g. HOx1)'''
        
        kwargs2 = {**self.kwargs.copy(), **kwargs.copy()}
        obj2file = fh.SDT2FileDict()
        if not name in obj2file:
            raise ValueError(f'Unknown object requested: {name}')
        file = obj2file[name]
        if 'x' in name:
            s2 = ''
        else:
            s2 = 'p3'
        for ss in ['o1', s2]:
            allIn = [file]
            overlay = {'dx':-0.7, 'dy':-0.7}
            tag = self.getTag(name, overlay, ss)
            self.getCrops(name, kwargs2, kwargs)
            allIn, xvar, yvar = self.checkFreeVars(allIn, kwargs2, kwargs)
            concat = self.getConcat(name)
            self.getOverlay(name, overlay)
            exportFolder = self.getExportFolder(name)
            pp = picPlots(self.folder, exportFolder
                            , allIn, [], tag, showFig=showFig, export=export
                            , overlay=overlay
                            , xvar=xvar, yvar=yvar, concat=concat
                            , **kwargs2)
            pp.picPlots0()
   