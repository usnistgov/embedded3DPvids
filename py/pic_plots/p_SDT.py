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
    
    def __init__(self, folder:str, exportFolder:str, transpose:bool=False, **kwargs):
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
            for s in ['HIx', 'HOx', 'HOh', 'V', 'HIh']:
                self.plot(f'{s}{i+1}', spacing=0.875, **kwargs)
                self.plot(f'{s}{i+1}', ink=self.inkList[-1], **kwargs)  
                
    def getTag(self, name:str, overlay:dict, ss:str, kwargs) -> list:
        '''get the lines to search for and the overlay shape'''
        if name.endswith('1'):
            tag = [f'l2w1{ss}', f'l2d1{ss}']
            overlay['shape'] = '2circles'
        elif name.endswith('2'):
            tag = [f'l2w1{ss}', f'l2w2{ss}', f'l2d2{ss}']
            overlay['shape'] = '2circles'
        elif name.endswith('3'):
            tag = [f'l2w1{ss}', f'l2w2{ss}', f'l2w3{ss}']
            overlay['shape'] = '3circles'
        else:
            raise ValueError(f'Unknown object requested: {name}')
        if 'o' in ss and not 'HIh' in name:
            overlay['color'] = 'black'
        elif name[:-1] in ['HOx', 'HIh']:
            overlay['color'] = 'white'
        if 'tag' in kwargs:
            tag = kwargs.pop('tag')
        return tag
    
    def getCrops(self, name:str, tag:str, kwargs2:dict, kwargs:dict) -> None:
        '''get the crop dimensions'''
        if 'crops' in kwargs:
            kwargs2['crops'] = kwargs['crops']
        else:
            crops = {'relative':True}
            if name[:-1]=='HIx':
                crops = {**crops, 'w':150, 'h':150, 'wc':50, 'hc':110}
            elif name[:-1]=='HOx':
                crops = {**crops, 'w':100, 'h':250, 'wc':50, 'hc':200}
            elif name[:-1]=='HOh':
                crops = {**crops, 'w':800, 'h':275, 'wc':400, 'hc':200}
            elif name[:-1]=='V':
                crops = {**crops, 'w':200, 'h':700, 'wc':60, 'hc':350}
            elif name[:-1]=='HIh':
                crops = {**crops, 'w':700, 'h':280, 'wc':250, 'hc':180}
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
            # xvar = self.yvar
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
        if 'transpose' in kwargs and kwargs['transpose']:
            return allIn, yvar, xvar
        else:
            return allIn, xvar, yvar

    def getConcat(self, name:str, kwargs) -> str:
        '''get the direction to concatenate images'''
        if 'concat' in kwargs:
            concat = kwargs.pop('concat')
            return concat
        if name[:-1] in ['HOh', 'HIx', 'HIh']:
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
        elif name[:-1] in ['HOx', 'HOh', 'HIh']:
            overlay['shiftdir'] = 'y'
            if name[:-1]=='HOx':
                overlay['dy'] = 0.1
            
    def getExportFolder(self, name:str) -> str:
        '''get the export folder and create subfolder if needed'''
        if 'exportBn' in self.kwargs:
            return self.exportFolder
        exportFolder =  os.path.join(self.exportFolder, name)
        if not os.path.exists(exportFolder):
            os.mkdir(exportFolder)
        return exportFolder

            
    def plot(self, name:str, getO:bool=True, getP:bool=True, showFig:bool=False, export:bool=True,  **kwargs):
        '''plot the values for name (e.g. HOx1)'''
        
        kwargs2 = {**self.kwargs.copy(), **kwargs.copy()}
        obj2file = fh.SDT2FileDict()
        if not name in obj2file:
            raise ValueError(f'Unknown object requested: {name}')
        file = obj2file[name]
        slist = []
        if 'tag' in kwargs:
            slist = [kwargs['tag']]
        else:
            if getO:
                slist.append('o1')
            if getP:
                if 'x' in name:
                    slist.append('')
                elif 'HIh' in name:
                    slist.append('p2')
                else:
                    slist.append('p3')
        allIn = [file]
        overlay = {'dx':-0.7, 'dy':-0.7}
        self.getOverlay(name, overlay)
        concat = self.getConcat(name, kwargs2)
        if 'HIh' in name:
            overlay['color'] = 'white'
            overlay['dy']=-0.3
        elif 'V' in name:
            kwargs2['rotate'] = True  # rotate 90 degrees clockwise
            overlay['shiftdir'] = 'y'
            concat = 'v'
        
        for ss in slist:
            tag = self.getTag(name, overlay, ss, kwargs2)
            self.getCrops(name, tag, kwargs2, kwargs)
            allIn, xvar, yvar = self.checkFreeVars(allIn, kwargs2, kwargs)
            exportFolder = self.getExportFolder(name)
            pp = picPlots(self.folder, exportFolder
                            , allIn, [], tag, showFig=showFig, export=export
                            , overlay=overlay
                            , xvar=xvar, yvar=yvar, concat=concat
                          , spacerThickness=0
                            , **kwargs2)
            pp.picPlots0()

class singleFilePicsSDT:
    '''plot pictures from a single folder, given tags and metadata'''    
    
    def __init__(self, row:Union[pd.Series, dict]):
        self.row = row
        self.caption = row['caption']
        self.phase = row['phase']
        self.direction = row['direction']
        self.line = row['line']
        self.num = self.line[-1]
        self.bn = f'{self.line}_{self.direction}_{self.phase}_{self.caption}'
        self.title = f'{self.line} {self.direction} {self.phase}: {self.caption}'
        self.folders = [os.path.join(cfg.path.server, row['folder'])]
        self.mp = multiPlotsSDT(self.folders[0], os.path.join(cfg.path.fig, 'SDT', 'images', 'archetypes'), exportBn=self.bn, folders=self.folders)
        
    def plot(self, imsize:float=2, times:bool=True, **kwargs) -> None:
        '''plot the picture'''
        tags = []
        for t in ['tag1', 'tag2', 'tag3']:
            if type(self.row[t]) is str:
                tags.append(self.row[t])
        dirtag = {'HOP':'HOh', 'HIP':'HIh', 'V':'V'}[self.direction]
        self.mp.plot(f'{dirtag}{self.num}', tag=tags, ink=self.mp.inkList[0], imsize=imsize, title=self.title, times=times, **kwargs)
        