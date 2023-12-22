#!/usr/bin/env python
'''functions for selecting colors'''

# external packages
import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mc
import pandas as pd
import colorsys
import seaborn as sns
import itertools
from typing import List, Dict, Tuple, Union, Any, TextIO
import logging
import traceback

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
sys.path.append(os.path.dirname(os.path.dirname(currentdir)))


# plotting
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rc('font', family='Arial')
matplotlib.rc('font', size='10.0')

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)


#-------------------------------------------------------------


class plotColors:
    '''for deciding on color values for plots'''
    
    def __init__(self, vallist:list, cvar:str, clabel:str, byIndices:bool=True, logScale:bool=False, defaultColor:str='#000000', **kwargs):
        self.vallist = vallist  # list of values used to determine color
        self.vallist.sort()
        self.cvar = cvar
        self.clabel = clabel
        self.defaultColor = defaultColor
        self.byIndices = byIndices
        
        # explicitly set the bounds of the range used for value scaling
        if 'minval' in kwargs:
            self.minval = kwargs['minval']
        else:
            if len(self.vallist)>0:
                self.minval = min(self.vallist)
        if 'maxval' in kwargs:
            self.maxval = kwargs['maxval']
        else:
            if len(self.vallist)>0:
                self.maxval = max(self.vallist)        
        
        if len(self.vallist)>0 and type(list(self.vallist)[0]) is str:
            # values are strings. no fractional scaling allowed
            self.byIndices=True
        
        if self.byIndices:
            # select based on the index in the list
            self.valFunc = self.indexFunc
        else:
            if logScale:
                # select based on log-scaled fractional value within a range
                self.valFunc = self.logFracFunc
            else:
                # select based on fractional value within a range
                self.valFunc = self.fracFunc
           
        if 'color' in kwargs or len(self.vallist)==0:
            # always one color
            self.cfunc = self.oneColor
            self.valfunc = self.exactFunc
            if 'color' in kwargs:
                self.color = kwargs['color']
            else:
                self.color = defaultColor
        elif 'colorList' in kwargs:
            # select value from list of colors
            self.colorList = kwargs['colorList']
            self.cfunc = self.listFunc
            self.valFunc = self.indexFunc
        elif 'colorDict' in kwargs:
            # select value from dictionary of colors
            self.colorDict = kwargs['colorDict']
            self.cfunc = self.dictFunc
            self.valFunc = self.exactFunc
        elif 'cname' in kwargs:
            # select value from color palette
            self.cname = kwargs['cname']
            if self.cname=='cubeHelix':
                self.makeCubeHelix()
            elif self.cname=='diverging':
                self.makeDiverging()
            else:
                self.makePalette()
        else:
            # use one color
            if len(self.vallist)==1:
                self.color = 'black'
                self.cfunc = self.oneColor
                self.valFunc = self.exactFunc
            else:
                self.cname = 'coolwarm'
                self.makePalette()
  
    def fracFunc(self, val:Any) -> float:
        '''get the position of the value scaled by value as a fraction from 0 to 1'''
        return (val-self.minval)/(self.maxval-self.minval)
        
    def logFracFunc(self, val:Any) -> float:
        return (np.log10(val)-np.log10(self.minval))/(np.log10(self.maxval)-np.log10(self.minval))
        
    def indexFunc(self, val:Any) -> float:
        '''get the position of the value scaled by order in the list as a fraction from 0 to 1'''
        if not val in self.vallist:
            return -1
        return self.vallist.index(val)/(len(self.vallist)-1)
    
    def exactFunc(self, val:Any) -> Any:
        return val
    
    #-------------------------
    
    def adjust_lightness(self, color, amount=0.5):
        '''https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib'''
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

    def adjust_saturation(self, color, amount=0.5):
        '''https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib'''
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], c[1], max(0, min(1, amount * c[1])))

    #-------------------------
    
    def oneColor(self, val:Any) -> str:
        '''always return the same color'''
        return self.color
                
    def listFunc(self, val:Any) -> str:
        ''' get a color from a list of values '''
        i = int(val*len(self.vallist))
        if i<0 or i>len(self.colorList):
            return self.defaultColor
        return self.colorList[i]
    
    def dictFunc(self, val:Any) -> str:
        '''get a color from a dictionary'''
        if not val in self.colorDict:
            return self.defaultColor
        return self.colorDict[val]
    
    def makeCubeHelix(self):
        self.cmap = sns.cubehelix_palette(as_cmap=True, rot=-0.4)
        self.cfunc = self.cmapFunc
    
    def makeDiverging(self):
        self.cmap = sns.diverging_palette(220, 20, as_cmap=True)
        self.cfunc = self.cmapFunc
    
    def makePalette(self):
        self.cmap = sns.color_palette(self.cname, as_cmap=True)
        self.cfunc = self.cmapFunc
    
    def cmapFunc(self, val:Any) -> str:
        if type(val) is str or val<0 or val>1:
            return self.defaultColor
        cmap = self.cmap
        color = cmap(val)
        
        if (val>0.35 and val<0.65) and self.cname=='coolwarm':
            if val==0.5:
                return 'gray'
            else:
                return self.adjust_saturation(self.adjust_lightness(color, 0.9), 0.5) # darken and desaturate middle color
        else:
            return color

    
    #-------------------------
    
    def getColor(self, val):
        '''get the color for a value'''
        return self.cfunc(self.valFunc(val))
             
