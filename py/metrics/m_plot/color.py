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
        self.swatch = True
        
        if self.cvar in [ 'l1w1', 'l1w1relax', 'l1d1', 'l1d1relax', 'l1w2', 'l1w2relax', 'l1d2', 'l1d2relax', 'l1w3', 'l1w3relax', 'change', 'l1w2w3', 'l1w3end']:
            # qualitative
            self.cfunc=self.qualityDict
            self.valFunc=self.exactFunc
            return
        
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
            
        if len(self.vallist)>10:
            self.swatch = False
        
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
            self.valFunc = self.exactFunc
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
        '''convert the value into a log-scaled fraction between the min and max val'''
        return (np.log10(val)-np.log10(self.minval))/(np.log10(self.maxval)-np.log10(self.minval))
        
    def indexFunc(self, val:Any) -> float:
        '''get the position of the value scaled by order in the list as a fraction from 0 to 1'''
        if not val in self.vallist:
            return -1
        return self.vallist.index(val)/(len(self.vallist)-1)
    
    def exactFunc(self, val:Any) -> Any:
        '''return the exact value'''
        return val
    
    #-------------------------
    
    def adjust_lightness(self, color, amount=0.5):
        '''adjust the lightness of the color. https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib'''
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

    def adjust_saturation(self, color, amount=0.5):
        '''adjust the saturation of the color. https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib'''
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
        '''create a colorfunction for the cubehelix palette'''
        self.cmap = sns.cubehelix_palette(as_cmap=True, rot=-0.4)
        self.cfunc = self.cmapFunc
    
    def makeDiverging(self):
        '''create a diverging palette'''
        self.cmap = sns.diverging_palette(220, 20, as_cmap=True)
        self.cfunc = self.cmapFunc
    
    def makePalette(self):
        '''create a color palette given a palette name'''
        self.cmap = sns.color_palette(self.cname, as_cmap=True)
        self.cfunc = self.cmapFunc
    
    def cmapFunc(self, val:Any) -> str:
        '''get a color from a value'''
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

    def qualityDict(self, val:str) -> str:
        '''get a color given a quality'''
        if val=='no change' or val=='no fusion':
            return '#bcbcbc'
        if val=='fuse' or val=='fuse 1 2 3' or val=='fuse all':
            return '#6081c5' # dark blue
        if val=='fuse last':
            return '#3477eb' #blue
        if val=='fuse 1 2':
            return '#2c8fd1' # blue
        if val=='fuse only 1st':
            return '#15853e' # green
        if val=='fuse 2 3':
            return '#808cd9' # periwinkle
        if val=='fuse 1 3':
            return '#5f9ac2' # blue
        if val=='partial fuse' or val=='partial fuse 1 2 3':
            return '#0f5e43'  # dark green
        if val=='partial fuse 2 3': 
            return '#5acca5'  # light green
        if val=='partial fuse 1 2':
            return '#08a6a8' # teal
        if val=='partial fuse 1 3':
            return '#468c74' # green
        if val=='partial fuse last':
            return '#73c957' # green
        if val=='partial fuse only 1st':
            return '#71990c' # oliver
        if val=='fuse droplets':
            return '#110c75' # indigo
        if val=='rupture' or val=='rupture combined':
            return '#e06b4a' # peach
        if val=='rupture 2':
            return '#9c2e1f' # burnt red
        if val=='rupture 2 step':
            return '#780839' # magenta
        if val=='rupture 1':
            return '#e0520b' # orange
        if val=='rupture 1st':
            return '#c98279' # pink
        if val=='rupture 3':
            return '#b58a09' # yellow
        if val=='rupture both' or val=='rupture 1 2' or val=='rupture all':
            return '#e06b4a' # peach
        if val=='fuse rupture' or val=='fuse 1 2 and rupture 12': 
            return '#18da3f' # green
        if val=='rupture both fuse droplets':
            return '#ab84e0' # light purple
        if val=='shrink':
            return '#486cc7'
        else:
            print(val)
            # return '#%02x%02x%02x' % tuple(np.random.choice(range(256), size=3))  # random color
            return '#888888'
        
    
    #-------------------------
    
    def getColor(self, val):
        '''get the color for a value'''
        return self.cfunc(self.valFunc(val))
             