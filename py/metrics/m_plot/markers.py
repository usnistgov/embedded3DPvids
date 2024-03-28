#!/usr/bin/env python
'''tools for handling plot markers'''

# external packages
import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
import itertools
from typing import List, Dict, Tuple, Union, Any, TextIO
import logging
import traceback

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

# plotting
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rc('font', family='Arial')
matplotlib.rc('font', size='10.0')

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)


#-------------------------------------------

class plotMarkers:
    '''for deciding on marker values for plots'''
    
    def __init__(self, vallist:list, mvar:str, mlabel:str
                 , markerSize:int 
                 , lineWidth:float
                 , markerList:list=['o','v','s','^','X','D', 'P', '<', '>', '8', 'p', 'h', 'H']
                 , filledMarker:bool=False
                 , lines:bool=False
                 , lineList:list = ['solid', 'dotted', 'dashed', 'dashdot']
                 , **kwargs):
        self.vallist = vallist  # list of values used to determine color
        self.vallist.sort()
        self.mvar = mvar
        self.mlabel = mlabel
        self.markerSize = markerSize
        self.lineWidth = lineWidth
        self.markerList = markerList
        self.filledMarker = filledMarker
        self.lineList = lineList
        self.line = lines
        
        if self.mvar in [ 'l1w1', 'l1w1relax', 'l1d1', 'l1d1relax', 'l1w2', 'l1w2relax', 'l1d2', 'l1d2relax', 'l1w3', 'l1w3relax', 'change', 'l1w2w3', 'l1w3end']:
            # qualitative
            self.mfunc=self.qualityDict
            self.mvalFunc=self.exactFunc
            return
        
        if mvar=='const' or mvar=='' or len(self.vallist)>len(markerList):
            if not 'marker' in kwargs:
                kwargs['marker'] = self.markerList[0]
            if not 'line' in kwargs:
                kwargs['line'] = self.lineList[0]
        if 'marker' in kwargs:
            self.mfunc = self.constMarker
            self.mvalFunc = self.constFunc
            self.marker = kwargs['marker']
        else:
            self.mfunc = self.listMarker
            self.mvalFunc = self.indexFunc
            
        if 'markerDict' in kwargs:
            self.mfunc = self.dictMarker
            self.mvalFunc = self.exactFunc
            self.mDict = kwargs['markerDict']
            
        if not lines:
            self.lfunc = self.constLine
            self.lvalFunc = self.constFunc
            self.line0 = 'None'
        else:
            if 'lineStyle' in kwargs:
                self.lfunc = self.constLine
                self.lvalFunc = self.constFunc
                self.line = kwargs['lineStyle']
            elif 'lineDict' in kwargs:
                self.lfunc = self.dictLine
                self.lvalFunc = self.exactFunc
                self.lDict = kwargs['lineDict']
            else:
                self.lfunc = self.listLine
                self.lvalFunc = self.indexFunc
            
    #---------------------------
            
    def indexFunc(self, val:Any) -> float:
        '''get the index of this value in the list'''
        if not val in self.vallist:
            return 0
        else:
            return self.vallist.index(val)
        
    def constFunc(self, val:Any) -> float:
        '''always return 0'''
        return 0
    
    def exactFunc(self, val:Any) -> Any:
        '''return the value given'''
        return val
    
    #----------------------------
            
    def listMarker(self, val):
        '''get the marker from a list'''
        return self.markerList[val]

    def constMarker(self, val):
        '''always return the same marker'''
        return self.marker
    
    def dictMarker(self, val):
        '''get the marker from a dictionary'''
        return self.mDict[val]
    
    #-----------------------------
                                    
    def listLine(self, val):
        '''get the line from a list'''
        return self.lineList[val]
    
    def constLine(self, val):
        '''always return the same line'''
        return self.line  
    
    def dictLine(self, val):
        '''get the line style from a dictionary'''
        return self.lDict[val]
    
    def qualityDict(self, val:str) -> str:
        '''get a color given a quality'''
        if val=='no change' or val=='no fusion':
            return 's'
        if val=='fuse' or val=='fuse 1 2 3' or val=='fuse all':
            return 'o' # dark blue
        if val=='fuse last':
            return '$V$' 
        if val=='fuse 1 2':
            return '$p$' # blue
        if val=='fuse only 1st':
            return '$T$'
        if val=='fuse 2 3':
            return '$ꟼ$' # periwinkle
        if val=='fuse 1 3':
            return '$Y$'
        if val=='partial fuse' or val=='partial fuse 1 2 3':
            return '$W$'  # dark green
        if val=='partial fuse 2 3': 
            return '$IU$'  # light green
        if val=='partial fuse 1 2':
            return '$UI$' # teal
        if val=='partial fuse 1 3':
            return '$H$' # teal
        if val=='partial fuse last':
            return '$M$' 
        if val=='partial fuse only 1st':
            return '$J$'
        if val=='fuse droplets':
            return 'P' # indigo
        if val=='rupture' or val=='rupture combined':
            return 'X' # red
        if val=='rupture 2':
            return '$E$' # burnt red
        if val=='rupture 1':
            return '$Ǝ$' # orange
        if val=='rupture 1st':
            return '$ↄ$'
        if val=='rupture 3':
            return '$K$' # yellow
        if val=='rupture both' or val=='rupture 1 2' or val=='rupture all':
            return '$8$' # peach
        if val=='rupture 2 step':
            return '$\#$'
        if val=='fuse rupture' or val=='fuse 1 2 and rupture 12': 
            return '<' # purple
        if val=='rupture both fuse droplets':
            return '>' # light purple
        if val=='shrink':
            return 'v'
        else:
            print(val)
            # return self.markerList[int(np.random.randint(len(self.markerList), size=1))]
            return '.'
    
    #-----------------------------
    
    def getMarker(self, val:Any) -> str:
        '''get the marker from a value'''
        return self.mfunc(self.mvalFunc(val))
    
    def markerStyle(self, val:Any, color:str) -> dict:
        '''get the marker styling'''
        out = {'marker':self.getMarker(val), 'color':color, 'linewidth':self.lineWidth, 's':self.markerSize}
        if self.filledMarker or val in ['no change', 'no fusion']:
            out['facecolors'] = color
            out['edgecolors'] = 'none'
        else:
            out['facecolors'] = 'none'
            out['edgecolors'] = color
        return out
            
    
    def getLine(self, val) -> str:
        '''get the line style from a value'''
        if not self.line:
            return 'None'
        return self.lfunc(self.lvalFunc(val))
