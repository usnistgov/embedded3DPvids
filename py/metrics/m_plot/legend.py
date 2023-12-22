#!/usr/bin/env python
'''For creating legends separately from the plots. This is useful if you have multiple axes that share the same legend'''

# external packages
import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
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
from m_plot.color import plotColors
from m_plot.markers import plotMarkers

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

class plotLegend:
    '''for creating an independent legend'''
    
    def __init__(self, colors:plotColors
                 , markers:plotMarkers
                 , ms
                 , filedf:pd.DataFrame
                 , line:bool=False
                 , fs:int=10
                 ,**kwargs):
        self.colors = colors
        self.markers = markers
        self.ms = ms
        self.filedf = self.ms.ss
        self.line = line
        self.fs = fs
        self.referenceHandles = {}

    def colorPatches(self) -> list:
        '''create a legend that is just color patches'''
        ph = [plt.plot([], marker="", ls="", label=self.colors.clabel)[0]]; # Canvas
        plist = [mpatches.Patch(color=self.colors.getColor(val)
                                , label=val)
                 for val in self.colors.vallist]
        self.plist = ph+plist
        return self.plist
    
    def idealLine(self):
        '''create a handle for the ideal value line'''
        return mlines.Line2D([], [], color='gray'
                               , marker='None'
                               , linestyle='dashed'
                               , label='Ideal')
        
    def oneColorMarkers(self, addRef:bool=False) -> list:
        '''create a legend that is just markers of one color'''
        ph = [plt.plot([], marker="", ls="", label=self.markers.mlabel)[0]]; # Canvas
        il = self.idealLine()
        plist = [mlines.Line2D([], [], markeredgecolor=self.colors.getColor(0)
                               , markerfacecolor='None'
                               , marker=self.markers.getMarker(val)
                               , linestyle='None'
                               , markersize=self.markers.markerSize
                               , label=val)
                 for val in self.markers.vallist]
        self.plist = ph+il+plist
        return self.plist
    
    def justC(self) -> bool:
        '''duplicated info in markers'''
        return (self.markers.mvar in self.colors.cvar or len(self.markers.mlabel)==0)
    
    def justM(self) -> bool:
        '''duplicated info in colors'''
        return (self.colors.cvar in self.markers.mvar or len(self.colors.clabel)==0)
    
    def legendLabel(self) -> str:
        '''get a label for the whole legend'''
        if self.justC():
            # duplicated info in markers
            s = self.colors.clabel
        elif self.justM():
            s = self.markers.mlabel
        else:
            s = f'{self.markers.mlabel}, {self.colors.clabel}'
        return self.ms.varSymbol(s)
    
    def markerLabel(self, cval:str, mval:str) -> str:
        '''determine a marker label'''
        if self.justC():
            # duplicated info in markers
            return cval
        elif self.justM():
            return mval
        else:
            return f'{cval}, {mval}'
        
    def getMultiMarker(self, cval, mval):
        '''get a marker object to add to handles, knowing that colors and markers depend on cval and mval'''
        return mlines.Line2D([], [], markeredgecolor=self.colors.getColor(cval)
                               , markerfacecolor='None'
                               , color=self.colors.getColor(cval)
                               , marker=self.markers.getMarker(mval)
                               , linestyle=self.markers.getLine(mval)
                               , label=self.markerLabel(cval, mval))
        
    def hasElements(self, cval, mval) -> bool:
        '''determine if there might be elements in this set. for example, if cval is sigma,ink_velocity and mval is sigma, and they have different sigma values, then there will be no elements'''
        fdf = self.filedf.copy()
        if self.colors.cvar in fdf:
            fdf = fdf[fdf[self.colors.cvar]==cval]
        if self.markers.mvar in fdf:
            fdf = fdf[fdf[self.markers.mvar]==mval]
        return len(fdf)>0

    def multiColorMarkers(self, addRef:bool) -> list:
        '''create a legend that has markers of multiple colors'''
        ph = [plt.plot([], marker="", ls=""
                       , label=self.legendLabel())[0]]; # Canvas
        plist = []
        plist.append(self.idealLine())
        for mval in self.markers.vallist:
            for cval in self.colors.vallist:
                if self.hasElements(cval, mval):
                    plist.append(self.getMultiMarker(cval, mval))
        self.plist = ph+plist
        return self.plist
    
    def patches(self, addRef:bool=False) -> list:
        '''make a legend label for the markers'''
        if len(self.colors.vallist)>1:
            if len(self.markers.vallist)>1:
                self.multiColorMarkers(addRef)
            else:
                self.colorPatches()
        elif len(self.markers.vallist)>1:
            self.oneColorMarkers(addRef)
        else:
            return []
        plt.close()
        return self.plist+list(self.referenceHandles.values())
    
    def swatchLegend(self, fig, ax, legendloc:str='above', **kwargs):
        '''make a legend of swatches'''
        if len(fig.axes)==1 or (legendloc=='right'):
            bbox = (1.05, 1)
            loc = 'upper left'
        elif (legendloc=='above'):
            bbox = (0,1)
            loc = 'lower left'
        elif (legendloc=='below'):
            bbox = (0, -0.5)
            loc = 'upper left'
        elif (legendloc=='inset'):
            bbox = (1,0)
            loc = 'lower right'
        elif (legendloc=='center'):
            bbox = (0.5, 0.5)
            loc = 'center'
        else:
            raise ValueError(f'Unexpected legendloc {legendloc}')
        
        self.swatches = ax.legend(handles=self.patches(), bbox_to_anchor=bbox, loc=loc, frameon=False, fontsize=self.fs)
            
    
    def colorBar(self, fig, legendy:float=-0.1, legendloc:str='below', **kwargs) -> None:
        '''add a color bar to the figure'''
        if legendloc=='below' or legendloc=='above':
            cbaxes = fig.add_axes([0.2, legendy, 0.6, 0.05])
        else:
            cbaxes = fig.add_axes([1, 0.2, 0.05, 0.6])
        nm = plt.Normalize(vmin=self.colors.minval, vmax=self.colors.maxval)
        sm = plt.cm.ScalarMappable(cmap=self.colors.cmap, norm=nm)
        if legendloc=='below' or legendloc=='above':
            ori = 'horizontal'
        else:
            ori = 'vertical'
            
        cbar = plt.colorbar(sm, cax=cbaxes, orientation=ori)
        cbar.ax.set_xlabel(self.colors.clabel, fontsize=self.fs)