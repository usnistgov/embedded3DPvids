#!/usr/bin/env python
'''Functions for plotting still and video data. Adapted from https://github.com/usnistgov/openfoamEmbedded3DP'''

# external packages
import os, sys
import traceback
import logging
import pandas as pd
import matplotlib
from typing import List, Dict, Tuple, Union, Any, TextIO
import csv

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
from m_summary.summary_metric import summaryMetric
from m_stats import *
from p_multi import multiPlot
from p_scatter import scatterPlot
from p_mesh import meshPlot
from p_contour import contourPlot

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
# plotting
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rc('font', family='Arial')
matplotlib.rc('font', size='10.0')

#-------------------------------------------------------------

def simplifyOutliers(ss:pd.DataFrame) -> None:
    '''simplify rare morphologies'''
    ss.replace({'fuse 1 2 bubbles':'fuse 1 2'
                , 'fuse 1 2 partial fuse 2 3':'fuse 1 2 3'
                , 'fuse 12 3':'fuse 2 3'
                , 'rupture 1 fuse 2 3':'fuse rupture'
                , 'rupture 1 fuse droplets':'fuse rupture'
                , 'rupture 2 fuse droplets':'fuse rupture'
                , 'rupture1':'rupture 1'
                , 'fuse bubbles':'fuse'}, inplace=True)
    
def simplifyMorphologies(ss:pd.DataFrame) -> None:
    '''simplify all morphologies to rupture, fusion, partial fusion'''
    ss.replace({'fuse 1 2':'fuse'
                , 'fuse 2 3':'fuse'
                , 'fuse 1 3':'fuse'
                , 'fuse 1 2 3':'fuse'
                , 'partial fuse 1 2 3':'partial fuse'
                , 'partial fuse 2 3':'partial fuse'
                , 'partial fuse 1 2':'partial fuse'
                , 'partial fuse 1 3':'partial fuse'
                , 'fuse droplets':'partial fuse'
                , 'rupture combined':'rupture'
                , 'rupture 1':'rupture'
                , 'rupture 2':'rupture'
                , 'rupture 3':'rupture'
                , 'rupture both':'rupture'
                , 'rupture 1 2':'rupture'
                , 'fuse 1 2 and rupture 12':'fuse rupture'
                , 'rupture both fuse droplets':'fuse rupture'}, inplace=True)
    
def plainEnglishQuality(cvar:str) -> str:
    '''convert the measured quality variable to plain english'''
    out = ''
    if 'relax' in cvar:
        s1 = 'After'
    else:
        s1 = 'While'
    if cvar[2]=='w':
        s2 = 'writing'
    else:
        s2 = 'disturbing'
    line = cvar[3]
    return f'{s1} {s2} line {line}'
    

class qualityScatter(multiPlot):
    '''for plotting quality as color on the same x and y var on different axes'''
    
    def __init__(self, ms:summaryMetric, ss:pd.DataFrame, xvar:str, yvar:str, simplify:bool=False, dx:float=0.05, dy:float=0.05, rigid:bool=True, **kwargs):
        self.xvar = xvar
        self.yvar = yvar
        self.legendMade = False
        self.ms = ms
        self.ss = ss.copy()
        simplifyOutliers(self.ss)
        if simplify:
            simplifyMorphologies(self.ss)
        self.getRC()
        if 'sharex' in kwargs:
            self.sharex = kwargs['sharex']
            kwargs.pop('sharex')
        else:
            self.sharex = True
        if 'sharey' in kwargs:
            self.sharey = kwargs['sharey']
            kwargs.pop('sharey')
        super().__init__(self.rows, self.cols, sharex=self.sharex, sharey=self.sharey, dx=dx, dy=dy, errorBars=False, rigid=rigid, **kwargs)
        self.combineLegendBox()
        self.plots()
        
    def getRC(self) -> None:
        '''get rows and columns'''
        self.sharex = True
        self.sharey = True
        self.rows = 4
        self.cols = 3
        self.cvars = np.array([['l1w1', 'l1w2', 'l1w3'], ['l1w1relax', 'l1w2relax', 'l1w3relax'], ['l1d1', 'l1d2', ''], ['l1d1relax', 'l1d2relax', '']])
        
    def combineLegendBox(self) -> None:
        '''combine the bottom right plots to put the legend'''
        gs = self.axs[2, 2].get_gridspec()
        # remove the underlying axes
        for ax in self.axs[2:, -1]:
            ax.remove()
        self.axbig = self.fig.add_subplot(gs[2:, -1])
        
        uniqueVals = pd.unique(self.ss[['l1w1', 'l1w2', 'l1w3', 'l1w1relax', 'l1w2relax', 'l1w3relax', 'l1d1', 'l1d2', 'l1d1relax', 'l1d2relax']].values.ravel('K'))
        sslegen = pd.DataFrame([{'change':c} for c in uniqueVals])  # create a dataframe with all the possible morphologies
        sslegen.dropna(inplace=True)
        self.legendObj = scatterPlot(self.ms, sslegen, justLegend=True, ax=self.axbig, cvar='change', yvar='change', xvar='change', legendVals=list(sslegen['change']), fs=self.fs)

    
    def plot(self, i:int, j:int) -> None:
        cvar = self.cvars[i,j]
        if not cvar in self.ss:
            return
        kwargs = {**self.kwargs0, **{'xvar':self.xvar, 'yvar':self.yvar, 'cvar':cvar
                                     , 'ax':self.axs[i,j], 'fig':self.fig
                                     , 'plotType':self.plotType}}
        self.objs[i,j] = scatterPlot(self.ms, self.ss
                                 , set_xlabel=(i==self.rows-1), set_ylabel=(j==0)
                                 , legend=False
                                 , legendloc='right'   
                                 , **kwargs)
        self.axs[i,j].set_title(plainEnglishQuality(cvar))

    def plots(self):
        '''plot all plots'''
        for i in range(self.rows):
            for j in range(self.cols):
                self.plot(i,j)
        self.clean()

                
class qualityScatterSpacing(multiPlot):
    '''for plotting quality as color on the same x and y var on different axes, for different spacings'''
    
    def __init__(self, ms:summaryMetric, ss:pd.DataFrame, xvar:str, yvar:str, cvar0:str, y0var:str='spacing_adj'
                 , write:bool=True, relax:bool=True, dx:float=0.05, dy:float=0.05, rigid:bool=True, **kwargs):
        self.xvar = xvar
        self.yvar = yvar
        self.y0var = y0var
        self.cvar0 = cvar0  # l1w1, l1w2, l1w3, l1d2, or l1d2 or l1w2w3
        self.write=write
        self.relax=relax
        self.legendMade = False
        self.ms = ms
        self.ss = ss.copy()
        simplifyOutliers(self.ss)
        if self.cvar0=='l1w2w3':
            self.createCompositeCol()   # combine w2, w2relax, and w3 to categorize overall shape
            self.relaxVar = 'l1w3relax'
        elif self.cvar0=='l1w3end':
            self.createEndCol()
            self.relax = False
            self.relaxVar = ''
        else:
            self.relaxVar = f'{self.cvar0}relax'
        self.sssimple = self.ss.copy()
        simplifyMorphologies(self.sssimple)
        self.getRC()
        
        # override sharing
        if 'sharex' in kwargs:
            self.sharex = kwargs['sharex']
            kwargs.pop('sharex')
        if 'sharey' in kwargs:
            self.sharey = kwargs['sharey']
            kwargs.pop('sharey')
        super().__init__(self.rows, self.cols, sharex=self.sharex, sharey=self.sharey, dx=dx, dy=dy, errorBars=False, rigid=rigid, **kwargs)
        self.combineLegendBox()
        self.groupCols()
        self.plots()
        
    def findComposite(self, i:int, row:pd.Series) -> str:
        '''find the final w2 w3 status for a row'''
        w2 = row['l1w2']
        w2r = row['l1w2relax']
        w3 = row['l1w3']
        if not (type(w2) is str and type(w2r) is str and type(w3) is str):
            return np.nan
        if w3=='fuse 1 2 3':
            return 'fuse 1 2 3'
        if w3=='fuse droplets':
            return 'fuse droplets'
        if ('partial fuse' in w2 and ('no change' in w2r or 'partial fuse' in w2r)) or ('partial fuse' in w2r):
            # partial fusion during w2
            if 'partial fuse 1 2' in w3:
                return 'partial fuse 1 2'
            if 'fuse 1 2' in w3:
                return 'fuse 1 2'
            if 'partial fuse 2 3' in w3:
                return 'partial fuse last'
            if 'fuse 2 3' in w3:
                return 'partial fuse last'
            if 'rupture' in w3:
                return 'fuse rupture'
            if 'no fusion' in w3:
                return 'partial fuse only 1st'
        if ('fuse' in w2 or 'fuse' in w2r):
            # full fuse during w2
            if 'partial fuse 2 3' in w3:
                return 'partial fuse last'
            if 'fuse 2 3' in w3 or 'fuse' in w3:
                return 'fuse last'
            if 'rupture' in w3:
                return 'fuse rupture'
            if 'no fusion' in w3:
                return 'fuse only 1st'
        if 'rupture' in w2 or 'rupture' in w2r:
            # rupture during w2
            if 'no fusion' in w3:
                return 'rupture 1st'
            if 'rupture' in w3:
                return 'rupture 2 step'
            if 'fuse' in w3:
                return w3
        if 'no fusion' in w2 and ('no change' in w2r or 'no fusion' in w2r):
            # no fusion during w2
            return w3              
        
        raise ValueError(f'Unexpected combination {w2}, {w2r}, {w3}')
        
    def findEndpoint(self, i:int, row:pd.Series) -> str:
        '''find the final w2 w3 status for a row'''
        w2 = row['l1w2']
        w2r = row['l1w2relax']
        w3 = row['l1w3']
        w3r = row['l1w3relax']
        if not (type(w2) is str and type(w2r) is str and type(w3) is str):
            return np.nan
        if w3=='fuse 1 2 3':
            return 'fuse all'
        if w3=='fuse droplets':
            return 'fuse rupture'
        if ('partial fuse' in w2 and ('no change' in w2r or 'partial fuse' in w2r)) or ('partial fuse' in w2r):
            # partial fusion during w2
            if 'partial fuse 1 2' in w3 or 'partial fuse 1 2' in w3r:
                return 'partial fuse 1 2'
            if 'fuse 1 2' in w3 or 'fuse 1 2' in w3r:
                return 'fuse 1 2'
            if 'partial fuse 2 3' in w3 or 'partial fuse 2 3' in w3r:
                return 'partial fuse'
            if 'fuse 2 3' in w3 or 'fuse 2 3' in w3r:
                return 'partial fuse'
            if 'rupture' in w3 or 'rupture' in w3r:
                return 'fuse rupture'
            if 'no fusion' in w3 and 'no change' in w3r:
                return 'partial fuse 1 2'
        if ('fuse' in w2 or 'fuse' in w2r):
            # full fuse during w2
            if 'partial fuse 2 3' in w3 or 'partial fuse 2 3' in w3r:
                return 'partial fuse'
            if 'fuse' in w3 or 'fuse' in w3r:
                return 'fuse all'
            if 'rupture' in w3 or 'rupture' in w3r:
                return 'fuse rupture'
            if 'no fusion' in w3 and 'no change' in w3r:
                return 'fuse 1 2'
        if 'rupture' in w2 or 'rupture' in w2r:
            # rupture during w2
            if 'no fusion' in w3 and 'no change' in w3r:
                if 'rupture' in w2:
                    return w2
                else:
                    return w2r
            if 'rupture' in w3 or 'rupture' in w3r:
                return 'rupture all'
            if 'fuse' in w3 or 'fuse' in w3r:
                return 'fuse rupture'
        if 'no fusion' in w2 and ('no change' in w2r or 'no fusion' in w2r):
            # no fusion during w2
            if 'no fusion' in w3 and ('no change' in w3r or 'no fusion' in w3r):
                return 'no fusion'
            if 'partial fuse 2 3' in w3 or 'partial fuse 2 3' in w3r:
                return 'partial fuse 2 3'
            if 'fuse' in w3 or 'fuse' in w3r:
                return 'fuse 2 3'
            if 'rupture' in w3 or 'rupture' in w3r:
                if 'rupture' in w3:
                    return w3
                else:
                    return w3r          
        
        raise ValueError(f'Unexpected combination {w2}, {w2r}, {w3}')
        
    def createCompositeCol(self):
        '''combine write2 and write3 to describe overall print for all rows'''
        if 'l1w2w3' in self.ss:
            return
        for i,row in self.ss.iterrows():
            name = self.findComposite(i,row)
            self.ss.loc[i, 'l1w2w3'] = name
            
    def createEndCol(self):
        '''combine write2 and write3 to describe overall print for all rows'''
        if 'l1w3end' in self.ss:
            return
        for i,row in self.ss.iterrows():
            name = self.findEndpoint(i,row)
            self.ss.loc[i, 'l1w3end'] = name
       
    def getRows(self):
        '''get the number of rows'''
        if self.relax and self.write:
            self.cvarShort = [self.cvar0, self.relaxVar]
        elif self.relax:
            self.cvarShort = [self.relaxVar]
        elif self.write:
            self.cvarShort = [self.cvar0]
        else:
            raise ValueError('No plots requested. write and/or relax must be True.')
        self.rows = len(self.cvarShort)
        self.cvars = np.array([[var for i in range(self.cols-1)] for var in self.cvarShort])
        
    def getRC(self) -> None:
        '''get rows and columns'''
        self.sharex = True
        self.sharey = False
        self.cols = 5
        self.getRows()
        
    def combineLegendBox(self) -> None:
        '''combine the bottom right plots to put the legend'''
        gs = self.axs[0,self.cols-1].get_gridspec()
        # remove the underlying axes
        for ax in self.axs[:, -1]:
            ax.remove()
        self.axbig = self.fig.add_subplot(gs[:, -1])
        
        if self.cols>2 or (hasattr(self, 'spacing') and self.spacing>0):
            ssi = self.ss[self.ss.spacing.isin([0.5, 0.875, 1.25])]
            uniqueVals = pd.unique(ssi[self.cvarShort].values.ravel('K'))  # get the unique values in all columns together
        else:
            uniqueVals = pd.unique(self.sssimple[self.cvarShort].values.ravel('K'))  # get the unique values in all columns together
        sslegen = pd.DataFrame([{'change':c} for c in uniqueVals])  # create a dataframe with all the possible morphologies
        sslegen.dropna(inplace=True)
        self.legendObj = scatterPlot(self.ms, sslegen, justLegend=True
                                     , ax=self.axbig, cvar='change', yvar='change', xvar='change'
                                     , legendVals=list(sslegen['change']), fs=self.fs)
        
    def groupCols(self) -> None:
        '''group columns 1:'''
        for j in range(2,self.cols-1):
            for i in range(self.rows): 
                self.axs[0,1].get_shared_y_axes().join(self.axs[i,j], self.axs[0,1])  # share the y axes for these columns
                self.axs[0,1].get_shared_x_axes().join(self.axs[i,j], self.axs[0,1])  # share the x axes for these columns
      #  self.axs[0,0].sharex(self.axs[1,0])  # share the y axes for these columns
      #  self.axs[0,0].sharey(self.axs[1,0])  # share the x axes for these columns
    
    def plot(self, i:int, j:int) -> None:
        cvar = self.cvars[i,j]
        if not cvar in self.ss:
            return
        kwargs = {**self.kwargs0, **{'xvar':self.xvar, 'yvar':self.yvar, 'cvar':cvar
                                     , 'ax':self.axs[i,j], 'fig':self.fig
                                     , 'plotType':self.plotType}}
        if j==0:
            if hasattr(self, 'spacing') and self.spacing>0:
                ss = self.ss[self.ss.spacing==self.spacing]
                title = f'spacing = {self.spacing} $d_i$'
            else:
                title = 'simplified'
                ss = self.sssimple
                kwargs['yvar'] = self.y0var
                kwargs['logy'] = False
                kwargs['dy'] = 0.15
        elif j==1:
            ss = self.ss[self.ss.spacing==0.5]
            title = 'spacing = 0.5 $d_i$'
        elif j==2:
            ss = self.ss[self.ss.spacing==0.875]
            title = 'spacing = 0.875 $d_i$'
        elif j==3:
            ss = self.ss[self.ss.spacing==1.250]
            title = 'spacing = 1.250 $d_i$'
        self.objs[i,j] = scatterPlot(self.ms, ss
                                 , set_xlabel=(not self.sharex or (i==self.rows-1)), set_ylabel=(j<2)
                                 , legend=False
                                 , legendloc='right'   
                                 , **kwargs)
        self.axs[i,j].set_title(f'{title}\n{plainEnglishQuality(cvar)}', fontsize=self.fs)

    def plots(self):
        '''plot all plots'''
        for i in range(self.rows):
            for j in range(self.cols-1):
                self.plot(i,j)
        self.fig.tight_layout()
        self.clean()
        
        
class qualityScatterSimple(qualityScatterSpacing):
    
    def __init__(self, ms:summaryMetric, ss:pd.DataFrame, xvar:str, yvar:str, cvar0:str, spacing:float=0, **kwargs):
        self.spacing=spacing
        super().__init__(ms, ss, xvar, yvar, cvar0, **kwargs)
        
    def getRC(self) -> None:
        '''get rows and columns'''
        self.sharex = True
        self.sharey = False
        self.cols = 2
        self.getRows()

        
class qualityScatterXY(multiPlot):
    '''for plotting quality  as color on arbitrary x and y var'''
    
    def __init__(self, ms:summaryMetric, ss:pd.DataFrame, xvars:np.array, yvars:np.array, cvar:str, simplify:bool=True, dx:float=0.05, dy:float=0.05, rigid:bool=True, **kwargs):
        if type(xvars) is np.array:
            self.xvars = xvars
        else:
            self.xvars = np.array(xvars)
        if type(yvars) is np.array:
            self.yvars = yvars
        else:
            self.yvars = np.array(yvars)
        if not self.xvars.shape==self.yvars.shape:
            raise ValueError('xvars and yvars must be same shape')
        self.cvar = cvar  # l1w1, l1w1relax, l1w2, l1w2relax, etc.
        self.legendMade = False
        self.ms = ms
        self.ss = ss.copy()
        simplifyOutliers(self.ss)
        if simplify:
            simplifyMorphologies(self.ss)
        self.getRC()
        
        # override sharing
        if 'sharex' in kwargs:
            self.sharex = kwargs['sharex']
            kwargs.pop('sharex')
        if 'sharey' in kwargs:
            self.sharey = kwargs['sharey']
            kwargs.pop('sharey')
        super().__init__(self.rows, self.cols, sharex=self.sharex, sharey=self.sharey, dx=dx, dy=dy, errorBars=False, legendAbove=True, tightLayout=False, rigid=rigid, **kwargs)
        # self.combineLegendBox()
        self.plots()

    def getRC(self) -> None:
        '''get rows and columns'''
        self.sharex = False
        self.sharey = False
        self.rows, self.cols = self.xvars.shape
        # self.cols = self.cols+1
        
    def makeLegend(self) -> None:
        '''combine the bottom right plots to put the legend'''
        uniqueVals = pd.unique(self.ss[self.cvar].values.ravel('K'))  # get the unique values in all columns together
        sslegen = pd.DataFrame([{'change':c} for c in uniqueVals])  # create a dataframe with all the possible morphologies
        sslegen.dropna(inplace=True)
        self.legendObj = scatterPlot(self.ms, sslegen, justLegend=True
                                     , ax=self.legendAx, cvar='change', yvar='change', xvar='change'
                                     , legendVals=list(sslegen['change']), ncol=6, fs=self.fs)
    
    def plot(self, i:int, j:int) -> None:
        cvar = self.cvar
        if not cvar in self.ss:
            return
        kwargs = {**self.kwargs0, **{'xvar':self.xvars[i,j], 'yvar':self.yvars[i,j], 'cvar':cvar
                                     , 'ax':self.axs[i,j], 'fig':self.fig
                                     , 'plotType':self.plotType}}
        self.objs[i,j] = scatterPlot(self.ms, self.ss
                                 , set_xlabel=(not self.sharex or (i==self.rows-1)), set_ylabel=(not self.sharey or (i==0))
                                 , legend=False
                                 , legendloc='right'   
                                 , **kwargs)
        self.axs[i,j].set_title(plainEnglishQuality(cvar), fontsize=self.fs)

    def plots(self):
        '''plot all plots'''
        for i in range(self.rows):
            for j in range(self.cols):
                self.plot(i,j)
        # self.fig.tight_layout()
        self.makeLegend()
        self.clean()