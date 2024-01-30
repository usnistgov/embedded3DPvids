#!/usr/bin/env python
'''Functions for plotting video and image data. Adapted from https://github.com/usnistgov/openfoamEmbedded3DP'''

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
from val.v_print import printVals
from val.v_fluid import fluidVals

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
# plotting
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rc('font', family='Arial')
matplotlib.rc('font', size='10.0')


#----------------------------------------------


class folderPlots:
    '''A generic class used for plotting many folders at once. Subclasses are comboPlot, which puts everything on one plot, and gridOfPlots, which puts everythign in separate plots based on viscosity.'''
    
    def __init__(self, folders:List[str], imsize:float, fontsize:int=10, **kwargs):
        '''topFolder is the folder we're plotting
            imsize is the size of the total image in inches
            split is true to split into separate plots by surface tension'''
        self.kwargs = kwargs
        self.ab = not 'adjustBounds' in self.kwargs or self.kwargs['adjustBounds']==True
        self.flist = folders
        self.imsize = imsize
        self.fontsize = fontsize
        plt.rc('font', size=fontsize) 
        self.plotsLists(**kwargs) 
        
    def plotsLists(self, xvar:str='ink.var', yvar:str='sup.var', **kwargs):
        '''plotsLists initializes variable names for gridOfPlots and comboPlots objects. 
        vname=val for fluid composition data. vname=v for speed data'''
        self.pvlists = [printVals(f) for f in self.flist]
        
        self.xvar = xvar
        self.yvar = yvar
        self.xfunc = xvar
        self.yfunc = yvar
        
        self.getBases()
        for s in ['x', 'y']:
            self.getLabels(s)
        
        try:
            # find lists of unique x values and y values
            self.xlists = self.unqList('x')
            self.ylists = self.unqList('y')
            self.xlistsreal = [[]]*len(self.bases)
            self.ylistsreal = [[]]*len(self.bases)
            self.indicesreal = pd.DataFrame({'x':[],'y':[]})
            self.xylistsreal = [[[]]]*len(self.bases)
        except:
            raise ValueError('Failed to identify x and y variables')
            traceback.print_exc()
        return self
    
    def getBases(self) -> None:
        '''get the list of materials bases (e.g. mineral oil) to determine how many plots to make'''
        self.bases = []
        for pv in self.pvlists:
            base = pv.base()
            if not base in self.bases:
                self.bases.append(base)
        return True
    
    def unqList(self, var:str) -> List:
        '''get a list of unique x or y variables for each plot. var should be x or y'''
        func = getattr(self, f'{var}func')
        pvlist = self.pvlists
        
        unqs = [[]]*len(self.bases)
        for pv in pvlist:
            axnum = pv.ax
            try:
                pvval = pv.value(func, var)
            except AttributeError:
                pass
            else:
                if not pvval in unqs[axnum]:
                    unqs[axnum].append(pvval)
        for i in range(len(self.bases)):
            unqs[i].sort()
        return unqs
    
    def getLabels(self, var:str) -> List[str]:
        '''get the axis labels and store axis numbers for each folder. vname=var for composition data. vname=v for velocity data'''
        labels = ['']*len(self.bases)
        func = getattr(self, f'{var}func')
        for pv in self.pvlists:
            base = pv.base()
            for i,b in enumerate(self.bases): # fill the list of labels corresponding to list of bases
                if b==base:
                    pv.ax = i # store the axis number for this folder
                    if len(labels[i])==0: # fill this label if not already filled
                        labels[i] = pv.label(func)
        setattr(self, f'{var}labels', labels)
        
    
    def checkVals(self, xbase:bool=True, xvar:bool=True, ybase:bool=True, yvar:bool=True, **kwargs) -> bool:
        '''check that the samples all have the same materials space'''
        
        if xbase:
            xbase0 = getattr(self.pvlists[0], self.xfluid).base
        if xvar:
            xvar0 = getattr(self.pvlists[0], self.xfluid).var
        if ybase:
            ybase0 = getattr(self.pvlists[0], self.xfluid).base
        if yvar:
            yvar0 = getattr(self.pvlists[0], self.xfluid).var
        for pv in self.pvlists[1:]:
            if xbase:
                xbase = getattr(pv, self.xfluid).base
                if not (xbase==xbase0):
                    return False
            if xvar:
                xvar = getattr(pv, self.xfluid).var
                if not (xvar==xvar0):
                    return False
            if ybase:
                ybase = getattr(pv, self.xfluid).base
                if not (ybase==yvar0):
                    return False
            if yvar:
                yvar = getattr(pv, self.xfluid).var
                if not (yvar==yvar0):
                    # if base and variable are not the same, these are not in the same materials space
                    return False
        return True
     
#-------------------------------------------------


class comboPlot(folderPlots):
    '''stores variables needed to create big combined plots across several folders '''
    
    def __init__(self, topFolder:str, xr:List[float], yr:List[float], imsize:float, gridlines:bool=True, **kwargs):
        '''topFolder is a full path name. topFolder contains all of the folders we want to plot
        xr is the min and max x value for each section of the plot, e.g. [-0.7, 0.7]
        yr is the min and max y value for each section of the plot
        imsize is the size of the whole image in inches
        gridlines true to show gridlines'''
        
        super().__init__(topFolder, imsize, **kwargs)
        self.type="comboPlot"
        
        self.figtitle = ""
        self.xr = xr # x bounds for each plot
        self.yr = yr
        self.dx = xr[1]-xr[0] # size of each plot chunk
        self.dy = yr[1]-yr[0]
        self.legdy = 0
        self.xrtots = [[xr[0], xr[0]+(len(xlist))*self.dx] for xlist in self.xlists] # total bounds of the whole plot
        self.yrtots = [[yr[0], yr[0]+(len(ylist))*self.dy] for ylist in self.ylists]
        self.xmlists = [[xr[0]+(i+1/2)*self.dx for i in range(len(xlist))] for xlist in self.xlists]
            # x displacement list. this is the midpoint of each section of the plot
        self.ymlists = [[yr[0]+(i+1/2)*self.dy for i in range(len(ylist))] for ylist in self.ylists] # y displacement list

        # if split, make a row of 3 plots. If not split, make one plot
        ncol = len(self.bases)
        self.ncol = ncol
        self.imsize = imsize
        ars = [len(self.ylists[i])/len(self.xlists[i]) for i in range(ncol)]
        fig, axs = plt.subplots(nrows=1, ncols=ncol, figsize=(imsize,imsize*max(ars)), sharey=True)
        fig.subplots_adjust(wspace=0)
        if ncol==1:
            axs = [axs]
        
        # vert/horizontal grid
        if gridlines:
            for ax in axs:
                ax.grid(linestyle='-', linewidth='0.25', color='#949494')

        # set position of titley
        if ncol==1:
            self.titley = 1
        else:
            self.titley = 0.8
        # store variables
        self.axs = axs
        self.fig = fig 
        
    def adjustBounds(self, indices:List[int], xr:List[float], legdy:float):
        '''adjust the bounds of the plot.
        indices is a list of indices which were included in the plot
        xr is the [min, max] position of each segment, e.g. [-0.7, 0.7]
        legdy is the block height for the legend'''
        if len(indices)>0:
            pos1 = min(indices)
            pos2 = max(indices)
            dx = (xr[1]-xr[0])
            if legdy>0:
                x0 = pos1-legdy/2
            else:
                x0 = xr[0]+pos1*dx
            xrtot = [x0, xr[0]+pos2*dx+dx]
        else:
            xrtot = xr
        return xrtot

        
    def clean(self):
        '''post-processes the plot to add components after all plots are added '''
        
        # adjust the bounds of the plot to only include plotted data
        # each time we added a folder to the plot, we added the 
        # x and y values of the centers to xlistreal and ylistreal
        # this is particularly useful if a big section of the plot 
        # is unplottable. e.g., very low support viscosity/ink viscosity
        # values produce filaments which curl up on the nozzle, so they don't produce cross-sections.
        # This step cuts out the space we set out for those folders that didn't end up in the final plot
        # if we were given adjustBounds=False during initialization, don't adjust the bounds
        if self.ab:
#             self.xrtots = self.adjustBounds(self.xlistsreal, self.xr, self.xlists)
#             self.yrtots = self.adjustBounds(self.ylistsreal, self.yr, self.ylists)
            self.xrtot = self.adjustBounds(self.indicesreal.x, self.xr, 0)
            self.yrtot = self.adjustBounds(self.indicesreal.y, self.yr, self.legdy)
        else:
            self.xrtot[1] = self.xrtot[1]-self.dx
            self.yrtot[1] = self.yrtot[1]-self.dy
            self.yrtot[0] = self.yrtot[0]/2+self.indicesreal.y.min()*self.dy
        

        # put x labels on all plots
        for i, ax in enumerate(self.axs):
            ax.set_xlabel(self.xlabels[i], fontname="Arial", fontsize=self.fontsize)
            
            if i==0 or (not self.ylabels[i]==self.ylabels[i-1]):
                # if first plot or ylabel is different:
                ax.set_ylabel(self.ylabels[i], fontname="Arial", fontsize=self.fontsize)
                ax.yaxis.set_major_locator(mticker.FixedLocator(self.ymlists[i]))
                ax.set_yticklabels(self.ylists[i], fontname="Arial", fontsize=self.fontsize)

            # the way comboPlots is set up, it has one big plot, 
            # and each folder is plotted in a different section of the plot
            # because the viscosity inputs are on a log scale, 
            # it is more convenient to make these plots in some real space
            # ,e.g. if we're plotting cross-sections, make the whole
            # plot go from 0-8 mm, and give the sections centers at 1, 2, 3... mm
            # then go back in later and relabel 1,2,3 to the actual viscosities, 10, 100, 1000... Pa s
            # this is the relabeling step
            ax.set_xticks(self.xmlists[i], minor=False)
            ax.set_yticks(self.ymlists[i], minor=False)

            ax.xaxis.set_major_locator(mticker.FixedLocator(self.xmlists[i]))
            ax.set_xticklabels(self.xlists[i], fontname="Arial", fontsize=self.fontsize)  
            if len(self.xrtots[i])==2:
                ax.set_xlim(self.xrtots[i]) # set the limits to the whole bounds
            if len(self.yrtots[i])==2:
                ax.set_ylim(self.yrtots[i])

            # make each section of the plot square
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(self.bases[i], fontname="Arial", fontsize=self.fontsize)
        
            if self.ab:
                # reset the figure size so the title is in the right place
                if len(self.xlistsreal[0])>0 and len(self.ylistsreal[0])>0:
                    width = self.imsize
                    height = width*(self.yrtot[1]-self.yrtot[0])/(self.xrtot[1]-self.xrtot[0])
                    self.fig.set_size_inches(width,h=height, forward=True)
                    
       
        self.titley = 1
        self.fig.suptitle(self.figtitle, y=self.titley, fontname="Arial", fontsize=self.fontsize)
 
        
        return
    
    def vvplot(self, pv:printVals) -> Tuple[float, float, float]:
        '''find the position of the file in the plot. x0, y0 is the position in the plot, in plot coordinates'''
        axnum = pv.ax
        x = pv.xval
        y = pv.yval
        xpos = findPos(self.xlists[axnum], x)
        ypos = findPos(self.ylists[axnum], y)
        x0 = self.xmlists[axnum][xpos]
        y0 = self.ymlists[axnum][ypos]
        if x not in self.xlistsreal[axnum]:
            self.xlistsreal[axnum].append(x)
        if y not in self.ylistsreal[axnum]:
            self.ylistsreal[axnum].append(y)
        if [x,y] not in self.xylistsreal[axnum]:
            self.xylistsreal[axnum].append([x,y])
        else:
            raise ValueError('Square already filled')
        self.indicesreal = self.indicesreal.append({'x':int(xpos), 'y':int(ypos)}, ignore_index=True)
        return x0, y0, axnum
    
            
    def removeFrame(self, ax) -> None:
        '''remove the outside frame from the axis'''
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
    def removeFrames(self) -> None:
        for ax in self.axs:
            self.removeFrame(ax)




def findPos(l:List, v:Any) -> Any:
    '''find the position of v in list l. l is a list. v is a value in the list.
    used by vv'''
    try:
        p = l.index(v)
    except ValueError:
        return -1
    return p
    


#-----------------------------------------------

class multiPlots:
    '''given a sample type folder, plot values'''
    
    def __init__(self, folder:str, exportFolder:str, dates:list, **kwargs):
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
            if os.path.isdir(os.path.join(self.folder, subfolder)):
                self.addToLists(subfolder)
        if len(self.inkList)==0:
            self.addToLists(os.path.basename(self.folder))
            if len(self.inkList)==0:
                self.addToLists(os.path.basename(os.path.dirname(self.folder)))
                if len(self.inkList)==0:
                    if 'P_vs' in self.folder:
                        if '8_3.50' in self.folder:
                            self.inkList.append('PDMSS8-S85-0.05')
                            self.supList.append('3.50')
        
        self.inkList = self.sortList(self.inkList)         
        # determine how many variables must be defined for a 2d plot
        self.freevars = 1
        self.freevarList = ['spacing']
        for s in ['ink', 'sup', 'inkv', 'supv']:
            l = getattr(self, f'{s}List')
            if len(l)>1:
                self.freevars+=1
                self.freevarList.append(s) 
                
    def addToLists(self, subfolder:str) -> None:
        '''split the elements of the subfolder basename and use it to lay out the list of inks and supports'''
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
                
    def sortList(self, l:list) -> list:
        '''sort the list by values'''
        fvl = pd.DataFrame([fluidVals(s, 'ink').metarow()[0] for s in l])
        fvl.rename(columns={'val':'rheWt'}, inplace=True)
        fvl['rheWt'] = pd.to_numeric(fvl['rheWt'])
        fvl['surfactantWt'] = pd.to_numeric(fvl.surfactantWt)
        fvl.sort_values(by=['rheWt', 'surfactantWt'], inplace=True)
        fvl.reset_index(inplace=True)
        return list(fvl.shortname)

    def spacingPlots(self, name:str, showFig:bool=False, export:bool=True):
        '''run all plots for object name (e.g. HOB, HIPxs)'''
        for spacing in self.spacingList:
            self.plot(spacing=spacing, showFig=showFig, export=export)
