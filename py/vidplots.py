#!/usr/bin/env python
'''Functions for plotting video data. Adapted from https://github.com/usnistgov/openfoamEmbedded3DP'''

# external packages
import os, sys
import traceback
import logging
import pandas as pd
from matplotlib import pyplot as plt
from typing import List, Dict, Tuple, Union, Any, TextIO
import re
import numpy as np
import cv2 as cv

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
import fileHandling as fh
import stitchBas as sb

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)

# info
__author__ = "Leanne Friedrich"
__copyright__ = "This data is publicly available according to the NIST statements of copyright, fair use and licensing; see https://www.nist.gov/director/copyright-fair-use-and-licensing-statements-srd-data-and-software"
__credits__ = ["Leanne Friedrich"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Leanne Friedrich"
__email__ = "Leanne.Friedrich@nist.gov"
__status__ = "Development"

#----------------------------------------------

class fluidVals:
    '''class that holds info about fluid'''
    
    def __init__(self, fluid:str):
        '''convert the shorthand sample name to info about the fluid'''
        self.shortname = fluid
        if fluid[0]=='M':
            self.var = 'w% silica'
            self.val = fluid[1:]
            if fluid[-1]=='S':
                self.base = 'mineral oil + Span 20'
                self.val = self.val[:-1]
            else:
                self.base = 'mineral oil'
        elif fluid[:4]=='PDMS':
            self.var = 'w% silica'
            self.val = fluid[4:]
            if fluid[-1]=='S':
                self.base = 'mineral oil + PDMS + Span 20'
                self.val = self.val[:-1]
            else:
                self.base = 'mineral oil + PDMS'
        elif fluid[:3]=='PEG':
            self.var = 'w% silica'
            self.val = fluid[3:]
            self.base = 'water + 40% PEGDA'
        else:
            self.var = 'w% Laponite RD'
            self.val = fluid
            if fluid[-1]=='T':
                self.base = 'water + Tween 80'
                self.val = self.val[:-1]
            else:
                self.base = 'water'
        try:
            self.val = float(self.val)
        except:
            logging.warning(f'Failed to convert fluid value to float: {fluid}, {self.val}')

#------   

class printVals:
    '''class that holds info about experiment'''
    
    def __init__(self, folder:str):
        '''get the ink and support names from the folder name'''
        self.folder = folder
        self.bn = os.path.basename(folder)
        split = re.split('_', self.bn)
        ink = split[1]
        sup = split[3]
        self.ink = fluidVals(ink)
        self.sup = fluidVals(sup)
        
    def base(self, xfluid:str, yfluid:str) -> str:
        '''get the plot title'''
        self.xval = getattr(self, xfluid).val
        self.yval = getattr(self, yfluid).val
        xbase = getattr(self, xfluid).base
        ybase = getattr(self, yfluid).base
        base = xbase + ', '+ybase
        return base
    
#------
    


#------

class folderPlots:
    '''A generic class used for plotting many folders at once. Subclasses are comboPlot, which puts everything on one plot, and gridOfPlots, which puts everythign in separate plots based on viscosity.'''
    
    def __init__(self, folders:List[str], imsize:float, **kwargs):
        '''topFolder is the folder we're plotting
            imsize is the size of the total image in inches
            split is true to split into separate plots by surface tension'''
        self.kwargs = kwargs
        self.ab = not 'adjustBounds' in self.kwargs or self.kwargs['adjustBounds']==True
        self.flist = folders
        self.imsize = imsize
        self.plotsLists(**kwargs) 
        
    def plotsLists(self, **kwargs):
        '''plotsLists initializes gridOfPlots and comboPlots objects, creating the initial figure'''
        self.pvlists = [printVals(f) for f in self.flist]
#         if not self.checkVals(**kwargs):
#             raise ValueError(f'Inconsistent variables: {[f.bn for f in self.pvlists]}')
        
        self.xfluid ='ink'
        self.yfluid = 'sup'
        self.xvar = self.xfluid+'.var'
        self.yvar = self.yfluid+'.var'
        self.xfunc = self.xfluid+'.val'
        self.yfunc = self.yfluid+'.val'
        
        self.getBases()
        for s in ['x', 'y']:
            self.getLabels(s)
        
        try:
            # find lists of unique x values and y values
            self.xlists = self.unqList('x')
            self.ylists = self.unqList('y')
            self.xlistsreal = [[]]*len(self.bases)
            self.ylistsreal = [[]]*len(self.bases)
        except:
            raise ValueError('Failed to identify x and y variables')
            traceback.print_exc()
        return self
    
    def getBases(self) -> None:
        '''get the list of materials bases to determine how many plots to make'''
        self.bases = []
        for pv in self.pvlists:
            base = pv.base(self.xfluid, self.yfluid)
            if not base in self.bases:
                self.bases.append(base)
        return True
    
    def unqList(self, var:str) -> List:
        '''get a list of unique values for each plot. var should be x or y'''
        func = getattr(self, var+'func')
        pvlist = self.pvlists
        split = re.split('\.', func)
        fluid = split[0]
        val = split[1]
        unqs = [[]]*len(self.bases)
        for pv in pvlist:
            axnum = pv.ax
            pvval = getattr(getattr(pv, fluid),val)
            if not pvval in unqs[axnum]:
                unqs[axnum].append(pvval)
        for i in range(len(self.bases)):
            unqs[i].sort()
        return unqs
    
    def getLabels(self, var:str) -> List[str]:
        '''get the axis labels and store axis numbers for each folder'''
        labels = ['']*len(self.bases)
        if var=='x':
            fluid = self.xfluid
        else:
            fluid = self.yfluid
        for pv in self.pvlists:
            base = pv.base(self.xfluid, self.yfluid)
            for i,b in enumerate(self.bases): # fill the list of labels corresponding to list of bases
                if b==base:
                    pv.ax = i # store the axis number for this folder
                    if len(labels[i])==0: # fill this label if not already filled
                        labels[i] = getattr(pv, fluid).var
        setattr(self, var+'labels', labels)
        
    
    def checkVals(self, xbase:bool=True, xvar:bool=True, ybase:bool=True, yvar:bool=True, **kwargs) -> bool:
        '''check that the samples all have the same materials space'''
        
        if xbase:
            xbase0 = getattr(self.pvlists[0], self.xfluid).base
        if xvar:
            xvar0=getattr(self.pvlists[0], self.xfluid).var
        if ybase:
            ybase0=getattr(self.pvlists[0], self.xfluid).base
        if yvar:
            yvar0=getattr(self.pvlists[0], self.xfluid).var
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
            self.xrtots = self.adjustBounds(self.xlistsreal, self.xr, self.xlists)
            self.yrtots = self.adjustBounds(self.ylistsreal, self.yr, self.ylists)
        

        # put x labels on all plots
        for i, ax in enumerate(self.axs):
            ax.set_xlabel(self.xlabels[i], fontname="Arial", fontsize=10)
            if i==0 or (not self.ylabels[i]==self.ylabels[i-1]):
                # if first plot or ylabel is different:
                ax.set_ylabel(self.ylabels[i], fontname="Arial", fontsize=10)
                ax.set_yticklabels(self.ylists[i], fontname="Arial", fontsize=10)

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
            
            ax.set_xticklabels(self.xlists[i], fontname="Arial", fontsize=10)  
            if len(self.xrtots[i])==2:
                ax.set_xlim(self.xrtots[i]) # set the limits to the whole bounds
            if len(self.yrtots[i])==2:
                ax.set_ylim(self.yrtots[i])

            # make each section of the plot square
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(self.bases[i], fontname="Arial", fontsize=10)
        
            # reset the figure size so the title is in the right place
            if self.ab and len(self.xlistsreal[i])>0 and len(self.ylistsreal[i])>0:
                width = self.imsize
                height = width*len(self.ylistsreal[i])/(len(self.xlistsreal[i])*len(self.axs))
                self.fig.set_size_inches(width, height)
       
        self.fig.suptitle(self.figtitle, y=self.titley, fontname="Arial", fontsize=10)
        
#         self.fig.tight_layout()
        
        return
    
    def adjustBounds(self, xlistsreal:List[List[float]], xr:List[float], xlists:List[float]) -> List[List[float]]:
        '''adjust the bounds of the plot.
        xlistreal is a list of x points to be included in the plot
        xr is the [min, max] position of each segment, e.g. [-0.7, 0.7]
        xlist is the initial list of x points we included in the plot'''
        xrtot = [[]]*len(self.bases)
        for i in range(len(self.bases)):
            if len(xlistsreal[i])>1:
                xmin = min(xlistsreal[i])
                xmax = max(xlistsreal[i])
                pos1 = xlists[i].index(min(xlistsreal[i]))
                pos2 = xlists[i].index(max(xlistsreal[i]))+1
                dx = xr[1]-xr[0]
                xrtot[i] = [xr[0]+pos1*dx, xr[0]+pos2*dx]
            else:
                xrtot[i] = [0]
        return xrtot

def findPos(l:List, v:Any) -> Any:
    '''find the position of v in list l. l is a list. v is a value in the list.
    used by vv'''
    try:
        p = l.index(v)
    except ValueError:
        return -1
    return p
    
def vvplot(pv:printVals, cp:comboPlot) -> Tuple[float, float, float]:
    '''find the position of the file in the plot'''
    axnum = pv.ax
    x = pv.xval
    y = pv.yval
    xpos = findPos(cp.xlists[axnum], x)
    ypos = findPos(cp.ylists[axnum], y)
    x0 = cp.xmlists[axnum][xpos]
    y0 = cp.ymlists[axnum][ypos]
    if x not in cp.xlistsreal[axnum]:
        cp.xlistsreal[axnum].append(x)
    if y not in cp.ylistsreal[axnum]:
        cp.ylistsreal[axnum].append(y)
    return x0, y0, axnum
    
#-----------------------------------------------

def imFn(exportfolder:str, topfolder:str, label:str, **kwargs) -> str:
    '''Construct an image file name with no extension. Exportfolder is the folder to export to. Label is any given label. Topfolder is the folder this image refers to, e.g. singlelines. Insert any extra values in kwargs as keywords'''
    bn = os.path.basename(topfolder)
    s = ''
    for k in kwargs:
        if not k in ['adjustBounds'] and type(kwargs[k]) is not dict:
            s = s + k + '_'+str(kwargs[k])+'_'
    s = s[0:-1]
    s = s.replace('*', 'x')
    s = s.replace('/', 'div')
    s = s.replace(' ', '-')
    return os.path.join(exportfolder, bn, label+'_'+bn+'_'+s)

def exportIm(fn:str, fig) -> None:
    '''export an image. fn is a full path name, without the extension. fig is a matplotlib figure'''
    dires = [os.path.dirname(fn)]
    while not os.path.exists(dires[-1]):
        dires.append(os.path.dirname(dires[-1]))
    for dire in dires[:-1]:
        os.mkdir(dire)
    for s in ['.svg', '.png']:
        fig.savefig(fn+s, bbox_inches='tight', dpi=300)
    print('Exported ', fn)

def picFromFolder(folder:str, tag:str) -> np.array:
    '''gets one picture from a folder
    returns the picture
    tag is the name of the image type, e.g. 'xs1'. Used to find images. '''
    imfile = ''
    for f in os.listdir(folder):
        if tag in f:
            imfile = os.path.join(folder, f)
    if not os.path.exists(imfile):
        # no file in the main folder. search archives
        raw = os.path.join(folder, 'raw')
        if os.path.exists(raw):
            archive = os.path.join(raw, tag)
            if not os.path.exists(archive):
                archive = os.path.join(raw, tag[:-1]) # remove last char, e.g. for xs1 just use xs
            if os.path.exists(archive):
                l = os.listdir(archive)
                if len(l)>0:
                    imfile = os.path.join(archive, l[0]) # if there is a file in the archive folder, use it
    if os.path.exists(imfile):
        im = cv.imread(imfile)
        return im
    else:
        return []

def cropImage(im:np.array, crops:dict) -> np.array:
    '''crop the image and return width, height, image. Crops must contain x0, xf, y0, yf. x0 and xf are from left. y0 and yf are from bottom.'''
    height,width = im.shape[0:2]
    if 'x0' in crops:
        x0 = crops['x0']
    else:
        x0 = 0
    if x0>width:
        logging.warning('Cropx is larger than image')
        return
    if 'y0' in crops:
        yf = height-crops['y0']
    else:
        yf = height
    if yf<0:
        logging.warning('Cropy is larger than image')
        return
    if 'xf' in crops:
        if crops['xf']<0:
            xf = width+crops['xf']
        else:
            xf = min(width, crops['xf'])
    if 'yf' in crops:
        y0 = max(0,height-crops['yf'])
    else:
        y0 = 0
    im = im[y0:yf, x0:xf]
    return im

def importAndCrop(folder:str, tag:str, **kwargs) -> np.array:
    '''import and crop an image from a folder, given a tag. crops can be in kwargs'''
    im = picFromFolder(folder, tag)
    if type(im) is list:
#         logging.debug(f'Image missing: {folder}')
        return []
    if 'crops' in kwargs:
        crops = kwargs['crops']
        im = cropImage(im, crops)
    return im

def picPlot(pv:printVals, cp:comboPlot, dx0:float, tag:str, **kwargs) -> None:
    '''plots picture from just one folder. 
    folder is the full path name
    cp is the comboPlot object that stores the plot
    dx0 is the spacing between images in plot space, e.g. 0.7
    tag is the name of the image type, e.g. 'y_umag'. Used to find images. '''
    im = []
    try:
        if not type(tag) is list:
            tag = [tag]
        for t in tag:
            im1 = importAndCrop(pv.folder, t, **kwargs)
            if len(im1)>0:
                try:
                    im = cv.hconcat([im, im1]) # add to the right
                except Exception as e:
                    im = im1 # set imtot to first image
    except Exception as e:
        logging.error(f'Cropping error: {str(e)}')
        return
    if len(im)==0:
        return
    
    height,width = im.shape[0:2]
    if 'crops' in kwargs:
        crops = kwargs['crops']
        if 'yf' in crops and 'y0' in crops and 'xf' in crops and 'x0' in crops:
            if crops['yf']<0:
                heightI = height
            else:
                heightI = crops['yf']-crops['y0']
            if crops['xf']<0:
                widthI = width
            else:
                widthI = (crops['xf']-crops['x0'])*len(tag)
            # use intended height/width to scale all pictures the same
    else:
        widthI = width
        heightI = height
    
    # fix the scaling
    pxperunit = max(widthI, heightI) # image pixels per image block
    dx = dx0*(width/pxperunit)
    dy = dx0*(height/pxperunit)
        
    try:
        x0, y0, axnum = vvplot(pv, cp)
    except Exception as e:
        logging.error(f'Positioning error: {str(e)}')
        return
    s = 0.95 # scale images to leave white space
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    cp.axs[axnum].imshow(im, extent=[x0-dx*s, x0+dx*s, y0-dy*s, y0+dy*s])


def picPlots(cp:comboPlot, dx:float, tag:str, **kwargs) -> None:
    '''plot all pictures for simulations in a folder
    folderList is a list of paths
    cp holds the plot
    dx is the spacing between images in plot space, e.g. 0.7
    cropx is the size to crop from the left and right edges of the picture in px
    cropy is the size to crop from top and bottom in px
    tag is the name of the image type, e.g. 'y_umag'. Used to find images. '''
    for pv in cp.pvlists:
        picPlot(pv, cp, dx, tag, **kwargs)
    cp.figtitle = tag
    cp.clean()


def picPlots0(topFolder:str, exportFolder:str, dates:List[str], tag:str, overwrite:bool=False, **kwargs) -> None:
    '''plot all pictures for simulations in a folder, but use automatic settings for cropping and spacing and export the result
    topFolder is the folder that holds the simulations
    exportFolder is the folder to export the images to
    tag is the name of the image type, e.g. xs1. Used to find images.
    other kwargs can be used to style the plot
    '''

    if not os.path.isdir(topFolder):
        return
    if type(tag) is list:
        taglabel = "".join(tag)
    else:
        taglabel = tag
    fn = imFn(exportFolder, topFolder, taglabel, **kwargs)
    if not overwrite and os.path.exists(fn+'.png'):
        return
    
    flist = fh.subFolders(topFolder, tags=dates, **kwargs)
    if len(flist)==0:
        return
    
    dx = 0.7
    cp = comboPlot(flist, [-dx, dx], [-dx, dx], 6.5, gridlines=False, **kwargs)
    picPlots(cp, dx, tag, **kwargs)
    
    if not ('export' in kwargs and not kwargs['export']):
        exportIm(fn, cp.fig)