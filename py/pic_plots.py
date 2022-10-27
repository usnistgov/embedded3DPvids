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
import file_handling as fh
from val_print import *
import im_morph as vm

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
    
    def __init__(self, folders:List[str], imsize:float,**kwargs):
        '''topFolder is the folder we're plotting
            imsize is the size of the total image in inches
            split is true to split into separate plots by surface tension'''
        self.kwargs = kwargs
        self.ab = not 'adjustBounds' in self.kwargs or self.kwargs['adjustBounds']==True
        self.flist = folders
        self.imsize = imsize
        self.plotsLists(**kwargs) 
        
    def plotsLists(self, xvar:str='ink.var', yvar:str='sup.var', **kwargs):
        '''plotsLists initializes variable names for gridOfPlots and comboPlots objects. 
        vname=val for fluid composition data. vname=v for speed data'''
        self.pvlists = [printVals(f) for f in self.flist]
#         if not self.checkVals(**kwargs):
#             raise ValueError(f'Inconsistent variables: {[f.bn for f in self.pvlists]}')
        
        self.xvar = xvar
        self.yvar = yvar
        self.xfunc = xvar
        self.yfunc = yvar
#         else:

#             self.vname = vname
#             self.xfluid ='ink'
#             self.yfluid = 'sup'
#             if vname=='val':
#                 self.xvar = f'{self.xfluid}.var'
#                 self.yvar = f'{self.yfluid}.var'
#             elif vname=='v':
#                 self.xvar = f'{self.xfluid}.v'
#                 self.yvar = f'{self.yfluid}.v'
#             self.xfunc = f'{self.xfluid}.{vname}'
#             self.yfunc = f'{self.yfluid}.{vname}'
        
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
            self.xrtot = adjustBounds(self.indicesreal.x, self.xr, 0)
            self.yrtot = adjustBounds(self.indicesreal.y, self.yr, self.legdy)
        else:
            self.xrtot[1] = self.xrtot[1]-self.dx
            self.yrtot[1] = self.yrtot[1]-self.dy
            self.yrtot[0] = self.yrtot[0]/2+self.indicesreal.y.min()*self.dy
        

        # put x labels on all plots
        for i, ax in enumerate(self.axs):
            ax.set_xlabel(self.xlabels[i], fontname="Arial", fontsize=10)
            
            if i==0 or (not self.ylabels[i]==self.ylabels[i-1]):
                # if first plot or ylabel is different:
                ax.set_ylabel(self.ylabels[i], fontname="Arial", fontsize=10)
                ax.yaxis.set_major_locator(mticker.FixedLocator(self.ymlists[i]))
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

            ax.xaxis.set_major_locator(mticker.FixedLocator(self.xmlists[i]))
            ax.set_xticklabels(self.xlists[i], fontname="Arial", fontsize=10)  
            if len(self.xrtots[i])==2:
                ax.set_xlim(self.xrtots[i]) # set the limits to the whole bounds
            if len(self.yrtots[i])==2:
                ax.set_ylim(self.yrtots[i])

            # make each section of the plot square
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(self.bases[i], fontname="Arial", fontsize=10)
        
#             # reset the figure size so the title is in the right place
#             if self.ab and len(self.xlistsreal[i])>0 and len(self.ylistsreal[i])>0:
#                 width = self.imsize
#                 height = width*len(self.ylistsreal[i])/(len(self.xlistsreal[i])*len(self.axs))
#                 self.fig.set_size_inches(width, height)
            if self.ab:
                # reset the figure size so the title is in the right place
                if len(self.xlistsreal[0])>0 and len(self.ylistsreal[0])>0:
                    width = self.imsize
                    height = width*(self.yrtot[1]-self.yrtot[0])/(self.xrtot[1]-self.xrtot[0])
                    self.fig.set_size_inches(width,h=height, forward=True)
       
        self.titley = 1
        self.fig.suptitle(self.figtitle, y=self.titley, fontname="Arial", fontsize=10)
        
#         self.fig.tight_layout()
        
        return
    
#     def adjustBounds(self, xlistsreal:List[List[float]], xr:List[float], xlists:List[float]) -> List[List[float]]:
#         '''adjust the bounds of the plot.
#         xlistreal is a list of x points to be included in the plot
#         xr is the [min, max] position of each segment, e.g. [-0.7, 0.7]
#         xlist is the initial list of x points we included in the plot'''
#         xrtot = [[]]*len(self.bases)
#         for i in range(len(self.bases)):
#             if len(xlistsreal[i])>1:
#                 xmin = min(xlistsreal[i])
#                 xmax = max(xlistsreal[i])
#                 pos1 = xlists[i].index(min(xlistsreal[i]))
#                 pos2 = xlists[i].index(max(xlistsreal[i]))+1
#                 dx = xr[1]-xr[0]
#                 xrtot[i] = [xr[0]+pos1*dx, xr[0]+pos2*dx]
#             else:
#                 xrtot[i] = [0]
#         return xrtot

def adjustBounds(indices:List[int], xr:List[float], legdy:float):
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

def findPos(l:List, v:Any) -> Any:
    '''find the position of v in list l. l is a list. v is a value in the list.
    used by vv'''
    try:
        p = l.index(v)
    except ValueError:
        return -1
    return p
    
def vvplot(pv:printVals, cp:comboPlot) -> Tuple[float, float, float]:
    '''find the position of the file in the plot. x0, y0 is the position in the plot, in plot coordinates'''
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
    if [x,y] not in cp.xylistsreal[axnum]:
        cp.xylistsreal[axnum].append([x,y])
    else:
        raise ValueError('Square already filled')
    cp.indicesreal = cp.indicesreal.append({'x':int(xpos), 'y':int(ypos)}, ignore_index=True)
    return x0, y0, axnum
    
#-----------------------------------------------

def imFn(exportfolder:str, topfolder:str, label:str, **kwargs) -> str:
    '''Construct an image file name with no extension. Exportfolder is the folder to export to. Label is any given label. Topfolder is the folder this image refers to, e.g. singlelines. Insert any extra values in kwargs as keywords'''
    bn = os.path.basename(topfolder)
    s = ''
    for k in kwargs:
        if not k in ['adjustBounds', 'overlay', 'overwrite', 'removeBorders', 'whiteBalance', 'normalize', 'crops', 'export'] and type(kwargs[k]) is not dict:
            s = f'{s}{k}_{kwargs[k]}_'
    s = s[0:-1]
    s = s.replace('*', 'x')
    s = s.replace('/', 'div')
    s = s.replace(' ', '-')
    return os.path.join(exportfolder, bn, f'{label}_{bn}_{s}')

def exportIm(fn:str, fig) -> None:
    '''export an image. fn is a full path name, without the extension. fig is a matplotlib figure'''
    dires = [os.path.dirname(fn)]
    while not os.path.exists(dires[-1]):
        dires.append(os.path.dirname(dires[-1]))
    for dire in dires[:-1]:
        os.mkdir(dire)
    for s in ['.svg', '.png']:
        fig.savefig(fn+s, bbox_inches='tight', dpi=300, transparent=True)
    logging.info(f'Exported {fn}')
    
def parseTag(tag:str) -> dict:
    '''parse the tag into relevant names'''
    out = {'raw':False, 'tag':'', 'num':0}
    if 'vid' in tag:
        out['tag']=tag
        return out
    if '_' in tag:
        spl = re.split('_',tag)
        if 'raw' in spl:
            out['raw'] = True
            spl.remove('raw')
        for s in spl:
            try:
                num = int(s)
            except:
                out['tag']=s
            else:
                out['num']=num
    else:
        out['tag'] = tag
    return out
    
def picFileFromFolder(folder:str, tag:str) -> str:
    '''get the file name of the image'''
    imfile = ''
    if not 'raw' in tag:
        for f in os.listdir(folder):
            if tag in f and not ('vid' not in tag and 'vid' in f):
                # use last image in folder that has the tag
                imfile = os.path.join(folder, f)
    if not os.path.exists(imfile):
        # no file in the main folder. search archives
        raw = os.path.join(folder, 'raw')
        if os.path.exists(raw):
            out = parseTag(tag)
            archive = os.path.join(raw, out['tag'])
            if not os.path.exists(archive):
                archive = os.path.join(raw, tag[:-1]) # remove last char, e.g. for xs1 just use xs
            if os.path.exists(archive):
                l = os.listdir(archive)
                if len(l)>0:
                    imfile = os.path.join(archive, l[out['num']]) # if there is a file in the archive folder, use it
    return imfile

def picFromFolder(folder:str, tag:str) -> np.array:
    '''gets one picture from a folder
    returns the picture
    tag is the name of the image type, e.g. 'xs1'. Used to find images. '''
    imfile = picFileFromFolder(folder, tag)
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
    im = im[int(y0):int(yf), int(x0):int(xf)]
    return im

def importAndCrop(folder:str, tag:str, whiteBalance:bool=True, normalize:bool=True, removeBorders:bool=False, **kwargs) -> np.array:
    '''import and crop an image from a folder, given a tag. crops can be in kwargs
    whiteBalance=True to do auto white balancing
    normalize=True to do auto brightness adjustment
    removeBorders to remove the dark borders from the edges of the image, e.g. above the bath or the walls of the bath'''
    im = picFromFolder(folder, tag)
    if type(im) is list:
#         logging.debug(f'Image missing: {folder}')
        if 'crops' in kwargs:
            crops = kwargs['crops']
            if 'yf' in crops and 'y0' in crops and 'xf' in crops and 'x0' in crops:
                return 255*np.ones((int(crops['yf']-crops['y0']), int(crops['xf']-crops['x0']), 3), dtype=np.uint8)  # no image. return an empty image
            else:
                return []
        else:
            return []
    if 'crops' in kwargs:
        crops = kwargs['crops']
        im = cropImage(im, crops)
    if removeBorders:
        im = vm.removeBorders(im, normalizeIm=False)
    if whiteBalance:
        im = vm.white_balance(im)
    if normalize:
        im = vm.normalize(im)
    return im

def getImages(pv:printVals, tag:str, concat:str='h', **kwargs) -> np.array:
    '''get the images, crop them, and put them together into one image'''
    im = []
    try:
        if not type(tag) is list:
            tag = [tag]
        for i,t in enumerate(tag):
            if 'crops' in kwargs and type(kwargs['crops']) is list:
                k2 = kwargs.copy()
                if i-1>len(kwargs['crops']):
                    raise Exception('Image list is longer than crop list')
                k2['crops'] = kwargs['crops'][i]
            else:
                k2 = kwargs
            im1 = importAndCrop(pv.printFolder, t, **k2)
            if len(im1)>0:
                if len(im)==0:
                    im = im1
                else:
                    # pad the images to make them the same shape
                    
                    if concat=='h':
                        if im1.shape[0]>im.shape[0]:
                            pad = im1.shape[0]-im.shape[0]
                            im = cv.copyMakeBorder(im, pad, 0, 0,0, cv.BORDER_CONSTANT, value=(255,255,255)) 
                        elif im.shape[0]>im1.shape[0]:
                            pad = im.shape[0]-im1.shape[0]
                            im1 = cv.copyMakeBorder(im1, pad, 0, 0,0, cv.BORDER_CONSTANT, value=(255,255,255))
                        im = cv.hconcat([im, im1]) # add to the right
                    elif concat=='v':
                        if im1.shape[1]>im.shape[1]:
                            pad = im1.shape[1]-im.shape[1]
                            im = cv.copyMakeBorder(im, 0, pad, 0,0, cv.BORDER_CONSTANT, value=(255,255,255)) 
                        elif im.shape[1]>im1.shape[1]:
                            pad = im.shape[1]-im1.shape[1]
                            im1 = cv.copyMakeBorder(im1, 0, pad, 0,0, cv.BORDER_CONSTANT, value=(255,255,255))
                        im = cv.vconcat([im1, im])
                    else:
                        raise ValueError(f'Bad value for concat: {concat}')
    except Exception as e:
        logging.error(f'Cropping error: {str(e)}')
        traceback.print_exc()
        raise e
    if len(im)==0:
        raise e
    return im, t, tag


def wfull(im:np.array, tag:List[str], concat:str='h', **kwargs) -> Tuple[float,float]:
    '''get the width and height of the combined image'''
    height,width = im.shape[0:2]
    if 'crops' in kwargs:
        
        crops = kwargs['crops']
        if type(crops) is list:
            # multiple crop zones indicated for different images
            c = pd.DataFrame(crops)
            if not 'x0' in c:
                c['x0'] = 0
            if not 'xf' in c:
                c['xf'] = width
            if not 'y0' in c:
                c['y0'] = 0
            if not 'yf' in c:
                c['yf'] = 0
            c['w'] = c['xf']-c['x0']
            c['h'] = c['yf']-c['y0']
            widthI = c.w.sum()
            heightI = c.h.max()
        else:
            # single crop zone indicated
            if concat=='h':
                hs = 1
                if type(tag) is list:
                    ws = len(tag)
                else:
                    ws = 1
            elif concat=='v':
                ws = 1
                if type(tag) is list:
                    hs = len(tag)
                else:
                    hs = 1
            if 'yf' in crops and 'y0' in crops and 'xf' in crops and 'x0' in crops:
                if crops['yf']<0:
                    heightI = height*hs
                else:
                    heightI = (crops['yf']-crops['y0'])*hs
                if crops['xf']<0:
                    widthI = width*ws
                else:
                    widthI = (crops['xf']-crops['x0'])*ws

                # use intended height/width to scale all pictures the same
    else:
        widthI = width
        heightI = height
        
    return widthI, heightI, width,height
    
def getWidthScaling(im:np.array, dx0:float, dy0:float, tag:List[str], concat:str='h', **kwargs) -> Tuple[float,float,float]:
    '''determine how to scale the image. dx0 is the space between images on the plot, in plot coordinates. tag is the line type, e.g. xs or horiz'''
    
    widthI, heightI, width,height = wfull(im, tag, concat=concat, **kwargs)
    
    # fix the scaling
    pxperunit = max(widthI, heightI) # image pixels per image block
    if widthI>heightI:
        dx = dx0*(width/pxperunit)
        dy = dx*height/width
    else:
        dy = dy0*height/pxperunit
        dx = dy*width/height
        
    return dx, dy, pxperunit

def picPlotOverlay(pv:printVals, pxperunit:float, t:dict, dx0:float, dy0:float, x0:float, y0:float, s:float, ax, concat:str='h', **kwargs):
    '''draw an annotation, e.g. a circular or rectangular scale bar, over the image
    pxperunit is the image pixels per image block
    t is a dictionary holding info about the line, including, e.g. {'tag':'xs1'}
    dx0 is the spacing between images in plot coords
    x0 is the x position of the annotation
    y0 is the y position of the annotation
    s is the scaling to use to add white space between images
    ax is the axis to plot this annotation on
    '''
    overlay = kwargs['overlay'] # get overlay dictionary from kwargs
    file = picFileFromFolder(pv.printFolder, parseTag(t)['tag'])
    scale = float(fh.fileScale(file))
    pxPerBlock = pxperunit/s # rescale px to include white space
    realPxPerBlock = pxPerBlock/scale # rescale to include shrinkage of original image
    mmPerBlock = realPxPerBlock/pv.geo.pxpmm # mm per block: pixels per image block, scaled by s is displayed size
    mmPerPlotUnit = mmPerBlock/(2*max(dx0, dy0)) # convert to dimensions of the plot
    if not overlay['shape'] in ['rectangle', 'circle', '3circles', '2circles']:
        return

    circlewMM = pv.dEst   # estimated filament diameter
    circlewPlotUnits = circlewMM/mmPerPlotUnit # diameter in plot units
    if 'dx' in overlay:
        x0 = x0+overlay['dx']*dx0
    if 'dy' in overlay:
        y0 = y0+overlay['dy']*dy0
    
    if 'color' in overlay:
        color = overlay['color']
    else:
        color = 'black'
        
    if overlay['shape']=='circle':
        # plot circle
        circle2 = plt.Circle((x0, y0), circlewPlotUnits/2, color=color, fill=False)
        ax.add_patch(circle2)
    elif overlay['shape']=='3circles' or overlay['shape']=='2circles':
        # plot circle
        if hasattr(pv,'spacing'):
            spacing = pv.spacing
            num = int(overlay['shape'][0])
        else:
            spacing = 0
            num = 1
        for i in range(num):
            if concat=='h':
                x = x0+i*circlewPlotUnits*spacing
                y = y0
            else:
                y = y0+i*circlewPlotUnits*spacing
                x = x0
            circle2 = plt.Circle((x, y), circlewPlotUnits/2, color=color, fill=False)
            ax.add_patch(circle2)
    elif overlay['shape']=='rectangle' and ('w' in overlay or 'h' in overlay):
        # plot rectangle
        if 'w' in overlay:
            w = overlay['w']/mmPerPlotUnit
            h = circlewPlotUnits
        else:
            w = circlewPlotUnits
            h = overlay['h']/mmPerPlotUnit
        y0 = y0-h/2
        x0 = x0-w/2
        rect = plt.Rectangle((x0, y0), w, h, color=color, fill=True, edgecolor=None)
        ax.add_patch(rect)
    
def picPlot(pv:printVals, cp:comboPlot, dx0:float, dy0:float, tag:str, **kwargs) -> None:
    '''plots picture from just one folder. 
    folder is the full path name
    cp is the comboPpositioninglot object that stores the plot
    dx0 is the spacing between images in plot space, e.g. 0.7
    tag is the name of the image type, e.g. 'y_umag'. Used to find images. '''
    
    # determine where to plot the image
    try:
        x0, y0, axnum = vvplot(pv, cp)
    except ValueError:
        return
    except Exception as e:
        if not 'already filled' in str(e):
            logging.error(f'Positioning error: {str(e)}')
        traceback.print_exc()
        return
    
    # get the images
    try:
        im, t, tag = getImages(pv, tag, **kwargs)
    except:
        return
    
    # get scaling
    dx, dy, pxperunit = getWidthScaling(im, dx0, dy0, tag, **kwargs)
    
    s = 0.95 # scale images to leave white space
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    
    # plot the image
    cp.axs[axnum].imshow(im, extent=[x0-dx*s, x0+dx*s, y0-dy*s, y0+dy*s])
    
    if 'overlay' in kwargs:
        picPlotOverlay(pv, pxperunit, t, dx0, dy0, x0, y0, s, cp.axs[axnum], **kwargs)
        
        
def picPlots(cp:comboPlot, dx:float, dy:float, tag:str, **kwargs) -> None:
    '''plot all pictures for simulations in a folder
    folderList is a list of paths
    cp holds the plot
    dx is the spacing between images in plot space, e.g. 0.7
    tag is the name of the image type, e.g. 'y_umag'. Used to find images. '''
    for pv in cp.pvlists:
        picPlot(pv, cp, dx, dy, tag, **kwargs)
    cp.figtitle = tag
    cp.clean()


def picPlots0(topFolder:str, exportFolder:str, allIn:List[str], dates:List[str], tag:str, overwrite:bool=False, showFig:bool=True, **kwargs):
    '''plot all pictures for simulations in a folder, but use automatic settings for cropping and spacing and export the result
    topFolder is the folder that holds the simulations
    exportFolder is the folder to export the images to
    tag is the name of the image type, e.g. xs1. Used to find images.
    other kwargs can be used to style the plot
    '''

    if not os.path.isdir(topFolder):
        logging.error(f'{topFolder} is not a directory')
        return
    if type(tag) is list:
        taglabel = "".join(tag)
    else:
        taglabel = tag
    fn = imFn(exportFolder, topFolder, taglabel, dates=dates[0], **kwargs)
    if not overwrite and os.path.exists(f'{fn}.png'):
        return

    flist = fh.printFolders(topFolder, tags=allIn, someIn=dates, **kwargs)
    flist.reverse()
    
    if len(flist)==0:
        logging.debug(f'No folders to plot: {dates}')
        return
    

    widthI, heightI, _, _ = wfull(np.array([[0]]), tag,  **kwargs)
    if widthI<10:
        dx = 0.5
        dy = 0.5
    else:
        if heightI>widthI:
            dy = 0.5
            dx = dy*widthI/heightI
        else:
            dx = 0.5
            dy = dx*heightI/widthI
    
#     dx = 0.7
    cp = comboPlot(flist, [-dx, dx], [-dy, dy], 6.5, gridlines=False, **kwargs)
    picPlots(cp, dx, dy, tag, **kwargs)
    
    if not ('export' in kwargs and not kwargs['export']) and os.path.exists(exportFolder):
        exportIm(fn, cp.fig)
        
    if not showFig:
        plt.close()
        
    return cp.fig
