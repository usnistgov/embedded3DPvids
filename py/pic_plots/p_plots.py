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
import file.file_handling as fh
from val.v_print import *
from p_folderImages import folderImages
from p_comboPlot import comboPlot, multiPlots

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

class picPlots:
    '''this creates a figure that is a grid of pictures'''
    
    def __init__(self, topFolder:str, exportFolder:str, allIn:List[str], dates:List[str], tag:str, overwrite:bool=False, showFig:bool=True, imsize:float=6.5, export:bool=True, **kwargs):
        self.concat = 'h'
        self.topFolder = topFolder
        self.bn = os.path.basename(self.topFolder)
        self.exportFolder = exportFolder
        self.allIn = allIn
        self.dates = dates
        self.tag = tag
        self.overwrite = overwrite
        self.showFig = showFig
        self.imsize = imsize
        self.kwargs = kwargs
        self.export = export
            
    def imFn(self) -> str:
        '''Construct an image file name with no extension. Exportfolder is the folder to export to. Label is any given label. Topfolder is the folder this image refers to, e.g. singlelines. Insert any extra values in kwargs as keywords'''
        if type(self.tag) is list:
            label = "_".join(self.tag)
        else:
            label = self.tag
        s = ''
        
        for k,val in self.kwargs.items():
            if not k in ['adjustBounds', 'overlay', 'overwrite', 'removeBorders', 'whiteBalance', 'normalize', 'crops', 'export', 'removeBackground', 'concat'] and type(val) is not dict:
                s = f'{s}{k}_{val}_'
        s = s[0:-1]
        s = s.replace('*', 'x')
        s = s.replace('/', 'div')
        s = s.replace(' ', '-')
        if len(label)>0:
            out = f'{label}_{self.bn}_{s}'
        else:
            out = f'{self.bn}_{s}'
        self.fn = os.path.join(self.exportFolder, self.bn, out)
        return self.fn
    
    def tag2List(self, tag:str) -> dict:
        '''convert the tag to a dictionary describing the image'''
        out = {}
        if not tag[0]=='l':
            return {'tag':tag}
        out['line'] = int(tag[1])
        if 'o' in tag:
            spl = re.split('o', tag[2:])
            ln = spl[0]
            out['observation'] = int(spl[1][-1])
        else:
            ln = tag[2:]
        if ln[0]=='d':
            out['extrude'] = 'disturb'
        else:
            out['extrude'] = 'write'
        return out
        
    
    def tags2Title(self) -> None:
        '''convert the tag to a human readable title'''
        s = ''
        if type(self.tag) is list:
            tag = self.tag
        else:
            tag = [self.tag]
        df = pd.DataFrame([self.tag2List(t) for t in tag])
        if len(df.line.unique())>1:
            s = f'Lines {tuple(df.line.unique())}'
        else:
            l = df.loc[0,'line']
            s = f'Line {l}'
        if len(df.extrude.unique())>1:
            s = f'{s}, ('
            for s1 in df.extrude:
                s = f'{s}{s1}, '
            s = f'{s[:-2]})'
        else:
            l = df.loc[0, 'extrude']
            s = f'{s}, {l}'
        for si in self.allIn:
            s = f'{s}, {si}'
        return s
            

    def picPlots(self) -> None:
        '''plot all pictures for simulations in a folder'''
        for pv in self.cp.pvlists:
            fi = folderImages(pv, self.tag, **self.kwargs)
            fi.getImages()
            fi.picPlot(self.cp, self.dx, self.dy)
        self.cp.figtitle = self.tags2Title()
        self.cp.clean()
        
    def getFolders(self) -> None:
        self.flist = fh.printFolders(self.topFolder, tags=self.allIn, someIn=self.dates, **self.kwargs)
        self.flist.reverse()
        
    def getDims(self) -> None:
        '''get the spacing between images'''
        fi = folderImages(printVals(self.flist[0]), self.tag, **self.kwargs)
        widthI, heightI, _, _ = fi.wfull()
        if widthI<10:
            self.dx = 0.5
            self.dy = 0.5
        else:
            if heightI>widthI:
                self.dy = 0.5
                self.dx = self.dy*widthI/heightI
            else:
                self.dx = 0.5
                self.dy = self.dx*heightI/widthI
                
                
    def exportIm(self) -> None:
        '''export an image. fn is a full path name, without the extension. fig is a matplotlib figure'''
        if not self.export:
            return
        if not os.path.exists(self.exportFolder):
            raise ValueError(f'Export folder does not exist: {self.exportFolder}')
        
        dires = [os.path.dirname(self.fn)]
        while not os.path.exists(dires[-1]):
            dires.append(os.path.dirname(dires[-1]))
        for dire in dires[:-1]:
            os.mkdir(dire)
        for s in ['.svg', '.png']:
            self.cp.fig.savefig(f'{self.fn}{s}', bbox_inches='tight', dpi=300, transparent=True)
        logging.info(f'Exported {self.fn}')
        
            
    def picPlots0(self):
        '''plot all pictures for simulations in a folder, but use automatic settings for cropping and spacing and export the result
    topFolder is the folder that holds the simulations
    exportFolder is the folder to export the images to
    tag is the name of the image type, e.g. xs1. Used to find images.
    other kwargs can be used to style the plot'''

        if not os.path.isdir(self.topFolder):
            logging.error(f'{self.topFolder} is not a directory')
            return
        if len(self.dates)==0:
            self.dates = ['']
        self.imFn()
        if self.export and not self.overwrite and os.path.exists(f'{self.fn}.png'):
            return
        
        self.getFolders()
        if len(self.flist)==0:
            logging.debug(f'No folders to plot: some {self.someIn} all {self.allIn} dates {self.dates}')
            return

        self.getDims()
        self.cp = comboPlot(self.flist, [-self.dx, self.dx], [-self.dy, self.dy], self.imsize, gridlines=False, **self.kwargs)
        self.picPlots()
        self.exportIm()

        if not self.showFig:
            plt.close()

        return self.cp.fig
