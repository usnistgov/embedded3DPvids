#!/usr/bin/env python
'''Functions for displaying images in jupyter notebooks'''

# external packages
from matplotlib import pyplot as plt
import cv2 as cv
from typing import List, Dict, Tuple, Union, Any, TextIO
import logging
import numpy as np

# local packages

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)


#----------------------------------------------

####### DISPLAY TOOLS

def imshow(*args, scale:float=8, axesVisible:bool=True, numbers:bool=False, perRow:int=6, maxwidth:float=20, **kwargs) -> None:
    '''displays cv image(s) in jupyter notebook using matplotlib'''
    aspect = args[0].shape[0]/args[0].shape[1]
    rows = int(np.ceil(len(args)/perRow))
    cols = min(len(args), perRow)
    
    if aspect>1:
        w = scale*cols/aspect
        h = scale*rows
    else:
        w = scale*cols
        h = scale*aspect*rows
    if w>maxwidth:
        w = maxwidth
        h = w*aspect*cols/rows
    f, axarr = plt.subplots(rows, cols, figsize=(w,h))
    f.subplots_adjust(wspace=0)
    if len(args)>1:
        if rows==1:
            axs = [axarr]
            axlist = axarr
        else:
            axs = axarr
            axlist = axarr.flatten()
        for axrow in axs:
            for ax in axrow:
                ax.get_xaxis().set_visible(axesVisible)
                ax.get_yaxis().set_visible(axesVisible)
                ax.set_aspect(aspect)
                ax.spines['bottom'].set_color('white')
                ax.spines['top'].set_color('white') 
                ax.spines['right'].set_color('white')
                ax.spines['left'].set_color('white')
        for i, im in enumerate(args):
            if type(im) is str:
                axlist[i].text(0.1,0.1,im, family='Monospace', linespacing=2)
            else:
                if len(im.shape)>2:
                    # color
                    axlist[i].imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
                else:
                    # B&W
                    axlist[i].imshow(im, cmap='Greys')
                if numbers:
                    axlist[i].text(0,0,str(i))
            
    else:
        ax = axarr
        ax.get_xaxis().set_visible(axesVisible)
        ax.get_yaxis().set_visible(axesVisible)
        im = args[0]
        if len(im.shape)>2:
            ax.imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
        else:
            ax.imshow(im, cmap='Greys')
    if 'title' in kwargs:
        # put title on figure
        if len(args)>1:
            axlist[0].set_title(kwargs['title'])
        else:
            ax.set_title(kwargs['title'])
    if 'titles' in kwargs:
        for i,t in enumerate(kwargs['titles']):
            axlist[i].set_title(t)
    f.tight_layout()