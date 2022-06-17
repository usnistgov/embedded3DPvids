#!/usr/bin/env python
'''Functions for displaying images in jupyter notebooks'''

# external packages
from matplotlib import pyplot as plt
import cv2 as cv
from typing import List, Dict, Tuple, Union, Any, TextIO
import logging

# local packages

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)


#----------------------------------------------

####### DISPLAY TOOLS

def imshow(*args, scale:float=8, axesVisible:bool=True, **kwargs) -> None:
    '''displays cv image(s) in jupyter notebook using matplotlib'''
    aspect = args[0].shape[0]/args[0].shape[1]
    if aspect>1:
        f, axs = plt.subplots(1, len(args), figsize=(scale*len(args)/aspect, scale))
    else:
        f, axs = plt.subplots(1, len(args), figsize=(scale*len(args), scale*aspect))
    if len(args)>1:
        
        
        for ax in axs:
            ax.get_xaxis().set_visible(axesVisible)
            ax.get_yaxis().set_visible(axesVisible)
        for i, im in enumerate(args):
            if len(im.shape)>2:
                # color
                axs[i].imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
            else:
                # B&W
                axs[i].imshow(im, cmap='Greys')
    else:
        ax = axs
        ax.get_xaxis().set_visible(axesVisible)
        ax.get_yaxis().set_visible(axesVisible)
        im = args[0]
        if len(im.shape)>2:
            ax.imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
        else:
            ax.imshow(im, cmap='Greys')
    if 'title' in kwargs:
        # put title on figure
        f.suptitle(kwargs['title'])
    f.tight_layout()