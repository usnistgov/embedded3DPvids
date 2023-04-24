#!/usr/bin/env python
'''functions for getting colors and markers for a plot'''

# external packages
import os, sys
import traceback
import logging
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.markers import MarkerStyle
import matplotlib.cm as cm
import matplotlib.colors as mc
import colorsys
import seaborn as sns
from typing import List, Dict, Tuple, Union, Any, TextIO

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
sys.path.append(os.path.dirname(os.path.dirname(currentdir)))

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

def cubehelix1(val:float):
    '''val should be 0-1. returns a color'''
    cm = sns.cubehelix_palette(as_cmap=True, rot=-0.4)
    return cm(val)

def adjust_lightness(color, amount=0.5):
    '''https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib'''
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def adjust_saturation(color, amount=0.5):
    '''https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib'''
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], c[1], max(0, min(1, amount * c[1])))

class colorModes:
    '''gradColor 0 means color by discrete values of zvar, gradColor 1 means means to use a gradient color scheme by values of zvar, gradColor 2 means all one color, one type of marker. gradColor 0 with 'color' in kwargs means make it all one color, but change markers.'''
    discreteZvar = 0
    gradientZvar = 1
    constant = 2
