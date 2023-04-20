#!/usr/bin/env python
'''Functions for plotting still and video data'''

# external packages
import os, sys
import traceback
import logging
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.markers import MarkerStyle
import matplotlib.cm as cm
import matplotlib.colors as mc
import colorsys
from typing import List, Dict, Tuple, Union, Any, TextIO
import re
import numpy as np
import seaborn as sns
import string
from scipy import stats
import csv

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
from m_plots_metric import *
from m_plots_color import *
from m_plots_contour import *
from m_plots_mesh import *
from m_plots_scatter import *
from m_stats_table import *
from m_plots_multi import *
from m_plots_multi_SDT import *

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