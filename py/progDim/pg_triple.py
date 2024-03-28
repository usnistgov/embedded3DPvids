#!/usr/bin/env python
'''Functions for handling tables of programmed timings for triple lines'''

# external packages
import os, sys
import traceback
import logging
from typing import List, Dict, Tuple, Union, Any, TextIO
import re
import pandas as pd
import numpy as np
import csv

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
from tools.config import cfg
from pg_tools import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)

#----------------------------------------------

class progDimsTripleLine(progDim):
    '''for programmed dimensions of triple line prints'''
    
    def __init__(self, printFolder:str, pv:printVals, **kwargs):
        super().__init__(printFolder, pv)
    
    def initializeProgDims(self):
        '''initialize programmed dimensions table'''

        if 'crossDoubleVert_0.5' in self.sbp:
            # wrote 2 sets of vertical lines and a zigzag
            self.progDims.name = ['v00', 'v01', 'v10', 'v11', 'zig']
        elif 'crossDoubleVert_0' in self.sbp:
            # wrote 2 sets of vertical lines with 6 crosses for each set
            self.progDims.name = ['v00', 'v01'
                                  , 'h00', 'h01', 'h02', 'h03', 'h04', 'h05'
                                  , 'v10', 'v11'
                                 , 'h10', 'h11', 'h12', 'h13', 'h14', 'h15']
        elif 'crossDoubleHoriz_0.5' in self.sbp:
            # wrote zigzag and 4 vertical lines
            self.progDims.name = ['zig', 'v0', 'v1', 'v2', 'v3']
        elif 'crossDoubleHoriz_0' in self.sbp:
            # wrote 3 zigzags with 4 cross lines each
            self.progDims.name = ['hz0', 'hc00', 'hc01', 'hc02', 'hc03'
                                 , 'hz1', 'hc10', 'hc11', 'hc12', 'hc13'
                                 , 'hz2', 'hc20', 'hc21', 'hc22', 'hc23']
        elif 'underCross_0.5' in self.sbp:
            # wrote 2 zigzags
            self.progDims.name = ['z1', 'z2']
        elif 'underCross_0' in self.sbp:
            # wrote 3 double line zigzag, then 3 sets of 4 vertical lines 
            self.progDims.name = ['zig'
                                 , 'v00', 'v01', 'v02', 'v03'
                                 , 'v10', 'v11', 'v12', 'v13'
                                 , 'v20', 'v21', 'v22', 'v23']
        elif 'tripleLines' in self.sbp:
            # wrote 4 groups with 3 lines each
            self.progDims.name = ['l00', 'l01', 'l02'
                                 ,'l10', 'l11', 'l12'
                                 ,'l20', 'l21', 'l22'
                                 ,'l30', 'l31', 'l32']
