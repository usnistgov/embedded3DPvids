#!/usr/bin/env python
'''class that holds ideal values for measurements'''

# external packages
import os, sys
import traceback
import logging
from typing import List, Dict, Tuple, Union, Any, TextIO
import re
import numpy as np

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


#----------------------------------------------

class ideals:
    '''holds ideal values for measurements'''
    
    def __init__(self):
        return
    
    def yideal(self, yvar):
        if yvar.startswith('delta') or ('d' in yvar and 'dt' in yvar):
            # all changes should be 0
            return 0
        
        for s in ['yBot', 'xLeft', 'xshift', 'yshift', 'yTop', 'xRight'
                  , 'xc', 'yc', 'emptiness', 'roughness', 'x0', 'dxprint'
                  , 'xf', 'dx0', 'dxf', 'space_a', 'space_at', 'ldiff', 'stdevT', 'minmaxT'
                 , 'dy0l', 'dyfl', 'dy0lr', 'dyflr', 'space_l', 'space_b', 'y0', 'yf']:
            if s in yvar:
                # all positions are measured relative to the ideal position and should be 0
                return 0
            
        for s in ['segments', 'area', 'w', 'h', 'aspectI', 'meanT']:
            if s in yvar:
                # dimensions are measured relative to the ideal position and should be 1
                return 1
            
        for s in ['spacing', 'spacing_adj']:
            if s in yvar:
                return np.sqrt(np.pi)/2
            
        raise AttributeError(f'Ideal value not found for {yvar}')


class XSSDTIdeals(ideals):
    '''holds ideal values for XS SDT measurements'''
    
    def __init__(self, dire:str):
        self.dire = dire
        super().__init__()
    
    def yideal(self, yvar:str):
        try:
            return super().yideal(yvar)
        except AttributeError:
            if 'aspect' in yvar:
                if 'w1' in yvar or 'd1' in yvar:
                    return 1
                if 'w2' in yvar or 'd2' in yvar:
                    # this should really depend on spacing, but let's say we are using the wrong spacing but expecting an ideal aspect ratio
                    if self.dire=='+y':
                        return 1/2
                    else:
                        return 2
                if 'w3' in yvar or 'd3' in yvar:
                    if self.dire=='+y':
                        return 1/3
                    else:
                        return 3
            
        raise AttributeError(f'Ideal value not found for {yvar}')
        