#!/usr/bin/env python
'''Functions for setting dimensions of figures'''

# external packages
from typing import List, Dict, Tuple, Union, Any, TextIO

# local packages


# logging

#-------------------------------------------------------------


class sizes:
    '''for setting sizes of figures, fonts, and markers'''
    
    def __init__(self, rows:int, cols:int, plotType:str='notebook'):
        self.rows = rows
        self.cols = cols
        self.plotType = plotType
        if self.plotType=='ppt':
            self.fs = 18
            if self.cols==1:
                maxwidth=4
            else:
                maxwidth = 14
            self.getFigSize(maxwidth, 8)
            self.markersize=100
            self.linewidth = 2
        elif self.plotType=='paper':
            self.fs = 8
            if self.cols<=2:
                # maxwidth = 3.25
                maxwidth=5
            else:
                maxwidth = 6.5
            self.getFigSize(maxwidth, 8.5)
            self.markersize=20
            self.linewidth = 0.75
        elif self.plotType=='paperhalf':
            self.fs = 8
            self.getFigSize(3.25, 3.25)
            self.markersize = 20
            self.linewidth = 0.75
        elif self.plotType=='notebook':
            self.fs = 10
            self.getFigSize(10, 10)
            self.markersize = 40
            self.linewidth = 1
        else:
            raise ValueError(f'Unknown plot type {self.plotType}')
            
    def values(self) -> Tuple[int, tuple, int, int]:
        '''return all sizes'''
        return self.fs, self.figsize, self.markersize, self.linewidth
            
            
    def getFigSize(self, wmax:float, hmax:float) -> None:
        '''get figure size, given maximum dimensions'''
        self.ar = self.rows/self.cols*1.1
        wider = [wmax, wmax*self.ar]
        if wider[1]>hmax:
            wider = [w*hmax/wider[1] for w in wider]
        self.figsize = tuple(wider)
        