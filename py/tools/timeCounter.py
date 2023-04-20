#!/usr/bin/env python
'''Functions for collecting data from stills of single lines, for a single image'''

# external packages
import time

# local packages

# logging
#----------------------------------------------

def timeCounter(t0:float, s:str) -> float:
    tt = time.time()
    if len(s)>0:
        print(f'{s} {(tt-t0):0.4f} seconds')
    return tt

class timeObject:
    '''this gives functions to any subclass that let us track how long functions take'''
    
    def __init__(self):
        return
    
    def initializeTimeCounter(self, name:str):
        self.timeCount = time.time()
        self.timeName = name
        
    def timeCounter(self, s:str):
        if not hasattr(self, 'timeCount'):
            self.initializeTimeCounter('')
        tt = time.time()
        print(f'{self.timeName} {(tt-self.timeCount):0.4f} seconds {s} ')
        self.timeCount = tt