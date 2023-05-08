#!/usr/bin/env python
'''Morphological operations applied to images'''

# external packages
import cv2 as cv
import numpy as np 
import os
import sys
import logging
from typing import List, Dict, Tuple, Union, Any, TextIO
import pandas as pd
import matplotlib.pyplot as plt

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
from s_segmenterDF import segmenterDF
from morph import *
import im_fill as fi
from tools.timeCounter import timeObject

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)

#----------------------------------------------

class segmentCombiner(timeObject):
    '''combine segmentation from ML model and unsupervised model'''
    
    def __init__(self, imML:np.array, imU:np.array, acrit:int, largeCrit:int=1000, rErode:int=5
                 , rDilate:int=10, eraseTop:bool=True, diag:int=0, **kwargs):
        super().__init__()
        sML = segmenterDF(imML, acrit=acrit) # ML DF
        sU = segmenterDF(imU, acrit=acrit) # unsupervised DF
        sMLm = sML.commonMask(sU)         # parts from ML that are also in U
        sUm = sU.commonMask(sML)          # parts from U that are also in ML
        both = cv.bitwise_and(sMLm, sUm)  # parts that are in both
        tot = cv.add(sMLm, sUm)           # parts that are in either, but with overlapping components

        MLadd = cv.subtract(sMLm, sUm)   # parts that are in ML but not U
        if MLadd.sum().sum()>0 and eraseTop:
            e2 = segmenterDF(MLadd, acrit=0)
            e2.eraseTopBorder(margin=5, checks=False)               # remove parts from the ML image that are touching the top edge
            if hasattr(e2, 'labelsBW'):       # ML model has tendency to add reflection at top
                MLadd = e2.labelsBW
        dif = cv.subtract(sUm, sMLm)
        Uadd = dilate(erode(dif, rErode),rDilate)   # parts that are in U but not ML, opened
        Uexc = cv.bitwise_and(tot, Uadd)
        if Uexc.sum().sum()>0 and largeCrit<10000:
            e1 = segmenterDF(Uexc, acrit=0)
            e1.eraseLargeComponents(largeCrit, checks=False)
            if hasattr(e1, 'labelsBW'):
                Uexc = e1.labelsBW
        exc = cv.add(MLadd, Uexc)
        filled = cv.add(both, exc)
        filled = fi.filler(filled, fi.fillMode.fillTiny, acrit=50).filled
        segmenter = segmenterDF(filled, acrit=acrit, diag=diag)
        segmenter.imML = imML
        segmenter.imU = imU
        segmenter.exc = exc
        segmenter.dif = dif
        self.segmenter = segmenter