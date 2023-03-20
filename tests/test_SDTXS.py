#!/usr/bin/env python
'''Script for testing measurements of SDTed cross-sections'''

# external packages
import os, sys
import traceback
import logging
from typing import List, Dict, Tuple, Union, Any, TextIO
import re
import numpy as np
import unittest
import pandas as pd
__unittest = True

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
parentdir = os.path.dirname(currentdir)
sys.path.append(os.path.join(parentdir, 'py'))
from file_handling import *
import tools.logs as logs
import metrics.m_SDT as me

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
LOGGERDEFINED = logs.openLog('test_SDTXS.py', False, level='DEBUG', exportLog=True) # export logs to file


#----------------------------------------------


class TestSDTXS(unittest.TestCase):
    '''test for correct print file labeling'''
    
    def parameterize(self, folder:str='', file:str='', w:float=0, h:float=0, xc:float=0, yc:float=0, test:int=0, **kwargs):
        self.folder = folder
        self.file = file
        self.w = w
        self.h = h
        self.xc = xc
        self.yc = yc
        self.test = test

    def setUp(self):
        self.me, self.units = me.xsSDTMeasure(self.file)
        
    def test_generic(self, s:str):
        val = getattr(self, s)
        self.assertTrue(s in self.me, f'Nothing measured in {self.file}: {self.test}')
        if abs(val)>0.5:
            self.assertTrue(abs(self.me[s]-val)/val<0.02, f'test_{s} failed in {self.file}: {self.test}')
        else:
            self.assertTrue(abs(self.me[s]-val)<0.02, f'test_{s} failed in {self.file}: {self.test}')
        
    def test_w(self):
        self.test_generic('w')
        
    def test_h(self):
        self.test_generic('h')
            
    def test_xc(self):
        self.test_generic('xc')
            
    def test_yc(self):
        self.test_generic('xc')

    def runTest(self):
        self.test_w()
        self.test_h()
        self.test_xc()
        self.test_yc()
        
    def tearDown(self):
        if hasattr(self, '_outcome'):  # Python 3.4+
            result = self.defaultTestResult()  # These two methods have no side effects
            self._feedErrorsToResult(result, self._outcome.errors)
        else:  # Python 3.2 - 3.3 or 3.0 - 3.1 and 2.7
            result = getattr(self, '_outcomeForDoCleanups', self._resultForDoCleanups)
        pass

def suite():
    suite = unittest.TestSuite()
    cdir = os.path.dirname(os.path.realpath(__file__))
    testcsv = os.path.join(cdir,'test_SDTXS.csv')
    testlist = pd.read_csv(testcsv, dtype={'folder':'str', 'file':'str', 'w':'float', 'h':'float', 'xc':'float', 'yc':'float'})
    testlist.fillna('', inplace=True)
    for i,row in testlist.iterrows():
        row['folder'] = os.path.join(cfg.path.server, row['folder'])
        row['file'] = os.path.join(row['folder'], row['file'])
        row['test'] = i
        s = dict(row)
        t = TestSDTXS()
        t.parameterize(**s)
        suite.addTest(t)
    return suite
    
    
if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    result = runner.run(suite())
    failedFiles = [int(re.split(': ', str(s))[-1][:-4]) for s in result.failures]  # indices of failed files
    print(failedFiles)
    sys.exit(failedFiles)