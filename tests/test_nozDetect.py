#!/usr/bin/env python
'''Script for testing nozzle detection in videos. Tests specific files with known nozzle location'''

# external packages
import os, sys
import traceback
import logging
import pandas as pd
from typing import List, Dict, Tuple, Union, Any, TextIO
import re
import numpy as np
import unittest
__unittest = True

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
from py.tools.config import cfg
import py.tools.logs as logs
import py.vid_tools as vt
import py.vid_noz_detect as nt
from py.tools.imshow import imshow

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
LOGGERDEFINED = logs.openLog('test_nozDetect.py', False, level='DEBUG', exportLog=True) # export logs to file


#----------------------------------------------


class TestNozDetect(unittest.TestCase):
    '''tests that nozzles are correctly detected'''
    
    def parameterize(self, margin:int=20, **kwargs):
        self.folder = kwargs['folder']
        self.xL = kwargs['xL']
        self.xR = kwargs['xR']
        self.yB = kwargs['yB']
        self.margin = margin
    
    def setUp(self):
        self.vd = nt.nozData(self.folder)
        self.vd.detectNozzle(diag=1, suppressSuccess=True, export=False, overwrite=True)
    
    def test_xL(self):
        errmess = f'text_xL failed on {self.folder}, expected-actual = {self.xL-self.vd.xL}'
        self.assertTrue(self.xL-self.margin <= self.vd.xL <= self.xL+self.margin, errmess)
        
    def test_xR(self):
        errmess = f'text_xR failed on {self.folder}, expected-actual = {self.xR-self.vd.xR}'
        self.assertTrue(self.xR-self.margin <= self.vd.xR <= self.xR+self.margin, errmess)
        
    def test_yB(self):
        errmess = f'text_yB failed on {self.folder}, expected-actual = {self.yB-self.vd.yB}'
        self.assertTrue(self.yB-self.margin <= self.vd.yB <= self.yB+self.margin, errmess)
        
    def runTest(self):
        self.test_xL()
        self.test_xR()
        self.test_yB()
        
    def tearDown(self):
#         logging.info(f'{self.folder} passed nozzle tests')
        if hasattr(self, '_outcome'):  # Python 3.4+
            result = self.defaultTestResult()  # These two methods have no side effects
            self._feedErrorsToResult(result, self._outcome.errors)
        else:  # Python 3.2 - 3.3 or 3.0 - 3.1 and 2.7
            result = getattr(self, '_outcomeForDoCleanups', self._resultForDoCleanups)
        if len(result.failures)>0:
            self.vd.drawDiagnostics(2)
        pass

def suite():
    suite = unittest.TestSuite()
    cdir = os.path.dirname(os.path.realpath(__file__))
    testcsv = os.path.join(cdir,'test_nozDetect.csv')
    testlist = pd.read_csv(testcsv, dtype={'s':'str', 'xL':'float', 'xR':'float', 'yB':'float'})
    for i,row in testlist.iterrows():
        row['folder'] = os.path.join(cfg.path.server, row['s'])
        s = dict(row)
        t = TestNozDetect()
        t.parameterize(**s)
        suite.addTest(t)
    return suite
    
    
if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
    