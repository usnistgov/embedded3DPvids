#!/usr/bin/env python
'''Script for testing geometry metadata labeling. Tests specific files with known pxpmm'''

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
import py.tools.logs as logs
from py.tools.config import cfg
from py.val_geometry import *


# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
LOGGERDEFINED = logs.openLog('test_nozDetect.py', False, level='DEBUG', exportLog=True) # export logs to file


#----------------------------------------------


class TestGeometryVals(unittest.TestCase):

    
    def parameterize(self, folder:str='', pxpmm:int=0, **kwargs):
        self.folder = folder
        self.pxpmm = pxpmm
    
    def setUp(self):
        self.gval = geometryVals(self.folder)
    
    def test_pxpmm(self):
        errmess = f'test_pxpmm failed on {self.folder}, expected = {self.pxpmm}, actual = {self.gval.pxpmm}'
        self.assertEqual(self.pxpmm, self.gval.pxpmm, errmess)
        
    def runTest(self):
        self.test_pxpmm()
        
    def tearDown(self):
#         logging.info(f'{self.folder} passed nozzle tests')
        if hasattr(self, '_outcome'):  # Python 3.4+
            result = self.defaultTestResult()  # These two methods have no side effects
            self._feedErrorsToResult(result, self._outcome.errors)
        else:  # Python 3.2 - 3.3 or 3.0 - 3.1 and 2.7
            result = getattr(self, '_outcomeForDoCleanups', self._resultForDoCleanups)
        pass

def suite():
    suite = unittest.TestSuite()
    cdir = os.path.dirname(os.path.realpath(__file__))
    testcsv = os.path.join(cdir,'test_geometryVals.csv')
    testlist = pd.read_csv(testcsv, dtype={'folder':'str', 'pxpmm':'int'})
    for i,row in testlist.iterrows():
        row['folder'] = os.path.join(cfg.path.server, row['folder'])
        s = dict(row)
        t = TestGeometryVals()
        t.parameterize(**s)
        suite.addTest(t)
    return suite
    
    
if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
    