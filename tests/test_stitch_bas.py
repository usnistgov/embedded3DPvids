#!/usr/bin/env python
'''Script for testing that bas files are correctly labeled for stitching'''

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
import pic_stitch_bas as sb

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
LOGGERDEFINED = logs.openLog('test_stitch_bas.py', False, level='DEBUG', exportLog=True) # export logs to file


#----------------------------------------------


class TestStitchBasSingle(unittest.TestCase):
    '''test for correct print file labeling'''
    
    def parameterize(self, folder:str, **kwargs):
        self.folder = folder

    def setUp(self):
        self.fl = sb.stitchSortDecide(self.folder)
        
    def test_xs(self):
        xsTest = self.fl.testGroup('xs', index=0)
        self.assertEqual(xsTest, 0, 'test_xs failed')
        
    def test_horiz(self):
        horizTest = self.fl.testGroup('horiz', index=-1)
        self.assertEqual(horizTest, 0, 'test_horiz failed')
        
    def test_vert(self):
        vertTest = self.fl.testGroup('vert', index=0)
        self.assertEqual(vertTest, 0, 'test_vert failed')

    def runTest(self):
        self.test_xs()
        self.test_horiz()
        self.test_vert()
        
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
    testcsv = os.path.join(cdir,'test_stitch_bas.csv')
    testlist = pd.read_csv(testcsv, dtype=str)
    testlist.fillna('', inplace=True)
    for i,row in testlist.iterrows():
        row['folder'] = os.path.join(cfg.path.server, row['folder'])
        s = dict(row)
        t = TestStitchBasSingle()
        t.parameterize(**s)
        suite.addTest(t)
    return suite
    
    
if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
    