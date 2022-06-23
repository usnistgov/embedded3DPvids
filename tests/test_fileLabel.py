#!/usr/bin/env python
'''Script for testing file sorting and labeling'''

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

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
LOGGERDEFINED = logs.openLog('test_fileHandling.py', False, level='DEBUG', exportLog=True) # export logs to file


#----------------------------------------------


class TestFileLabel(unittest.TestCase):
    '''test for correct folder level labeling'''
    
    def parameterize(self, file:str, level:str, **kwargs):
        self.file = file
        self.level = level
    
    def setUp(self):
        self.levels = labelLevels(self.file)
        self.currentLevel = self.levels.currentLevel
    
    def test_level(self):
        errmess = f'test_level failed on {self.file}, expected = {self.level}, found = {self.currentLevel}'
        self.assertEqual(self.level, self.currentLevel, errmess)

    def runTest(self):
        self.test_level()
        
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
    testcsv = os.path.join(cdir,'test_fileLabel.csv')
    testlist = pd.read_csv(testcsv, dtype={'file':'str', 'level':'str'})
    for i,row in testlist.iterrows():
        row['file'] = os.path.join(cfg.path.server, row['file'])
        s = dict(row)
        t = TestFileLabel()
        t.parameterize(**s)
        suite.addTest(t)
    return suite
    
    
if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())