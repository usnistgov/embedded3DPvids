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
from fileHandling import *
from py.tools.config import cfg
import tools.logs as logs

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
LOGGERDEFINED = logs.openLog('test_fileHandling.py', False, level='DEBUG', exportLog=True) # export logs to file


#----------------------------------------------


class TestDateAndTime(unittest.TestCase):
    '''test for correct date and time detection'''
    
    def parameterize(self, file:str, date:int, time:int, time_v:str, **kwargs):
        self.file = file
        self.date_s = str(date)
        self.date_i = toInt(date)
        self.time_s = str(time)
        self.time_i = toInt(time)
        self.t_v = time_v
    
    def setUp(self):
        self.dts = fileDateAndTime(self.file)
        self.dti = fileDateAndTime(self.file, out='int')
        self.ftv = fileTimeV(self.file)
    
    def test_date_s(self):
        errmess = f'test_date_s failed on {self.file}, expected = {self.date_s}, found = {self.dts[0]}'
        self.assertEqual(self.date_s, self.dts[0], errmess)
        
    def test_date_i(self):
        errmess = f'test_date_i failed on {self.file}, expected = {self.date_i}, found = {self.dti[0]}'
        self.assertEqual(self.date_i, self.dti[0], errmess)
        
    def test_time_s(self):
        errmess = f'test_time_s failed on {self.file}, expected = {self.time_s}, found = {self.dts[1]}'
        self.assertEqual(self.time_s, self.dts[1], errmess)
        
    def test_time_i(self):
        errmess = f'test_date_i failed on {self.file}, expected = {self.time_i}, found = {self.dti[1]}'
        self.assertEqual(self.time_i, self.dti[1], errmess)
        
    def test_time_v(self):
        errmess = f'test_time_v failed on {self.file}, expected = {self.t_v}, found = {self.ftv}'
        self.assertEqual(self.t_v, self.ftv, errmess)

    def runTest(self):
        self.test_date_s()
        self.test_date_i()
        self.test_time_s()
        self.test_time_i()
        self.test_time_v()
        
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
    testcsv = os.path.join(cdir,'test_fileDateAndTime.csv')
    testlist = pd.read_csv(testcsv, dtype={'file':'str', 'date':'str', 'time':'str', 'time_v':'str'})
    testlist.fillna('', inplace=True)
    for i,row in testlist.iterrows():
        row['file'] = os.path.join(cfg.path.server, row['file'])
        s = dict(row)
        t = TestDateAndTime()
        t.parameterize(**s)
        suite.addTest(t)
    return suite
    
    
if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())