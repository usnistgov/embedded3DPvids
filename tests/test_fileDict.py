#!/usr/bin/env python
'''Script for testing print file labeling'''

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


class TestFileDict(unittest.TestCase):
    '''test for correct print file labeling'''
    
    def parameterize(self, folder:str, still:str, meta:str, time:str, vid:str, summary:str, stitch:str, printType:str, **kwargs):
        self.folder = folder
        self.still = self.fullFile(still)
        self.meta = self.fullFile(meta)
        self.time = self.fullFile(time)
        self.vid = self.fullFile(vid)
        self.summary = self.fullFile(summary)
        self.stitch = self.fullFile(stitch)
        self.printType = printType
            
    def fullFile(self, s:str) -> str:
        '''full file name of file'''
        if len(s)>0:
            return os.path.join(self.folder, s)
        else:
            return s
    
    def setUp(self):
        self.pfd = printFileDict(self.folder)
        
    def errmess(self, s:str) -> str:
        '''error message'''
        return f'test_{s} failed on {self.folder}, expected = {getattr(self, s)}, found = {self.pfd.first(s)}'
    
    def assertEq(self, s:str):
        self.assertEqual(getattr(self, s), self.pfd.first(s), self.errmess(s))
    
    def test_still(self):
        self.assertEq('still')
        
    def test_meta(self):
        errMess = f'test_{s} failed on {self.folder}, expected = {self.meta}, found = {self.pfd.metaFile()}'
        self.assertEqual(self.meta, self.pfd.metaFile(), errMess)
        
    def test_time(self):
        errMess = f'test_{s} failed on {self.folder}, expected = {self.time}, found = {self.pfd.timeFile()}'
        self.assertEqual(self.time, self.pfd.timeFile(), errMess)
        
    def test_vid(self):
        errMess = f'test_{s} failed on {self.folder}, expected = {self.vid}, found = {self.pfd.vidFile()}'
        self.assertEqual(self.vid, self.pfd.vidFile(), errMess)
        
    def test_summary(self):
        self.assertEq('summary')
        
    def test_stitch(self):
        self.assertEq('stitch')
        
    def test_printType(self):
        errmess = f'test_printType failed on {self.folder}, expected = {self.printType}, found = {self.pfd.printType}'
        self.assertEqual(self.printType, self.pfd.printType, errmess)

    def runTest(self):
        self.test_still()
        self.test_meta()
        self.test_time()
        self.test_vid()
        self.test_summary()
        self.test_stitch()
        self.test_printType()
        
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
    testcsv = os.path.join(cdir,'test_fileDict.csv')
    testlist = pd.read_csv(testcsv, dtype=str)
    testlist.fillna('', inplace=True)
    for i,row in testlist.iterrows():
        row['folder'] = os.path.join(cfg.path.server, row['folder'])
        s = dict(row)
        t = TestFileDict()
        t.parameterize(**s)
        suite.addTest(t)
    return suite
    
    
if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())