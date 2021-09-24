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
from py.config import cfg
import py.logs as logs
import py.vidTools as vt
from py.imshow import imshow

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
LOGGERDEFINED = logs.openLog('test_nozDetect.py', False, level='DEBUG', exportLog=True) # export logs to file

# info
__author__ = "Leanne Friedrich"
__copyright__ = "This data is publicly available according to the NIST statements of copyright, fair use and licensing; see https://www.nist.gov/director/copyright-fair-use-and-licensing-statements-srd-data-and-software"
__credits__ = ["Leanne Friedrich"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Leanne Friedrich"
__email__ = "Leanne.Friedrich@nist.gov"
__status__ = "Development"

#----------------------------------------------


class TestNozDetect(unittest.TestCase):

    
    def parameterize(self, margin:int=20, **kwargs):
        self.folder = kwargs['folder']
        self.xL = kwargs['xL']
        self.xR = kwargs['xR']
        self.yB = kwargs['yB']
        self.margin = margin
    
    def setUp(self):
        self.vd = vt.vidData(self.folder)
        self.vd.detectNozzle(diag=2, suppressSuccess=True)
    
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
    testlist = [{'s':r'LapRD LapRD 1day\I_2.75_S_3.00\I_2.75_S_3.00_210727', 'xL':343, 'xR':470, 'yB':354},
               {'s':r'LapRD LapRD 3day\I_3.00_S_2.50\I_3.00_S_2.50_210517', 'xL':322, 'xR':436, 'yB':386.5}, 
                
                {'s':r'mineral812 LapRD\I_M4_S_2.50\I_M4_S_2.50_210519', 'xL':336, 'xR':463, 'yB':368.5},
                {'s':r'mineral812 LapRD\I_M6_S_2.50\I_M6_S_2.50_210921', 'xL':329, 'xR':457, 'yB':375},
                {'s':r'mineral812 LapRD\I_M6_S_3.50\I_M6_S_3.50_210519', 'xL':326, 'xR':449, 'yB':297},
                {'s':r'mineral812 LapRD\I_M9_S_2.75\I_M9_S_2.75_210921', 'xL':283, 'xR':403, 'yB':385.5},
                
                {'s':r'mineral812S LapRDT\I_M4S_S_2.25T\I_M4S_S_2.25T_210922', 'xL':326, 'xR':455, 'yB':373},
                {'s':r'mineral812S LapRDT\I_M4S_S_2.75T\I_M4S_S_2.75T_210922', 'xL':333, 'xR':458, 'yB':363.5},
                {'s':r'mineral812S LapRDT\I_M4S_S_3.00T\I_M4S_S_3.00T_210922', 'xL':327, 'xR':440, 'yB':375.5},
                {'s':r'mineral812S LapRDT\I_M5S_S_2.25T\I_M5S_S_2.25T_210518', 'xL':322, 'xR':441, 'yB':347},
                {'s':r'mineral812S LapRDT\I_M5S_S_2.25T\I_M5S_S_2.25T_210922', 'xL':345, 'xR':476, 'yB':343},
                {'s':r'mineral812S LapRDT\I_M5S_S_2.50T\I_M5S_S_2.50T_210518', 'xL':383, 'xR':501, 'yB':351},
                {'s':r'mineral812S LapRDT\I_M5S_S_2.75T\I_M5S_S_2.75T_210518', 'xL':327, 'xR':451, 'yB':348},
                {'s':r'mineral812S LapRDT\I_M6S_S_2.25T\I_M6S_S_2.25T_210518', 'xL':285, 'xR':415, 'yB':292},
                {'s':r'mineral812S LapRDT\I_M6S_S_2.25T\I_M6S_S_2.25T_210922', 'xL':298, 'xR':428, 'yB':365},
                {'s':r'mineral812S LapRDT\I_M6S_S_2.50T\I_M6S_S_2.50T_210518', 'xL':278, 'xR':399, 'yB':312},
                {'s':r'mineral812S LapRDT\I_M7S_S_2.50T\I_M7S_S_2.50T_210518', 'xL':291, 'xR':414, 'yB':371},
                {'s':r'mineral812S LapRDT\I_M7S_S_2.75T\I_M7S_S_2.75T_210922', 'xL':306, 'xR':429, 'yB':340},
                {'s':r'mineral812S LapRDT\I_M9S_S_2.75T\I_M9S_S_2.75T_210922', 'xL':324, 'xR':453, 'yB':350},
                
                {'s':r'PDMSM LapRD\I_PDMSM7.5_S_4.00\I_PDMSM7.5_S_4.00_210630', 'xL':291, 'xR':421, 'yB':346.5},
               ]
    for s in testlist:
        s['folder'] = os.path.join(cfg.path.server, 'singleLines', s['s'])
        t = TestNozDetect()
        t.parameterize(**s)
        suite.addTest(t)
    return suite
    
    
if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())