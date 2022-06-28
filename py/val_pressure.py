#!/usr/bin/env python
'''Functions for storing metadata about print folders'''

# external packages
import os, sys
import traceback
import logging
from typing import List, Dict, Tuple, Union, Any, TextIO
import re
import pandas as pd
import numpy as np
import csv

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
from config import cfg
from plainIm import *
import file_handling as fh

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)



#----------------------------------------------

class pressureVals:
    '''holds info about pressure calibration'''
    
    def __init__(self, folder, channel:int=0, **kwargs):
        self.printFolder = folder
        self.channel = channel   # 0-indexed
        self.bn = fh.twoBN(self.printFolder)
        self.caliba = 0
        self.calibb = 0
        self.calibc = 0
        
        # default velocity values
        self.vink = cfg.const.vink
        self.vsup = cfg.const.vsup
        
        self.targetPressure = 0
        # dictionary of files in the printFolder
        if 'pfd' in kwargs:
            self.pfd = kwargs['pfd']
        else:
            self.pfd = fh.printFileDict(self.printFolder) 
        
        # get calib a,b,c, ink speed, support speed
        out = self.importMetaFile()
        if out>0:
            out = self.readCalibFile()
            if out>0:
                logging.warning(f'No pressure calibration found for {self.printFolder}')
            else:
                self.exportSpeedFile()
        
            
    def calculateP(self, s:float) -> float:
        '''calculate the target pressure from the calibration curve and given flow speed'''
        a = self.caliba
        b = self.calibb
        c = self.calibc
        if abs(a)>0:
            d = b**2-4*a*(c-s)
            if d>0:
                p = (-b+np.sqrt(d))/(2*a)
            else:
                logging.warning(f'{self.bn}: Speed cannot be reached')
                p = 0
        elif abs(b)>0:
            p = (s-c)/b
        else:
            p = c
        return p
    
    def calculateTargetPressure(self) -> float:
        self.targetPressure = self.calculateP(self.vink)
        return self.targetPressure
        
    def calculateSpeed(self, p:float) -> float:
        '''calculate speed from pressure'''
        if (self.caliba==0 or self.calibb==0 or self.calibc==0):
            return 0
        else:
            return self.caliba*p**2+self.calibb*p+self.calibc
        
    def metarow(self) -> Tuple[dict,dict]:
        '''row holding metadata'''
        pvals = {f'pressureCh{self.channel}':self.targetPressure*100} # convert from mbar to Pa
        punits = {f'pressureCh{self.channel}':'Pa'}
        return pvals, punits
        
    #-----------------
    # ShopbotPyQt after addition of _speed_ and _meta_ files
    def importMetaFile(self) -> int:
        '''find the metadata file. returns 0 if successful'''
        if not hasattr(self.pfd, 'meta') or len(self.pfd.meta)==0:
            return 1
        file = self.pfd.meta[0]
        with open(file, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                for s in ['a', 'b', 'c']:
                    # calib params
                    if row[0] in [f'calib{s}_channel_{self.channel}', f'calib{s}']:
                        # store calibration constant
                        setattr(self, f'calib{s}', float(row[2]))
                        
                        # check units
                        if not row[1] in ['mm/s/mbar^2', 'mm/s/mbar','mm/s']:
                            logging.warning(f'Bad units in {fh.twoBN(file)}: calib{s}: {row[1]}')
                            
                # ink speed
                if row[0] in ['ink speed', f'ink_speed_channel_{self.channel}']:
                    self.vink = float(row[2])
                    if not row[1]=='mm/s':
                        logging.warning(f'Bad units in {fh.twoBN(file)}: ink speed: {row[1]}')

                # support speed
                elif row[0] in ['support speed', 'speed_move_xy']:
                    self.vsup = float(row[2])
                    if not row[1]=='mm/s':
                        logging.warning(f'Bad units in {fh.twoBN(file)}: support speed: {row[1]}')
        self.calculateTargetPressure()
        return 0
    
    #-----------------                        
    # ShopbotPyQt before addition of _speed_ and _meta_ files
    def findCalibFile(self, ink) -> str:
        '''find the pressure calibration file'''
        folder = cfg.path.pCalibFolder
        if not os.path.exists(folder):
            raise NameError(f'Pressure calibration folder does not exist {folder}')
        if self.ink.shortname[0]=='M':
            shortname = f'mineral_812_{self.ink.shortname[1]}'
        elif self.ink.shortname[0:4]=='PDMS':
            if self.ink.shortname[4]=='S':
                s0 = 'silicone'
            else:
                s0 = 'mineral'
            sn = self.ink.shortname.replace(self.ink.shortname[0:5],'PDMS_3_812_')
            shortname = f'{sn}_{s0}_25'
        elif self.ink.shortname[0:3]=='PEG':
            shortname = self.ink.shortname.replace('PEG', 'PEGDA_40_200_')
        else:
            shortname = self.ink.shortname
        key = f'{shortname}_{self.date}'
        validfiles = []
        for f in os.listdir(folder):
            if key in f:
                validfiles.append(os.path.join(folder, f))
        return validfiles
        
    def getTargetPressureFromCalib(self, calibFile:str, justCalib:bool=False) -> dict:
        '''find the target pressure from the calibration file'''
        with open(calibFile, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                if not justCalib:
                    if row[0]=='target speed':
                        if not self.vink==float(row[2]):
                            return False
                    elif row[0]=='target pressure':
                        self.targetPressure = float(row[2])
                if row[0]=='a':
                    self.caliba = float(row[2])
                elif row[0]=='b':
                    self.calibb = float(row[2])
                elif row[0]=='c':
                    self.calibc = float(row[2])
                    self.calculateTargetPressure()
                    return True
        return False
  
    def readCalibFile(self, justCalib:bool=False) -> int:
        '''read target pressure and calibration curve from calibration file'''
        self.calibFile = False
        # find the file
        try:
            cfiles = self.findCalibFile()   # get the pressure calibration file
        except:
            return 1
        if len(cfiles)==0:
#             logging.warning(f'{self.bn}: No calibration file found')
            return
        # read the file
        read = False
        while not read and len(cfiles)>0:
            read = self.getTargetPressureFromCalib(cfiles.pop(), justCalib=justCalib)
        if self.targetPressure>0:
            self.calibFile = True
            
    #----------------------
    
    def exportSpeedFile(self) -> None:
        '''export a speed file based on expected values. version control for shopbotpyqt files before mid-2021'''
        file = self.pfd.newFileName('speeds', 'csv')
        with open(file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for k,v in self.targetPressures.items():
                writer.writerow([f'ink pressure channel {k}', 'mbar', str(v)])
            writer.writerow(['ink speed', 'mm/s', str(self.vink)])
            writer.writerow(['support speed', 'mm/s', str(self.vsup)])
            writer.writerow(['caliba', 'mm/s/mbar^2', str(self.caliba)])
            writer.writerow(['calibb', 'mm/s/mbar', str(self.calibb)])
            writer.writerow(['calibc', 'mm/s', str(self.calibc)])
        logging.info(f'Exported {file}')