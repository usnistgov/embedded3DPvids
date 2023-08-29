#!/usr/bin/env python
'''Functions for storing pressure and speed metadata about print folders'''

# external packages
import os, sys
import traceback
import logging
from typing import List, Dict, Tuple, Union, Any, TextIO
import re
import pandas as pd
import numpy as np
import csv
import shutil

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
from tools.config import cfg
from tools.plainIm import *
import file.file_handling as fh
from v_fluid import fluidVals
from tools.regression import regPD

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
        self.calibFile = False
        self.calibFile = True
        
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
                self.calibFile = False
            else:
                self.exportSpeedFile()
                self.calibFile = True
        else:
            self.calibFile = True
            
        self.correctCalib()  # check that the calibration is valid
        
            
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
    
    def localMin(self) -> float:
        '''get the pressure at which the speed has a local minimum'''
        if self.caliba==0:
            return 0
        return -self.calibb/(2*self.caliba)
    
    def calculateTargetPressure(self) -> float:
        self.targetPressure = self.calculateP(self.vink)
        return self.targetPressure
        
    def calculateSpeed(self, p:float) -> float:
        '''calculate speed from pressure'''
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
        file = self.pfd.metaFile()
        if len(file)==0:
            return 1
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
    def findCalibFile(self, ink) -> list:
        '''find the pressure calibration file'''
        folder = cfg.path.pCalibFolder
        if not os.path.exists(folder):
            raise NameError(f'Pressure calibration folder does not exist {folder}')
        if ink.shortname[0]=='M':
            shortname = f'mineral_812_{ink.shortname[1]}'
        elif ink.shortname[0:4]=='PDMS':
            if ink.shortname[4]=='S':
                s0 = 'silicone'
            else:
                s0 = 'mineral'
            sn = ink.shortname.replace(ink.shortname[0:5],'PDMS_3_812_')
            shortname = f'{sn}_{s0}_25'
        elif ink.shortname[0:3]=='PEG':
            shortname = ink.shortname.replace('PEG', 'PEGDA_40_200_')
        elif ink.base=='silicone oil':
            shortname = 'silicone_oil'
            if len(ink.surfactant)>0:
                ss = ink.surfactant.lower().replace(' ','')
                shortname = f'{shortname}_{ss}_{ink.surfactantWt}'
            if int(ink.val)==ink.val:
                iv = int(ink.val)
            else:
                iv = ink.val
            shortname = f'{shortname}_812_{iv}'
        else:
            shortname = ink.shortname
        key = f'{shortname}_{self.pfd.date}'
        validfiles = []
        for f in os.listdir(folder):
            if key in f:
                validfiles.append(os.path.join(folder, f))
        return validfiles
    
    def findCalibFile0(self) -> list:
        '''find the calibration file using only stored values'''
        if not hasattr(self, 'ink'):
            ink0 = re.split('_S_', re.split('I_', self.bn)[1])[0]
            self.ink = fluidVals(ink0, 'ink', properties=False)
        return self.findCalibFile(self.ink)
    
    def fileTime(self, file:str) -> int:
        '''get the time of the file'''
        return int(re.split('_', os.path.basename(file))[-2])
    
    def adoptNewCalib(self, file:str) -> int:
        '''create new linear model fit from the file and adopt the values. return 0 if succeeded'''
        pts = self.getPointsFromCalib(file)
        newreg = regPD(pts, ['pressure (mbar)'], 'speed (mm/s)', order=1, log=False, intercept='')
        if not 'r2' in newreg or newreg['r2']<0.8:
            return 1
        self.caliba = 0
        self.calibb = newreg['b']
        self.calibc = newreg['c']
        if self.localMin()<=0:
            return 0
        else:
            return 1
    
    def correctCalib(self) -> None:
        '''check that calibration makes sense and switch to a linear model if it doesn't. return 0 if values were changed'''
        self.changed = False
        if self.localMin()<=0:
            return 
        files = self.findCalibFile0()
        if len(files)==0:
            return
        time = self.fileTime(self.pfd.metaFile())
        vals = pd.DataFrame([{'file':file, 'time':self.fileTime(file)} for file in files])
        vless = vals[vals.time<time]
        file = vless[vless.time==vless.time.max()].iloc[0]['file']
        out = self.adoptNewCalib(file)
        files.remove(file)
        while out>0 and len(files)>0:
            file = files.pop(-1)
            out = self.adoptNewCalib(file)
        if out==0:
            self.changed = True
            self.changeFit()
        
        
    def changeFit(self):
        '''change the pressure vs speed model used in progDims and measurements for a single folder'''
        mf = self.pfd.meta[0]
        meta,u = plainImDict(mf, unitCol=1, valCol=2)
        if meta['caliba_channel_0'] == self.caliba and meta['calibb_channel_0'] == self.calibb and meta['calibc_channel_0'] == self.calibc:
            return
        meta['caliba_channel_0'] = self.caliba
        meta['calibb_channel_0'] = self.calibb
        meta['calibc_channel_0'] = self.calibc
        shutil.copyfile(mf, mf.replace('meta', 'metOrig'))
        plainExpDict(mf, meta, u, quotechar='"')
        return
  
    def getFitFromCalib(self, calibFile:str) -> dict:
        '''get the a,b,c from the calibration file'''
        d = {'file':calibFile}
        with open(calibFile, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                if row[0]=='a':
                    d['a'] = float(row[2])
                elif row[0]=='b':
                    d['b'] = float(row[2])
                elif row[0]=='c':
                    d['c'] = float(row[2])
                    return True
        return False
    
    def getPointsFromCalib(self, calibFile:str) -> pd.DataFrame:
        '''get the original calibration points from the calibration file'''
        with open(calibFile, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                if len(row)>0 and 'init wt' in row[0]:
                    df = pd.read_csv(calibFile, skiprows=reader.line_num-1)
                    df.dropna(inplace=True)
                    return df
        return pd.DataFrame([])
        
    def getTargetPressureFromCalib(self, calibFile:str, justCalib:bool=False) -> bool:
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
            writer.writerow([f'ink pressure channel {self.channel}', 'mbar', str(self.targetPressure)])
            writer.writerow(['ink speed', 'mm/s', str(self.vink)])
            writer.writerow(['support speed', 'mm/s', str(self.vsup)])
            writer.writerow(['caliba', 'mm/s/mbar^2', str(self.caliba)])
            writer.writerow(['calibb', 'mm/s/mbar', str(self.calibb)])
            writer.writerow(['calibc', 'mm/s', str(self.calibc)])
        logging.info(f'Exported {file}')