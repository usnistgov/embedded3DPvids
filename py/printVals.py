#!/usr/bin/env python
'''Functions for plotting video data. Adapted from https://github.com/usnistgov/openfoamEmbedded3DP'''

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

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)

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

class fluidVals:
    '''class that holds info about fluid'''
    
    def __init__(self, fluid:str, ftype:str):
        '''convert the shorthand sample name to info about the fluid. ftype is 'ink' or 'sup' '''
        self.shortname = fluid
        self.days = 1
        self.rheModifier = 'fumed silica'
        self.surfactant = ''
        self.dye = ''
        if fluid[0]=='M':
            self.var = 'w% silica'
            self.val = fluid[1:]
            self.base = 'mineral oil'
            if ftype=='ink':
                self.dye = 'red'
            if fluid[-1]=='S':
                self.surfactant = 'Span 20'
                self.val = self.val[:-1]
        elif fluid[:4]=='PDMS':
            self.var = 'w% silica'
            if ftype=='ink':
                self.dye = 'red'
            if fluid[4]=='M':
                self.base = 'PDMS_3_mineral_25'
            else:
                self.base = 'PDMS_3_silicone_25'
            self.val = fluid[5:]
            if fluid[-1]=='S':
                self.surfactant = 'Span 20'
                self.val = self.val[:-1]
        elif fluid[:3]=='PEG':
            if ftype=='ink':
                self.dye = 'blue'
            self.var = 'w% silica'
            self.val = fluid[3:]
            self.base = 'PEGDA_40'
        else:
            self.var = 'w% Laponite RD'
            self.rheModifier = 'Laponite RD'
            self.base = 'water'
            self.val = fluid
            if ftype=='ink':
                self.dye='blue'
            if fluid[-1]=='T':
                self.surfactant = 'Tween 80'
                self.val = self.val[:-1]
        try:
            self.val = float(self.val)
        except:
            logging.warning(f'Failed to convert fluid value to float: {fluid}, {self.val}')
        if len(self.surfactant)>0:
            self.type = self.base + '_'+self.surfactant
        else:
            self.type = self.base
        self.findRhe()
        self.findDensity()
        
    def metarow(self, tag:s='') -> Tuple[dict,dict]:
        '''row containing metadata'''
        mlist = ['shortname', 'days', 'rheModifier', 'surfactant' ,'dye', 'var', 'val', 'base', 'type']
        meta = [[tag+i,getattr(self,i)] for i in mlist]
        munits = [[tag+i, ''] for i in mlist]
        rhelist = ['tau0', 'eta0']
        rhe = [[tag+i,getattr(self,i)] for i in rhelist]
        rheunits = [[tag+i, self.rheUnits[i]] for i in rhelist]
        clist = ['v', 'visc0', 'CaInv', 'Re', 'WeInv', 'OhInv', 'rPR']
        const = [[tag+i,getattr(self,i)] for i in clist]
        constunits = [[tag+i, self.constUnits[i]] for i in clist]
        out = dict(meta+rhe+const)
        units = dict(munits+rheunits+constunits)
        return out,units
            
    def findRhe(self) -> dict:
        '''find Herschel-Bulkley fit from file'''
        if not os.path.exists(cfg.path.rheTable):
            logging.error(f'No rheology table found: {cfg.path.rheTable}')
            return
        rhe = pd.read_excel(cfg.path.rheTable)
        rhe = rhe.fillna('') 
        entry = rhe[(rhe.base==self.base)&(rhe.rheModifier==self.rheModifier)&(rhe.rheWt==self.val)&(rhe.surfactant==self.surfactant)&(rhe.dye==self.dye)&(rhe.days==self.days)]
        if len(entry)==0:
            logging.error(f'No rheology fit found for fluid {self.shortname}')
            return
        if len(entry)>1:
            print(entry)
            logging.error(f'Multiple rheology fits found for fluid {self.shortname}')
        entry = entry.iloc[0]
        self.tau0 = entry['y2_tau0'] # Pa
        self.k = entry['y2_k'] 
        self.n = entry['y2_n']
        self.eta0 = entry['y2_eta0'] # these values are for Pa.s, vs. frequency in 1/s
        self.rheUnits = {'tau0':'Pa', 'k':'Pa*s^n', 'n':'','eta0':'Pa*s'}
        return
    
    def findDensity(self) -> dict:
        '''find density from file'''
        if not os.path.exists(cfg.path.densityTable):
            logging.error(f'No rheology table found: {cfg.path.densityTable}')
            return
        tab = pd.read_excel(cfg.path.densityTable)
        tab = tab.fillna('') 
        entry = tab[(tab.base==self.base)&(tab.rheModifier==self.rheModifier)&(tab.rheWt==self.val)]
        if len(entry)==0:
            logging.error(f'No density fit found for fluid {self.shortname}')
            return
        if len(entry)>1:
            print(entry)
            logging.error(f'Multiple density fits found for fluid {self.shortname}')
        entry = entry.iloc[0]
        self.density = entry['density'] # g/mL
        return
    
    def visc(self, gdot:float) -> float:
        '''get the viscosity of the fluid in Pa*s at shear rate gdot in Hz'''
        try:
            mu = self.k*(abs(gdot)**(self.n-1)) + self.tau0/(abs(gdot))
            nu = min(mu, self.eta0)
        except:
            raise ValueError(f'Rheology is not defined for {self.shortname}')
        else:
            return nu
        
    def constants(self, v:float, diam:float, sigma:float) -> None:
        '''find shear rate, viscosity during printing, capillary number.
        v is in mm/s, diam is in mm, sigma is in mJ/m^2'''
        self.v = v                                              # mm/s
        self.rate = v/diam                                      # 1/s
        self.visc0 = self.visc(self.rate)                       # Pa*s
        self.CaInv = sigma/(self.visc0*self.v)                  # capillary number ^-1
        self.Re = 10**3*(self.density*self.v*diam)/(self.visc0) # reynold's number
        self.WeInv = 10**-3*sigma/(self.density*self.v**2*diam) # weber number ^-1
        self.OhInv = np.sqrt(self.WeInv)*self.Re                # Ohnesorge number^-1
        self.rPR = sigma/self.tau0                              # characteristic length for Plateau rayleigh instability in mm
        self.constUnits = {'v':'mm/s','rate':'1/s','visc0':'Pa*s', 'CaInv':'','Re':'','WeInv':'','OhInv':'','rPR':'mm'}
        

#------   

class printVals:
    '''class that holds info about the experiment'''
    
    def __init__(self, folder:str, di:float=cfg.const.di, do:float=cfg.const.do):
        '''get the ink and support names from the folder name. 
        vink and vsup are in mm/s. 
        di (inner nozzle diameter) and do (outer) are in mm'''
        self.folder = folder
        self.bn = os.path.basename(folder)
        self.fluFile = False
        self.calibFile = False
        split = re.split('_', self.bn)
        inkShortName = split[1]
        supShortName = split[3]
        self.date = int(split[-1][0:6])
        self.ink = fluidVals(inkShortName, 'ink')
        self.sup = fluidVals(supShortName, 'sup')
        if '3day' in folder:
            self.ink.days = 3
            self.sup.days = 3
            self.ink.findRhe()
            self.sup.findRhe()
        self.tension()
        self.di = di
        self.do = do
        self.findVelocities()
        self.ink.constants(self.vink, self.di, self.sigma)
        self.sup.constants(self.vsup, self.do, self.sigma)
        self.const()
        self.units = {'bn':'','folder':'','date':'YYMMDD','di':'mm','do':'mm', 'sigma':'mN/m', 'fluFile':'', 'calibFile':''}
        self.progDims = {}
        self.progDimsUnits = {}
        
                    
        
    def const(self) -> None:
        '''define dimensionless numbers and critical values'''
        self.viscRatio = self.ink.visc0/self.sup.visc0 
            # viscosity ratio
        ddiff = self.ink.density-self.sup.density
        if abs(ddiff)>0:
            self.rGrav = 10**6*(self.sup.tau0)/((ddiff)*9.8) 
        else:
            self.rGrav = 0
            # characteristic sagging radius in mm, missing scaling factor, see O'Brien MRS Bulletin 2017
        self.vRatio = self.ink.v/self.sup.v 
        self.dEst = self.di*np.sqrt(self.vRatio)
            # expected filament diameter in mm
        self.ReRatio = self.ink.Re/self.sup.Re
        l = 12.5 # mm, depth of nozzle in bath
        dn = 2*np.sqrt(self.do*l/np.pi)
        Kn = 1/3+2/3*np.sqrt(np.pi)/2
        self.hDragP = 3*np.pi*self.sup.visc0*self.sup.v*(dn*Kn)/(self.do*l)
            # horizontal line stokes drag pressure
        self.vDragP = 3*self.sup.visc0*self.sup.v*4/(self.do)
            # vertical line stokes drag pressure
        self.constUnits = {'viscRatio':'', 'vRatio':'', 'ReRatio':'', 'rGrav':'mm','dEst':'mm', 'hDragP':'Pa', 'vDragP':'Pa'}
        
    def metarow(self) -> Tuple[dict,dict]:
        '''row holding metadata'''
        mlist = ['bn', 'date', 'sigma', 'di', 'do', 'fluFile', 'calibFile']
        meta = [[i,getattr(self,i)] for i in mlist]
        munits = [[i, self.units[i]] for i in mlist]
        clist = [ 'viscRatio','vRatio',  'ReRatio', 'rGrav', 'dEst', 'hDragP', 'vDragP']
        const = [[i,getattr(self,i)] for i in clist]
        cunits = [[i, self.constUnits[i]] for i in clist]
        pvals = dict(meta+const)
        punits = dict(munits+cunits)
        for i,p in self.targetPressures.items():
            pvals[f'pressureCh{i}']=p*100
            punits[f'pressureCh{i}']='Pa'
        inkvals, inkunits = self.ink.metarow('ink_')
        supvals, supunits = self.sup.metarow('sup_')
        out = {**pvals, **inkvals, **supvals}
        units = {**punits, **inkunits, **supunits}
        return out, units
        
    def base(self, xfluid:str, yfluid:str, vname:str='val') -> str:
        '''get the plot title'''
        self.xval = getattr(getattr(self, xfluid),vname)
        self.yval = getattr(getattr(self, yfluid),vname)
        xbase = getattr(self, xfluid).base
        ybase = getattr(self, yfluid).base
        base = xbase + ', '+ybase
        return base
    
    def tension(self) -> float:
        '''estimate the surface tension'''
        if not os.path.exists(cfg.path.sigmaTable):
            logging.error(f'No sigma table found: {cfg.path.sigmaTable}')
            return
        sigt = pd.read_excel(cfg.path.sigmaTable)
        sigt = sigt.fillna('') 
        entry = sigt[(sigt.ink_base==self.ink.base)&(sigt.sup_base==self.sup.base)&(sigt.ink_surfactant==self.ink.surfactant)&(sigt.sup_surfactant==self.sup.surfactant)]
        if len(entry)==0:
            logging.error(f'No surface tension fit found for fluid {self.bn}')
            return
        if len(entry)>1:
            logging.error(f'Multiple surface tension fits found for fluid {self.bn}')
        entry = entry.iloc[0]
        self.sigma = entry['sigma'] # mJ/m^2
        return self.sigma
    
    def vidFile(self) -> str:
        '''get the path of the video file taken during the print'''
        for f in os.listdir(self.folder):
            if 'singleLinesNoZig' in f and 'avi' in f and 'Basler camera' in f:
                return os.path.join(self.folder, f)
        return ''
                    
    
    #--------------------------------------------------
   
    def findCalibFile(self) -> str:
        '''find the pressure calibration file'''
        folder = cfg.path.pCalibFolder
        if not os.path.exists(folder):
            raise NameError(f'Pressure calibration folder does not exist {folder}')
        if self.ink.shortname[0]=='M':
            shortname = 'mineral_812_'+str(self.ink.shortname[1])
        elif self.ink.shortname[0:4]=='PDMS':
            if self.ink.shortname[4]=='S':
                s0 = 'silicone'
            else:
                s0 = 'mineral'
            shortname = self.ink.shortname.replace(self.ink.shortname[0:5],'PDMS_3_812_')+'_'+s0+'_25'
        elif self.ink.shortname[0:3]=='PEG':
            shortname = self.ink.shortname.replace('PEG', 'PEGDA_40_200_')
        else:
            shortname = self.ink.shortname
        key = shortname+'_'+str(self.date)
        validfiles = []
        for f in os.listdir(folder):
            if key in f:
                validfiles.append(os.path.join(folder, f))
        return validfiles
        
    def getTargetPressureFromCalib(self, calibFile:str) -> dict:
        '''find the target pressure from the calibration file'''
        with open(calibFile, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                if row[0]=='target speed':
                    if not self.vink==float(row[2]):
                        return False
                if row[0]=='target pressure':
                    self.targetPressures[0] = float(row[2])
                elif row[0]=='a':
                    self.caliba = float(row[2])
                elif row[0]=='b':
                    self.calibb = float(row[2])
                elif row[0]=='c':
                    self.calibc = float(row[2])
                    return True
        return False
        
    def readCalibFile(self) -> None:
        '''read target pressure and calibration curve from calibration file'''
        self.calibFile = False
        # find the file
        try:
            cfiles = self.findCalibFile()   # get the pressure calibration file
        except:
            return
        if len(cfiles)==0:
            return
        # read the file
        read = False
        while not read and len(cfiles)>0:
            read = self.getTargetPressureFromCalib(cfiles.pop())
        if self.targetPressures[0]>0:
            self.calibFile = True
                
    def findSpeedFile(self) -> str:
        '''find the speed file in this folder'''
        for f in os.listdir(self.folder):
            if 'singleLinesNoZig' in f and 'speeds' in f:
                return os.path.join(self.folder, f)
        return ''
    
    def readSpeedFile(self, file:str) -> None:
        '''read values from the speed file'''
        with open(file, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            self.targetPressures = {}
            for row in reader:
                if 'pressure' in row[0]:
                    # for each channel, get target pressure
                    channel = re.split('channel ', row[0])[-1]
                    self.targetPressures[int(channel)] =  float(row[2])
                elif 'ink speed' in row:
                    self.vink = float(row[2])
                elif 'support speed' in row:
                    self.vsup = float(row[2])
                elif 'caliba' in row:
                    self.caliba = float(row[2])
                elif 'calibb' in row:
                    self.calibb = float(row[2])
                elif 'calibc' in row:
                    self.calibc = float(row[2])
                

    def exportSpeedFile(self) -> None:
        '''export a speed file based on expected values'''
        vf = self.vidFile()
        if os.path.exists(vf):
            file = vf.replace('avi', 'csv')
            file = file.replace('Basler camera', 'speeds')
        else:
            return
        with open(file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for k,v in self.targetPressures.items():
                writer.writerow(['ink pressure channel '+str(k), 'mbar', str(v)])
            writer.writerow(['ink speed', 'mm/s', str(self.vink)])
            writer.writerow(['support speed', 'mm/s', str(self.vsup)])
            writer.writerow(['caliba', 'mm/s/mbar^2', str(self.caliba)])
            writer.writerow(['calibb', 'mm/s/mbar', str(self.calibb)])
            writer.writerow(['calibc', 'mm/s', str(self.calibc)])
        
    def findVelocities(self) -> None:
        '''find the target velocities'''
        self.targetPressures = {0:0}
        self.caliba = 0
        self.calibb = 0
        self.calibc = 0
        
        file = self.findSpeedFile() # file that holds metadata about speed for this run
        if os.path.exists(file):
            self.readSpeedFile(file) # get target pressure, ink speed, and sup speed from speed file
            self.calibFile = True 
        else:
            self.vink = cfg.const.vink    # use default ink and sup speed
            self.vsup = cfg.const.vsup
            self.targetPressures = {0:0}
            self.readCalibFile()
            self.exportSpeedFile()

    def importFluFile(self) -> pd.DataFrame:
        '''find and import the pressure-time table'''
        l = []
        for f in os.listdir(self.folder):
            if 'singleLinesN' in f and 'Fluigent' in f:
                l.append(f)
        if len(l)==0:
            raise NameError(f'No fluigent file found in {self.folder}')
        l.sort()
        file = os.path.join(self.folder, l[-1])
        ftable = pd.read_csv(file)
        ftable.rename(columns={'time (s)':'time','Channel 0 pressure (mbar)':'pressure'}, inplace=True)
        return ftable
    
                
                    
    def redoSpeedFile(self) -> None:
        '''read calibration curve from calibration file and add to speed file'''
        self.readCalibFile()
        if self.calibb==0:
            # if there is no calibration curve, assume speed = pressure * (intended speed)/(max pressure)
            try:
                ftable = self.importFluFile()
            except:
                # no fluigent file. not enough info to create speed file
                return
            self.targetPressures[0] = ftable.pressure.max()
            self.calibb = self.vink/self.targetPressures[0]
        self.exportSpeedFile()
            
                
    def initializeProgDims(self):
        '''initialize programmed dimensions table'''
        self.progDims = pd.DataFrame(columns=['name','l','w','t','a','vol'])
        self.progDims.name=['xs5','xs4','xs3','xs2','xs1','vert4','vert3','vert2','vert1','horiz2','horiz1','horiz0']
        self.progDimsUnits = {'name':'', 'l':'mm','w':'mm','t':'s','a':'mm^2','vol':'mm^3'}
                
    def useDefaultTimings(self):
        '''use the programmed line lengths'''
        self.initializeProgDims()
        for s,row in self.progDims.iterrows():
            if 'xs' in row['name']:
                self.progDims.loc[s,'l']=15         # intended filament length
            elif 'vert' in row['name']:
                self.progDims.loc[s,'l']=15
            elif 'horiz' in row['name']:
                self.progDims.loc[s,'l']=22.5 
            self.progDims.loc[s,'w']=self.dEst  # intended filament diameter
            self.progDims.loc[s,'t']=self.progDims.loc[s,'l']/self.sup.v # extrusion time
            self.progDims.loc[s,'a']=np.pi*(self.dEst/2)**2
            self.progDims.loc[s,'vol']=self.progDims.loc[s,'l']*self.progDims.loc[s,'a']
            
            
    def storeProg(self, i:int, vals:dict) -> None:
        '''store programmed values into the table'''
        self.progDims.iloc[i]['l'] = vals['l']
        self.progDims.iloc[i]['vol'] = vals['vol']
        self.progDims.iloc[i]['t'] = vals['ttot']
        self.progDims.iloc[i]['a']= vals['vol']/vals['l']
#         self.progDims.iloc[i]['w'] = 2*np.sqrt(self.progDims.iloc[i]['a']/np.pi) # a=pi*(w/2)^2
        self.progDims.iloc[i]['w'] = self.dEst 
    
            
    def readProgDims(self, df:pd.DataFrame, tp:float):
        '''read programmed line dimensions from fluigent table. df is time-pressure table from file. tp is target pressure'''
        self.initializeProgDims()
        i = 0
        vol = 0
        ttot = 0
        l = 0
        a = np.pi*(self.dEst/2)**2 # ideal cross-sectional area
        anoz = np.pi*(self.di/2)**2 # inner cross-sectional area of nozzle
        for j,row in df.iterrows():
            if row['pressure']>0:
                # flow is on
                if j>0:
                    dt = (row['time']-df.loc[j-1,'time']) # timestep size
                else:
                    dt = (-row['time']+df.loc[j+1,'time']) 
                ttot = ttot+dt # total time traveled
                dl = dt*self.sup.v # length traveled in this timestep
                l = l+dl  # total length traveled
                if dt==0:
                    dvol=0
                else:
#                     dvol = row['pressure']/tp*a*dl 
#                         # volume extruded in this timestep,
#                         # where a*dl is ideal and volume is scaled by 
#                         # ratio of intended pressure to actual pressure
                    p = row['pressure']
                    flux = max((self.caliba*p**2+self.calibb*p+self.calibc)*anoz,0)
                        # actual flow speed based on calibration curve (mm/s) * area of nozzle (mm^2) = flux (mm^3/s)
                    dvol = flux*dt 
                vol = vol+dvol # total volume extruded
            else:
                # flow is off
                if l>0:
                    self.storeProg(i, {'l':l, 'vol':vol, 'ttot':ttot, 'a':a})
                    ttot=0
                    vol=0
                    l=0
                    i = i+1
        if i<len(self.progDims):
            self.storeProg(i, {'l':l, 'vol':vol, 'ttot':ttot, 'a':a})
        self.progDimsUnits = {'name':'', 'l':'mm', 'w':'mm', 't':'s', 'a':'mm^2', 'vol':'mm^3'}
    
    
    def fluigent(self) -> None:
        '''get lengths of actual extrusion from fluigent'''
        try:
            ftable = self.importFluFile()
            self.fluFile = True
        except:
            self.useDefaultTimings()
            self.fluFile = False
        else:
            if len(self.targetPressures)==0 or self.targetPressures[0]==0:
                self.targetPressures[0] = ftable.pressure.max()
            self.readProgDims(ftable, self.targetPressures[0])
            
                    
    def progDimsFile(self) -> str:
        return os.path.join(self.folder, os.path.basename(self.folder)+'_progDims.csv')
    
    def importProgDims(self) -> str:
        '''import the progdims file'''
        fn = self.progDimsFile()
        if not os.path.exists(fn):
            self.fluigent()
            self.exportProgDims()
        else:
            self.progDims, self.progDimsUnits = plainIm(fn, ic=0)
        return self.progDims, self.progDimsUnits
            
    def exportProgDims(self) -> None:
        plainExp(self.progDimsFile(), self.progDims, self.progDimsUnits)
        
    def progDimsSummary(self) -> Tuple[pd.DataFrame,dict]:
        if len(self.progDims)==0:
            self.importProgDims()
        if len(self.progDims)==0:
            return {},{}
        df = self.progDims.copy()
        df2 = pd.DataFrame(columns=df.columns)
        for s in ['xs', 'vert', 'horiz']:
            df2.loc[s] = df[df.name.str.contains(s)].mean()
        df2 = df2.drop(columns=['name'])
        df2 = df2.T
        v = (df2).unstack().to_frame().sort_index(level=0).T
        v.columns = v.columns.map('_'.join) # single row dataframe
        vunits = dict([[k, self.progDimsUnits[re.split('_', k)[1]]] for k in v]) # units for this dataframe
        for s in ['bn', 'vink', 'vsup', 'sigma']:
            v.loc[0, s] = getattr(self, s)
        v.loc[0, 'pressure'] = self.targetPressures[0]
        vunits['bn']=''
        vunits['vink']='mm/s'
        vunits['vsup']='mm/s'
        vunits['sigma']='mJ/m^2'
        vunits['pressure']='mbar'
        return v,vunits
    
    def vertSum(self) -> Tuple[dict, dict]:
        '''summarize all vertical lines'''
        vertfile = os.path.join(self.folder, self.bn+'_vertSummary.csv')
        if not os.path.exists(vertfile):
            return {}, {}
        tab, units = plainIm(vertfile, ic=0)
        if len(tab)==0:
            return {},{}
        t1 = []
        for i,row in tab.iterrows():
            progrow = self.progDims[self.progDims.name=='vert'+str(int(row['line']))]
            if len(progrow)==0:
                raise ValueError('Line name not found in progDims')
            progrow = progrow.iloc[0]
            t2 = {}
#             for s in ['w', 'h', 'vintegral', 'meanT']:
#                 t2[s] = row[s]
            t2['wN'] = row['w']/cfg.const.pxpmm/progrow['w'] # convert to mm, divide by intended dimension
            t2['hN'] = row['h']/cfg.const.pxpmm/progrow['l'] # convert to mm
            t2['vN'] = row['vest']/cfg.const.pxpmm**3/progrow['vol'] # convert to mm^3. vN is the estimated volume of the bounding box
            t2['viN'] = row['vintegral']/cfg.const.pxpmm**3/progrow['vol'] # convert to mm^3. viN is the estimated volume by integrating over the length of the line
            t2['vleakN'] = row['vleak']/cfg.const.pxpmm**3/(progrow['a']*(15-progrow['l'])) 
                # convert to mm^3. viN is the estimated volume by integrating past the programmed length of the line, 
                # normalized by the remaining allowed volume after flow stops
            t2['roughness'] = row['roughness']
            t2['meanTN'] = row['meanT']/cfg.const.pxpmm/progrow['w'] # convert to mm
            t2['stdevTN'] = row['stdevT']/cfg.const.pxpmm/progrow['w'] # convert to mm
            t2['minmaxTN'] = row['minmaxT']/cfg.const.pxpmm/progrow['w'] # convert to mm
            t1.append(t2)
        t1 = pd.DataFrame(t1)
        t3 = [['vert_'+s, t1[s].mean()] for s in t1] # averages
        t4 = [['vert_'+s+'_SE', t1[s].sem()] for s in t1] # standard errors
        t3 = dict(t3+t4)
        units = dict([[s,''] for s in t3.keys()])
        return t3, units
    
    def horizSum(self) -> Tuple[dict, dict]:
        '''summarize all horizontal lines'''
        file = os.path.join(self.folder, self.bn+'_horizSummary.csv')
        if not os.path.exists(file):
            return {}, {}
        tab, units = plainIm(file, ic=0)
        if len(tab)==0:
            return {},{}
        t1 = []
        for i,row in tab.iterrows():
            progrow = self.progDims[self.progDims.name=='horiz'+str(int(row['line']))]
            if len(progrow)==0:
                raise ValueError('Line name not found in progDims')
            progrow = progrow.iloc[0]
            t2 = {}
            t2['segments'] = row['segments']
            t2['maxlenN'] = row['maxlen']/cfg.const.pxpmm/progrow['l']
            t2['totlenN'] = row['totlen']/cfg.const.pxpmm/progrow['l']
            t2['vN'] = row['vest']/cfg.const.pxpmm**3/progrow['vol'] # convert to mm^3
            t2['roughness'] = row['roughness']
            t2['meanTN'] = row['meanT']/cfg.const.pxpmm/progrow['w'] # convert to mm
            t2['stdevTN'] = row['stdevT']/cfg.const.pxpmm/progrow['w'] # convert to mm
            t2['minmaxTN'] = row['minmaxT']/cfg.const.pxpmm/progrow['w'] # convert to mm
            t1.append(t2)
        t1 = pd.DataFrame(t1)
        t3 = [['horiz_'+s, t1[s].mean()] for s in t1] # averages
        t4 = [['horiz_'+s+'_SE', t1[s].sem()] for s in t1] # standard errors
        t3 = dict(t3+t4)
        t3['horiz_segments'] = t1.segments.sum()
        t3.pop('horiz_segments_SE')
        units = dict([[s,''] for s in t3.keys()])
        return t3, units
    
    def xsSum(self) -> Tuple[dict, dict]:
        '''summarize all horizontal lines'''
        file = os.path.join(self.folder, self.bn+'_xsSummary.csv')
        if not os.path.exists(file):
            return {}, {}
        tab, units = plainIm(file, ic=0)
        t1 = []
        if len(tab)==0:
            return {},{}
        tab = tab[tab.line<5] # ignore the first line
        if 'PDMS' in self.bn or 'mineral' in self.bn:
            tab = tab[tab.roughness<0.01] 
            # remove rough cross-sections for non-zero surface tension, 
            # indicating that we may be looking at two droplets, one behind the other
        for i,row in tab.iterrows():
            progrow = self.progDims[self.progDims.name=='xs'+str(int(row['line']))] # programmed dimensions
            if len(progrow)==0:
                raise ValueError('Line name not found in progDims')
            progrow = progrow.iloc[0]
            t2 = {}
            t2['aspect'] = row['aspect']
            t2['xshift'] = row['xshift']
            t2['yshift'] = row['yshift']
            t2['areaN'] = row['area']/(cfg.const.pxpmm**2)/progrow['a']
            t2['wN'] = row['w']/cfg.const.pxpmm/progrow['w']
            t2['hN'] = row['h']/cfg.const.pxpmm/progrow['w']
            t2['roughness'] = row['roughness']
            t1.append(t2)
        t1 = pd.DataFrame(t1)
        t3 = [['xs_'+s, t1[s].mean()] for s in t1] # averages
        t4 = [['xs_'+s+'_SE', t1[s].sem()] for s in t1] # standard errors
        t3 = dict(t3+t4)
        units = dict([[s,''] for s in t3.keys()])
        return t3, units
    
    def summary(self) -> Tuple[dict,dict]:
#         logging.info(self.folder)
#         self.fluigent()
        self.importProgDims()
        meta,metaunits = self.metarow()
        xs,xsunits = self.xsSum()
        vert,vertunits = self.vertSum()
        horiz,horizunits = self.horizSum()
        try:
            vHorizEst = {'vHorizEst':xs['xs_areaN']*horiz['horiz_totlenN']}
        except:
            vHorizEst = {}
        vHorizEstUnits = {'vHorizEst':''}
        out = {**meta, **xs, **vert, **horiz, **vHorizEst}
        outunits = {**metaunits, **xsunits, **vertunits, **horizunits, **vHorizEstUnits}
        self.exportSummary(out, outunits)
        return out, outunits
    
    def summaryFile(self) -> str:
        return os.path.join(self.folder, os.path.basename(self.folder)+'_summary.csv')
    
    def exportSummary(self, out:dict, outunits:dict) -> None:
        '''export a summary file based on expected values'''
        file = self.summaryFile()
        with open(file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for k,v in out.items():
                writer.writerow([k, outunits[k], str(v)])

    
#------
