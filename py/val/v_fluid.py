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
sys.path.append(os.path.dirname(currentdir))
from tools.config import cfg
from tools.plainIm import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)



#----------------------------------------------

class fluidVals:
    '''class that holds info about fluid'''
    
    def __init__(self, fluid:str, ftype:str, properties:bool=True):
        '''convert the shorthand sample name to info about the fluid. ftype is 'ink' or 'sup' '''
        self.shortname = fluid
        self.days = 1
        self.rheModifier = 'fumed silica'
        self.surfactant = ''
        self.surfactantWt = ''
        self.dye = ''
        self.ftype=ftype
        if fluid[0]=='M':
            self.var = 'w% silica'
            self.val = fluid[1:]
            self.base = 'mineral oil'
            if ftype=='ink':
                self.dye = 'red'
            if fluid[-1]=='S':
                self.surfactant = 'Span 20'
                self.surfactantWt = 0.5
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
                self.surfactantWt = 0.5
                self.val = self.val[:-1]
        elif fluid[:3]=='PEG':
            if ftype=='ink':
                self.dye = 'blue'
            self.var = 'w% silica'
            self.val = fluid[3:]
            self.base = 'PEGDA_40'
        elif fluid[0:2]=='SO':
            self.var = 'w% silica'
            self.base = 'silicone oil'
            self.rheModifier = 'Aerosil R812S'
            spl = re.split('-', fluid)
            self.val = spl[0][2:]
            self.dye = 'red'
            if len(spl)>1:
                if spl[1]=='S20':
                    self.surfactant = 'Span 20'
                elif spl[1]=='S85':
                    self.surfactant = 'Span 85'
                else:
                    raise ValueError(f'Unexpected surfactant in {fluid}')
                if len(spl)>2:
                    self.surfactantWt=float(spl[2])
                else:
                    raise ValueError(f'Missing surfactant wt in {fluid}')
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
            if self.base=='silicone oil':
                self.type = f'{self.base}_{self.surfactant}_{self.surfactantWt}'
            else:
                self.type = f'{self.base}_{self.surfactant}'
        else:
            self.type = self.base
        if properties:
            self.properties = True
            self.findRhe()
            self.findDensity()
        else:
            self.properties = False
        
    def metarow(self, tag:s='') -> Tuple[dict,dict]:
        '''row containing metadata'''
        mlist = ['shortname', 'days', 'rheModifier', 'surfactant', 'surfactantWt', 'dye', 'var', 'val', 'base', 'type']
        meta = [[f'{tag}{i}',getattr(self,i)] for i in mlist]  # metadata
        munits = [[f'{tag}{i}', ''] for i in mlist]            # metadata units
        
        if hasattr(self, 'rheUnits'):
            rhelist = ['tau0', 'eta0']             
            rhe = [[f'{tag}{i}',getattr(self,i) if hasattr(self, i) else ''] for i in rhelist]        # rheology data
            rheunits = [[f'{tag}{i}', self.rheUnits[i]] for i in rhelist] # rheology units
        else:
            rhe = []
            rheunits = []
        if hasattr(self, 'constUnits'):
            clist = self.constUnits.keys()
            const = [[f'{tag}{i}',getattr(self,i) if hasattr(self, i) else ''] for i in clist]           # constants data
            constunits = [[f'{tag}{i}', self.constUnits[i]] for i in clist]  # constants units
        else:
            const = []
            constunits = []
        out = dict(meta+rhe+const)
        units = dict(munits+rheunits+constunits)
        units[f'{tag}val'] = self.var
        return out,units
    
    def findRheTable(self, table:str) -> int:
        if not os.path.exists(table):
            logging.error(f'No rheology table found: {table}')
            return
        rhe = pd.read_excel(table)
        rhe = rhe.fillna('') 
        criterion = rhe.rheWt==self.val
        for s in ['base', 'rheModifier', 'surfactant', 'surfactantWt', 'dye', 'days']:
            if s in rhe:
                criterion = criterion&(rhe[s]==getattr(self, s))
        entry = rhe[criterion]
        if len(entry)==0:
            return 1
        if len(entry)>1:
            print(entry)
            logging.error(f'Multiple rheology fits found for fluid {self.shortname}')
        entry = entry.iloc[0]
        self.tau0 = entry['y2_tau0'] # Pa, use the 2% offset criterion
        self.k = entry['y2_k'] 
        self.n = entry['y2_n']
        self.eta0 = entry['y2_eta0'] # these values are for Pa.s, vs. frequency in 1/s
        self.rheUnits = {'tau0':'Pa', 'k':'Pa*s^n', 'n':'','eta0':'Pa*s'}
        return 0
            
    def findRhe(self) -> dict:
        '''find Herschel-Bulkley fit from file'''
        for table in cfg.path.rheTable.values():
            out = self.findRheTable(table)
            if out==0:
                return
        logging.error(f'No rheology fit found for fluid {self.shortname}')
        return
    
    def findDensityTable(self, table:str) -> int:
        if not os.path.exists(table):
            logging.error(f'No density table found: {table}')
            return 1
        tab = pd.read_excel(table)
        tab = tab.fillna('') 
        
        criterion = tab.rheWt==self.val
        for s in ['base', 'rheModifier', 'surfactant', 'surfactantWt']:
            if s in tab:
                criterion = criterion&(tab[s]==getattr(self, s))
        entry = tab[criterion]
        if len(entry)==0:
            return 1
        if len(entry)>1:
            print(entry)
            logging.error(f'Multiple rheology fits found for fluid {self.shortname}')
        entry = entry.iloc[0]
        self.density = entry['density'] # g/mL
        return 0
    
    def findDensity(self) -> dict:
        '''find density from file'''
        for table in cfg.path.densityTable.values():
            out = self.findDensityTable(table)
            if out==0:
                return
        logging.error(f'No density fit found for fluid {self.shortname}')
        return
    
    def visc(self, gdot:float) -> float:
        '''get the viscosity of the fluid in Pa*s at shear rate gdot in Hz'''
        if gdot<=0:
            return -1
        if not hasattr(self, 'k') or not hasattr(self, 'n') or not hasattr(self, 'tau0') or not hasattr(self, 'eta0'):
            return -1
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
        self.v = v  # mm/s
        self.rate = v/diam                                      # 1/s (approximation, gives scaling, but shear rate actually spatially varies b/c of non-Newtonian)
        if self.properties:
            self.visc0 = self.visc(self.rate)                       # Pa*s
            self.CaInv = sigma/(self.visc0*self.v)                  # capillary number ^-1
            self.Re = 10**-3*(self.density*self.v*diam)/(self.visc0) # reynold's number
            self.WeInv = 10**3*sigma/(self.density*self.v**2*diam)  # weber number ^-1
            self.OhInv = np.sqrt(self.WeInv)*self.Re                # Ohnesorge number^-1
            self.dPR = sigma/self.tau0                              # characteristic diameter for Plateau rayleigh instability in mm
            self.Bm = self.tau0*diam/(self.visc0*self.v)            # Bingham number
        self.constUnits = {'density':'g/mL', 'v':'mm/s','rate':'1/s','visc0':'Pa*s', 'CaInv':'','Re':'','WeInv':'','OhInv':'','dPR':'mm', 'dnormInv':'', 'Bm':''}