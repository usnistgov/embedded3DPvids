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

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)



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
        meta = [[f'{tag}{i}',getattr(self,i)] for i in mlist]  # metadata
        munits = [[f'{tag}{i}', ''] for i in mlist]            # metadata units
        
        rhelist = ['tau0', 'eta0']             
        rhe = [[f'{tag}{i}',getattr(self,i)] for i in rhelist]        # rheology data
        rheunits = [[f'{tag}{i}', self.rheUnits[i]] for i in rhelist] # rheology units
        clist = self.constUnits.keys()
        const = [[f'{tag}{i}',getattr(self,i)] for i in clist]           # constants data
        constunits = [[f'{tag}{i}', self.constUnits[i]] for i in clist]  # constants units
        out = dict(meta+rhe+const)
        units = dict(munits+rheunits+constunits)
        units[f'{tag}val'] = self.var
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
        self.tau0 = entry['y2_tau0'] # Pa, use the 2% offset criterion
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
        self.Re = 10**-3*(self.density*self.v*diam)/(self.visc0) # reynold's number
        self.WeInv = 10**3*sigma/(self.density*self.v**2*diam)  # weber number ^-1
        self.OhInv = np.sqrt(self.WeInv)*self.Re                # Ohnesorge number^-1
        self.dPR = sigma/self.tau0                              # characteristic diameter for Plateau rayleigh instability in mm
        self.Bm = self.tau0*diam/(self.visc0*self.v)            # Bingham number
        self.constUnits = {'density':'g/mL', 'v':'mm/s','rate':'1/s','visc0':'Pa*s', 'CaInv':'','Re':'','WeInv':'','OhInv':'','dPR':'mm', 'dnormInv':'', 'Bm':''}