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
import time

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
from tools.config import cfg
from tools.plainIm import *
import file.file_handling as fh
from val.v_fluid import fluidVals
from val.v_pressure import pressureVals
from val.v_geometry import geometryVals
from val.v_tables import valTables

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)



#----------------------------------------------


class printVals:
    '''class that holds info about the experiment'''
    
    def __init__(self, folder:str, fluidProperties:bool=True, **kwargs0):
        '''get the ink and support names from the folder name. 
        vink and vsup are in mm/s. 
        di (inner nozzle diameter) and do (outer) are in mm'''
        kwargs = kwargs0.copy()
        self.fluidProperties = fluidProperties
        tic = time.perf_counter()
        self.printFolder = folder
        self.printFolderR = os.path.relpath(folder, cfg.path.server)
        if 'pfd' in kwargs:
            self.pfd = kwargs['pfd']
        else:
            self.pfd = fh.printFileDict(self.printFolder)
        if 'valTable' in kwargs:
            self.valTable = kwargs.pop('valTable')
        else:
            self.valTable = valTables(self.pfd.printType)
            
        # get the sample folder
        self.findSampleName()
        self.date = self.pfd.date
        
        if len(self.pfd.timeSeries)>0:
            self.fluFile = True
        else:
            self.fluFile = False

        self.constUnits = {}
        self.pxpmm = self.pfd.pxpmm()
        

        self.constUnits['date'] = 'yymmdd'
        
        self.ink = fluidVals(self.inkShortName, 'ink', properties=fluidProperties, valTable=self.valTable, **kwargs)
        self.sup = fluidVals(self.supShortName, 'sup', properties=fluidProperties, valTable=self.valTable, **kwargs)
        
        if self.pfd.printType in ['tripleLine', 'singleDisturb', 'SDT']:
            split = re.split('_', os.path.basename(self.pfd.printFolder))
            if len(split)>1:
                if 'disturbXS' in self.printFolder:
                    self.spacing = float(split[3])
                else:
                    self.spacing = float(split[2])
                self.constUnits['spacing'] = '$d_i$'
            else:
                return
        self.press = pressureVals(self.printFolder, pfd=self.pfd)
        self.geo = geometryVals(self.printFolder, pfd=self.pfd)
        self.tension()
        self.ink.constants(self.press.vink, self.geo.di, self.sigma)
        self.sup.constants(self.press.vsup, self.geo.do, self.sigma)
        self.const()
        
    def findSampleName(self):
        fi = self.printFolder
        fi0 = ''
        while not fh.sampleInName(fi) and not fi0==fi:
            fi0 = fi
            fi = os.path.dirname(fi)
        if fi0==fi:
            # no sample folder found
            for f in os.listdir(self.printFolder):
                if '_I_' in f and '_S_' in f:
                    spl = re.split('I_', f)[1]
                    spl = re.split('_', spl)
                    self.inkShortName = spl[0]
                    self.supShortName = spl[2]
                    self.bn = f'I_{self.inkShortName}_S_{self.supShortName}'
                    return
        else:
            # sample folder found
            self.bn = os.path.basename(os.path.dirname(fi))
            split = re.split('_', self.bn)
            self.inkShortName = split[1]
            self.supShortName = split[3]
  
    def const(self) -> None:
        '''define dimensionless numbers and critical values'''
        self.vRatio = self.ink.v/self.sup.v 
        self.constUnits['vRatio'] = ''
        self.dEst = self.geo.di*np.sqrt(self.vRatio)
            # expected filament diameter in mm
        self.constUnits['dEst'] = 'mm'
            
        if self.fluidProperties:
            # viscosity ratio
            self.viscRatio = self.ink.visc0/self.sup.visc0   # local viscosity ratio
            self.tau0aRatio = self.ink.tau0a/self.sup.tau0a
            self.tau0dRatio = self.ink.tau0d/self.sup.tau0d
            self.GaRatio = self.ink.Gstora/self.sup.Gstora
            self.GdRatio = self.ink.Gstord/self.sup.Gstord
            self.GtaRatio = self.ink.Gstora/self.sup.tau0a
            self.tGdRatio = self.ink.tau0d/self.sup.Gstord
            for s in ['viscRatio', 'tau0aRatio', 'tau0dRatio', 'GaRatio', 'GdRatio', 'GtaRatio', 'tGdRatio']:
                self.constUnits[s] = ''
            # velocity ratio

            # crit radius  
            ddiff = self.ink.density-self.sup.density
            for dire in ['a', 'd']:
                if abs(ddiff)>0:
                    setattr(self, f'rGrav{dire}', 10**6*(getattr(self.sup, f'tau0{dire}'))/((ddiff)*9.8)) 
                else:
                    setattr(self, f'rGrav{dire}', 0) 
                    # characteristic sagging radius in mm, missing scaling factor, see O'Brien MRS Bulletin 2017
                self.constUnits[f'rGrav{dire}'] = 'mm'

                
                setattr(self.ink, f'dnormInv{dire}', getattr(self.ink, f'dPR{dire}')/self.dEst)  # keep this inverted to avoid divide by 0 problems
                setattr(self.sup, f'dnormInv{dire}', getattr(self.sup, f'dPR{dire}')/self.dEst)
                for fluid in [self.ink, self.sup]:
                    fluid.constUnits[f'dnormInv{dire}'] = ''

            # Reynolds number
            self.int_Re = 10**-3*(self.ink.density*self.ink.v*self.geo.di)/(self.sup.visc0)
            self.constUnits['int_Re'] = ''
            self.ReRatio = self.ink.Re/self.sup.Re
            self.constUnits['ReRatio'] = ''

            # drag
            l = self.geo.lBath
            dn = 2*np.sqrt(self.geo.do*l/np.pi)
            Kn = 1/3+2/3*np.sqrt(np.pi)/2
            self.hDragP = 3*np.pi*self.sup.visc0*self.sup.v*(dn*Kn)/(self.geo.do*l)
                # horizontal line stokes drag pressure
            self.vDragP = 3*self.sup.visc0*self.sup.v*4/(self.geo.do)
                # vertical line stokes drag pressure
            self.constUnits['hDragP'] = 'Pa'
            self.constUnits['vDragP'] = 'Pa'

            # capillary number    
            self.int_CaInv = self.sigma/(self.sup.visc0*self.ink.v)
            self.constUnits['int_CaInv'] = ''
           
    def metarow(self) -> Tuple[dict,dict]:
        '''row holding metadata'''
        mlist = ['printFolderR', 'bn', 'date', 'sigma', 'fluFile']
        meta = dict([[i,getattr(self,i)] for i in mlist])
        munits = dict([[i, self.constUnits.get(i, '')] for i in mlist])
        meta['calibFile'] = self.press.calibFile
        munits['calibFile'] = ''
        clist = self.constUnits.keys()
        const = dict([[i,getattr(self,i)] for i in clist])
        cunits = dict([[i, self.constUnits[i]] for i in clist])
        
        pvals, punits = self.press.metarow()
        inkvals, inkunits = self.ink.metarow('ink_')
        supvals, supunits = self.sup.metarow('sup_')
        out = {**meta, **const, **pvals, **inkvals, **supvals}
        units = {**munits, **cunits, **punits, **inkunits, **supunits}
        return out, units
    
    def value(self, varfunc:str, var:str) -> float:
        '''get the value of a given function. var is x or y'''
        split = re.split('\.', varfunc)
        if 'ink' in varfunc or 'sup' in varfunc:
            fluid = split[0]
            valname = split[1]
            fobject = getattr(self, fluid)
            if valname=='var':
                valname = 'val'
            value = getattr(fobject, valname)
        elif 'self' in varfunc:
            value = getattr(self, split[1])
        else:
            raise ValueError(f'Value not found for {var}: {varfunc}')
        setattr(self, f'{var}val', value)
        return value
        
    def label(self, varfunc:str) -> float:
        split = re.split('\.', varfunc)
        if 'ink' in varfunc or 'sup' in varfunc:
            fluid = split[0]
            valname = split[1]
            fobject = getattr(self, fluid)
            if valname=='var' or valname=='val':
                return f'{fluid} {fobject.var}'
            elif valname=='v':
                return f'{fluid} speed (mm/s)'
            else:
                return ''
        elif 'self' in varfunc:
            valname = split[1]
            return f'{valname} ({self.constUnits[valname]})'

    def base(self) -> str:
        '''get the plot title'''
#         if vname in ['val', 'v']:
#             self.xval = getattr(getattr(self, xfluid),vname)
#             self.yval = getattr(getattr(self, yfluid),vname)
        base = f'{self.ink.base}, {self.sup.base}'
        return base
    
    def tension(self) -> float:
        '''pull the surface tension from a table'''
        sigt = self.valTable.sigmaDF()
        criterion = sigt.ink_base==self.ink.base
        for s in ['sup_base', 'ink_surfactant', 'ink_surfactantWt', 'sup_surfactant']:
            if s in sigt:
                spl = re.split('_', s)
                criterion = criterion&(sigt[s]==getattr(getattr(self, spl[0]), spl[1]))
        entry = sigt[criterion]
        if len(entry)==0:
            print(sigt)
            logging.error(f'No surface tension fit found for fluid {self.bn}')
            return
        if len(entry)>1:
            logging.error(f'Multiple surface tension fits found for fluid {self.bn}')
        entry = entry.iloc[0]
        self.sigma = entry['sigma'] # mN/m
        self.constUnits['sigma'] = 'mN/m'
        return self.sigma
    
    def vidFile(self) -> str:
        '''get the path of the video file taken during the print'''
        return self.pfd.vidFile()   
    

                
class pvSingle(printVals):
    '''class that holds info about single line experiments'''
    
    def __init__(self, folder:str, di:float=cfg.const.di, do:float=cfg.const.do):
        super(pvSingle, self).__init__(folder, di=di, do=do)
        self.units = {'bn':'','folder':'','date':'YYMMDD','di':'mm','do':'mm', 'sigma':'mN/m', 'fluFile':'', 'calibFile':''}
        if '3day' in folder:
            # 3 days
            self.ink.days = 3
            self.sup.days = 3
            self.ink.findRhe()
            self.sup.findRhe()
            
    def summaryFile(self) -> str:
        return os.path.join(self.printFolder, f'{os.path.basename(self.printFolder)}_summary.csv')

        
    def vertSum(self) -> Tuple[dict, dict]:
        '''summarize all vertical lines'''
        vertfile = os.path.join(self.printFolder, f'{self.bn}_vertSummary.csv')
        if not os.path.exists(vertfile):
            return {}, {}
        tab, units = plainIm(vertfile, ic=0)
        if len(tab)==0:
            return {},{}
        t1 = []
        if 15 in list(self.progDims.l):
            # using default timings. throw this out
            return {},{}
        for i,row in tab.iterrows():
            progrow = self.progDims[self.progDims.name=='vert'+str(int(row['line']))]
            if len(progrow)==0:
                raise ValueError('Line name not found in progDims')
            progrow = progrow.iloc[0]
            t2 = {}
#             for s in ['w', 'h', 'vintegral', 'meanT']:
#                 t2[s] = row[s]
            t2['wN'] = row['w']/self.pxpmm/progrow['w'] # convert to mm, divide by intended dimension
            t2['hN'] = row['h']/self.pxpmm/progrow['l'] # convert to mm
            t2['vN'] = row['vest']/self.pxpmm**3/progrow['vol'] # convert to mm^3. vN is the estimated volume of the bounding box
            t2['vintegral'] = row['vintegral']/self.pxpmm**3
            t2['viN'] = t2['vintegral']/progrow['vol'] # convert to mm^3. viN is the estimated volume by integrating over the length of the line
            t2['vleak'] = row['vleak']/self.pxpmm**3
            t2['vleakN'] = t2['vleak']/(progrow['a']*(15-progrow['l'])) 
                # convert to mm^3. viN is the estimated volume by integrating past the programmed length of the line, 
                # normalized by the remaining allowed volume after flow stops
            t2['roughness'] = row['roughness']
            t2['meanTN'] = row['meanT']/self.pxpmm/progrow['w'] # convert to mm
            t2['stdevTN'] = row['stdevT']/self.pxpmm/progrow['w'] # convert to mm
            t2['minmaxTN'] = row['minmaxT']/self.pxpmm/progrow['w'] # convert to mm
            t1.append(t2)
        t1 = pd.DataFrame(t1)
        t3 = [[f'vert_{s}', t1[s].mean()] for s in t1] # averages
        t4 = [[f'vert_{s}_SE', t1[s].sem()] for s in t1] # standard errors
        t3 = dict(t3+t4)
        units = dict([[s,units[s.replace('vert_','')].replace('px','mm') if s.replace('vert_','') in units else ''] for s in t3.keys()])
        return t3, units
    
    def horizSum(self) -> Tuple[dict, dict]:
        '''summarize all horizontal lines'''
        file = os.path.join(self.printFolder, f'{self.bn}_horizSummary.csv')
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
            t2['maxlenN'] = row['maxlen']/self.pxpmm/progrow['l']
            t2['totlenN'] = row['totlen']/self.pxpmm/progrow['l']
            t2['vN'] = row['vest']/self.pxpmm**3/progrow['vol'] # convert to mm^3
            t2['roughness'] = row['roughness']
            t2['meanTN'] = row['meanT']/self.pxpmm/progrow['w'] # convert to mm
            t2['stdevTN'] = row['stdevT']/self.pxpmm/progrow['w'] # convert to mm
            t2['minmaxTN'] = row['minmaxT']/self.pxpmm/progrow['w'] # convert to mm
            t1.append(t2)
        t1 = pd.DataFrame(t1)
        t3 = [[f'horiz_{s}', t1[s].mean()] for s in t1] # averages
        t4 = [[f'horiz_{s}_SE', t1[s].sem()] for s in t1] # standard errors
        t3 = dict(t3+t4)
        t3['horiz_segments'] = t1.segments.sum()
        t3.pop('horiz_segments_SE')
        units = dict([[s,''] for s in t3.keys()])
        return t3, units
    
    def xsSum(self) -> Tuple[dict, dict]:
        '''summarize all horizontal lines'''
        file = os.path.join(self.printFolder, self.bn+'_xsSummary.csv')
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
            t2['area'] = row['area']/(self.pxpmm**2)
            t2['areaN'] = t2['area']/progrow['a']
            t2['wN'] = row['w']/self.pxpmm/progrow['w']
            t2['hN'] = row['h']/self.pxpmm/progrow['w']
            t2['roughness'] = row['roughness']
            t1.append(t2)
        t1 = pd.DataFrame(t1)
        t3 = [[f'xs_{s}', t1[s].mean()] for s in t1] # averages
        t4 = [[f'xs_{s}_SE', t1[s].sem()] for s in t1] # standard errors
        t3 = dict(t3+t4)
        units['area']='mm^2'
        units = dict([[s,units[s.replace('xs_','')] if s.replace('xs_','') in units else ''] for s in t3.keys()])
        return t3, units
    
    def summary(self) -> Tuple[dict,dict]:
#         logging.info(self.printFolder)
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
    
    def exportSummary(self, out:dict, outunits:dict) -> None:
        '''export a summary file based on expected values'''
        file = self.summaryFile()
        with open(file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for k,v in out.items():
                writer.writerow([k, outunits[k], str(v)])
        

    
#------

class pvTriple(printVals):
    '''class that holds info about triple line experiments'''
    
    def __init__(self, folder:str, di:float=cfg.const.di, do:float=cfg.const.do):
        super().__init__(folder, di=di, do=do)
        self.units = {'bn':'','folder':'','date':'YYMMDD','di':'mm','do':'mm', 'sigma':'mN/m', 'fluFile':'', 'calibFile':''}


    def summary(self) -> Tuple[dict,dict]:
        meta,metaunits = self.metarow()
        out = {**meta, **xs}
        outunits = {**metaunits, **xsunits}
        self.exportSummary(out, outunits)
        return out, outunits
    


        