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
from fluidVals import *
import fileHandling as fh
from pressureVals import pressureVals
from geometryVals import geometryVals

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)



#----------------------------------------------


class printVals:
    '''class that holds info about the experiment'''
    
    def __init__(self, folder:str, di:float=cfg.const.di, do:float=cfg.const.do):
        '''get the ink and support names from the folder name. 
        vink and vsup are in mm/s. 
        di (inner nozzle diameter) and do (outer) are in mm'''
        self.printFolder = folder
        self.pfd = printFileDict(self.printFolder)
        self.bn = os.path.basename(folder)
        self.fluFile = False
        self.calibFile = False
        split = re.split('_', self.bn)
        inkShortName = split[1]
        supShortName = split[3]
        self.date = int(split[-1][0:6])
        self.ink = fluidVals(inkShortName, 'ink')
        self.sup = fluidVals(supShortName, 'sup')

        self.press = pressureVals(self.printFolder, pfd=self.pfd)
        self.geo = geometryVals(self.printFolder, pfd=self.pfd)
        self.tension()
        self.ink.constants(self.vink, self.geo.di, self.sigma)
        self.sup.constants(self.vsup, self.geo.do, self.sigma)
        self.const()
        
  
    def const(self) -> None:
        '''define dimensionless numbers and critical values'''
        self.constUnits = {}
        
        # viscosity ratio
        self.viscRatio = self.ink.visc0/self.sup.visc0 
        self.constUnits['viscRatio'] = ''
        # velocity ratio
        self.vRatio = self.ink.v/self.sup.v 
        self.constUnits['vRatio'] = ''
            
        # crit radius  
        ddiff = self.ink.density-self.sup.density
        if abs(ddiff)>0:
            self.rGrav = 10**6*(self.sup.tau0)/((ddiff)*9.8) 
        else:
            self.rGrav = 0
            # characteristic sagging radius in mm, missing scaling factor, see O'Brien MRS Bulletin 2017
        self.constUnits['rGrav'] = 'mm'
        self.dEst = self.geo.di*np.sqrt(self.vRatio)
            # expected filament diameter in mm
        self.constUnits['dEst'] = 'mm'
        self.ink.dnormInv = self.ink.dPR/self.dEst  # keep this inverted to avoid divide by 0 problems
        self.sup.dnormInv = self.sup.dPR/self.dEst
        
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
        mlist = ['folder', 'bn', 'date', 'sigma', 'fluFile', 'calibFile']
        meta = [[i,getattr(self,i)] for i in mlist]
        munits = [[i, self.units[i]] for i in mlist]
        clist = self.constUnits.keys()
        const = [[i,getattr(self,i)] for i in clist]
        cunits = [[i, self.constUnits[i]] for i in clist]
        
        pvals, punits = self.press.metarow()
        inkvals, inkunits = self.ink.metarow('ink_')
        supvals, supunits = self.sup.metarow('sup_')
        out = {**meta, **const, **pvals, **inkvals, **supvals}
        units = {**munits, **cunits, **punits, **inkunits, **supunits}
        return out, units
        
    def base(self, xfluid:str, yfluid:str, vname:str='val') -> str:
        '''get the plot title'''
        self.xval = getattr(getattr(self, xfluid),vname)
        self.yval = getattr(getattr(self, yfluid),vname)
        xbase = getattr(self, xfluid).base
        ybase = getattr(self, yfluid).base
        base = f'{xbase}, {ybase}'
        return base
    
    def tension(self) -> float:
        '''pull the surface tension from a table'''
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
        return self.pfd.vidFile()        
    
    #--------------------------------------------------
    

                
#     def findSpeedFile(self) -> str:
#         '''find the speed file in this folder'''
#         if hasattr(self.pfd, 'meta'):
#             return self.pfd.meta
#         else:
#             return ''
    
#     def readSpeedFile(self, file:str) -> None:
#         '''read values from the speed file'''
#         correctSpeedFile=False
#         with open(file, newline='') as csvfile:
#             reader = csv.reader(csvfile, delimiter=',', quotechar='|')
#             self.targetPressures = {}
#             for row in reader:
#                 if 'pressure' in row[0]:
#                     # for each channel, get target pressure
#                     channel = re.split('channel ', row[0])[-1]
#                     self.targetPressures[int(channel)] =  float(row[2])
#                 elif 'ink speed' in row:
#                     self.vink = float(row[2])
#                 elif 'support speed' in row:
#                     self.vsup = float(row[2])
#                 elif 'caliba' in row and self.caliba==0:
#                     self.caliba = float(row[2])
#                     self.calibFile=True
#                 elif 'calibb' in row or ('caliba' in row and not self.caliba==0):
#                     self.calibb = float(row[2])
#                     if not 'calibb' in row:
#                         correctSpeedFile = True # there was an error in the original file. re-export correct labels
#                 elif 'calibc' in row or ('caliba' in row and not self.calibb==0):
#                     self.calibc = float(row[2])
#                     if not 'calibc' in row:
#                         correctSpeedFile = True # there was an error in the original file. re-export correct labels
#         if correctSpeedFile:
#             self.exportSpeedFile()
            
    
#     def readMeta(self) -> None:
#         '''read metadata into the object'''
#         self.meta= {}
#         if not hasattr(self.pfd, 'meta'):
#             return
#         with open(self.pfd.meta, newline='') as csvfile:
#             reader = csv.reader(csvfile, delimiter=',', quotechar='|')
#             for row in reader:
#                 key = row[0]
#                 unit = row[1]
#                 val = row[2]
#                 # handle duplicate keys
#                 if key in self.meta:
#                     ii = 1
#                     while f'{key}_{ii}' in self.meta:
#                         ii+=1
#                 self.meta[key] = {'unit':unit, 'val':val}   # store value
#                 for s in ['a', 'b', 'c']:
#                     if key==f'channel0_calib{s}' or key==f'calib{s}':
#                         setattr(self, f'calib{s}', value)        
#                 if key=='ink speed':
#                     self.vink = value
#                 if key=='support speed':
#                     self.vsup = value
#                 if 'pressure' in key:
#                     # for each channel, get target pressure
#                     channel = re.split('channel ', key)[-1]
#                     self.targetPressures[int(channel)] =  float(row[2])
#         if not ('caliba' in self.meta and 'calibb' in self.meta and 'calibc' in self.meta):
#             # include calib values in the file if they're not in there
#             self.exportSpeedFile()


        
#     def findVelocities(self) -> None:
#         '''find the target velocities'''
#         self.targetPressures = {0:0}
#         self.caliba = 0
#         self.calibb = 0
#         self.calibc = 0
        
#         file = self.findSpeedFile() # file that holds metadata about speed for this run
#         if os.path.exists(file):
#             self.readSpeedFile(file) # get target pressure, ink speed, and sup speed from speed file
#             self.calibFile = True 
#         else:
#             self.vink = cfg.const.vink    # use default ink and sup speed
#             self.vsup = cfg.const.vsup
#             self.targetPressures = {0:0}
#             self.readCalibFile()
#             self.exportSpeedFile()
            

    
    
    def summaryFile(self) -> str:
        return os.path.join(self.printFolder, f'{os.path.basename(self.printFolder)}_summary.csv')
    
    def exportSummary(self, out:dict, outunits:dict) -> None:
        '''export a summary file based on expected values'''
        file = self.summaryFile()
        with open(file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for k,v in out.items():
                writer.writerow([k, outunits[k], str(v)])
                
                
                
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
        
        
#     def redoSpeedFile(self) -> None:
#         '''read calibration curve from calibration file and add to speed file. version control for shopbotpyqt files before fall 2021'''
#         self.readCalibFile(justCalib=True)
#         if self.calibb==0:
#             # if there is no calibration curve, assume speed = pressure * (intended speed)/(max pressure)
#             try:
#                 ftable = self.importFluFile()
#             except:
#                 # no fluigent file. not enough info to create speed file
#                 return
#             self.targetPressures[0] = ftable.pressure.max()
#             self.calibb = self.vink/self.targetPressures[0]
#         self.exportSpeedFile()
        
        
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
            t2['wN'] = row['w']/cfg.const.pxpmm/progrow['w'] # convert to mm, divide by intended dimension
            t2['hN'] = row['h']/cfg.const.pxpmm/progrow['l'] # convert to mm
            t2['vN'] = row['vest']/cfg.const.pxpmm**3/progrow['vol'] # convert to mm^3. vN is the estimated volume of the bounding box
            t2['vintegral'] = row['vintegral']/cfg.const.pxpmm**3
            t2['viN'] = t2['vintegral']/progrow['vol'] # convert to mm^3. viN is the estimated volume by integrating over the length of the line
            t2['vleak'] = row['vleak']/cfg.const.pxpmm**3
            t2['vleakN'] = t2['vleak']/(progrow['a']*(15-progrow['l'])) 
                # convert to mm^3. viN is the estimated volume by integrating past the programmed length of the line, 
                # normalized by the remaining allowed volume after flow stops
            t2['roughness'] = row['roughness']
            t2['meanTN'] = row['meanT']/cfg.const.pxpmm/progrow['w'] # convert to mm
            t2['stdevTN'] = row['stdevT']/cfg.const.pxpmm/progrow['w'] # convert to mm
            t2['minmaxTN'] = row['minmaxT']/cfg.const.pxpmm/progrow['w'] # convert to mm
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
            t2['maxlenN'] = row['maxlen']/cfg.const.pxpmm/progrow['l']
            t2['totlenN'] = row['totlen']/cfg.const.pxpmm/progrow['l']
            t2['vN'] = row['vest']/cfg.const.pxpmm**3/progrow['vol'] # convert to mm^3
            t2['roughness'] = row['roughness']
            t2['meanTN'] = row['meanT']/cfg.const.pxpmm/progrow['w'] # convert to mm
            t2['stdevTN'] = row['stdevT']/cfg.const.pxpmm/progrow['w'] # convert to mm
            t2['minmaxTN'] = row['minmaxT']/cfg.const.pxpmm/progrow['w'] # convert to mm
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
            t2['area'] = row['area']/(cfg.const.pxpmm**2)
            t2['areaN'] = t2['area']/progrow['a']
            t2['wN'] = row['w']/cfg.const.pxpmm/progrow['w']
            t2['hN'] = row['h']/cfg.const.pxpmm/progrow['w']
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
        

    
#------

class pvTriple(printVals):
    '''class that holds info about triple line experiments'''
    
    def __init__(self, folder:str, di:float=cfg.const.di, do:float=cfg.const.do):
        super(pvSingle, self).__init__(folder, di=di, do=do)
        self.units = {'bn':'','folder':'','date':'YYMMDD','di':'mm','do':'mm', 'sigma':'mN/m', 'fluFile':'', 'calibFile':''}


    def summary(self) -> Tuple[dict,dict]:
        meta,metaunits = self.metarow()
        out = {**meta, **xs}
        outunits = {**metaunits, **xsunits}
        self.exportSummary(out, outunits)
        return out, outunits
    
    
    
    
#--------------------

class progDim:
    '''class that holds timing for the video'''
    
    def __init__(self, printFolder:str, pv:printVals, **kwargs):
        self.printFolder = printFolder
        self.pv = pv
        
        # dictionary of files in the printFolder
        self.pfd = pv.pfd
        self.geo = pv.geo
        self.press = pv.press
            
        self.sbp = self.pfd.sbpName()    # name of shopbot file
        self.progDims = pd.DataFrame(columns=['name','l','w','t','a','vol', 't0','tf'])
        self.units = {'name':'', 'l':'mm','w':'mm','t':'s'
                      ,'a':'mm^2','vol':'mm^3','t0':'s', 'tf':'s'}
        
        
    def initializeProgDims(self):
        '''initialize programmed dimensions table'''

        if 'singleLinesNoZig' in self.sbp:
            self.progDims.name=['xs5','xs4','xs3','xs2','xs1',
                                'vert4','vert3','vert2','vert1',
                                'horiz2','horiz1','horiz0']
        elif 'crossDoubleVert_0.5' in self.sbp:
            self.progDims.name = ['v00', 'v01', 'v10', 'v11', 'zig']
        elif 'crossDoubleVert_0' in self.sbp:
            self.progDims.name = ['v00', 'v01'
                                  , 'h00', 'h01', 'h02', 'h03', 'h04', 'h05'
                                  , 'v10', 'v11'
                                 , 'h10', 'h11', 'h12', 'h13', 'h14', 'h15']
        elif 'crossDoubleHoriz_0.5' in self.sbp:
            self.progDims.name = ['zig', 'v0', 'v1', 'v2', 'v3']
        elif 'crossDoubleVert_0' in self.sbp:
            self.progDims.name = ['hz0', 'hz1', 'hz2'
                                 , 'hc00', 'hc01', 'hc02', 'hc03'
                                 , 'hc10', 'hc11', 'hc12', 'hc13'
                                 , 'hc20', 'hc21', 'hc22', 'hc23']
        elif 'underCross_0.5' in self.sbp:
            self.progDims.name = ['z1', 'z2']
        elif 'underCross_0' in self.sbp:
            self.progDims.name = ['z1'
                                 , 'v00', 'v01', 'v02', 'v03'
                                 , 'v10', 'v11', 'v12', 'v13'
                                 , 'v20', 'v21', 'v22', 'v23']
        elif 'tripleLines' in self.sbp:
            self.progDims.name = ['l00', 'l01', 'l02'
                                 ,'l10', 'l11', 'l12'
                                 ,'l20', 'l21', 'l22'
                                 ,'l30', 'l31', 'l32']
            
            
    def findTimeFile(self) -> str:
        '''get the name of the pressure-time table'''
        if len(self.pfd.timeSeries)==0:
            return ''
        if len(self.pfd.timeSeries)==1:
            return self.pfd.timeSeries[0]
        if len(self.pfd.timeSeries)>1:
            l = []
            for f in self.pfd.timeSeries:
                if 'singleLinesN' in f and 'Fluigent' in f:
                    l.append(f)
            if len(l)==0:
                raise NameError(f'No fluigent file found in {self.printFolder}')
            l.sort()
            file = os.path.join(self.printFolder, l[-1]) # select last fluigent file
            return file

    def importTimeFile(self) -> pd.DataFrame:
        '''find and import the pressure-time table'''
        file = self.findTimeFile()
        if not os.path.exists(file):
            return []
        
        ftable = pd.read_csv(file)
        ftable.rename(columns={'time (s)':'time', 'time(s)':'time', 'Channel 0 pressure (mbar)':'pressure', 'Channel_0_pressure(mbar)':'pressure', 'x(mm)':'x', 'y(mm)':'y', 'z(mm)':'z'}, inplace=True)
        return ftable

                
    def useDefaultTimings(self):
        '''use the programmed line lengths'''
        self.initializeProgDims()
        if 'singleLinesNoZig' in self.sbp:
            for s,row in self.progDims.iterrows():
                if 'xs' in row['name']:
                    self.progDims.loc[s,'l']=15         # intended filament length in mm
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
        self.progDims.iloc[i]['tf'] = vals['tf']
        self.progDims.iloc[i]['t0'] = vals['t0']
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
        a = np.pi*(self.pv.dEst/2)**2 # ideal cross-sectional area
        anoz = np.pi*(self.geo.di/2)**2 # inner cross-sectional area of nozzle
        for j,row in df.iterrows():
            if row['pressure']>0:
                if ttot==0:
                    t0 = row['time']
                # flow is on
                if j>0:
                    dt = (row['time']-df.loc[j-1,'time']) # timestep size
                else:
                    dt = (-row['time']+df.loc[j+1,'time']) 
                ttot = ttot+dt # total time traveled
                if self.date<=210929 and 'vert' in self.progDims.loc[i,'name']:
                    # bug in initial code wasn't changing vertical velocity
                    dl = dt*5
                else:
                    dl = dt*self.sup.v # length traveled in this timestep
                l = l+dl  # total length traveled
                if dt==0:
                    dvol=0
                else:
                    p = row['pressure']
                    vcalc = self.press.calculateSpeed(p)
                    if not vcalc==0:
                        flux = max(vcalc*anoz,0)
                        # actual flow speed based on calibration curve (mm/s) * area of nozzle (mm^2) = flux (mm^3/s)
                    else:
                        flux = p/self.press.targetPressure*anoz*self.ink.v
                    dvol = flux*dt 
                vol = vol+dvol # total volume extruded
            else:
                # flow is off
                if l>0:
                    self.storeProg(i, {'l':l, 'vol':vol, 'ttot':ttot, 'a':a, 't0':t0, 'tf':row['time']})
                    ttot=0
                    vol=0
                    l=0
                    i = i+1
        if i<len(self.progDims):
            self.storeProg(i, {'l':l, 'vol':vol, 'ttot':ttot, 'a':a, 't0':t0, 'tf':row['time']})
        
        # start times at 0
        self.progDims.tf = self.progDims.tf-self.progDims.t0.min()
        self.progDims.t0 = self.progDims.t0-self.progDims.t0.min()
        
    
    
    def fluigent(self) -> None:
        '''get lengths of actual extrusion from fluigent'''
        try:
            ftable = self.importTimeFile()
            self.fluFile = True
        except:
            self.useDefaultTimings()
            self.fluFile = False
        else:
            if len(self.targetPressures)==0 or self.targetPressures[0]==0:
                self.targetPressures[0] = ftable.pressure.max()
            self.readProgDims(ftable, self.targetPressures[0])
            
                    
    def progDimsFile(self) -> str:
        bn = os.path.basename(self.printFolder)
        return os.path.join(self.printFolder, f'{bn}_progDims.csv')
    
    def importProgDims(self, overwrite:bool=False) -> str:
        '''import the progdims file'''
        fn = self.progDimsFile()
        if not os.path.exists(fn) or overwrite:
            self.fluigent()
            self.exportProgDims()
        else:
            self.progDims, self.units = plainIm(fn, ic=0)
            if not 0 in list(self.progDims.vol):
                self.fluFile=True
        return self.progDims, self.units
            
    def exportProgDims(self) -> None:
        plainExp(self.progDimsFile(), self.progDims, self.units)
        
    def progDimsSummary(self) -> Tuple[pd.DataFrame,dict]:
        if len(self.progDims)==0:
            self.importProgDims()
        if len(self.progDims)==0:
            return {},{}
        if 0 in list(self.progDims.vol):
            # redo programmed dimensions if there are zeros in the volume column
            self.redoSpeedFile()
            self.fluigent
            self.exportProgDims()
        df = self.progDims.copy()
        df2 = pd.DataFrame(columns=df.columns)
        for s in ['xs', 'vert', 'horiz']:
            df2.loc[s] = df[df.name.str.contains(s)].mean()
        df2 = df2.drop(columns=['name'])
        df2 = df2.T
        v = (df2).unstack().to_frame().sort_index(level=0).T
        v.columns = v.columns.map('_'.join) # single row dataframe
        vunits = dict([[k, self.units[re.split('_', k)[1]]] for k in v]) # units for this dataframe
        for s in ['bn', 'vink', 'vsup', 'sigma']:
            v.loc[0, s] = getattr(self, s)
        v.loc[0, 'pressure'] = self.targetPressures[0]
        vunits['bn']=''
        vunits['vink']='mm/s'
        vunits['vsup']='mm/s'
        vunits['sigma']='mJ/m^2'
        vunits['pressure']='mbar'
        return v,vunits
