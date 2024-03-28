#!/usr/bin/env python
'''holds data and functions for handling metric summary tables for single lines'''

# external packages
import os, sys
import traceback
import logging
import pandas as pd
from matplotlib import pyplot as plt
from typing import List, Dict, Tuple, Union, Any, TextIO
import re
import numpy as np
import cv2 as cv
import shutil
import subprocess

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
sys.path.append(os.path.dirname(os.path.dirname(currentdir)))
from summary_metric import *
from m_tools import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)

#--------------------------------

class summarySingle(summaryMetric):
    '''holds data and functions for handling metric summary tables for single lines'''
    
    def __init__(self, file:str):
        super().__init__(self, file)
        
    def importStillsSummary(file:str='stillsSummary.csv', diag:bool=False) -> pd.DataFrame:
        '''import the stills summary and convert sweep types, capillary numbers'''
        self.file = os.path.join(cfg.path.fig, file)
        ss,u = plainIm(self.file, ic=0)

        ss = ss[ss.date>210500]       # no good data before that date
        ss = ss[ss.ink_days==1]       # remove 3 day data
        ss.date = ss.date.replace(210728, 210727)   # put these dates together for sweep labeling
        k = ss.keys()
        k = k[~(k.str.contains('_SE'))]
        k = k[~(k.str.endswith('_N'))]
        idx = self.idx0(k)
        controls = k[:idx]
        deps = k[idx:]
        self.flipInv()
        ss.insert(idx+2, 'sweepType', ['visc_'+self.fluidAbbrev(row) for j,row in ss.iterrows()])
        ss.loc[ss.bn.str.contains('I_3.50_S_2.50_VI'),'sweepType'] = 'speed_W_high_visc_ratio'
        ss.loc[ss.bn.str.contains('I_2.75_S_2.75_VI'),'sweepType'] = 'speed_W_low_visc_ratio'
        ss.loc[ss.bn.str.contains('I_3.00_S_3.00_VI'),'sweepType'] = 'speed_W_int_visc_ratio'
        ss.loc[ss.bn.str.contains('VI_10_VS_5_210921'), 'sweepType'] = 'visc_W_high_v_ratio'
        ss.loc[ss.bn.str.contains('I_M5_S_3.00_VI'), 'sweepType'] = 'speed_M_low_visc_ratio'
        ss.loc[ss.bn.str.contains('I_M6_S_3.00_VI'), 'sweepType'] = 'speed_M_high_visc_ratio'
    #     ss.loc[ss.ink_type=='PEGDA_40', 'sweepType'] = 'visc_PEG'

        # remove vertical data for speed sweeps with inaccurate vertical speeds

        for key in k[k.str.startswith('vert_')]:
            ss.loc[(ss.sweepType.str.startswith('speed'))&(ss.date<211000), key] = np.nan

        if diag:
            printStillsKeys(ss)
        self.ss = ss
        self.u = u
        return ss,u
    
    def fluidAbbrev(self, row:pd.Series) -> str:
        '''get a short abbreviation to represent fluid name'''
        it = row['ink_type']
        if it=='water':
            return 'W'
        elif it=='mineral oil':
            return 'M'
        elif it=='mineral oil_Span 20':
            return 'MS'
        elif it=='PDMS_3_mineral_25':
            return 'PM'
        elif it=='PDMS_3_silicone_25':
            return 'PS'
        elif it=='PEGDA_40':
            return 'PEG'

    def firstDepCol(self) -> str:
        '''get the name of the first dependent column'''
        k = self.ss.columns
        if 'xs_aspect' in k:
            return 'xs_aspect'
        elif 'projectionN' in k:
            return 'projectionN'
        elif 'horiz_segments' in k:
            'horiz_segments'
        else:
            return ''
    
    
    def addRatios() -> pd.DataFrame:
        '''add products and ratios of nondimensional variables. operator could be Prod or Ratio'''
        return super().addRatios('xs_aspect')

    def addLogs(ss:pd.DataFrame, varlist:List[str]) -> pd.DataFrame:
        '''add log values for the list of variables to the dataframe'''
        return super().addLogs('xs_aspect')
    
    
    
    def varSymbol(s:str, lineType:bool=True, commas:bool=True, **kwargs) -> str:
        '''get a symbolic representation of the variable
        lineType=True to include the name of the line type in the symbol
        commas = True to use commas, otherwise use periods'''
        if s.startswith('xs_'):
            varlist = {'xs_aspect':'XS height/width'
                       , 'xs_xshift':'XS horiz shift/width'
                       , 'xs_yshift':'XS vertical shift/height'
                       , 'xs_area':'XS area'
                       , 'xs_areaN':'XS area/intended'
                       , 'xs_wN':'XS width/intended'
                       , 'xs_hN':'XS height/intended'
                       , 'xs_roughness':'XS roughness'}
        elif s.startswith('vert_'):
            varlist = {'vert_wN':'vert bounding box width/intended'
                    , 'vert_hN':'vert length/intended'
                       , 'vert_vN':'vert bounding box volume/intended'
                   , 'vert_vintegral':'vert integrated volume'
                   , 'vert_viN':'vert integrated volume/intended'
                   , 'vert_vleak':'vert leak volume'
                   , 'vert_vleakN':'vert leak volume/line volume'
                   , 'vert_roughness':'vert roughness'
                   , 'vert_meanTN':'vert diameter/intended'
                       , 'vert_stdevTN':'vert stdev(diameter)/diameter'
                   , 'vert_minmaxTN':'vert diameter variation/diameter'}
        elif s.startswith('horiz_') or s=='vHorizEst':
            varlist = {'horiz_segments':'horiz segments'
                   , 'horiz_segments_manual':'horiz segments'
                   , 'horiz_maxlenN':'horiz droplet length/intended'
                   , 'horiz_totlenN':'horiz total length/intended'
                   , 'horiz_vN':'horiz volume/intended'
                   , 'horiz_roughness':'horiz roughness'
                   , 'horiz_meanTN':'horiz height/intended'
                   , 'horiz_stdevTN':'horiz stdev(height)/intended'
                   , 'horiz_minmaxTN':'horiz height variation/diameter'
                   , 'vHorizEst':'horiz volume'}
        elif s.startswith('proj'):
            varlist = {'projectionN':'projection into bath/intended'
                       , 'projShiftN':'$x$ shift of lowest point/$d_{est}$'}
        elif s.startswith('vertDisp'):
            varlist = {'vertDispBotN':'downstream $z_{bottom}/d_{est}$'
                      ,'vertDispBotN':'downstream $z_{middle}/d_{est}$'
                      ,'vertDispBotN':'downstream $z_{top}/d_{est}$'}
        elif s.endswith('Ratio') or s.endswith('Prod'):
            if s.endswith('Ratio'):
                symb = '/'
                var1 = s[:-5]
            else:
                symb = r'\times '
                var1 = s[:-4]
                
            inkSymbol = self.indVarSymbol(var1, 'ink', commas=commas)[:-1]
            supSymbol = self.indVarSymbol(var1, 'sup', commas=commas)[1:]
            return f'{inkSymbol}{symb}{supSymbol}'
        elif s=='int_Ca':
            return r'$Ca=v_{ink}\eta_{sup}/\sigma$'
        elif s.startswith('ink_') or s.startswith('sup_'):
            fluid = s[:3]
            var = s[4:]
            return self.indVarSymbol(var, fluid, commas=commas)
        else:
            if s=='pressureCh0':
                return 'Extrusion pressure (Pa)'
            else:
                return s

        if lineType:
            return varlist[s]
        else:
            s1 = varlist[s]
            typ = re.split('_', s)[0]
            s1 = s1[len(typ)+1:]
            return s1
        