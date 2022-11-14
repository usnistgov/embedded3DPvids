#!/usr/bin/env python
'''Functions for collecting data from stills of single lines'''

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
from pic_stitch_bas import stitchSorter
from file_handling import isSubFolder, fileScale
import im_crop as vc
import im_morph as vm
from tools.imshow import imshow
from tools.plainIm import *
from val_print import *
from vid_noz_detect import nozData
from metrics_xs import *
from metrics_vert import *
from metrics_horiz import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
pd.set_option("display.precision", 2)


#--------------------------------


def stitchMeasure(file:str, st:str, progDims:pd.DataFrame, diag:int=0, **kwargs) -> Union[Tuple[dict,dict], Tuple[pd.DataFrame,dict]]:
    '''measure one stitched image
    st is a line type, e.g. xs, vert, or horiz
    progDims holds timing info about the print
    '''
    if st=='xs':
        return xsMeasure(file, diag=diag)
    elif st=='vert':
        return vertMeasure(file, progDims, diag=diag, **kwargs)
    elif st=='horiz':
        return horizMeasure(file, progDims, diag=diag, **kwargs)
    
def fnMeasures(folder:str, st:str) -> str:
    '''get a filename for summary table. st is xs, vert, or horiz'''
    return os.path.join(folder, f'{os.path.basename(folder)}_{st}Summary.csv')

    
def importProgDims(folder:str) -> Tuple[pd.DataFrame, dict]:
    '''import the programmed dimensions table to a dataframe, and get units'''
    pv = printVals(folder)
    progDims, units = pv.importProgDims()
    for s in ['l', 'w']:
        progDims[s] = progDims[s]*cfg.const.pxpmm # convert to mm
        units[s] = 'px'
    return progDims, units  


def stitchFile(folder:str, st:str, i:int) -> str:
    '''get the name of the stitch file, where st is vert, horiz, or xs, and i is a line number'''
    try:
        fl = stitchSorter(folder)
    except:
        return
    if st=='horiz':
        sval = 'horizfullStitch'
    else:
        sval = f'{st}{i+1}Stitch'
    file = getattr(fl, sval)
    if len(file)>0:
        return file[0]
    else:
        return ''

def measure1Line(folder:str, st:str, i:int, diag:int=0, **kwargs) -> Union[Tuple[dict,dict], Tuple[pd.DataFrame,dict]]:
    '''measure just one line. st is vert, horiz, or xs. i is the line number'''
    progDims, units = importProgDims(folder)
    file = stitchFile(folder, st, i)
    if os.path.exists(file):
        return stitchMeasure(file, st, progDims, diag=diag, **kwargs)
    else:
        return {},{}
    
def copyImage(folder:str, st:str, i:int) -> None:
    '''make a copy of the image. st is vert, horiz, or xs. i is the line number'''
    file = stitchFile(folder, st, i)
    if not os.path.exists(file):
        return
    if not '00' in file:
        return
    newfile = file.replace('_00', '_01')
    if os.path.exists(newfile):
        return
    shutil.copyfile(file, newfile)
    logging.info(f'Created new file {newfile}')


def measureStillsSingle(folder:str, overwrite:bool=False, diag:int=0, overwriteList:List[str]=['xs', 'vert', 'horiz'], **kwargs) -> None:
    '''measure the stills in folder. 
    overwrite=True to overwrite files. 
    overwriteList is the list of files that should be overwritten'''
    if not isSubFolder(folder):
        return
    try:
        fl = stitchSorter(folder)
    except Exception as e:
        return
    if fl.date<210500:
        # no stitches to measure
        return
    if 'dates' in kwargs and not fl.date in kwargs['dates']:
        return
    progDims, units = importProgDims(folder)
    
    # measure xs and vert
    logging.info(f'Measuring {os.path.basename(folder)}')
    for st in ['xs', 'vert']:
        fn = fnMeasures(folder, st)
        if overwrite and st in overwriteList and os.path.exists(fn):
            os.remove(fn)
        if not os.path.exists(fn):
            xs = []
            for i in range(getattr(fl, f'{st}Cols')):
                file = getattr(fl, f'{st}{i+1}Stitch')
                if len(file)>0:
                    ret = stitchMeasure(file[0], st, progDims, diag=diag, **kwargs)
                    if len(ret[0])>0:
                        sm, units = ret
                        xs.append(sm)
            if len(xs)>0:
                xs = pd.DataFrame(xs)
#                 exportMeasures(xs, st, folder, units)
                plainExp(fnMeasures(folder, st), xs, units)
    
    # measure horiz
    fn = fnMeasures(folder, 'horiz')
    if overwrite and 'horiz' in overwriteList and os.path.exists(fn):
        os.remove(fn)
    if not os.path.exists(fn):
        file = fl.horizfullStitch
        if len(file)>0:
            hm, units = horizMeasure(file[0], progDims,  diag=diag, **kwargs)
            if len(hm)>0:
                plainExp(fnMeasures(folder, 'horiz'), hm, units)
#                 exportMeasures(hm, 'horiz', folder, units)
            
def measureStillsSingleRecursive(topfolder:str, overwrite:bool=False, diag:int=0, **kwargs) -> None:
    '''measure stills recursively in all folders'''
    
    if isSubFolder(topfolder):
        try:
            measureStills(topfolder, overwrite=overwrite, diag=diag, **kwargs)
        except:
            traceback.print_exc()
            pass
    else:
        for f1 in os.listdir(topfolder):
            f1f = os.path.join(topfolder, f1)
            if os.path.isdir(f1f):
                measureStillsRecursive(f1f, overwrite=overwrite, diag=diag, **kwargs)


class metricList:
    '''holds info about measured lines'''
    
    def __init__(self, folder:str):
        if not os.path.isdir(folder):
            raise NameError(f'Cannot create metricList: {folder} is not directory')
        self.folder = folder
        self.bn = os.path.basename(folder)
        self.findSummaries()
        
    def validS(self) -> List[str]:
        return ['vert', 'horiz', 'xs']
        
    def findSummaries(self) -> None:
        '''import summary data'''
        for s in self.validS():
            fn = os.path.join(self.folder, f'{self.bn}_{s}Summary.csv')
            if os.path.exists(fn):
                t,u = plainIm(fn,0)
                setattr(self, f'{s}Sum', t)
                setattr(self, f'{s}SumUnits', u)
            else:
                setattr(self, f'{s}Sum', [])
                setattr(self, f'{s}SumUnits', {})
                
    def findRhe(self, vink:float=5, vsup:float=5, di:float=0.603, do:float=0.907) -> None:
        '''find viscosity for ink and support at flow speed vink and translation speed vsup for nozzle of inner diameter di and outer diameter do'''
        pv = printVals(folder)
        inkrate = vink/di # 1/s
        suprate = vsup/do # 1/s
        inknu = pv.ink.visc(inkrate)
        supnu = pv.sup.visc(suprate)
        return {'ink':pv.ink.shortname, 'sup':pv.sup.shortname, 'nuink':inknu, 'nusup':supnu}
        
        
                
    #-----------------------------
        
    def checkS(self, s:str) -> None:
        '''check if s is valid. s is a line type, i.e. vert, horiz, or xs'''
        if not s in self.validS():
            raise NameError(f'Line name must be in {self.validS()}')
        
    def numLines(self, s:str) -> int:
        '''number of lines where s is vert, horiz, or xs'''
        self.checkS(s)
        return len(getattr(self, f'{s}Sum'))

    def missingLines(self, s:str) -> list:
        '''indices of missing lines where s is vert, horiz, or xs'''
        self.checkS(s)
        if s=='xs':
            allL = [1,2,3,4,5]
        elif s=='horiz':
            allL = [0,1,2]
        elif s=='vert':
            allL = [1,2,3,4]
        tab = getattr(self, f'{s}Sum')
        if len(tab)==0:
            return allL
        else:
            return set(allL) - set(tab.line)
    
    def inconsistent(self, s:str, vlist:List[str], tolerance:float=0.25) -> list:
        '''get a list of variables in which the cross-sections are inconsistent, i.e. range is greater than median*tolerance
         where s is vert, horiz, or xs
         vlist is a list of variable names
         tolerance is a fraction of median value
         '''
        self.checkS(s)
        out = []
        for v in vlist:
            t = getattr(self, f'{s}Sum')
            if len(t)>0:
                ma = t[v].max()
                mi = t[v].min()
                me = t[v].median()
                if ma-mi > me*tolerance:
                    out.append(v)
        return out  
    
    def summarize(self) -> dict:
        '''summarize all of the summary data into one line'''
        rhe = self.findRhe()

#-----------------------------------------

def checkMeasurements(folder:str) -> None:
    '''check the measurements in the folder and identify missing or unusual data'''
    problems = []
    ml = metricList(folder)
    for i,s in enumerate(ml.validS()):
        missing = ml.missingLines(s)
        if len(missing)>0:
            problems.append({'code':i+1, 'description':f'Missing # {s} lines', 'value':missing, 'st':s})
    verti = ml.inconsistent('vert', ['xc', 'yc', 'area'], tolerance=0.5)
    if len(verti)>0:
        problems.append({'code':4, 'description':f'Inconsistent vert values', 'value':verti, 'st':'vert'})
    horizi = ml.inconsistent('horiz', ['totarea'], tolerance=0.5)
    if len(horizi)>0:
        problems.append({'code':5, 'description':f'Inconsistent horiz values', 'value':horizi, 'st':'horiz'})
    xsi = ml.inconsistent('xs', ['xc', 'yc'])
    if len(xsi)>0:
        problems.append({'code':6, 'description':f'Inconsistent xs values', 'value':xsi, 'st':'xs'})
    return pd.DataFrame(problems)

def checkAndDiagnose(folder:str, redo:bool=False) -> None:
    '''check the folder and show images to diagnose problems
    redo=True if you want to re-measure any bad values
    '''
    problems = checkMeasurements(folder)
    if len(problems)==0:
        logging.info(f'No problems detected in {folder}')
    else:  
        logging.info(f'Problems detected in {folder}')
        display(problems)
        
    relist = []
    if redo:
        # get a list of images to re-analyze
        for i, row in problems.iterrows():
            if row['code']<4:
                for m in row['value']:
                    relist.append([row['st'], m-1])
            else:
                if row['st']=='horiz':
                    mlist = [0]
                elif row['st']=='vert':
                    mlist = [0,1,2,3]
                elif row['st']=='xs':
                    mlist = [0,1,2,3,4]
                for m in mlist:
                    val = [row['st'], m]
                    if not val in relist:
                        relist.append(val)
        relist.sort()
        for r in relist:
    #         logging.info(f'Measuring {r[0]}{r[1]}')
            measure1Line(folder, r[0], r[1], diag=1)
    
    
def checkAndDiagnoseRecursive(topfolder:str, redo:bool=False) -> None:
    '''go through all folders recursively and check and diagnose measurements. redo=True to redo any measurements that are bad'''
    if isSubFolder(topfolder):
        try:
            checkAndDiagnose(topfolder, redo=redo)
        except:
            traceback.print_exc()
            pass
    elif os.path.isdir(topfolder):
        for f in os.listdir(topfolder):
            f1f = os.path.join(topfolder, f)
            if os.path.isdir(f1f):
                checkAndDiagnoseRecursive(f1f, redo=redo) 
                
def returnNewSummary(pv) -> Tuple[pd.DataFrame, dict]:
    '''get summary data from a printVals object'''
    t,u = pv.summary()
    return pd.DataFrame([t]),u
                

def stillsSummaryRecursive(topfolder:str) -> Tuple[pd.DataFrame, dict]:
    '''go through all of the folders and summarize the stills'''
    if isSubFolder(topfolder):
        try:
            pv = printVals(topfolder)
            t,u = pv.summary()
            return pd.DataFrame([t]),u
        except:
            traceback.print_exc()
            logging.warning(f'failed to summarize {topfolder}')
            return {}, {}
    elif os.path.isdir(topfolder):
        tt = []
        u = {}
        logging.info(topfolder)
        for f in os.listdir(topfolder):
            f1f = os.path.join(topfolder, f)
            if os.path.isdir(f1f):
                # recurse into next folder level
                t,u0=stillsSummaryRecursive(f1f)
                if len(t)>0:
                    if len(tt)>0:
                        # combine dataframes
                        tt = pd.concat([tt,t], ignore_index=True)
                    else:
                        # adopt this dataframe
                        tt = t
                    if len(u0)>len(u):
                        u = dict(u, **u0)
        return tt, u
    
def stillsSummary(topfolder:str, exportFolder:str, newfolders:list=[], filename:str='stillsSummary.csv') -> pd.DataFrame:
    '''go through all of the folders and summarize the stills'''
    outfn = os.path.join(exportFolder, filename) # file to export to
    if os.path.exists(outfn) and len(newfolders)>0:
        # import existing table and only get values from new folders
        ss,u = plainIm(outfn, ic=0)
        for f in newfolders:
            tt,units = stillsSummaryRecursive(f)
            newrows = []
            for i,row in tt.iterrows():
                if row['folder'] in list(ss.folder):
                    # overwrite existing row
                    ss.loc[ss.folder==row['folder'], ss.keys()] = row[ss.keys()]
                else:
                    # add new row
                    newrows.append(i)
            if len(newrows)>0:
                # add new rows to table
                ss = pd.concat([ss, tt.loc[newrows]])
    else:
        # create an entirely new table
        ss,units = stillsSummaryRecursive(topfolder)
        
    # export results
    if os.path.exists(exportFolder):
        plainExp(outfn, ss, units)
    return ss,units

def idx0(k:list) -> int:
    '''get the index of the first dependent variable'''
    if 'xs_aspect' in k:
        idx = int(np.argmax(k=='xs_aspect'))
    elif 'projectionN' in k:
        idx = int(np.argmax(k=='projectionN'))
    elif 'horiz_segments' in k:
        idx = int(np.argmax(k=='horiz_segments'))
    else:
        idx = 1
    return idx

def printStillsKeys(ss:pd.DataFrame) -> None:
    '''sort the keys into dependent and independent variables and print them out'''
    k = ss.keys()
    k = k[~(k.str.endswith('_SE'))]
    k = k[~(k.str.endswith('_N'))]
    idx = idx0(k)
    controls = k[:idx]
    deps = k[idx:]
    print(f'Independents: {list(controls)}')
    print()
    print(f'Dependents: {list(deps)}')
    
def fluidAbbrev(row:pd.Series) -> str:
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
    
def indVarSymbol(var:str, fluid:str, commas:bool=True) -> str:
    '''get the symbol for an independent variable, eg. dnorm, and its fluid, e.g ink
    commas = True to use commas, otherwise use periods'''
    if commas:
        com = ','
    else:
        com = '.'
    if var=='visc' or var=='visc0':
        return '$\eta_{'+fluid+'}$'
    elif var=='tau0':
        return '$\tau_{y'+com+fluid+'}$'
    elif var=='dPR':
        return '$d_{PR'+com+fluid+'}$'
    elif var=='dnorm':
        return '$\overline{d_{PR'+com+fluid+'}}$'
    elif var=='dnormInv':
        return '$1/\overline{d_{PR'+com+fluid+'}}$'
    elif var=='rate':
        return '$\dot{\gamma}_{'+fluid+'}$'
    else:
        if var.endswith('Inv'):
            varsymbol = '1/'+var[:-3]
        else:
            varsymbol = var
        return '$'+varsymbol+'_{'+fluid+'}$'
    
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
        return indVarSymbol(var1, 'ink', commas=commas)[:-1]+symb+indVarSymbol(var1, 'sup', commas=commas)[1:]
    elif s=='int_Ca':
        return r'$Ca=v_{ink}\eta_{sup}/\sigma$'
    elif s.startswith('ink_') or s.startswith('sup_'):
        fluid = s[:3]
        var = s[4:]
        return indVarSymbol(var, fluid, commas=commas)
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

def importStillsSummary(file:str='stillsSummary.csv', diag:bool=False) -> pd.DataFrame:
    '''import the stills summary and convert sweep types, capillary numbers'''
    ss,u = plainIm(os.path.join(cfg.path.fig, file), ic=0)
    
    ss = ss[ss.date>210500]       # no good data before that date
    ss = ss[ss.ink_days==1]       # remove 3 day data
    ss.date = ss.date.replace(210728, 210727)   # put these dates together for sweep labeling
    k = ss.keys()
    k = k[~(k.str.contains('_SE'))]
    k = k[~(k.str.endswith('_N'))]
    idx = idx0(k)
    controls = k[:idx]
    deps = k[idx:]
    ss = flipInv(ss)
    ss.insert(idx+2, 'sweepType', ['visc_'+fluidAbbrev(row) for j,row in ss.iterrows()])
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
    return ss,u

def plainTypes(sslap:pd.DataFrame, incSweep:int=1, abbrev:bool=True) -> pd.DataFrame:
    '''convert types to cleaner form for plot legends
    incSweep=2 for a long sweep name, 1 for a short type of sweep, 0 for no sweep type label
    abbrev=True to use short names, False to use long names
    '''
    if incSweep==2:
        vsweep = '$v$ sweep, '
        etasweep = '$\eta$ sweep, '
    elif incSweep==1:
        vsweep = '$v$, '
        etasweep = '$\eta$, '
    else:
        vsweep = ''
        etasweep = ''
    if not abbrev:
        waterlap = 'water/Lap'
        mo = 'mineral oil'
        peg = 'PEGDA'
    else:
        waterlap = 'water'
        mo = 'MO'
        peg = 'PEG'
        
    
    sslap.loc[sslap.sweepType=='speed_M', 'sweepType'] = vsweep + mo
    sslap.loc[sslap.sweepType=='visc_W', 'sweepType'] = etasweep + waterlap
    sslap.loc[sslap.sweepType=='visc_W_high_v_ratio', 'sweepType'] = etasweep + waterlap + ', high $v_i/v_s$'
    sslap.loc[sslap.sweepType=='visc_M', 'sweepType'] = etasweep + mo
    sslap.loc[sslap.sweepType=='visc_PEG', 'sweepType'] = etasweep + peg
    sslap.loc[sslap.sweepType=='speed_W', 'sweepType'] = vsweep + waterlap
    sslap.loc[sslap.sweepType=='speed_M_low_visc_ratio', 'sweepType'] = vsweep + mo+ ', low $\eta_i/\eta_s$'
    sslap.loc[sslap.sweepType=='speed_M_high_visc_ratio', 'sweepType'] = vsweep + mo+', high $\eta_i/\eta_s$'
    sslap.loc[sslap.sweepType=='speed_W_high_visc_ratio', 'sweepType'] = vsweep + waterlap + ', high $\eta_i/\eta_s$'
    sslap.loc[sslap.sweepType=='speed_W_low_visc_ratio', 'sweepType'] = vsweep + waterlap + ', low $\eta_i/\eta_s$'
    sslap.loc[sslap.sweepType=='speed_W_int_visc_ratio', 'sweepType'] = vsweep + waterlap + ', med $\eta_i/\eta_s$'
    for s in ['sweepType', 'ink_type']:
        if s=='sweepType':
            sap = etasweep + ''
        else:
            sap = ''
        if not abbrev:
            sslap.loc[sslap.ink_type=='PDMS_3_mineral_25',s] = sap+'PDMS/mineral oil'
            sslap.loc[sslap.ink_type=='PDMS_3_silicone_25', s] = sap+'PDMS/silicone oil'
            sslap.loc[sslap.ink_type=='mineral oil_Span 20', s] = sap+'mineral oil/Span'
            sslap.loc[sslap.ink_type=='PEGDA_40', s] = sap+'PEGDA'
        else:
            sslap.loc[sslap.ink_type=='PDMS_3_mineral_25',s] = sap+'PDMS/MO'
            sslap.loc[sslap.ink_type=='PDMS_3_silicone_25', s] = sap+'PDMS/SO'
            sslap.loc[sslap.ink_type=='mineral oil_Span 20', s] = sap+'MO/Span'
            sslap.loc[sslap.ink_type=='PEGDA_40', s] = sap+'PEG'



def flipInv(ss:pd.DataFrame, varlist = ['Ca', 'dPR', 'dnorm', 'We', 'Oh']) -> pd.DataFrame:
    '''find inverse values and invert them (e.g. WeInv)'''
    k = ss.keys()
    idx = idx0(k)
    for j, s2 in enumerate(varlist):
        for i,s1 in enumerate(['sup', 'ink']):
            xvar = f'{s1}_{s2}'
            if f'{s1}_{s2}Inv' in ss and not xvar in ss:
                ss.insert(idx, xvar, 1/ss[f'{s1}_{s2}Inv'])
                idx+=1
    if 'int_Ca' not in ss:
        ss.insert(idx, 'int_Ca', 1/ss['int_CaInv'])
    return ss

def addRatios(ss:pd.DataFrame, varlist = ['Ca', 'dPR', 'dnorm', 'We', 'Oh', 'Bm'], operator:str='Prod') -> pd.DataFrame:
    '''add products and ratios of nondimensional variables. operator could be Prod or Ratio'''
    k = ss.keys()
    idx = int(np.argmax(k=='xs_aspect'))
    for j, s2 in enumerate(varlist):
        xvar =  f'{s2}{operator}'
        if not xvar in ss:
            if not f'ink_{s2}' in ss or not  'sup_{s2}' in ss:
                ss = flipInv(ss)
            if operator=='Prod':
                ss.insert(idx, xvar, ss[f'ink_{s2}']*ss[f'sup_{s2}'])
            elif operator=='Ratio':
                ss.insert(idx, xvar, ss[f'ink_{s2}']/ss[f'sup_{s2}'])
            idx+=1
    return ss

def addLogs(ss:pd.DataFrame, varlist:List[str]) -> pd.DataFrame:
    '''add log values for the list of variables to the dataframe'''
    k = ss.keys()
    idx = int(np.argmax(k=='xs_aspect'))
    for j, s2 in enumerate(varlist):
        xvar = f'{s2}_log'
        if not xvar in s2:
            ss.insert(idx, xvar, np.log10(ss[s2]))
            idx+=1
    return ss
    

def speedTableRecursive(topfolder:str) -> pd.DataFrame:
    '''go through all of the folders and summarize the stills'''
    if isSubFolder(topfolder):
        try:
            pv = printVals(topfolder)
            t = {'bn':pv.bn, 'vink':pv.vink, 'vsup':pv.vsup, 'pressure':pv.targetPressures[0]}
            u = {'bn':'','vink':'mm/s', 'vsup':'mm/s','pressure':'mbar'}
        except:
            traceback.print_exc()
            logging.warning(f'failed to get speed from {topfolder}')
            return {}, {}
        return [t],u
    elif os.path.isdir(topfolder):
        tt = []
        u = {}
        for f in os.listdir(topfolder):
            f1f = os.path.join(topfolder, f)
            if os.path.isdir(f1f):
                t,u0=speedTableRecursive(f1f)
                if len(t)>0:
                    tt = tt+t
                    if len(u)==0:
                        u = u0
        return tt, u

def speedTable(topfolder:str, exportFolder:str, filename:str) -> pd.DataFrame:
    '''go through all the folders, get a table of the speeds and pressures, and export to fn'''
    tt,units = speedTableRecursive(topfolder)
    tt = pd.DataFrame(tt)
    if os.path.exists(exportFolder):
        plainExp(os.path.join(exportFolder, filename), tt, units)
    return tt,units



                
    