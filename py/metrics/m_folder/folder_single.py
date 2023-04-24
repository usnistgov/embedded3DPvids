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
sys.path.append(os.path.dirname(currentdir))
sys.path.append(os.path.dirname(os.path.dirname(currentdir)))
from pic_stitch.p_bas import stitchSorter
from file.file_handling import isSubFolder
from tools.plainIm import *
from val.v_print import *
from vid.noz_detect import nozData
from m_tools import *

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
    
def openImageInPaint(folder:str, st:str, i:int) -> None:
    '''open the image in paint. this is useful for erasing smudges or debris that are erroneously detected by the algorithm as filaments'''
    file = stitchFile(folder, st, i)
    if not os.path.exists(file):
        return
    openInPaint(file)

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





        
        
    

                
    