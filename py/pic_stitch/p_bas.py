#!/usr/bin/env python
'''Functions for sorting bascam stills to be stitched'''

# external packages
import os, sys
import traceback
import logging
import pandas as pd
from typing import List, Dict, Tuple, Union, Any, TextIO
import re
import numpy as np
import shutil
import cv2 as cv

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
import file_handling as fh
from p_stillGroup import stillGroup

# logging

#----------------------------------------------


def flatten(l:list) -> list:
    '''flatten a list of lists'''
    return [food for sublist in l for food in sublist]

            
#------------------------

        
        
class stitchSorter:
    '''class that holds, sorts, and stitches lists of stills'''
    
    def __init__(self, subFolder):
        '''folder should be a subfolder holding many sbp folders'''
        
        if not fh.isSubFolder(subFolder):
            raise ValueError('input to stitchSorter must be subfolder')
        
        self.subFolder = subFolder
        self.date = fh.fileDate(subFolder)
        
    
    def printGroup(self, st:str) -> None:
        '''print a single group, given a stitch label'''
        if not st in self.stlist:
            raise ValueError(f'Input to printGroup must be in {self.stlist}')
        
        logging.info(f'\n{st}\n-----')
        grps = getattr(self, f'{st}groups') # dictionary will be spacing -> folder
        for grp in grps:
            logging.info(grp.printVals())
        
        
    def printGroups(self) -> None:
        '''print all of the groups available'''
        for st in self.stlist:
            self.printGroup(st)
            
            
    def grpDone(self, st:str, index:int):
        '''see if a single group or all groups in the object type are done'''
        grps = getattr(self, f'{st}groups') # dictionary will be spacing -> folder
        if len(grps)==0:
            # no groups to stitch
            return
        if index<0:
            # check all groups
            for grp in grps:
                if not grp.stitchDone():
                    return False
            return True
        else:
            # check a single group
            if index>len(grps):
                raise ValueError(f'requested stitch {st}{index} does not exist')
            return grps[index].stitchDone()
        
    def stitchGroup(self, st:str, index:int=-1, overwrite:bool=False, **kwargs) -> None:
        '''stitch a single list of groups, given a stitch label'''
        if not st in self.stlist:
            raise ValueError(f'Input to stitchGroup must be in {self.stlist}')
        if not overwrite and self.grpDone(st, index):
            # these groups are done
            return
    
        grps = getattr(self, f'{st}groups') # dictionary will be spacing -> folder
        if len(grps)==0:
            # no groups to stitch
            return
        
        logging.info(f'\n{st}\n-----')
        if index<0:
            # stitch all groups
            for grp in grps:
                try:
                    grp.stitchAndCrop(overwrite=overwrite, **kwargs)
                except Exception as e:
                    logging.error(f'Stitching error in {grp.name()}: {e}')
        else:
            if index>len(grps):
                raise ValueError(f'requested stitch {st}{index} does not exist')
            grps[index].stitchAndCrop(overwrite=overwrite, **kwargs)
            
    def testGroup(self, st:str, index:int=-1) -> int:
        '''test if the stitching is correct'''
        out = 0
        if not st in self.stlist:
            raise ValueError(f'Input to testGroup must be in {self.stlist}')
        grps = getattr(self, f'{st}groups') # dictionary will be spacing -> folder
        if len(grps)==0:
            # no groups to stitch
            return
        
        logging.info(f'\n{st}\n-----')
        if index<0:
            # stitch all groups
            for grp in grps:
                out1 = grp.testStitch()
                if out1>0:
                    out = out1
        else:
            if index>len(grps):
                raise ValueError(f'requested stitch {st}{index} does not exist')
            out1 = grps[index].testStitch()
            if out1>0:
                out = out1
        return out
            
    def stitchAll(self, overwrite:bool=False, **kwargs) -> None:
        '''stitch all of the folders'''
        if not overwrite and self.stitchDone():
            return
        logging.info(f'\n--------------\n--------------\n{self.subFolder}\n-------')
        for sti in self.stlist:
            if 'st' in kwargs:
                if sti in kwargs['st']:
                    kwargs2 = kwargs.copy()
                    kwargs2.pop('st')
                    self.stitchGroup(sti, overwrite=overwrite, **kwargs2)
            else:
                self.stitchGroup(sti, overwrite=overwrite, **kwargs)
            
    def stitchDone(self) -> bool:
        '''determine if the folder is done stitching'''
        for st in self.stlist:
            grps = getattr(self, f'{st}groups')
            for grp in grps:
                if not grp.stitchDone():
                    # if one group is not done, the folder is not done
                    return False
        return True
    
    def selectFiles(self, pf:str, files:List[str], rows:int, cols:int, offset:int, i:int, j:int, obsPerSet:int, lastSkip:bool=False) -> List[str]:
        '''get a list of full path names'''
        picsPerSet = rows*cols
        i0 = int(picsPerSet*(i*obsPerSet+j)+offset)
        i1 = int(picsPerSet*(i*obsPerSet+j)+picsPerSet+offset)
        if lastSkip:
            i1 = i1-1
        f1 = files[i0:i1]
        return [os.path.join(pf, f) for f in f1]
        
    
    def setGroup(self, st:str, pf:str, files:List[str], rows:int, cols:int, rc:str, offset:int=0, **kwargs) -> None:
        '''set the groups value to a list of stillGroups
        st is the type of object, e.g. HOB
        pf is the still folder
        files is the list of stills
        rows,cols are the dimensions of the grid
        rc = 'r' if pics are left-right, bottom-top, 'c' if pics are bottom-top, left-right
        offset is number of pics to skip at the beginning of the stills list
        lastSkip = True if there is one pic missing at the end
        '''
        
        grps = f'{st}groups'
        folders = getattr(self, f'{st}StitchFolders')
        picsPerSet = rows*cols
        if len(list(set(folders.values())))==len(folders):
            # all different folders: don't increment num
            numlist = [0 for f in folders]
        else:
            numlist = [i for i in range(len(folders))]
        lst = [stillGroup(self.selectFiles(pf, files, rows, cols, offset, i, 0, 1), 
                                rows, cols, rc, 
                                folders[key], st, num=numlist[i], **kwargs) for i,key in enumerate(folders)]
        
        setattr(self, grps, lst)
        
    def setGroup2D(self, st:str, pf:str, files:List[str], rows:int, cols:int, rc:str, obsPerSet:int, offset:int=0, lastSkip:bool=False, **kwargs) -> None:
        '''set the groups value to a list of stillGroups
        st is the type of object, e.g. HOB. 
        obs per set is the number of objects per set, e.g. 4 for HIPxs0, HIPxs1, HIPxs1, HIPxs3'''
        grps = f'{st}groups'
        folders = getattr(self, f'{st}StitchFolders')
        picsPerSet = rows*cols
        
        lst = flatten([[stillGroup(
            self.selectFiles(pf, files, rows, cols, offset
                             , i, j, obsPerSet, lastSkip=lastSkip), 
                                rows, cols, rc, 
                                folders[key], st, num=j, checkRC=(not lastSkip), **kwargs) 
                            for j in range(obsPerSet)]
                           for i,key in enumerate(folders)])
        setattr(self, grps, lst)
        
        
#----------------------------------------

class stitchSorterSingle(stitchSorter):
    '''class that holds, sorts, and stitches lists of stills for singleLines prints'''
    
    def __init__(self, subFolder):
        '''folder should be a subfolder holding many sbp folders'''
        super(stitchSorterSingle,self).__init__(subFolder)
        self.pfd = fh.printFileDict(self.subFolder)
        # number of columns for each type. 
        
        self.labelPics()
        

        
        
    def resetFolders(self):
        '''reset all of the folder lists and dictionaries'''
        
        self.horizCols = 12
        self.vertCols = 4
        self.xsCols = 5
        self.horizPerCol=0
        self.vertPerCol=0
        self.xsPerCol=0
        
        # horiz, vert1, vert2, ... xs1, xs2...
        self.stlist = fh.singleLineSt()
        
        for st in self.stlist:
            setattr(self, f'{st}groups', [])  # list of stillGroup objects
        
        self.horizStitchFolders = {-1:self.subFolder}
        self.vertStitchFolders = dict([[i+1,self.subFolder] for i in range(self.vertCols)])
        self.xsStitchFolders = dict([[i+1,self.subFolder] for i in range(self.xsCols)])
        
        self.basStill = list(filter(lambda f: 'Basler' in f, self.pfd.still))  # get list of still images
        self.basStill.sort(key=lambda f:os.path.basename(f))   # sort by date

    
    def labelPics(self) -> None:
        '''find the stills in the subfolder and sort them'''
        self.resetFolders()
        self.detectNumCols()
        self.createGroups()
        
    def createGroups(self) -> None:
        '''create stillGroups'''
        kwargs = {}
        if self.horizCols==12:
            kwargs['dxrows'] = 274
        elif self.horizCols==6:
            kwargs['dxrows'] = 2*274
        elif self.horizCols==8:
            kwargs['dxrows'] = 424
        kwargs['dycols'] = -280
        kwargs['scale'] = round(3/self.horizPerCol,3)
        self.setGroup2D('horiz', self.subFolder, self.basStill, self.horizPerCol, self.horizCols, 'c', 1, offset=0, lastSkip=self.lastSkip, **kwargs)
        offset = self.horizPerCol*self.horizCols
        if self.lastSkip:
            offset = offset-1
            
        # vert
        kwargs = {}
        kwargs['dxcols'] = -1
        kwargs['dycols'] = -277
        kwargs['scale'] = round(3/self.vertPerCol,3)
        self.setGroup('vert', self.subFolder, self.basStill, self.vertPerCol, 1, 'c', offset=offset, **kwargs)
        offset = offset + self.vertCols*self.vertPerCol    
        
        # xs
        kwargs = {}
        kwargs['dxcols'] = 4.24781116
        kwargs['dycols'] = -265.683797
        self.setGroup('xs', self.subFolder, self.basStill, self.xsPerCol, 1, 'c', offset=offset, **kwargs)
            
    
    def detectNumCols(self) -> None:
        '''determine how many columns and rows there are in each group, depending on which shopbot file created the pics'''
        self.lastSkip = False
#         logging.info(len(self.basStill))
        if len(self.basStill)>0:
            if len(self.basStill)==49 and 'singleLinesPics' in self.basStill[0]:
                # we used the shopbot script to generate these images
                self.horizCols=1
                self.horizPerCol=9
                self.vertPerCol=5
                self.xsPerCol=4
            elif len(self.basStill)==48 and 'singleLinesPics' in self.basStill[0]:
                # we used the shopbot script to generate these images
                self.horizCols=1
                self.horizPerCol=10
                self.vertPerCol=7
                self.xsPerCol=2
            elif len(self.basStill)==58 and 'singleLinesPics' in self.basStill[0]:
                # we used the shopbot script to generate these images
                self.horizCols=1
                self.horizPerCol=10
                self.vertPerCol=7
                self.xsPerCol=4
            elif len(self.basStill)==174 and 'singleLinesPics3' in self.basStill[0]:
                # we used the shopbot script to generate these images
                self.horizCols=12
                self.horizPerCol=11
                self.vertPerCol=7
                self.xsPerCol=3
                self.lastSkip = True
            elif len(self.basStill)==108 and 'singleLinesPics4' in self.basStill[0]:
                # we used the shopbot script to generate these images
                self.horizCols=6
                self.horizPerCol=11
                self.vertPerCol=7
                self.xsPerCol=3
                self.lastSkip = True
            elif len(self.basStill)==130 and 'singleLinesPics5' in self.basStill[0]:
                # we used the shopbot script to generate these images
                self.horizCols=8
                self.horizPerCol=11
                self.vertPerCol=7
                self.xsPerCol=3
                self.lastSkip = True
            elif len(self.basStill)==131 and 'singleLinesPics6' in self.basStill[0]:
                # we used the shopbot script to generate these images
                self.horizCols=8
                self.horizPerCol=10
                self.vertPerCol=9
                self.xsPerCol=3
            elif len(self.basStill)==121 and 'singleLinesPics7' in self.basStill[-1]:
                # we used the shopbot script to generate these images
                self.horizCols=8
                self.horizPerCol=10
                self.vertPerCol=9
                self.xsPerCol=1
            elif len(self.basStill)==131 and 'singleLinesPics8' in self.basStill[-1]:
                # we used the shopbot script to generate these images
                self.horizCols=8
                self.horizPerCol=10
                self.vertPerCol=9
                self.xsPerCol=3
            elif len(self.basStill)==136 and 'singleLinesPics9' in self.basStill[-1]:
                # we used the shopbot script to generate these images
                self.horizCols=8
                self.horizPerCol=10
                self.vertPerCol=9
                self.xsPerCol=4
            return 
        else:
            # unknown sorting: check folders
            for s in ['horiz', 'vert', 'xs']:
                numfiles = len(getattr(self, f'{s}1Still'))
                if numfiles>0:
                    setattr(self, f'{s}PerCol', numfiles)
                else:
                    return ValueError(f'{self.folder}: Cannot calculate files per column for {s}')
            return 

    

            
            
#-----------

class stitchSorterTriple(stitchSorter):
    '''class that holds, sorts, and stitches lists of stills for tripleLines prints'''
    
    def __init__(self, subFolder):
        '''folder should be a subfolder holding many sbp folders'''
        super(stitchSorterTriple,self).__init__(subFolder)
        self.labelFolders()
        self.labelPics()
        

    def resetFolders(self) -> None:
        '''reset all of the folder lists and dictionaries'''
        # H = horiz, V = vertical
        # I = in layer, O = out of layer
        # P = parallel, B = bridge, C = cross
        # ['HIB', 'HIPh', 'HIPxs', 'HOB', 'HOC', 'HOPh', 'HOPxs', 'VB', 'VC', 'VP']
        self.stlist = fh.tripleLineSt()
        
        # C = cross, D = double, H = horiz, V = vert, U = under, TL = triple line, X = cross-section
        # ['CDH', 'CDV', 'CU', 'TLH', 'TLX', 'TLV', 'TLU']
        self.sbFiles = list(fh.tripleLineSBPfiles().values())
        
        for st in self.stlist:
            setattr(self, f'{st}StitchFolders', {}) # dictionary will be spacing -> folder
            setattr(self, f'{st}groups', [])  # list of stillGroup objects
            
        for sb in self.sbFiles:
            setattr(self, f'{sb}picFolder', '') # path of folder
 
    def labelFolders(self) -> None:
        '''find the shopbot folders in the subfolder and sort them'''
        self.resetFolders()
        d = fh.tripleLine2FileDict()
        sbp2obj = {v: k for k, v in d.items()}
        sbpfiles = fh.tripleLineSBPfiles()
        
        for f in os.listdir(self.subFolder):
            ffull = os.path.join(self.subFolder, f)
            spl = re.split('_', f)
            if len(spl)==1 and 'Pics' in f:
                # assign pics folder
                lineName = re.split('Pics', f)[0]
                s = sbpfiles[lineName]
                setattr(self, f'{s}picFolder', ffull)
            else:
                # assign object/spacing folder
                s = '_'.join(spl[:-1])
                if s in sbp2obj:
                    stitchFolders = getattr(self, f'{sbp2obj[s]}StitchFolders')
                    stitchFolders[spl[-1]]=ffull
 
                    
                    
    def labelPics(self) -> None:
        '''sort the raw stills into folders'''
        for sb in self.sbFiles:
#         for sb in ['TLH']:
            try:
                getattr(self, f'label{sb}pics')()
            except Exception as e:
                print(e)
                traceback.print_exc()
                pass
        
    def picsPerSet(self, pf:str, objects:int, offset:bool=False) -> Tuple[List[str], int]:
        '''get the number of pictures per set given a path to the picture folder pf and a number of objects to image'''
        if not os.path.exists(pf):
            return
        files = os.listdir(pf) # all files in folder
        files = list(filter(lambda f: 'Basler' in f, files))  # only select images by the basler camera
        picsPerSet = len(files)/objects                           # 12 objects to image
        skip = 0
        if picsPerSet-np.floor(picsPerSet)>10**-6:            # should be same number of pics in each object
            if offset:
            # allow us to throw out the first pic
                skip = 1
                picsPerSet = (len(files)-1)/objects
                if picsPerSet-np.floor(picsPerSet)>10**-6:
                    raise ValueError(f'Uneven number of stills in {pf}:{len(files)}/{picsPerSet}')
            else:
                raise ValueError(f'Uneven number of stills in {pf}:{len(files)}/{picsPerSet}')
        picsPerSet = int(picsPerSet)  
        files = files[skip:]
        return files, picsPerSet

    def labelCDHpics(self) -> None:
        '''sort the crossDoubleHoriz pics'''
        pf = self.CDHpicFolder
        if not os.path.exists(pf):
            return
        files, picsPerSet = self.picsPerSet(pf, 12)
        kwargs = {}
        
        if picsPerSet==2:   
            rows = 2
            cols = 1
            rc = 'c'
            if 'Pics2' in pf:
                kwargs['dycols'] = -259
            else:
                kwargs['dycols'] = -172
            kwargs['dxcols'] = 0
        elif picsPerSet==9:
            rows = 3
            cols = 3
            rc = 'r'
            kwargs['dxrows'] = 351
            kwargs['dycols'] = -351
        elif picsPerSet==1:
            rows=1
            cols=1
            rc='r'
        else:
            raise ValueError(f'Unexpected number of stills in {pf}:{picsPerSet}')
            
        self.setGroup('HOB', pf, files, rows, cols, rc, offset=0, **kwargs)
        self.setGroup('HOC', pf, files, rows, cols, rc, offset=6*rows*cols, **kwargs)
        
        
    def labelCUpics(self) -> None:
        '''sort the crossUnder pics'''
        pf = self.CUpicFolder
        if not os.path.exists(pf):
            return
        files, picsPerSet = self.picsPerSet(pf, 12)
        
        kwargs = {}
        
        if picsPerSet==2:
            rows = 2
            cols = 1
            rc = 'c'
            kwargs['dycols'] = -172 #### fix this
            kwargs['dxcols'] = 0
            kwargs['scale'] = 0.3
        elif picsPerSet==1:
            rows=1
            cols=1
            rc='r'
        else:
            raise ValueError(f'Unexpected number of stills in {pf}:{picsPerSet}')
            
        self.setGroup('HIB', pf, files, rows, cols, rc, offset=0, **kwargs)
        self.setGroup('HIC', pf, files, rows, cols, rc, offset=6*rows*cols, **kwargs)
        
        
    def labelCDVpics(self) -> None:
        '''sort the crossDoubleVert pics'''
        pf = self.CDVpicFolder
        if not os.path.exists(pf):
            return
        files, picsPerSet = self.picsPerSet(pf, 12)
        
        kwargs = {}
        
        if picsPerSet==8:
            kwargs['dycols'] = -236
            dxrowskey = {0.5:436, 0.625:458, 0.75:480, 0.875:504, 1:524, 1.25:568}
            kwargs['scale'] = 0.5
            kwargs['dyrows'] =0
            rows = 4
            cols = 2
            rc = 'c'
            self.VBgroups = [stillGroup(files[(picsPerSet*i):(picsPerSet*i+cols*rows)], 
                                rows, cols, rc, 
                                self.VBStitchFolders[key], 'VB', dxrows=dxrowskey[float(key)], **kwargs)
                             for i,key in enumerate(self.VBStitchFolders)]
            self.VCgroups = [stillGroup(files[(picsPerSet*(i+6)):(picsPerSet*(i+6)+cols*rows)], 
                                rows, cols, rc, 
                                self.VCStitchFolders[key], 'VC', dxrows=dxrowskey[float(key)], **kwargs) 
                             for i,key in enumerate(self.VCStitchFolders)]
            return
        elif picsPerSet==2:
            
                
            rows=2
            cols=1
            rc='r'
            if 'Pics2' in pf:
                kwargs['dycols'] = -280
            else:
                kwargs['dycols'] = -73
        elif picsPerSet==1:
            rows=1
            cols=1
            rc='r'
        else:
            raise ValueError(f'Unexpected number of stills in {pf}:{picsPerSet}')

        self.setGroup('VB', pf, files, rows, cols, rc, offset=0, **kwargs)
        self.setGroup('VC', pf, files, rows, cols, rc, offset=6*rows*cols, **kwargs)
        
        
    def labelTLHpics(self) -> None:
        '''sort the tripleLinesHoriz pics'''
        pf = self.TLHpicFolder
        if not os.path.exists(pf):
            return
        files, picsPerSet = self.picsPerSet(pf, 24)
        
        kwargs = {}
        
        if picsPerSet==4:
            # extra picture at beginning
            remove = set(files[0::4])
            files = [f for f in files if f not in remove]  # remove every 1st of 4 elements
            picsPerSet=3
        
        if picsPerSet==3:
            rows = 1
            cols = 3
            rc = 'r'
            kwargs['dxrows'] = 282
        elif picsPerSet==1:
            rows=1
            cols=1
            rc='r'
            kwargs['croptop'] = 100
            kwargs['cropbot'] = 100
        else:
            raise ValueError(f'Unexpected number of stills in {pf}:{picsPerSet}')

        self.setGroup2D('HOPh', pf, files, rows, cols, rc, 4, offset=0, **kwargs)
        
    def labelTLUpics(self) -> None:
        '''sort the tripleLinesUnder pics'''
        pf = self.TLUpicFolder
        if not os.path.exists(pf):
            return
        files, picsPerSet = self.picsPerSet(pf, 24)
        
        kwargs = {}
        
        if picsPerSet==3:
            rows = 1
            cols = 3
            rc = 'r'
            kwargs['dxrows'] = 282 ## fix this
        elif picsPerSet==1:
            rows=1
            cols=1
            rc='r'
        else:
            raise ValueError(f'Unexpected number of stills in {pf}:{picsPerSet}')

        self.setGroup2D('HIPh', pf, files, rows, cols, rc, 4, offset=0, **kwargs)
        
        
    def labelTLVpics(self) -> None:
        '''sort the tripleLinesVert pics'''
        pf = self.TLVpicFolder
        if not os.path.exists(pf):
            return
        files, picsPerSet = self.picsPerSet(pf, 12, offset=True) # check whole column which has 2 objects
        
        kwargs = {}
        
        kwargs['cropleft'] = 200
        kwargs['cropright'] = 250
        
        
        if picsPerSet==4:
            rows = 2
            cols = 1
            rc = 'c'
            kwargs['dycols'] = -277
        elif picsPerSet==2:
            rows=1
            cols=1
            rc='r'
        elif picsPerSet==9:
            # 5 on bottom, 4 on top
            kwargs['dycols'] = -280
            self.VPgroups = flatten(flatten([[[stillGroup(files[(picsPerSet*(i*2+j)):(picsPerSet*(i*2+j)+5)], 
                                5, 1, 'c', 
                                self.VPStitchFolders[key], 'VP', num=j*2, **kwargs), 
                                      stillGroup(files[(picsPerSet*(i*2+j)+5):(picsPerSet*(i*2+j)+9)], 
                                4, 1, 'c', 
                                self.VPStitchFolders[key], 'VP', num=j*2+1, **kwargs)]
                            for j in range(2)]
                           for i,key in enumerate(self.VPStitchFolders)]))
            return
        else:
            raise ValueError(f'Unexpected number of stills per set in {pf}:{picsPerSet}')

        picsPerSet = picsPerSet/2
        self.setGroup2D('VP', pf, files, rows, cols, rc, 4, offset=0, **kwargs)
        
    def labelTLXpics(self) -> None:
        '''sort the tripleLinesXS pics'''
        pf = self.TLXpicFolder
        if not os.path.exists(pf):
            return
        files, picsPerSet = self.picsPerSet(pf, 48)
        
        kwargs = {}
        
        if picsPerSet==2:
            kwargs['dycols'] = -361
            kwargs['cropleft'] = 250
            kwargs['cropright'] = 50
            kwargs['croptop'] = 50
            kwargs['cropbot'] = 200
            HIPxsfiles = flatten([files[1+8*i:1+8*i+7]+[''] for i in range(6)])
            self.setGroup2D('HIPxs', pf, HIPxsfiles, 2, 1, 'c', 4, offset=0, **kwargs)
            kwargs['cropleft'] = 150
            kwargs['cropright'] = 150
            kwargs['croptop'] = 50
            kwargs['cropbot'] = 150
            HOPxsfiles = flatten([files[49+8*i:49+8*i+7]+[''] for i in range(6)])
            self.setGroup2D('HOPxs', pf, HOPxsfiles, 2, 1, 'c', 4, offset=0, **kwargs)
            return
        elif picsPerSet==1:
            rows=1
            cols=1
            rc='r'
            kwargs['cropleft'] = 250
            kwargs['cropright'] = 250
            kwargs['croptop'] = 100
            kwargs['cropbot'] = 100
        else:
            raise ValueError(f'Unexpected number of stills in {pf}:{picsPerSet}')

        self.setGroup2D('HIPxs', pf, files, rows, cols, rc, 4, offset=0, **kwargs)
        self.setGroup2D('HOPxs', pf, files, rows, cols, rc, 4, offset=picsPerSet*24, **kwargs)       


            
#--------------------------------------------------


def stitchSortDecide(folder:str) -> stitchSorter:
    '''get a stitchSorter object with the correct print type'''
    pfd = fh.printFileDict(folder)
    if pfd.printType=='singleLine':
        return stitchSorterSingle(folder)
    elif pfd.printType=='tripleLine':
        return stitchSorterTriple(folder)
            
def stitchSubFolder(folder:str, **kwargs) -> None:
    '''stitches images in the subfolder'''

    try:
        pfd = fh.printFileDict(folder)
        if pfd.printType=='singleLine':
            fl = stitchSorterSingle(folder)
            fl.stitchAll(archive=True, **kwargs)
        elif pfd.printType=='tripleLine':
            fl = stitchSorterTriple(folder)
            fl.stitchAll(archive=False, **kwargs)
    except Exception as e:
        logging.error(f'Error stitching {folder}: {e}')
        traceback.print_exc()
        raise e
            
def stitchRecursive(folder:str, **kwargs) -> None:
    '''for all folders in the folder, stitch images in the subfolders'''
    if not os.path.isdir(folder):
        return
    if fh.isSubFolder(folder):
        try:
            stitchSubFolder(folder,  **kwargs)
        except:
            pass
    else:
        for f in os.listdir(folder):
            stitchRecursive(os.path.join(folder, f), **kwargs)
            

    
            
