#!/usr/bin/env python
'''Functions for stitching bascam stills'''

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
import fileHandling as fh
import stitching

# logging



#----------------------------------------------

def colNum(file:str)->int:
    '''get the column number for this stitched image'''
    bn = os.path.basename(file)
    spl = re.split('_', bn)
    tag = spl[-3] # if there is a scaling factor in the file name
    if not tag[0].isalpha():
        tag = spl[-2] # no scaling factor
    i0 = 0
    for i in range(len(tag)):
        if tag[i].isalpha():
            i0+=1
    val = tag[i0:]
    try:
        out = int(val)
    except:
        raise ValueError('Could not determine column number')
    else:
        return out
    


def flatten(l:list) -> list:
    '''flatten a list of lists'''
    return [food for sublist in l for food in sublist]





class stillGroup:
    '''class that holds lists of stills to be stitched into a single image'''
    
    def __init__(self, stillFolder:str, stillList:List[str], rows:int, cols:int, rc:str, targetFolder:str, st:str, dxrows:int=0, dxcols:int=0, dyrows:int=0, dycols:int=0, scale:float=1, num:int=0, cropleft:int=0, cropright:int=0, cropbot:int=0, croptop:int=0):
        '''stillFolder is the folder that the stills are in
        stillList is a list of file basenames (not full path)
        cols is the number of columns
        rows is the number of rows
        rc is 'r' if we read left to right, bottom to top
        rc is 'c' if we read bottom to top, left to right
        targetFolder is the folder we're saving to
        tag is the type of object, e.g. HOB, HIP, VP
        dxrows is the horizontal spacing between images in px when assembling rows
        dyrows is the vertical spacing between images in px when assembling rows
        dxcols is the horizontal spacing between images in px when assembling columns
        dycols is the vertical spacing between images in px when assembling columns
        scale is the factor to scale stitched images by
        num is the line number
        '''
        
        if not len(stillList) == cols*rows:
            raise ValueError('File list does not match number of rows and cols')
            
        self.dxrows = dxrows
        self.dyrows = dyrows
        self.dxcols = dxcols
        self.dycols = dycols
        self.cols = cols
        self.rows = rows
        self.rc = rc.lower()
        self.stillFolder = stillFolder
        self.targetFolder = targetFolder
        tag = os.path.basename(self.targetFolder)
        self.spacing = re.split('_', tag)[-1]
        self.scale = scale
        self.st = st
        self.num = num
        self.levels = fh.labelLevels(self.targetFolder)       # file hierarchy
        self.cropleft = cropleft
        self.cropright = cropright
        self.cropbot = cropbot
        self.croptop = croptop
        
        arr = np.array([['' for i in range(cols)] for j in range(rows)], dtype=object)
        
        for j in range(rows):
            for i in range(cols):
                if self.rc=='r':
                    k = cols*j+i
                else:
                    k = rows*i+j
                ffull = os.path.join(stillFolder, stillList[k])
                arr[j,i]=ffull
        self.stillArr = arr # array of stills
        
    def name(self) -> str:
        '''get the name of this stitch'''
        tag = os.path.basename(self.targetFolder)        # type and spacing of print
        return f'{tag}_{self.st}_{self.num}'
        
    def stitchFN(self) -> str:
        '''get the filename for the stitch'''
        # generate file name
        
        sample = 'I_'+re.split('I_',os.path.basename(self.stillArr[0,0]))[-1]      # sample and date
        fnstitch = os.path.join(self.targetFolder, f'{self.name()}_stitch_{self.scale}_{sample}')
        return fnstitch
        
        
    def printVals(self) -> str:
        '''return a string with information about the group'''
        fnstitch = self.stitchFN()
        if not os.path.exists(fnstitch):
            fnstitch = f'(({os.path.basename(fnstitch)}))'
        else:
            fnstitch = os.path.basename(fnstitch)

        vfunc = np.vectorize(fh.fileTimeV)
        return f'{self.st}_{self.spacing}_{self.num}: Stitch: {fnstitch}\n\tStills:{os.path.basename(self.stillArr[0,0])}\n{vfunc(self.stillArr)}'
    
    def stitchDone(self) -> bool:
        '''determine if this is done being stitched'''
        fnstitch = self.stitchFN()
        if os.path.exists(fnstitch):
            return True
        else:
            return False
        
    def stitchGrid(self, tempfolder:str, overwrite:bool) -> None:
        '''stitch the grid of images'''
        # stitch each column
        scale = self.scale
        ext = os.path.splitext(self.stillArr[0,0])[-1]
        
        for i in range(self.cols):
            files = list(self.stillArr[:,i])
            files = [x for x in files if os.path.exists(x) and len(os.path.basename(x))>0]
            tempfile = os.path.join(tempfolder, f'temp_{i}{ext}')
            if not os.path.exists(tempfile) or overwrite:
                if len(files)>1:
                    s = stitching.Stitch(files) # initialize the stitch object
                    s.matcher.setDefaults(self.dxcols*scale, self.dycols*scale)
                    s.matcher.resetLastH()
                    if not scale==1:
                        s.scaleImages(scale) # rescale images
                    try:
                    # stitch images and export
                        s.stitchTranslate(export=True, fn=tempfile)
                    except:
                        logging.warning('Stitching error')
                        traceback.print_exc()
                        return
                else:
                    shutil.copyfile(files[0], tempfile)
                
        # stitch columns together
        files = [os.path.join(tempfolder, f'temp_{i}{ext}') for i in range(self.cols)]
        s = stitching.Stitch(files) # initialize the stitch object
        s.matcher.setDefaults(self.dxrows*scale, self.dyrows*scale)
        s.matcher.resetLastH()
        try:
        # stitch images and export
            fnstitch = self.stitchFN()
            s.stitchTranslate(export=True, fn=fnstitch)
        except:
            logging.warning('Stitching error')
            traceback.print_exc()
            return
        

    def stitch(self, overwrite:bool=False) -> None:
        '''stitch the image and save it under the given file name'''
        
        fnstitch = self.stitchFN()
        if os.path.exists(fnstitch) and not overwrite:
            return
        
        if self.cols==1 and self.rows==1:
            # copy the image
            shutil.copyfile(self.stillArr[0,0], fnstitch)
            return
#             logging.debug(f'Copying file {os.path.basename(self.stillArr[0,0])} to {os.path.basename(fnstitch)}')
        
        # create folder to hold temp files
        tempfolder = os.path.join(os.path.dirname(fnstitch), 'temp')
        if not os.path.exists(tempfolder):
            os.mkdir(tempfolder)
            
        # stitch the images
        self.stitchGrid(tempfolder, overwrite)
        
        # remove temporary files
        for f in os.listdir(tempfolder):
            os.remove(os.path.join(tempfolder, f))
        os.rmdir(tempfolder)
        
    def cropStitch(self) -> None:
        '''crop the saved stitch file'''
        fnstitch = self.stitchFN()
        im = cv.imread(fnstitch)
        try:
            h = im.shape[0]
            w = im.shape[1]
        except:
            logging.error(f'Image read error on {file}')
            traceback.print_exc()
        if self.croptop+self.cropbot>h or self.cropleft+self.cropright>w:
            # crop is bigger than image. abort
            return
        im = im[self.croptop:h-self.cropbot, self.cropleft:w-self.cropright]
        cv.imwrite(fnstitch, im)
        
    def stitchAndCrop(self, overwrite:bool=False, **kwargs) -> None:
        '''stitch the image and crop it'''
        if overwrite or not self.stitchDone():
            crop=True
        else:
            crop=False
        self.stitch(overwrite=overwrite)
        if crop:
            self.cropStitch()
        


class fileList:
    '''class that holds lists of files of different type'''
    
    def __init__(self, folder):
        '''folder should be a subfolder holding many sbp folders'''
        
        if not fh.isSubFolder(folder):
            raise ValueError('input to fileList must be subfolder')
        
        self.subFolder = folder
        self.date = fh.fileDate(folder)
        self.labelFolders()
        self.labelPics()
        
    def resetFolders(self) -> None:
        '''reset all of the folder lists and dictionaries'''
        # H = horiz, V = vertical
        # I = in layer, O = out of layer
        # P = parallel, B = bridge, C = cross
        self.stlist = fh.tripleLineSt()
        
        # C = cross, D = double, H = horiz, V = vert, U = under, TL = triple line, X = cross-section
        self.sbFiles = list(fh.tripleLineSBPfiles().values())
        
        for st in self.stlist:
            setattr(self, f'{st}folders', {}) # dictionary will be spacing -> folder
            setattr(self, f'{st}groups', [])  # list of stillGroup objects
            
        for sb in self.sbFiles:
            setattr(self, f'{sb}picFolder', '') # path of folder
            
            
    def printGroup(self, st:str) -> None:
        '''print a single group, given a stitch label'''
        if not st in self.stlist:
            raise ValueError(f'Input to printGroup must be in {self.stlist}')
        
        print(f'\n{st}\n-----')
        grps = getattr(self, f'{st}groups') # dictionary will be spacing -> folder
        for grp in grps:
            print(grp.printVals())
        
        
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
            raise ValueError(f'Input to printGroup must be in {self.stlist}')
        if not overwrite and self.grpDone(st, index):
            # these groups are done
            return
    
        grps = getattr(self, f'{st}groups') # dictionary will be spacing -> folder
        if len(grps)==0:
            # no groups to stitch
            return
        
        print(f'\n{st}\n-----')
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
            
    def stitchAll(self, overwrite:bool=False, **kwargs) -> None:
        '''stitch all of the folders'''
        if not overwrite and self.stitchDone():
            return
        logging.info('\n--------------\n--------------{self.subFolder}\n-------')
        for st in self.stlist:
            self.stitchGroup(st, overwrite=overwrite, **kwargs)
            
    def stitchDone(self) -> bool:
        '''determine if the folder is done stitching'''
        for st in self.stlist:
            grps = getattr(self, f'{st}groups')
            for grp in grps:
                if not grp.stitchDone():
                    # if one group is not done, the folder is not done
                    return False
        return True
    

        
        
    def labelFolders(self) -> None:
        '''find the shopbot folders in the subfolder and sort them'''
        self.resetFolders()
        
        for f in os.listdir(self.subFolder):
            ffull = os.path.join(self.subFolder, f)
            spl = re.split('_', f)
            if 'crossDoubleHoriz' in f:
                if len(spl)==1:
                    self.CDHpicFolder = ffull
                else:
                    if spl[1]=='0.5':
                        # put into dictionary with spacing as key
                        self.HOCfolders[spl[2]] = ffull
                    else:
                        self.HOBfolders[spl[2]] = ffull
            elif 'crossDoubleVert' in f:
                if len(spl)==1:
                    self.CDVpicFolder = ffull
                else:
                    if spl[1]=='0.5':
                        # put into dictionary with spacing as key
                        self.VCfolders[spl[2]] = ffull
                    else:
                        self.VBfolders[spl[2]] = ffull
            elif 'underCross' in f:
                if len(spl)==1:
                    self.CUpicFolder = ffull
                else:
                    if spl[1]=='0.5':
                        # put into dictionary with spacing as key
                        self.HICfolders[spl[2]] = ffull
                    else:
                        self.HIBfolders[spl[2]] = ffull
            elif 'tripleLinesXS' in f:
                if len(spl)==1:
                    self.TLXpicFolder = ffull
                else:
                    if spl[1]=='+y':
                        # put into dictionary with spacing as key
                        self.HIPxsfolders[spl[2]] = ffull
                    else:
                        self.HOPxsfolders[spl[2]] = ffull
            elif 'tripleLinesUnder' in f:
                if len(spl)==1:
                    self.TLUpicFolder = ffull
                else:
                    # put into dictionary with spacing as key
                    self.HIPhfolders[spl[1]] = ffull
            elif 'tripleLinesHoriz' in f:
                if len(spl)==1:
                    self.TLHpicFolder = ffull
                else:
                    # put into dictionary with spacing as key
                    self.HOPhfolders[spl[1]] = ffull
            elif 'tripleLinesVert' in f:
                if len(spl)==1:
                    self.TLVpicFolder = ffull
                else:
                    # put into dictionary with spacing as key
                    self.VPfolders[spl[1]] = ffull
                    
                    
    def labelPics(self) -> None:
        '''sort the raw stills into folders'''
        for sb in self.sbFiles:
            try:
                getattr(self, f'label{sb}pics')()
            except:
                pass
        
    def picsPerSet(self, pf:str, objects:int) -> Tuple[List[str], int]:
        '''get the number of pictures per set given a path to the picture folder pf and a number of objects to image'''
        if not os.path.exists(pf):
            return
        files = os.listdir(pf) # all files in folder
        files = list(filter(lambda f: 'Basler' in f, files))  # only select images by the basler camera
        picsPerSet = len(files)/objects                           # 12 objects to image
        if picsPerSet-np.floor(picsPerSet)>10**-6:            # should be same number of pics in each object
            raise ValueError(f'Uneven number of stills in {pf}:{len(files)}/{picsPerSet}')
        picsPerSet = int(picsPerSet)  
        return files, picsPerSet
    
    def setGroup(self, st:str, pf:str, files:List[str], rows:int, cols:int, rc:str, offset:int=0, **kwargs) -> None:
        '''set the groups value to a list of stillGroups
        st is the type of object, e.g. HOB'''
        
        grps = f'{st}groups'
        folders = getattr(self, f'{st}folders')
        picsPerSet = rows*cols
        lst = [stillGroup(pf, files[(picsPerSet*i+offset):(picsPerSet*i+cols*rows+offset)], 
                                rows, cols, rc, 
                                folders[key], st, **kwargs) for i,key in enumerate(folders)]
        
        setattr(self, grps, lst)
        
    def setGroup2D(self, st:str, pf:str, files:List[str], rows:int, cols:int, rc:str, obsPerSet:int, offset:int=0, **kwargs) -> None:
        '''set the groups value to a list of stillGroups
        st is the type of object, e.g. HOB. obs per set is the number of identical objects per set, e.g. HOB1, HOB2, HOB3, HOB4'''
        grps = f'{st}groups'
        folders = getattr(self, f'{st}folders')
        picsPerSet = rows*cols
        
        lst = flatten([[stillGroup(pf, files[int(picsPerSet*(i*obsPerSet+j)+offset):int(picsPerSet*(i*obsPerSet+j)+cols*rows+offset)], 
                                rows, cols, rc, 
                                folders[key], st, num=j, **kwargs) 
                            for j in range(obsPerSet)]
                           for i,key in enumerate(folders)])
        setattr(self, grps, lst)
        
        
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
            self.VBgroups = [stillGroup(pf, files[(picsPerSet*i):(picsPerSet*i+cols*rows)], 
                                rows, cols, rc, 
                                self.VBfolders[key], 'VB', dxrows=dxrowskey[float(key)], **kwargs) for i,key in enumerate(self.VBfolders)]
            self.VCgroups = [stillGroup(pf, files[(picsPerSet*(i+6)):(picsPerSet*(i+6)+cols*rows)], 
                                rows, cols, rc, 
                                self.VCfolders[key], 'VC', dxrows=dxrowskey[float(key)], **kwargs) for i,key in enumerate(self.VCfolders)]
            return
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
        files, picsPerSet = self.picsPerSet(pf, 12) # check whole column which has 2 objects
        
        kwargs = {}
        
        kwargs['cropleft'] = 250
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
            self.VPgroups = flatten(flatten([[[stillGroup(pf, files[(picsPerSet*(i*2+j)):(picsPerSet*(i*2+j)+5)], 
                                5, 1, 'c', 
                                self.VPfolders[key], 'VP', num=j*2, **kwargs), 
                                      stillGroup(pf, files[(picsPerSet*(i*2+j)+5):(picsPerSet*(i*2+j)+9)], 
                                4, 1, 'c', 
                                self.VPfolders[key], 'VP', num=j*2+1, **kwargs)]
                            for j in range(2)]
                           for i,key in enumerate(self.VPfolders)]))
            return
        else:
            raise ValueError(f'Unexpected number of stills in {pf}:{picsPerSet}')

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
            
def stitchSubFolder(folder:str, **kwargs) -> None:
    '''stitches images in the subfolder'''

    try:
        fl = fileList(folder)
        fl.stitchAll(**kwargs)
    except Exception as e:
        logging.error(f'Error stitching {folder}: {e}')
#         traceback.print_exc()
            
def stitchRecursive(folder:str, **kwargs) -> None:
    '''for all folders in the folder, stitch images in the subfolders'''
    if not os.path.isdir(folder):
        return
    if fh.isSubFolder(folder):
        stitchSubFolder(folder,  **kwargs)
    else:
        for f in os.listdir(folder):
            stitchRecursive(os.path.join(folder, f), **kwargs)
                
                
def countFiles(folder:str, stills:bool=True, stitches:bool=True, subcall:bool=False) -> Union[dict, List[dict], pd.DataFrame]:
    '''count the types of files in each folder. 
    stills=True to print any folders which are missing stills. 
    stitches=True to print any folders which are missing stitched images. 
    If this is the top call, returns a dataframe. If this is a child call to a sample folder or above, returns a list of dicts. If this is a call to a subfolder, returns a dict.'''

    if not os.path.isdir(folder):
        return
    if fh.isSubFolder(folder):
        fl = fileList(folder)
        return fl.countFiles() # returns a dictionary
    
    # parent folder
    c = [] # list of dictionaries
    for f in os.listdir(folder):
        c1 = countFiles(os.path.join(folder, f), stills=stills, stitches=stitches, subcall=True)
        if type(c1) is list:
            c = c+c1
        elif type(c1) is dict:
            c.append(c1) 
    if subcall:
        return c # return the list of dictionaries
    
    # top folder
    df = pd.DataFrame(c)
    

    STLIST = ['horiz', 'vert1', 'vert2', 'vert3', 'vert4', 'xs1', 'xs2', 'xs3', 'xs4', 'xs5']
    
    missing = False
    
    if stills:
        for st in STLIST:
            tystr = 'Still'
            files = [os.path.basename(f) for f in df[df[st+tystr]==0]['folder']]
            if len(files)>0:
                missing=True
                logging.info(f'Missing {st} {tystr}: {files}')
                
    if stitches:
        for st in STLIST:
            tystr = 'Stitch'
            files = [os.path.basename(f) for f in df[(df[st+'Stitch']==0) & (df[st+'Still']>0)]['folder']]
            if len(files)>0:
                missing=True
                logging.info(f'Missing {st} {tystr}: {files}')
                
    if not missing:
        logging.info('No missing files')
        
    return df
    
            
