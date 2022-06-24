#!/usr/bin/env python
'''Functions for stitching a single group of bascam stills'''

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
import file_handling as fh
from pic_stitch import Stitch



#----------------------------------------

class stillGroup:
    '''class that holds lists of stills to be stitched into a single image'''
    
    def __init__(self, stillList:List[str], rows:int, cols:int, rc:str, targetFolder:str, st:str, dxrows:int=0, dxcols:int=0, dyrows:int=0, dycols:int=0, scale:float=1, num:int=0, cropleft:int=0, cropright:int=0, cropbot:int=0, croptop:int=0, checkRC:bool=True):
        '''
        stillList is a list of file full path
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
        
        self.skipLast = False
        if not len(stillList) == cols*rows:
            if checkRC:
                raise ValueError('File list does not match number of rows and cols')
            else:
                self.skipLast = True
            
        self.dxrows = dxrows
        self.dyrows = dyrows
        self.dxcols = dxcols
        self.dycols = dycols
        self.cols = cols
        self.rows = rows
        self.rc = rc.lower()
        self.targetFolder = targetFolder
        tag = os.path.basename(self.targetFolder)
        self.spacing = re.split('_', tag)[-1]
        self.scale = scale
        self.st = st
        self.num = num
        self.levels = fh.labelLevels(self.targetFolder)       # file hierarchy
        self.pfd = fh.printFileDict(self.levels.printFolder())
        self.cropleft = cropleft
        self.cropright = cropright
        self.cropbot = cropbot
        self.croptop = croptop
        self.stillList = stillList
        
        arr = np.array([['' for i in range(cols)] for j in range(rows)], dtype=object)
        
        for j in range(rows):
            for i in range(cols):
                if self.rc=='r':
                    k = cols*j+i
                else:
                    k = rows*i+j
                if k<len(stillList):
                    ffull = stillList[k]
                    arr[j,i]=ffull
        self.stillArr = arr # array of stills
        
    def nameShort(self) -> str:
        '''get the name with no tag'''
        if self.pfd.printType=='tripleLine':
            return f'{self.st}_{self.num}'
        elif self.pfd.printType=='singleLine':
            if self.st=='horiz':
                return f'horizfull'
            else:
                return f'{self.st}{self.num+1}'
        
    def name(self) -> str:
        '''get the name of this stitch'''
        tag = os.path.basename(self.targetFolder)        # type and spacing of print
        return f'{tag}_{self.nameShort()}'
        
    def stitchFN(self) -> str:
        '''get the filename for the stitch'''
        # generate file name
        firstIm = self.stillArr[0,0]        
        ext = os.path.splitext(firstIm)[-1]
        if self.pfd.printType=='tripleLine':
            s1 = re.split('I_', os.path.basename(firstIm))[-1]  # everything after ink
            sample = f'I_{s1}'                                  # sample and date
            fnstitch = os.path.join(self.targetFolder, f'{self.name()}_stitch_{self.scale}_{sample}')
        elif self.pfd.printType=='singleLine':
            scale = '{0:.4g}'.format(self.scale)
            fnstitch = os.path.join(self.targetFolder, f'{self.name()}_{scale}_00{ext}')
        return fnstitch
        
        
    def printVals(self) -> str:
        '''return a string with information about the group'''
        fnstitch = self.stitchFN()
        if not os.path.exists(fnstitch):
            fnstitch = f'(({os.path.basename(fnstitch)}))'
        else:
            fnstitch = os.path.basename(fnstitch)

        if self.cols>1 or self.rows>1:
            vfunc = np.vectorize(lambda f:(fh.fileTimeV(f)[4:]))
            arrprint = vfunc(self.stillArr)
        else:
            arrprint = ''
        if self.pfd.printType=='tripleLine':
            intro = f'{self.st}_{self.spacing}_{self.num}'
        elif self.pfd.printType=='singleLine':
            intro = self.nameShort()
        return f'{intro}: Stitch: {fnstitch}\n\tStills:{os.path.basename(self.stillArr[0,0])}\n{arrprint}'
    
    def stitchDone(self) -> bool:
        '''determine if this is done being stitched'''
        fnstitch = self.stitchFN()
        if os.path.exists(fnstitch):
            return True
        else:
            return False
        
    def stitchFiles(self, files:List[str], scale:float, dx:float, dy:float, fn:str, debug:bool, scaleIm:bool):
        '''stitch the files'''
        if len(files)==0:
            return
        if len(files)==1:
            shutil.copyfile(files[0], fn)
        s = Stitch(files) # initialize the stitch object
        s.matcher.setDefaults(dx*scale, dy*scale)
        s.matcher.resetLastH()
        if not scale==1 and scaleIm:
            s.scaleImages(scale) # rescale images
        try:
        # stitch images and export
            if debug:
                export=False
                disp = 1
            else:
                export=True
                disp = 0
            s.stitchTranslate(export=True, fn=fn)
        except Exception as e:
            logging.warning('Stitching error')
            traceback.print_exc()
            return
        
    def stitchGrid(self, tempfolder:str, overwrite:bool, debug:bool) -> None:
        '''stitch the grid of images
        overwrite to overwrite existing images
        debug to print diagnostics instead of exporting files'''
        # stitch each column
        scale = self.scale
        ext = os.path.splitext(self.stillArr[0,0])[-1]
        
        for i in range(self.cols):
            files = list(self.stillArr[:,i])
            files = [x for x in files if os.path.exists(x) and len(os.path.basename(x))>0] # only select real files
            tempfile = os.path.join(tempfolder, f'temp_{i}{ext}')
            if not os.path.exists(tempfile) or overwrite or debug:
                self.stitchFiles(files, scale, self.dxcols, self.dycols, tempfile, debug, True)
                
        # stitch columns together
        if not debug:
            if not self.skipLast:
                # stitch all columns together
                files = [os.path.join(tempfolder, f'temp_{i}{ext}') for i in range(self.cols)]
                fnstitch = self.stitchFN()
            else:
                # stitch last column separately
                files = [os.path.join(tempfolder, f'temp_{i}{ext}') for i in range(self.cols-1)]
                fnstitch = os.path.join(tempfolder, f'temp_most{ext}')
                
            self.stitchFiles(files, scale, self.dxrows, self.dyrows, fnstitch, debug, False)
            
            if self.skipLast:
                # add last column
                files = [os.path.join(tempfolder, f'temp_most{ext}'), os.path.join(tempfolder, f'temp_{self.cols-1}{ext}')]
                fnstitch = self.stitchFN()
                if self.dxrows==424:
                    self.dxrows = 422
                self.stitchFiles(files, scale, self.dxrows*(self.cols-1), -self.dycols*0.936, fnstitch, debug, False)
        

    def stitch(self, overwrite:bool=False, debug:bool=False, **kwargs) -> None:
        '''stitch the image and save it under the given file name
        overwrite to overwrite existing images
        debug to print diagnostics instead of exporting files'''
        
        fnstitch = self.stitchFN()
        if os.path.exists(fnstitch) and not overwrite and not debug:
            return
        
        if self.cols==1 and self.rows==1:
            # only one image, no stitching necessary
            if debug:
                # debug mode: print diagnostics
                logging.debug(f'Copying file {os.path.basename(self.stillArr[0,0])} to {os.path.basename(fnstitch)}')
            else:
                # copy the image
                shutil.copyfile(self.stillArr[0,0], fnstitch)
            return
       
        
        # create folder to hold temp files
        tempfolder = os.path.join(os.path.dirname(fnstitch), 'temp')
        if not os.path.exists(tempfolder) and not debug:
            os.mkdir(tempfolder)
            
        # stitch the images
        self.stitchGrid(tempfolder, overwrite, debug)
        
        # remove temporary files
        if not debug:
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
        except Exception as e:
            logging.error(f'Image read error on {file}')
            traceback.print_exc()
        if self.croptop+self.cropbot>h or self.cropleft+self.cropright>w:
            # crop is bigger than image. abort
            return
        im = im[self.croptop:h-self.cropbot, self.cropleft:w-self.cropright]
        cv.imwrite(fnstitch, im)
        
    def archive(self, debug:bool=False, **kwargs) -> None:
        '''put files in an archive folder'''
        files = self.stillList
        if 'raw' in files[0]:
            # already archived
            return
        dirname = os.path.dirname(files[0])
        rawfolder = os.path.join(dirname, 'raw')
        if not os.path.exists(rawfolder):
            if not debug:
                os.mkdir(rawfolder)
        st = self.nameShort()
        linefolder = os.path.join(rawfolder, st)
        if not os.path.exists(linefolder):
            if not debug:
                os.mkdir(linefolder)
        newnames = []
        for f in files:
            newname = os.path.join(linefolder, os.path.basename(f))
            if debug:
                logging.info(f'Old: {f}, New: {newname}')
            else:
                if os.path.exists(newname):
                    # remove existing file
                    os.remove(newname)
                os.rename(f, newname)
                newnames.append(newname)
#         if not debug:
#             setattr(self, f"{st}Still", newnames)
        
    def stitchAndCrop(self, overwrite:bool=False, archive:bool=False, debug:bool=False, **kwargs) -> None:
        '''stitch the image and crop it'''
        if overwrite or not self.stitchDone():
            crop=True
        else:
            crop=False
        self.stitch(overwrite=overwrite, debug=debug, **kwargs)
        if crop and not debug:
            self.cropStitch()
        if archive and not debug:
            self.archive(**kwargs)

    def clearTest(self, passed:bool, fn, testfn):
        '''clear test files'''
        if passed:
            os.remove(fn)    # remove new file
            os.rename(testfn, fn)  # put old file back
        else:
            os.rename(fn, os.path.join(self.targetFolder, f'failed_{os.path.basename(fn)}'))    # rename new file
            os.rename(testfn, fn)  # put old file back
        
            
    def testStitch(self) -> int:
        '''test if the stitch dimensions match the original file. 
        return -1 if no original file. return 0 if passed, return 1 if failed shape, return 2 if failed value'''
        fn = self.stitchFN()
        if not os.path.exists(fn):
            return -1
        testfn = os.path.join(self.targetFolder, 'test.png')
        if os.path.exists(testfn):
            os.remove(testfn)
        os.rename(fn, testfn)
        self.stitchAndCrop(overwrite=True, archive=False, debug=False)
        im1 = cv.imread(fn)
        im2 = cv.imread(testfn)
        if not im1.shape==im2.shape:
            if im1.shape[0]==im2.shape[0]-8 and im1.shape[0]==im2.shape[0]-8:
                logging.warning(f'New image cropped by 8 pixels: {self.targetFolder}')
                self.clearTest(True, fn, testfn)
                return 0
            elif im1.shape[0]==im2.shape[0]+8 and im1.shape[0]==im2.shape[0]+8:
                logging.warning(f'Old image cropped by 8 pixels: {self.targetFolder}')
                self.clearTest(True, fn, testfn)
                return 0
            else:
                logging.error(f'Test stitch failed: {self.targetFolder}, original shape = {im2.shape}, new shape = {im1.shape}')
                self.clearTest(False, fn, testfn)
                return 1
        if not (im1[0,0]==im2[0,0]).all():
            logging.error(f'Test stitch failed: {self.targetFolder}, pixel values different.')
            self.clearTest(False, fn, testfn)
            return 2
        # success
        logging.info(f'Test stitch passed: {self.targetFolder}')
        self.clearTest(True, fn, testfn)
        return 0