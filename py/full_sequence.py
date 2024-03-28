#!/usr/bin/env python
'''Functions for running full analysis workflow on a single folder'''

# external packages
import os, sys
import traceback
import logging

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
sys.path.append(os.path.dirname(os.path.dirname(currentdir)))
import file.file_handling as fh
import progDim.prog_dim as pg
import vid.v_tools as vt
import vid.noz_detect as nt
import metrics.m_SDT as me
from m_tools import *


# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


#----------------------------------------------

class SDTWorkflow:
    '''full workflow for a singleDoubleTriple folder'''
    
    def __init__(self, folder:str, **kwargs):
        self.folder = folder
        self.vs = {}
        self.imtag = ''
        
    def run(self, stillsAwayK:dict={}, progDimsK:dict={}, exportStillsK:dict={}, nozzleK:dict={}, backgroundK:dict={}, analyzeK:dict={}, **kwargs):
        '''go through all of the steps to analyze a single print folder'''
        self.putStillsAway(**stillsAwayK)      # put the original stills in the raw folder
        self.getProgDims(**progDimsK)          # generate progDims table
        self.exportStills(**exportStillsK)     # export stills from video
        self.detectNozzle(**nozzleK)           # detect the nozzle
        self.exportBackground(**backgroundK)   # export the background
        self.analyze(**analyzeK)               # segment and measure images, and summarize measurements
        
    def putStillsAway(self, **kwargs):
        '''put the shopbot-created stills in a folder'''
        fh.putStillsAway(self.folder, **kwargs)
        
    def getProgDims(self, **kwargs):
        '''get programmed dimensions'''
        self.pdim = pg.getProgDims(self.folder, **kwargs)
        self.pdim.exportAll(**kwargs)
        self.pfd = self.pdim.pfd
        
    def initPFD(self):
        '''initial the printFileDict object that holds file locations'''
        if not hasattr(self, 'pfd'):
            self.pfd = fh.printFileDict(self.folder)
        
    def initVD(self):
        '''initialize the vidData object that holds metadata about the print'''
        self.initPFD()
        self.vd = vt.vidData(self.folder, pfd = self.pfd)
        
    def exportStills(self, **kwargs):
        '''export new stills'''
        self.initVD()
        if not hasattr(self, 'pdim'):
            self.getProgDims()
        self.vd.exportStills(pdim=self.pdim, **kwargs)
        
    def detectNozzle(self, **kwargs):
        '''detect the nozzle and export background'''
        self.initPFD()
        self.nv = nt.nozData(self.folder, pfd = self.pfd)
        self.nv.detectNozzle(**kwargs)
        
    def shiftNozzleLeft(self, dx:int=3, **kwargs) -> None:
        '''shift the left edge of the nozzle to the left and save'''
        self.nv.nd.xL = self.nv.nd.xL-dx
        self.nv.nd.exportNozzleDims(overwrite=True)
        
    def exportBackground(self, **kwargs):
        '''export the background'''
        if not hasattr(self, 'nv'):
            self.nv = nt.nozData(self.folder)
        self.nv.exportBackground(**kwargs)
        
    def analyze(self, **kwargs):
        '''analyze all images'''
        if 'Horiz' in self.folder:
            func = me.folderHorizSDT
        elif 'Vert' in self.folder:
            func = me.folderVertSDT
        elif 'XS' in self.folder:
            func = me.folderXSSDT
        elif 'Under' in self.folder:
            func = me.folderUnderSDT
        else:
            raise ValueError(f'Could not determine folder type for {self.folder}')
        
        kwargs2 = kwargs.copy()
        if hasattr(self, 'pdim'):
            kwargs2['pv'] = self.pdim.pv
            kwargs2['pg'] = self.pdim
        self.fv = func(self.folder, **kwargs2)
        self.fv.measureFolder()
        self.fv.summarize()
        self.failuredf = self.fv.failures
        if hasattr(self, 'failuredfdisp'):
            delattr(self, 'failuredfdisp')
        if hasattr(self.fv, 'df'):
            self.measures = self.fv.df
        else:
            self.fv.importMeasure()
            self.measures = self.fv.df
        
    def testImage(self, tag:str, **kwargs):
        '''test a single image, given a tag that is in the file name'''
        if 'Horiz' in self.folder:
            func = me.fileHorizSDTFromTag
        elif 'Vert' in self.folder:
            func = me.fileVertSDTFromTag
        elif 'XS' in self.folder:
            func = me.fileXSSDTFromTag
        elif 'Under' in self.folder:
            func = me.fileUnderSDTFromTag
        else:
            raise ValueError(f'Could not determine folder type for {self.folder}')
        self.vs[tag] = func(self.folder, tag, **kwargs)
        self.imtag = tag
        
    def testFailure(self, i:int, **kwargs) -> None:
        '''test the failed file'''
        file = self.failuredf.loc[i,'file']
        print(os.path.basename(file))
        self.testImage(os.path.basename(file), **kwargs)
        
    def testAllFailures(self, **kwargs) -> None:
        '''test all failed files'''
        for i,row in self.failuredf.iterrows():
            if i>0:
                self.testFailure(i, **kwargs)
        
    def getTagFromFile(self, fn) -> str:
        '''get the still tag from the file name, e.g. l1w2o3'''
        bn = os.path.basename(fn)
        bn = re.split('vstill_', bn)[-1]
        bn = re.split('_I', bn)[0]
        bn = re.split('_', bn)[-1]
        return bn
        
    def showFailures(self) -> None:
        '''show a list of files that failed'''
        if not hasattr(self, 'failuredfdisp'):
            self.failuredfdisp = self.failuredf.copy()
            self.failuredfdisp['file'] = [self.getTagFromFile(f) for f in self.failuredfdisp.file]
        display(self.failuredfdisp)
        df2 = self.fv.df.dropna(subset=['usedML'])
        display(df2[df2.usedML][['line', 'usedML']])
        
        
    def openInPaint(self, tag:str, nmin:int=0, nmax:int=100, usegment:bool=False, **kwargs):
        '''open a still in MS paint, given a tag'''
        self.initPFD()
        self.pfd.findVstill()
        if usegment:
            kwargs['dropper'] = False
            kwargs['white'] = True
        for file in self.pfd.vstill:
            bn =  os.path.splitext(os.path.basename(file))[0]
            if tag in bn:
                spl = re.split('_', re.split(tag, bn)[-1])[0]
                if nmax<100:
                    try:
                        spl = int(spl)
                    except:
                        spl=0
                else:
                    spl = 0
                if spl<=nmax and spl>=nmin:
                    if usegment:
                        file = os.path.join(os.path.dirname(file), 'Usegment', os.path.basename(file).replace('vstill', 'Usegment'))
                    openInPaint(file, **kwargs)
        
    def openLastImage(self, **kwargs):
        '''open the last analyzed still in MS paint'''
        self.openInPaint(self.imtag, **kwargs)
        
    def openLastUsegment(self, **kwargs):
        '''open the last segmented image in MS paint'''
        self.openInPaint(self.imtag, usegment=True, **kwargs)
        
    def openLastSeries(self, **kwargs):
        '''open all files in the series for the last analyzed still in MS paint'''
        self.openInPaint(self.imtag[:5], **kwargs)
        
    def openLastUsegmentSeries(self, **kwargs):
        '''open all segmented images in the series for the last analyzed still in MS paint'''
        self.openInPaint(self.imtag[:5], usegment=True, **kwargs)
        
    def openBackground(self):
        '''show the background image in the notebook'''
        nfn = self.pfd.newFileName('background', 'png')
        if os.path.exists(nfn):
            im = cv.imread(nfn)
            imshow(im)
        
    def openExplorer(self):
        '''open the print folder in windows explorer'''
        fh.openExplorer(self.folder)
        
    def adjustNozzle(self) -> None:
        '''open the nozDims spreadsheet and a writing line image'''
        if os.path.exists(self.pfd.nozDims):
            openInExcel(self.pfd.nozDims)
            print(self.pfd.nozDims)
        self.pfd.findVstill()
        for file in self.pfd.vstill:
            if 'w1p3' in file:
                openInPaint(file)
                return
            
    def whiteOutFiles(self, canMatch:list=[], mustMatch:list=[]) -> None:
        '''white out the files'''
        whiteOutFiles(self.folder, canMatch=canMatch, mustMatch=mustMatch)
        
    def whiteOutLast(self) -> None:
        '''white out the last image tested'''
        whiteOutFiles(self.folder, mustMatch=[self.imtag])
        
    def approve(self) -> None:
        '''mark all segmentation failures as approved'''
        if not os.path.exists(self.folder):
            return
        if not hasattr(self, 'failuredf'):
            if not hasattr(self, 'pfd'):
                self.pfd = fh.printFileDict(folder)
            if hasattr(self.pfd, 'failures'):
                self.failuredf, _ = plainIm(self.pfd.failures, ic=0)
        self.failuredf['error'] = ['approved' for i in range(len(self.failuredf))]
        plainExp(self.pfd.failures, self.failuredf, {'file':'', 'error':''})
        
class fullSequencer(fh.folderLoop):
    '''recursively run all tests on all subfolders in the folder'''
    
    def __init__(self, folders:Union[str,list], **kwargs):
        super().__init__(folders, self.fullSequence, **kwargs)
        
    def fullSequence(self, folder:str, **kwargs) -> None:
        '''get summaries from a single folder and add them to the running list'''
        self.sw = SDTWorkflow(folder, **kwargs)
        self.sw.run(**kwargs)

    def exportFailures(self, fn:str) -> None:
        '''export a list of failed files'''
        if len(self.folderErrorList)>0:
            plainExp(fn.replace('Failures', 'Errors'), pd.DataFrame(self.folderErrorList), {}, index=False)

    def run(self):
        '''analyze all folders'''
        super().run()
