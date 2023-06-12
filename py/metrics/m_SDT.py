#!/usr/bin/env python
'''collects all metrics functions into one file'''

# external packages
import os,sys

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
from m_file.file_vert_SDT import *
from m_file.file_xs_SDT import *
from m_file.file_horiz_SDT import *
from m_file.file_ML import *
from m_file.file_unit import *

from m_folder.folder_vert_SDT import *
from m_folder.folder_xs_SDT import *
from m_folder.folder_horiz_SDT import *
from m_folder.folder_metric_exporter import *

from m_summarizer.summarizer_SDT import *

from m_summary.summary_SDT import *

from crop_locs import *
from m_plot.m_plots import *
from m_tools import *
import progDim.prog_dim as pgd

# logging


#--------------------------------

class fitChanger(fh.folderLoop):
    '''change the pressure vs speed model used in progDims and measurements for all folders and re-measure'''
    
    def __init__(self, folders:Union[str, list], a:float, b:float, c:float, canMatch:list=['Horiz', 'Vert', 'XS'], **kwargs) -> None:
        self.a = a
        self.b = b
        self.c = c
        super().__init__(folders, self.changeFitAndMeasure, canMatch=canMatch, **kwargs)
        
    def changeFit(self, folder:str):
        '''change the pressure vs speed model used in progDims and measurements for a single folder'''
        pfd = fh.printFileDict(folder)
        mf = pfd.meta[0]

        meta,u = plainImDict(mf, unitCol=1, valCol=2)
        if meta['caliba_channel_0'] == self.a and meta['calibb_channel_0'] == self.b and meta['calibc_channel_0'] == self.c:
            return pfd
        meta['caliba_channel_0'] = self.a
        meta['calibb_channel_0'] = self.b
        meta['calibc_channel_0'] = self.c
        shutil.copyfile(mf, mf.replace('meta', 'metOrig'))
        plainExpDict(mf, meta, u, quotechar='"')
        return pfd
        
    def changeFitAndMeasure(self, folder:str):
        pfd = self.changeFit(folder)
        pdim = getProgDims(folder, pfd=pfd)
        pdim.exportAll(overwrite=True)
        if 'Horiz' in folder:
            func = folderHorizSDT
        elif 'Vert' in folder:
            func = folderVertSDT
        elif 'XS' in folder:
            func = folderXSSDT
        else:
            raise ValueError(f'Could not determine folder type in {folder}')
        fvs = func(folder, overwriteMeasure=True, overwriteSummary=True, diag=0, pfd=pfd, pv=pdim.pv, pg=pdim)
        fvs.measureFolder()
        fvs.summarize()