#!/usr/bin/env python
'''Functions for plotting video and image data. Adapted from https://github.com/usnistgov/openfoamEmbedded3DP'''

# external packages
import os, sys
import traceback
import logging
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
from typing import List, Dict, Tuple, Union, Any, TextIO
import re
import numpy as np
import cv2 as cv
import matplotlib.ticker as mticker

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
import file.file_handling as fh
from val.v_print import *
from progDim.prog_dim import getProgDimsPV
from vid.noz_detect import nozData
import im.morph as vm
from im.crop import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)
    
# plotting
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rc('font', family='Arial')
matplotlib.rc('font', size='10.0')


#----------------------------------------------

class folderImages:
    '''imports, crops, and combines images given tags for a single folder'''
    
    def __init__(self, pv:printVals, tag:Union[str, list], concat:str='h', removeBackground:bool=False, whiteBalance:bool=True, normalize:bool=True, removeBorders:bool=False, times:bool=False, scale:float=0.95, rotate:bool=False, spacerThickness:int=10, **kwargs):
        self.concat = concat
        self.pv = pv
        self.removeBackground = removeBackground

        self.whiteBalance = whiteBalance
        self.normalize = normalize
        self.removeBorders = removeBorders
        self.folder = pv.printFolder
        self.tag = tag
        self.times = times
        self.imfiles = []
        self.scale = scale
        self.rotate = rotate
        self.spacerThickness=spacerThickness
        if not type(self.tag) is list:
            self.tag = [self.tag]
        self.kwargs = kwargs

    def picFileFromFolder(self, tag:str) -> str:
        '''get the file name of the image'''
        imfile = ''
        if not 'raw' in tag:
            for f in os.listdir(self.folder):
                if tag in f and not ('vid' not in tag and 'vid' in f):
                    # use last image in folder that has the tag
                    imfile = os.path.join(self.folder, f)
        if not os.path.exists(imfile):
            # no file in the main folder. search archives
            raw = os.path.join(self.folder, 'raw')
            if os.path.exists(raw):
                out = self.parseTag(tag)
                archive = os.path.join(raw, out['tag'])
                if not os.path.exists(archive):
                    archive = os.path.join(raw, tag[:-1]) # remove last char, e.g. for xs1 just use xs
                if os.path.exists(archive):
                    l = os.listdir(archive)
                    if len(l)>0:
                        imfile = os.path.join(archive, l[out['num']]) # if there is a file in the archive folder, use it
        return imfile

    def picFromFolder(self, tag:str) -> np.array:
        '''gets one picture from a folder
        returns the picture
        tag is the name of the image type, e.g. 'xs1'. Used to find images. '''
        imfile = self.picFileFromFolder(tag)
        if os.path.exists(imfile):
            self.imfiles.append(imfile)
            im = cv.imread(imfile)
            if self.removeBackground:
                self.getNozData()
                im = self.nd.subtractBackground(im)
            if self.whiteBalance:
                im = vm.white_balance(im)
            if self.normalize:
                im = vm.normalize(im)
            return im
        else:
            return []
        
    def getNozData(self):
        '''initialize the nozData object'''
        if not hasattr(self, 'nd'):
            self.nd = nozData(self.pv.printFolder)
            
    def getProgDims(self):
        '''initialize the progDims object'''
        if not hasattr(self, 'pg'):
            self.pg  = getProgDimsPV(self.pv)
        
    def getCrops(self, tag:str) -> dict:
        '''get the crop dictionary given a single tag, e.g. l1d'''
        if not 'crops' in self.kwargs:
            return {}
        self.getProgDims()
        self.getNozData()
        return relativeCrop(self.pg, self.nd, tag, self.kwargs['crops'])

    def cropImage(self, im:np.array, tag:str) -> np.array:
        '''crop the image and return width, height, image. Crops must contain x0, xf, y0, yf. x0 and xf are from left. y0 and yf are from bottom.'''
        crops = self.getCrops(tag)
        return imcropPad(im, crops)
    
    def emptyCrop(self, tag:str) -> np.array:
        '''create an empty image from the crops'''
        errorRet = []
        crops = self.getCrops(tag)
        if 'yf' in crops and 'y0' in crops and 'xf' in crops and 'x0' in crops:
            return 255*np.ones((int(crops['yf']-crops['y0']), int(crops['xf']-crops['x0']), 3), dtype=np.uint8)  # no image. return an empty image
        else:
            return errorRet

    def importAndCrop(self, tag:str) -> np.array:
        '''import and crop an image from a folder, given a tag. crops can be in kwargs
        whiteBalance=True to do auto white balancing
        normalize=True to do auto brightness adjustment
        removeBorders to remove the dark borders from the edges of the image, e.g. above the bath or the walls of the bath'''
        im = self.picFromFolder(tag)
        if type(im) is list:
    #         logging.debug(f'Image missing: {folder}')
            return self.emptyCrop(tag)
        im = self.cropImage(im, tag)
        if self.removeBorders:
            im = vm.removeBorders(im, normalizeIm=False)
        return im
    
    def combineImages(self, im1:np.array) -> None:
        '''combine all of the images into a single image'''
        if self.concat=='h':
            # stack horizontally
            if im1.shape[0]>self.im.shape[0]:
                # pad the combined image
                pad = im1.shape[0]-self.im.shape[0]
                self.im = cv.copyMakeBorder(self.im, pad, 0, 0,0, cv.BORDER_CONSTANT, value=(255,255,255)) 
            elif self.im.shape[0]>im1.shape[0]:
                # pad the new image
                pad = self.im.shape[0]-im1.shape[0]
                im1 = cv.copyMakeBorder(im1, pad, 0, 0,0, cv.BORDER_CONSTANT, value=(255,255,255))
                
            if self.spacerThickness>0:
                # add a spacer between the images
                spacer = 255*np.ones((im1.shape[0], self.spacerThickness, 3), dtype=np.uint8)  # create a white gap between images
                self.im = cv.hconcat([self.im, spacer, im1])
            else:
                # don't add a spacer
                self.im = cv.hconcat([self.im, im1])
        elif self.concat=='v':
            # stack vertically
            if im1.shape[1]>self.im.shape[1]:
                # pad the combined image
                pad = im1.shape[1]-self.im.shape[1]
                self.im = cv.copyMakeBorder(self.im, 0, 0,pad, 0, cv.BORDER_CONSTANT, value=(255,255,255)) 
            elif self.im.shape[1]>im1.shape[1]:
                # pad the new image
                pad = self.im.shape[1]-im1.shape[1]
                im1 = cv.copyMakeBorder(im1,0,0, pad,0, cv.BORDER_CONSTANT, value=(255,255,255))
                
            if self.spacerThickness>0:
                # add space between the images
                spacer = 255*np.ones((self.spacerThickness, im1.shape[1], 3), dtype=np.uint8)  # create a white gap between images
                self.im = cv.vconcat([self.im, spacer, im1])
            else:
                # stick iamges together with no gap
                self.im = cv.vconcat([self.im, im1])
        else:
            raise ValueError(f'Bad value for concat: {self.concat}')
        return 


    def getImages(self) -> np.array:
        '''get the images, crop them, and put them together into one image'''
        self.im = []
        try:
            for i,t in enumerate(self.tag):
                im1 = self.importAndCrop(t)
                if len(im1)>0:
                    if self.rotate:
                        im1 = np.rot90(im1, k=3, axes=(0,1))
                        im1 = np.flip(im1, axis=0)
                    if len(self.im)==0:
                        self.im = im1
                    else:
                        # pad the images to make them the same shape
                        self.combineImages(im1)
                        
        except Exception as e:
            logging.error(f'Cropping error: {str(e)}')
            traceback.print_exc()
            raise e
        if len(self.im)==0:
            raise 'No images found'
        return t
    
    def wfull(self) -> Tuple[float,float,float,float]:
        '''get the width and height of the combined image'''
        if hasattr(self, 'im') and len(self.im)>0:
            height,width = self.im.shape[0:2]
        else:
            height = 600
            width = 800
        if 'crops' in self.kwargs:
            crops = self.kwargs['crops']
            if type(crops) is dict and not ('xf' in crops or 'x0' in crops or 'y0' in crops or 'yf' in crops or 'w' in crops or 'h' in crops):
                # multiple crop zones indicated for different images
                c = pd.DataFrame(crops)
                if not 'x0' in c:
                    c['x0'] = 0
                if not 'xf' in c:
                    c['xf'] = width
                if not 'y0' in c:
                    c['y0'] = height
                if not 'yf' in c:
                    c['yf'] = 0
                c['w'] = c['xf']-c['x0']
                c['h'] = c['yf']-c['y0']
                widthI = c.w.sum()
                heightI = c.h.max()
            else:
                # single crop zone indicated
                if self.concat=='h':
                    hs = 1
                    if type(self.tag) is list:
                        ws = len(self.tag)
                    else:
                        ws = 1
                elif self.concat=='v':
                    ws = 1
                    if type(self.tag) is list:
                        hs = len(self.tag)
                    else:
                        hs = 1
                else:
                    raise ValueError(f'Unexpected concat direction: {self.concat}')
                if 'yf' in crops and 'y0' in crops and 'xf' in crops and 'x0' in crops:
                    if self.rotate:
                        w = crops['yf']-crops['y0']
                        h = crops['xf']-crops['x0']
                    else:
                        w = crops['xf']-crops['x0']
                        h = crops['yf']-crops['y0']
                    if crops['yf']<0:
                        heightI = height*hs
                    else:
                        heightI = (h)*hs
                    if crops['xf']<0:
                        widthI = width*ws
                    else:
                        widthI = (w)*ws
                    # use intended height/width to scale all pictures the same
                elif 'w' in crops and 'h' in crops:
                    # widthI = min(width, crops['w'])*ws
                    # heightI = min(height, crops['h'])*hs
                    if self.rotate:
                        h = crops['w']
                        w = crops['h']
                    else:
                        w = crops['w']
                        h = crops['h']
                    widthI = w*ws
                    heightI = h*hs
                else:
                    heightI = height*hs
                    widthI = width*ws
#                 print(widthI, heightI, width, height, hs, ws)
        else:
            widthI = width
            heightI = height

        return widthI, heightI, width,height
    
    def getWidthScaling(self, dx0:float, dy0:float) -> Tuple[float,float,float]:
        '''determine how to scale the image. dx0 is the space between images on the plot, in plot coordinates. tag is the line type, e.g. xs or horiz'''

        widthI, heightI, width, height = self.wfull()
        # fix the scaling
        pxperunit = max(widthI, heightI) # image pixels per image block
        if widthI>heightI:
            dx = dx0*(width/pxperunit)
            dy = dx*height/width
        else:
            dy = dy0*height/pxperunit
            dx = dy*width/height

        return dx, dy, pxperunit
    
    def parseTag(self, tag:str) -> dict:
        '''parse the tag into relevant names'''
        out = {'raw':False, 'tag':'', 'num':0}
        if 'vid' in tag:
            out['tag']=tag
            return out
        if '_' in tag:
            spl = re.split('_',tag)
            if 'raw' in spl:
                out['raw'] = True
                spl.remove('raw')
            for s in spl:
                try:
                    num = int(s)
                except:
                    out['tag']=s
                else:
                    out['num']=num
        else:
            out['tag'] = tag
        return out

    def picPlotOverlay(self, t:dict, s:float, dx0:float, dy0:float):
        '''draw an annotation, e.g. a circular or rectangular scale bar, over the image
        pxperunit is the image pixels per image block
        t is a dictionary holding info about the line, including, e.g. {'tag':'xs1'}
        dx0 is the spacing between images in plot coords
        x0 is the x position of the annotation
        y0 is the y position of the annotation
        s is the scaling to use to add white space between images
        ax is the axis to plot this annotation on
        '''
        if not 'overlay' in self.kwargs:
            return
        overlay = self.kwargs['overlay']
        file = self.picFileFromFolder(self.parseTag(t)['tag'])
        scale = float(fh.fileScale(file))
        pxPerBlock = self.pxperunit/s # rescale px to include white space
        realPxPerBlock = pxPerBlock/scale # rescale to include shrinkage of original image
        mmPerBlock = realPxPerBlock/self.pv.geo.pxpmm # mm per block: pixels per image block, scaled by s is displayed size
        mmPerPlotUnit = mmPerBlock/(2*max(dx0, dy0)) # convert to dimensions of the plot
        if not overlay['shape'] in ['rectangle', 'circle', '3circles', '2circles']:
            return

        circlewMM = self.pv.dEst   # estimated filament diameter
        circlewPlotUnits = circlewMM/mmPerPlotUnit # diameter in plot units
        x0 = self.x0
        y0 = self.y0
        if 'dx' in overlay:
            x0 = x0+overlay['dx']*dx0
        if 'dy' in overlay:
            y0 = y0+overlay['dy']*dy0

        if 'color' in overlay:
            color = overlay['color']
        else:
            color = 'black'

        if overlay['shape']=='circle':
            # plot circle
            circle2 = plt.Circle((x0, y0), circlewPlotUnits/2, color=color, fill=False)
            ax.add_patch(circle2)
        elif overlay['shape']=='3circles' or overlay['shape']=='2circles':
            # plot circle
            if hasattr(self.pv,'spacing'):
                spacing = self.pv.spacing
                num = int(overlay['shape'][0])
            else:
                spacing = 0
                num = 1
            for i in range(num):
                if 'shiftdir' in overlay:
                    shiftdir = overlay['shiftdir']
                else:
                    shiftdir = 'x'
                if shiftdir=='x':
                    x = x0+i*circlewPlotUnits*spacing
                    y = y0
                else:
                    y = y0+i*circlewPlotUnits*spacing
                    x = x0
                circle2 = plt.Circle((x, y), circlewPlotUnits/2, color=color, fill=False)
                self.ax.add_patch(circle2)
        elif overlay['shape']=='rectangle' and ('w' in overlay or 'h' in overlay):
            # plot rectangle
            if 'w' in overlay:
                w = overlay['w']/mmPerPlotUnit
                h = circlewPlotUnits
            else:
                w = circlewPlotUnits
                h = overlay['h']/mmPerPlotUnit
            y0 = y0-h/2
            x0 = x0-w/2
            rect = plt.Rectangle((x0, y0), w, h, color=color, fill=True, edgecolor=None)
            self.ax.add_patch(rect)
            
    def picPlotTimes(self) -> None:
        '''add timestamps to the plots'''
        if not self.times:
            return
        
        if 'overlay' in self.kwargs and 'color' in self.kwargs['overlay']:
            color = self.kwargs['overlay']['color']
        else:
            color = 'black'
        pg, u = plainIm(self.pv.pfd.progDims)
        for i,tag in enumerate(self.tag):
            ti = pg[pg.name==tag].iloc[0]['tpic']
            if 'XS' in self.folder:
                tag0 = tag[:4]
            else:
                tag0 = tag[:4]+'p5'
            t0 = pg[pg.name==tag0].iloc[0]['tpic']
            dt = float(ti)-float(t0)
            label = f't = {dt:0.2f} s'
            ypos = 1-(i/len(self.tag))-0.05
            self.ax.text(0.1, ypos, label, horizontalalignment='left', verticalalignment='top', transform=self.ax.transAxes, fontsize=8, color=color)

    def picPlot(self, cp, dx0:float, dy0:float) -> None:
        '''plots picture from just one folder.
        cp is the comboPlot object that stores the plot. 
        dx0 is the spacing between images in plot space, e.g. 0.7'''

        # determine where to plot the image
        try:
            x0, y0, axnum = cp.vvplot(self.pv)
            self.ax = cp.axs[axnum]
            self.x0 = x0
            self.y0 = y0
        except ValueError:
            return
        except Exception as e:
            if not 'already filled' in str(e):
                logging.error(f'Positioning error: {str(e)}')
            traceback.print_exc()
            return

        # get the images
        try:
            t = self.getImages()
        except:
            logging.error(f'Error getting images from {self.pv.printFolder}')
            traceback.print_exc()
            return

        # get scaling
        if self.rotate:
            temp = dx0
            dx0 = dy0
            dy0 = dx0
        dx, dy, self.pxperunit = self.getWidthScaling(dx0, dy0)

        s = self.scale # scale images to leave white space
        self.im = cv.cvtColor(self.im, cv.COLOR_BGR2RGB)
        # plot the image
        self.ax.imshow(self.im, extent=[self.x0-dx*s, self.x0+dx*s, self.y0-dy*s, self.y0+dy*s])
        self.picPlotOverlay(t, s, dx0, dy0)
        self.picPlotTimes()