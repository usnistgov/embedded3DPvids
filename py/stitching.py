#!/usr/bin/env python
'''Functions for stitching images. With code from: https://github.com/kushalvyas/Python-Multiple-Image-Stitching'''

# external packages
import cv2 as cv  # openCV version must be <=3.4.2.16, otherwise SIFT goes behind a paywall
import numpy as np 
import os
import sys
import time
import logging
from typing import List, Dict, Tuple, Union, Any, TextIO
from matplotlib import pyplot as plt

# local packages

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)

# info
__author__ = "Leanne Friedrich"
__copyright__ = "This data is publicly available according to the NIST statements of copyright, fair use and licensing; see https://www.nist.gov/director/copyright-fair-use-and-licensing-statements-srd-data-and-software"
__credits__ = ["Kushal Vyas", "Leanne Friedrich"]
__version__ = "1.0.0"
__maintainer__ = "Leanne Friedrich"
__email__ = "Leanne.Friedrich@nist.gov"
__status__ = "Development"



#----------------------------------

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        
def subfolder(fn:str, short:bool=False) -> str:
    '''get the subfolder basename from a file name'''
    if 'raw' in fn:
        # archived. need to get sample folder
        folder = os.path.dirname(os.path.dirname(os.path.dirname(fn)))
    else:
        folder = os.path.dirname(fn)
    if short:
        return os.path.basename(folder)
    else:
        return folder
        

def imshow(*args) -> None:
    '''displays cv image(s) in jupyter notebook using matplotlib'''
    if len(args)>1:
        f, axs = plt.subplots(1,len(args))
        for i, im in enumerate(args):
            if len(im.shape)>2:
                # color
                axs[i].imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
            else:
                # B&W
                axs[i].imshow(im, cmap='Greys')
    else:
        f, ax = plt.subplots(1,1)
        im = args[0]
        if len(im.shape)>2:
            ax.imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
        else:
            ax.imshow(im, cmap='Greys')
    
    
class matchers:
    '''this class contains functions'''
    
    
    def __init__(self):
        self.surf = cv.xfeatures2d.SURF_create()
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        self.flann = cv.FlannBasedMatcher(index_params, search_params)
        self.setDefaults(0,0)
        self.resetLastH()
        
    def setDefaults(self, dx:float, dy:float) -> None:
        '''set the default dx and dy'''
        self.defaultDx = dx
        self.defaultDy = dy
        
    def resetLastH(self) -> None:
        '''reset value for last Homography matrix'''
        self.lastH = np.float32([[1.,0,self.defaultDx],[0,1,self.defaultDy]])

    def showKeypoints(self, i1:np.ndarray, i2:np.ndarray, imageSet1:dict, imageSet2:dict) -> None:
        i1a = cv.drawKeypoints(i1, imageSet1['kp'], None, (255,0,0), 4)
        i2a = cv.drawKeypoints(i2, imageSet2['kp'], None, (255,0,0), 4)
        imshow(i1a, i2a)

    def checkH(self, H:np.ndarray, i2:np.ndarray) -> bool:
        '''check if H is valid for translation'''
        if abs(H[0,0]-1)>0.1 or abs(H[1,1]-1)>0.1:
            # too much scaling
            return False
        elif abs(H[0,1]>0.1) or abs(H[1,0]>0.1):
            # too much rotation
            return False
        elif abs(H[0,2])>i2.shape[1] or abs(H[1,2])>i2.shape[0]:
            # too much translation
            return False
        fudge = 5 # fudge factor
        if not (self.defaultDy)==0:
            if abs(H[1,2])-abs(self.defaultDy) > i2.shape[0]/fudge:
                # difference between expected and actual shift is greater than height/5
                return False
        if not self.defaultDx==0:
            if abs(H[0,2])-abs(self.defaultDx) > i2.shape[1]/fudge:
                return False
        else:
            return True
        
    def errorMatch(self, defaultToLastH:bool, debug:bool, i1:np.ndarray, i2:np.ndarray, imageSet1:dict, imageSet2:dict) -> np.ndarray:
        '''return error value'''
        if debug:
            self.showKeypoints(i1,i2,imageSet1, imageSet2)
        if defaultToLastH:
            # return previous H in the case of failure 
            H = self.lastH.copy()
            if H[0,2]>0: # if we moved up, we need to move up twice as much
                H[0,2] = H[0,2] + self.defaultDx
            if H[1,2]>0:
                H[1,2] = H[1,2] + self.defaultDy
            self.lastH = H
            return H
        else:
            return []

    def match(self, i1:np.ndarray, i2:np.ndarray, direction=None, rigid:bool=True, debug:bool=False, defaultToLastH:bool=True) -> np.ndarray:
        '''find homography matrix that relates the two images. set rigid true to only allow rigid transform, i.e. translation, rotation, scaling. set rigid false to allow warping'''
        imageSet1 = self.getSURFFeatures(i1)
        imageSet2 = self.getSURFFeatures(i2)
        matches = self.flann.knnMatch(imageSet2['des'],imageSet1['des'],k=2)
        good = []
        for i , (m, n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                good.append((m.trainIdx, m.queryIdx))

        if len(good) <= 4:
            # not enough points
            return self.errorMatch(defaultToLastH, debug, i1, i2, imageSet1, imageSet2)
            
        # enough points
        pointsCurrent = imageSet2['kp']
        pointsPrevious = imageSet1['kp']

        matchedPointsCurrent = np.float32([pointsCurrent[i].pt for (__, i) in good])
        matchedPointsPrev = np.float32([pointsPrevious[i].pt for (i, __) in good])

        if rigid:
            H, _ = cv.estimateAffinePartial2D(matchedPointsCurrent, matchedPointsPrev)
            # sanity check
            if not self.checkH(H, i2):
                # too much rotation, scaling, or translation
                return self.errorMatch(defaultToLastH, debug, i1, i2, {'kp':pointsPrevious}, {'kp':pointsCurrent})
            if debug:
                self.showKeypoints(i1,i2,{'kp':pointsPrevious}, {'kp':pointsCurrent})
            H = np.append(H, [[0,0,1]], axis=0)
        else:
            H, s = cv.findHomography(matchedPointsCurrent, matchedPointsPrev, cv.RANSAC, 4)
        self.lastH = H
        self.setDefaults(H[1,2], H[0,2])
        return H
            

    def getSURFFeatures(self, im):
        gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        kp, des = self.surf.detectAndCompute(gray, None)
        return {'kp':kp, 'des':des}
    
#----------------------------------

class Region:
    '''holds info about overlaps'''
    
    def __init__(self, im1:np.ndarray, im2:np.ndarray, dx:int, dy:int):
        '''get the overlap region'''
        self.dx = dx
        self.dy = dy
        self.Wim1 = im1.shape[1]
        self.Him1 = im1.shape[0]
        self.Dim1 = im1.shape[2]
        self.Wim2 = im2.shape[1]
        self.Him2 = im2.shape[0]
        self.Dim2 = im2.shape[2]
        self.Wnew = self.Wim1 + abs(dx)
        self.Hnew = self.Him1 + abs(dy)
        if not self.Dim1==self.Dim2:
            raise ValueError('Images are not in same color space')
        else:
            self.Dnew = self.Dim1
        
        # determine positions of images in new image
        self.im1x0, self.im1y0 = self.convertIm1Coords(0,0,dx,dy)
        self.im2x0, self.im2y0 = self.convertIm2Coords(0,0,dx,dy)

        self.im1xf = self.im1x0+self.Wim1
        self.im1yf = self.im1y0+self.Him1
        self.im2xf = self.im2x0+self.Wim2
        self.im2yf = self.im2y0+self.Him2
        
        self.ox0 = max(self.im1x0, self.im2x0)
        self.oxf = min(self.im1xf, self.im2xf)
        self.oy0 = max(self.im1y0, self.im2y0)
        self.oyf = min(self.im1yf, self.im2yf)
                    
        self.Ho = self.oyf-self.oy0
        self.Wo = self.oxf-self.ox0
        
        self.im1xc = int((self.im1xf+self.im1x0)/2)
        self.im1yc = int((self.im1y0+self.im1yf)/2)
        self.im2xc = int((self.im2xf+self.im2x0)/2)
        self.im2yc = int((self.im2y0+self.im2yf)/2)
        self.oxc = int((self.oxf+self.ox0)/2)
        self.oyc = int((self.oy0+self.oyf)/2)
        self.newyc = int(self.Hnew/2)
        self.newxc = int(self.Wnew/2)
        
    def convertIm1Coords(self, x0:int, y0:int, dx:float, dy:float) -> Tuple[int,int]:
        '''convert im1 coordinates to the new coordinate system'''
        if dx>0:
            im1x0 = 0 + x0
        else:
            im1x0 = abs(dx) + x0
        if dy>0:
            im1y0 = 0 + y0
        else:
            im1y0 = abs(dy) + y0
        return im1x0, im1y0
    
    def convertIm2Coords(self, x0:int, y0:int, dx:float, dy:float) -> Tuple[int, int]:
        '''convert im2 coordinates to the new coordinate system'''
        if dx>0:
            im2x0 = abs(dx) + x0
        else:
            im2x0 = 0 + x0
        if dy>0:
            im2y0 = abs(dy) + y0
        else:
            im2y0 = 0 + y0
        return im2x0, im2y0

    def table(self) -> None:
        '''print a table of values'''
        
        s = f'dx:{self.dx}, dy:{self.dy}\n'
        s+='\tim1\tim2\tst\toverlap\n'
        s+=f'H\t{self.Him1}\t{self.Him2}\t{self.Hnew}\t{self.Ho}\n'
        s+=f'W\t{self.Wim1}\t{self.Wim2}\t{self.Wnew}\t{self.Wo}\n'
        s+=f'D\t{self.Dim1}\t{self.Dim2}\t{self.Dnew}\t{self.Dnew}\n'
        s+=f'y0\t{self.im1y0}\t{self.im2y0}\t0\t{self.oy0}\n'
        s+=f'yc\t{self.im1yc}\t{self.im2yc}\t{self.newyc}\t{self.oyc}\n'
        s+=f'yf\t{self.im1yf}\t{self.im2yf}\t{self.Hnew}\t{self.oyf}\n'
        s+=f'x0\t{self.im1x0}\t{self.im2x0}\t0\t{self.ox0}\n'
        s+=f'xc\t{self.im1xc}\t{self.im2xc}\t{self.newxc}\t{self.oxc}\n'
        s+=f'xf\t{self.im1xf}\t{self.im2xf}\t{self.Wnew}\t{self.oxf}\n'
        
        logging.info(s)

    
#----------------------------------

class Stitch:
    '''Class that holds info about the stitched images'''
    
    
    def __init__(self, filenames:List[str], **kwargs) -> None:
        for file in filenames:
            self.readImage(file, **kwargs)
        self.killFrame = ''
        self.filenames = filenames
        self.imagesPerm = self.images.copy()
        self.filenamesPerm = self.filenames.copy()
        self.matcher = matchers()
        
    def readImage(self, file:str, crop:int=4, **kwargs) -> None:
        im = cv.imread(file)
        h = im.shape[0]
        w = im.shape[1]
        im = im[crop:h-crop, crop:w-crop]
        try:
            self.images.append(im)
        except:
            self.images = [im]
    
    def masks(self, r:Region) -> Tuple[np.ndarray, np.ndarray]:
        '''get blending masks for the two images'''
        new = np.zeros((r.Hnew, r.Wnew, r.Dnew), np.float32)
        
        # gradient mask
        c1 = [self.prevCenter[0], self.prevCenter[1], 0] # center of previous image
        c2 = [r.im2yc, r.im2xc, 0] # center of image 2
        v12 = np.subtract(c2,c1) # vector from 1 to 2
        v0 = [1,0,0] # vertical vector
        distC = np.linalg.norm(v12) # distance between centers
        theta = np.arccos(np.dot(v12, v0)/distC) # rotation angle, length of v0 is 1
        if v12[0]>0: 
            # vector to the right: theta<0
            theta = -theta

        Wgrad = int(np.sqrt(r.Wnew**2+r.Hnew**2))  + 4
            # initial gradient mask width bigger than it needs to be
        gradient0 = np.float32([[[i/distC]*r.Dnew]*Wgrad for i in range(int(distC))])
            # distO x Wgrad matrix gets more intense toward the bottom 
        # pad bottom with 1s, top with 0s
        padsize = int((Wgrad-distC)/2)
        toppad = np.float32([[[0]*r.Dnew]*Wgrad for i in range(padsize)])
        gradient0 = np.append(toppad, gradient0, axis=0)
        botpad = np.float32([[[1]*r.Dnew]*Wgrad for i in range(padsize)])
        gradient0 = np.append(gradient0, botpad, axis=0)
        Wg0 = gradient0.shape[1] # width of big gradient mask
        Hg0 = gradient0.shape[0] # height
        cyg0 = int(Hg0/2) # center in y
        cxg0 = int(Wg0/2) # center in x
        M = cv.getRotationMatrix2D((cyg0, cxg0), theta*180/np.pi, 1.0)
        rotated = cv.warpAffine(gradient0, M, (Hg0, Wg0)) 
            # rotate the mask so the gradient goes along the vector between centers
        rdym = int(np.floor(r.Ho/2)) # change in y, minus direction
        rdyp = int(np.ceil(r.Ho/2)) # change in y, plus direction
        rdxm = int(np.floor(r.Wo/2)) # change in x, minus direction
        rdxp = int(np.ceil(r.Wo/2))
        rotated = rotated[cyg0-rdym:cyg0+rdyp, cxg0-rdxm:cxg0+rdxp, :]
            # crop to size of overlap region
        
        im2mask = new.copy()
        im2mask[r.im2y0:r.im2yf, r.im2x0:r.im2xf, :] = 1
        im2mask[r.oy0:r.oyf, r.ox0:r.oxf, :] = rotated # set the overlap region to the mask
        
        im1mask = new.copy()
        im1mask[r.im1y0:r.im1yf, r.im1x0:r.im1xf] = 1
        im1mask[r.oy0:r.oyf, r.ox0:r.oxf, :] = 1-rotated # set the overlap region to the mask
        
        return im1mask, im2mask
        
        
    def blend(self, im2:np.ndarray, r:Region) -> None:
        '''blend the stitched image in the given region. blended is the final image. r is dictionary that holds info about regions'''
        new = np.zeros((r.Hnew, r.Wnew, r.Dnew), np.uint8)
        
        # move the images
        im2moved = new.copy()
        im2moved[r.im2y0:r.im2yf, r.im2x0:r.im2xf, :] = im2
        stitchedMoved = new.copy()
        stitchedMoved[r.im1y0:r.im1yf, r.im1x0:r.im1xf, :] = self.stitched
        
        # create masks
        im1mask, im2mask = self.masks(r)
        
        im2moved2 = np.uint8(np.multiply(im2moved,im2mask))
        stitchedMoved2 = np.uint8(np.multiply(stitchedMoved,im1mask))
        
        blended = cv.add(im2moved2, stitchedMoved2)
                    
        # keep new center as 
        self.prevCenter[0] = r.im2yc
        self.prevCenter[1] = r.im2xc
        return blended
    
    def newFN(self, duplicate:bool=True, tag:str='', **kwargs) -> str:
        '''get a file name. duplicate=True to get a new file name if there is already a file, duplicate=False to return existing file name'''
        fn = (self.filenames)[0]
        folder = subfolder(fn)
        i = 0
        ext = os.path.splitext(fn)[-1]
        while (duplicate and os.path.exists(fn)) or (not duplicate and i==0):
            # if duplicate is true, keep going until we get a new file name
            # if duplicate is false, stop at 1st iteration
            fn = os.path.join(folder, tag+"_{0:0=2d}".format(i)+ext)
            i+=1
        return fn
    
    def returnStitch(self, export:bool=False, tag:str='', debug:bool=False, clearEntries:bool=False, duplicate:bool=True, retval:int=0) -> int:
        '''return value from stitchTranslate. Return 0 if stitched, 1 if not'''

        if debug:
            imshow(self.stitched)
        if export:
            fn = self.newFN(duplicate=duplicate, tag=tag)
            cv.imwrite(fn, self.stitched)
            logging.info(f'Wrote image {os.path.basename(fn)}')
        if clearEntries:
            self.images = self.images[self.killFrame:]
            self.filenames = self.filenames[self.killFrame:]
        return retval
    
    def stitchTranslate(self, **kwargs) -> int:
        '''stitch the images, but only use translation. clearEntries=true to remove stitched images from queue when done. Returns 0 if stitched, 1 if not'''
        if 'duplicate' in kwargs:
            if not kwargs['duplicate']:
                # check if there is already a file
                fn = self.newFN(**kwargs)
                if os.path.exists(fn):
                    return 1
        
        logging.debug(f'Stitching {len(self.images)} images in {subfolder(self.filenames[0], short=True)}')
        self.stitched = self.images[0]
        if len(self.images)>1:
            self.matcher.resetLastH()
            for i, im in enumerate(self.images[1:]):
                H = self.matcher.match(self.stitched, im) # homography matrix, no warp
                if len(H)>0:
                    dx = int(H[0,2]) # translation in x
                    dy = int(H[1,2]) # translation in y
                    r = Region(self.stitched, im, dx, dy) # get positions
                    if i==0:
                        self.prevCenter = [r.im1yc, r.im1xc]
                    else:
                        pcx, pcy = r.convertIm1Coords(self.prevCenter[1], self.prevCenter[0], dx,dy)
                        self.prevCenter = [pcy, pcx]
                    if 'debug' in kwargs and kwargs['debug']:
                        r.table()
                    self.stitched = self.blend(im, r)
                else:
                    # no homography found, stop stitching
                    self.killFrame = i+1
                    logging.debug(f'Failed to find homography for image {i+1}')
                    return self.returnStitch(retval=1, **kwargs)
        return self.returnStitch(**kwargs)
    
    def stitchAll(self, **kwargs) -> None:
        '''stitch all images in the list, into separate images if necessary'''
        try:
            stitches = []
            while len(self.images)>0:
                self.stitchTranslate(**kwargs)   
                stitches.append(self.stitched.copy())
            if 'debug' in kwargs and kwargs['debug']:
                imshow(*stitches)
        except:
            pass
        self.images = self.imagesPerm.copy()
        self.filenames = self.filenamesPerm.copy()

        
        
#--------------------------------------------------
#             ARCHIVE 
    
    #     def move(self, im:np.ndarray, H:np.ndarray, r:Struct) -> np.ndarray:
#         if len(H)>2:
#             H2 = H.copy()
#             H2 = H2[0:2, :]
#         else:
#             H2 = np.array(H, np.float_)
#         dst = cv.warpAffine(im, H2, (r.Wnew, r.Hnew))
#         return dst


        
#     def emptyCropStruct(self) -> Struct:
#         '''get an empty crop struct'''
#         d = {'im1x0':0, 'im2x0':0, 'im1y0':0, 'im2y0':0}
#         return Struct(**d)
        
#     def cropToLast(self, im1:np.ndarray, im2:np.ndarray) -> Tuple[np.ndarray, np.ndarray, Struct]:
#         '''assume that the new homography matrix will be similar to the last. crop to the overlap region that would match that homography matrix. im1 is stitched, im2 is raw'''
#         dx = int(self.lastH[0,2]) # translation in x
#         dy = int(self.lastH[1,2]) # translation in y
#         s = self.emptyCropStruct()
#         if dx==0 and dy==0:
#             return im1, im2, s
#         h = im2.shape[0]
#         w = im2.shape[1]  
#         pad = 0.05
#         if dy>0:
#             s.im1x0 = max(0, int(dy-pad*h)) # pad estimate by 0.1*H
#             s.im1xf = h
#             s.im2x0 = 0
#             s.im2xf = min(h, int(h-dy+pad*h))
#         else:
#             s.im1x0 = 0
#             s.im1xf = min(h, int(h-abs(dy)+pad*h))
#             s.im2x0 = max(0, int(abs(dy)-pad*h))
#             s.im2xf = h
#         if dx>0:
#             s.im1y0 = max(0, int(dx-pad*w)) # pad estimate by 0.1*H
#             s.im1yf = w
#             s.im2y0 = 0
#             s.im2yf = min(w, int(w-dx+pad*w))
#         else:
#             s.im1y0 = 0
#             s.im1yf = min(w, int(w-abs(dx)+pad*w))
#             s.im2y0 = max(0, int(abs(dx)-pad*w))
#             s.im2yf = w
#         im1crop = im1[s.im1x0:s.im1xf, s.im1y0:s.im1yf]
#         im2crop = im2[s.im2x0:s.im2xf, s.im2y0:s.im2yf]
#         return im1crop, im2crop, s 