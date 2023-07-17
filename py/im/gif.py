#!/usr/bin/env python
'''Functions for combining images into videos'''

# external packages
import os, sys
import imageio
from typing import List, Dict, Tuple, Union, Any, TextIO
import logging

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(currentdir)
sys.path.append(parentdir)

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)


#----------------------------------------------

def convertToFastGif(vid:str, factor:int=1, speedup:float=1, tstart:float=0, tend:float=-1):
    '''convert the mp4 to a fast gif, reducing frames by a factor of int'''
    video = imageio.get_reader(vid,  'ffmpeg')
    newName = vid.replace('.mp4', '_fast.gif')
    if os.path.exists(newName):
        os.remove(newName)
    dat = video.get_meta_data()
    fps = int(dat['fps'])
    result = imageio.get_writer(newName, fps=int((fps/factor)*speedup))
    skip = factor
    i = 0
    for num , im in enumerate(video):
        frame = video.get_data(num)
        skip = skip-1
        if skip==0:
            # Write the frame into the
            # file 'filename.avi'
            result.append_data(frame)
            skip = factor

      # When everything done, release 
    # the video capture and video 
    # write objects
    video.close()
    result.close()

    logging.info(f'Exported {newName}')
    return