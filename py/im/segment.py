#!/usr/bin/env python
'''Morphological operations applied to images'''

# external packages
import os
import sys
import logging
from typing import List, Dict, Tuple, Union, Any, TextIO

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
from s_segmenter import *
from s_segmenterSingle import *
from s_segmentCombiner import *
from s_segmenterDF import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)

#----------------------------------------------
