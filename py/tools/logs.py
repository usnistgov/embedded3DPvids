#!/usr/bin/env python
'''Functions for handling log files'''

# external packages
import os
import sys
import logging, platform, socket

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
try:
    from config import cfg
except:
    pass

#------------------------------------------------------------------------------------------------- 

def logFN(scriptFile:str) -> str:
    '''Get a log file name, given a script file name'''
    compname = socket.gethostname()
    base = os.path.splitext(os.path.basename(scriptFile))[0]
    dirpath = os.path.dirname(os.path.realpath(__file__))
    try:
        cfgbase = cfg.path.logs
    except:
        cfgbase = 'logs'
    logfolder = os.path.join(dirpath, cfgbase)
    if not os.path.exists(logfolder):
        logfolder = os.path.join(os.path.dirname(dirpath), cfgbase)
        if not os.path.exists(logfolder):
            logfolder = dirpath
    return os.path.join(logfolder,f'{base}_{compname}.log')

def openLog(f:str, LOGGERDEFINED:bool, level:str="INFO", exportLog:bool=True) -> bool:
    '''this code lets you create log files, so you can track when you've moved files. f is the file name of the script calling the openLog function'''
    if not LOGGERDEFINED:
        loglevel = getattr(logging,level)
        root = logging.getLogger()
        if len(root.handlers)>0:
            # if handlers are already set up, don't set them up again
            return True
        root.setLevel(loglevel)

        # send messages to file
        if exportLog:
            logfile = logFN(f)
            filehandler = logging.FileHandler(logfile)
            filehandler.setLevel(loglevel)
            formatter = logging.Formatter("%(asctime)s/{}/%(levelname)s: %(message)s".format(socket.gethostname()), datefmt='%b%d/%H:%M:%S')
            filehandler.setFormatter(formatter)
            root.addHandler(filehandler)
            logging.info(f'Established log: {logfile}')

        # print messages
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(loglevel)
        formatter2 = logging.Formatter('%(levelname)s: %(message)s')
        handler.setFormatter(formatter2)
        root.addHandler(handler)
        LOGGERDEFINED = True
        
    return LOGGERDEFINED
