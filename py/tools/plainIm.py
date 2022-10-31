#!/usr/bin/env python
'''Functions for importing csv'''

# external packages
import os
import pandas as pd
from typing import List, Dict, Tuple, Union, Any, TextIO
import logging
import numpy as np
import re

# local packages


# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


#----------------------------------------------

def plainIm(file:str, ic:Union[int, bool]=0, checkUnits:bool=True) -> Tuple[Union[pd.DataFrame, List[Any]], Dict]:
    '''import a csv to a pandas dataframe. ic is the index column. Int if there is an index column, False if there is none. checkUnits=False to assume that there is no units row. Otherwise, look for a units row'''
    if os.path.exists(file):
        try:
            toprows = pd.read_csv(file, index_col=ic, nrows=2)
            toprows = toprows.fillna('')
            row1 = list(toprows.iloc[0])
            if checkUnits and all([(type(s) is str or pd.isnull(s)) for s in row1]):
                # row 2 is all str: this file has units
                unitdict = dict(toprows.iloc[0])
                skiprows=[1]
            else:
                unitdict = dict([[s,'undefined'] for s in toprows])
                skiprows = []
            try:
                d = pd.read_csv(file, index_col=ic, dtype=float, skiprows=skiprows)
            except:
                d = pd.read_csv(file, index_col=ic, skiprows=skiprows)
        except Exception as e:
#             logging.error(str(e))
            return [],{}
        return d, unitdict
    else:
        return [], {}
    
    
def splitUnits(df:pd.DataFrame) -> Tuple[pd.DataFrame,dict]:
    '''given a header row where units are in parentheses, rename the dataframe headers to have no units and return a dictionary with units'''
    header = {}
    units = {}
    for c in df.columns:
        if '(' in c:
            spl = re.split('\(', c)
            name = spl[0]
            if name[-1]==' ':
                name = name[:-1]
            if len(c)>1:
                u = spl[1][:-1]
        else:
            name = c
            u = ''
        header[c]=name
        units[name] = u
    df.rename(columns=header, inplace=True)
    return df,units

    
def plainExp(fn:str, data:pd.DataFrame, units:dict, index:bool=True) -> None:
    '''export the file'''
    if len(data)==0:
        return
    if len(units)==0:
        col = data.columns
    else:
        col = pd.MultiIndex.from_tuples([(k,units[k]) for k in data]) # index with units
    data = np.array(data)
    df = pd.DataFrame(data, columns=col)       
    df.to_csv(fn, index=index)
    logging.info(f'Exported {fn}')