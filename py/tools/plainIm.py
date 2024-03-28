#!/usr/bin/env python
'''Functions for importing and exporting tables and dictionaries'''

# external packages
import os
import pandas as pd
from typing import List, Dict, Tuple, Union, Any, TextIO
import logging
import numpy as np
import re
import csv

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

    
def plainExp(fn:str, data:pd.DataFrame, units:dict, index:bool=True, diag:bool=True) -> None:
    '''export the file'''
    # if len(data)==0:
    #     return
    if len(units)==0 or len(data)==0:
        col = data.columns
    else:
        col = pd.MultiIndex.from_tuples([(k,units[k] if k in units else '') for k in data]) # index with units
    data = np.array(data)
    df = pd.DataFrame(data, columns=col)       
    df.to_csv(fn, index=index)
    if diag:
        logging.info(f'Exported {fn}')
    
def tryfloat(val:Any) -> Any:
    try:
        val = float(val)
    except:
        pass
    return val
    
def plainImDict(fn:str, unitCol:int=-1, valCol:Union[int,list]=1) -> Tuple[dict,dict]:
    '''import values from a csv into a dictionary'''
    if type(valCol) is list:
        d = dict([[i,{}] for i in valCol])
    else:
        d = {}
    u = {}
    with open(fn, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            # save all rows as class attributes
            if unitCol>0:
                u[row[0]] = row[unitCol]
            if type(valCol) is int:
                val = (','.join(row[valCol:])).replace('\"', '')
                d[row[0]]=tryfloat(val)
            elif type(valCol) is list:
                for i in valCol:
                    d[i][row[0]]=tryfloat(row[i])
    return d,u

def plainExpDict(fn:str, vals:dict, units:dict={}, diag:bool=True, quotechar:str='|') -> None:
    '''export the dictionary to file'''
    with open(fn, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar=quotechar, quoting=csv.QUOTE_MINIMAL)
        for st,val in vals.items():
            if st in units:
                row = [st, units[st], val]
            else:
                if len(units)>0:
                    row = [st, '', val]
                else:
                    row = [st, val]
            writer.writerow(row)
    if diag:
        logging.info(f'Exported {fn}')
