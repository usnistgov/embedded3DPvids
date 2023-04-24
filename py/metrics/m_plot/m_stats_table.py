#!/usr/bin/env python
'''Functions for creating regression tables'''

# external packages
import os, sys
import traceback
import logging
import pandas as pd
from typing import List, Dict, Tuple, Union, Any, TextIO
import re
import numpy as np
import string
from scipy import stats
import csv
from IPython.display import display, Math

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
sys.path.append(os.path.dirname(os.path.dirname(currentdir)))
import tools.regression as rg
from tools.config import cfg
from m_summary.summary_metric import *
from m_stats import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)

#-------------------------------------------------------------

class regressionTable:
    '''for creating a single table of regression metrics for a single y variable'''
    
    def __init__(self, ms:summaryMetric, ss:pd.DataFrame, yvar:str, logy:bool=True, printOut:bool=True, export:bool=False, exportFolder:str='', tag:str='', package:str='pgfplot', **kwargs):
        self.ms = ms
        self.ss = ss.copy()
        self.yvar = yvar
        self.logy = logy
        self.defineY()
        self.printOut = printOut
        self.export = export
        self.exportFolder = exportFolder
        self.tag = tag
        self.package = package
        self.kwargs = kwargs
        self.smax = self.ss.sigma.max()
        if export and not os.path.exists(exportFolder):
            logging.warning(f'exportFolder {exportFolder} does not exist. Defaulting to {cfg.path.fig}')
            exportFolder = cfg.path.fig
        self.createTable()
        
    def defineY(self):
        # define y variables
        if self.logy:
            self.ss = self.ms.addLogs(self.ss, [self.yvar])
            self.ycol = self.yvar+'_log'
        else:
            self.ycol = self.yvar
                 
                 
    def checkYvar(self):
        '''not enough variables for regression. report failure and return'''
        if self.printOut:
            if self.smax>0:
                logging.info(f'All {self.yvar} values the same for nonzero surface tension\n---------------------------\n\n')
            else:
                logging.info(f'All {self.yvar} values the same for zero surface tension\n---------------------------\n\n')
        return

    def prepareSSI(self) -> Tuple[list, pd.DataFrame]:
        '''get the list of independent variables and add to the dataframe'''
        # define x variables
        if self.smax>0:
            self.varlist = ['Ca', 'dnorm', 'We', 'Oh', 'Re', 'Bm', 'visc0']
        else:
            self.varlist = ['Re', 'Bm', 'visc0']

        # add logs and ratios
        for i,s1 in enumerate(['sup', 'ink']):
            self.ss = self.ms.addLogs(self.ss, [f'{s1}_{v}' for v in self.varlist])
        for i,s1 in enumerate(['Prod', 'Ratio']):
            self.ss = self.ms.addRatios(self.ss, varlist=self.varlist, operator=s1)
            self.ss = self.ms.addLogs(self.ss, [f'{v}{s1}' for v in self.varlist])
            
        if 'spacing' in self.ss:
            self.varlist = ['spacing']+ self.varlist
    
    def regRow(self, df:list, xcol:str, title:str) -> None:
        '''get regression and correlation info for a single x,y variable combo'''
        if len(self.ss[xcol].unique())<2:
            return
        reg = rg.regPD(self.ss, [xcol], self.ycol)
        spear = rg.spearman(self.ss, xcol, self.ycol)
        reg['coeff'] = reg.pop('b')
        reg = {**reg, **spear}
        reg['title'] = title
        df.append(reg)
    
    def createVariableTable(self, scvar:str) -> pd.DataFrame:
        '''create a table of correlations for the scaling variable, in combos of ink, sup, ink*sup, and ink/sup'''
        df = []
        if scvar=='spacing':
            reg = self.regRow(df, 'spacing', 'spacing')
            return pd.DataFrame(df)
        
        if scvar=='Ca':
            self.ss = self.ms.addLogs(self.ss, ['int_Ca'])
            self.regRow(df, 'int_Ca_log', '$Ca$')            
        
        # single variable correlation
        for prefix in ['ink_', 'sup_']:
            xcol = f'{prefix}{scvar}_log'
            title = self.ms.varSymbol(f'{prefix}{scvar}', commas=False)
            self.regRow(df, xcol, title)

        # products and ratios
        for suffix in ['Prod', 'Ratio']:
            xcol = f'{scvar}{suffix}_log'
            title = self.ms.varSymbol(f'{scvar}{suffix}', commas=False)
            self.regRow(df, xcol, title)      
        df = pd.DataFrame(df)
        return df
    
    def labelBestFit(self, df:pd.DataFrame) -> pd.DataFrame:
        '''find the best fits and bold them in the table'''
        # label best fit
        crit = ((abs(df.spearman_corr)>0.9*abs(df.spearman_corr).max())&(df.spearman_p<0.05)&(abs(df.spearman_corr)>0.5))
        df.spearman_p = ['{:0.1e}'.format(i) for i in df.spearman_p]
        df.spearman_corr = ['{:0.2f}'.format(i) for i in df.spearman_corr]
        df.r2 = ['{:0.2f}'.format(i) for i in df.r2]
        for sname in ['title', 'r2', 'spearman_corr', 'spearman_p']:
            # bold rows that are best fit
            df.loc[crit,sname] = ['$\\bm{'+(i[1:-1] if i[0]=='$' else i)+'}$' for i in df.loc[crit,sname]]
        return df
    
    def addVariable(self, scvar:str) -> None:
        '''add the variable to the table'''
        df = self.createVariableTable(scvar)
        if len(df)>0:
            df = self.labelBestFit(df)       
            self.df = pd.concat([self.df, df])
        
    def addHeaders(self) -> None:
        '''add headers to the table'''
        self.df0 = self.df.copy()
        self.df = self.df[['title', 'r2', 'coeff', 'c', 'spearman_corr', 'spearman_p']]
        self.df = self.df.rename(columns={'r2': '$r^2$', 'title':'variables', 'coeff':'b', 'spearman_corr':'Spearman coeff', 'spearman_p':'Spearman p'})
    
    def getCaptions(self) -> None:
        '''get captions for a single table'''
        if 'nickname' in self.kwargs:
            self.nickname = self.kwargs['nickname']
        else:
            self.nickname = self.ms.varSymbol(self.yvar)
        self.shortCaption = f'Linear regressions for {self.nickname}'
        if self.smax>0:
            st = ' at nonzero surface tension.'
            self.label = f'tab:{self.yvar}{self.tag}RegNonZero'
        else:
            st =' in water/Laponite inks.'
            self.label = f'tab:{self.yvar}{self.tag}RegZero'
        self.shortCaption+=st
        if self.logy:
            regexample = '$y = 10^c*Re_{ink}^b$'
            v1 = 'x and y'
        else:
            regexample = '$y = b*log_{10}(Re_{ink}) + c$'
            v1 = 'x'
        self.longCaption = r'Table of linear regressions of log-scaled '+v1+' variables and Spearman rank correlations for \\textbf{'+self.nickname+r'}'+st+' For example, ${Re}_{ink}$ indicates a regression fit to '+regexample+'. A Spearman rank correlation coefficient of -1 or 1 indicates a strong correlation. Variables are defined in table \\ref{tab:variableDefs}.'
        
    def writeTextToFile(self, fn:str, text:str) -> None:
        '''write the text to file'''
        file2 = open(fn ,'w')
        file2.write(text)
        file2.close()
        logging.info(f'Exported {fn}\n---------------------------\n\n')

    
    def tabularText(self) -> str:
        '''for printing the table as a tabular latex structure'''
        dftext = self.df.to_latex(index=False, escape=False
                             , float_format = lambda x: '{:0.2f}'.format(x) if pd.notna(x) else '' 
                             , caption=(self.longCaption, self.shortCaption)
                             , label=self.label, position='H')
        dftext = dftext.replace('\\toprule\n', '')
        dftext = dftext.replace('\\midrule\n', '')
        dftext = dftext.replace('\\bottomrule\n', '')
        ctr = -10
        dftextOut = ''
        for line in iter(dftext.splitlines()):
            dftextOut = dftextOut+line+'\n'
            ctr+=1
            if 'variables' in line:
                ctr = 0
            if 'bm{Ca}' in line or '$Ca$' in line:
                ctr = 0
            if (ctr==4 and not line.startswith('\\end')) or 'spacing' in line:
                dftextOut = dftextOut+'\t\t\\hline\n'
                ctr=0
        self.dftextOut = dftextOut
        if self.printOut:
            print(self.dftextOut)
        if self.export:
            fn = os.path.join(self.exportFolder, self.label[4:]+'.tex')
            self.writeTextToFile(fn, self.dftextOut)
            
    def pgfText(self) -> str:
        '''for printing the table in latex using csv import. exports two files: one for the import command, and one for the values to export. both are .tex files, but the csv is inside one of the tex files'''
        label=self.label.replace('_', '')
        # import command
        
        dftext = r'\begin{filecontents*}{'
        dftext = dftext+self.label[4:]+'.csv'+'}\n'
        dftext = dftext+self.df.to_csv(index=False, float_format = lambda x: '{:0.2f}'.format(x) if pd.notna(x) else '')
        dftext = dftext+r'\end{filecontents*}'+'\n'
        dftext = dftext+ r'\pgfplotstableread[col sep=comma]{'+self.label[4:]+'.csv'+'}\\'+self.label.replace(':','')
        self.dftext = dftext
        if self.printOut:
            print(dftext)
            print('\n-------------\n')
        if self.export:
            fn = os.path.join(self.exportFolder, self.label[4:]+'Import.tex')
            writeTextToFile(fn, dftext)

        # displayed table
        dftextOut = '\\begin{table}\n\\centering\n\\caption['
        dftextOut = dftextOut+self.shortCaption+r']{'+self.longCaption+'}\n'
        dftextOut = dftextOut+'\\pgfplotstabletypeset[\n\tcol sep=comma,\n\tstring type,\n'
        for s in ['variables', '$r^2$','b','c','Spearman coeff','Spearman p']:
            dftextOut = dftextOut+'\tcolumns/'+s+'/.style={column type=l},\n'
        dftextOut = dftextOut+'\tevery head row/.style={after row=\hline},\n\tevery nth row={4'
        if 'Ca' in self.df.iloc[0]['variables']:
            # skip another row before hline
            dftextOut = dftextOut+'[+1]'
        dftextOut = dftextOut+'}{before row=\\hline}\n]'+'\\'+self.label.replace(':','')+'\n'
        dftextOut = dftextOut+'\\label{'+self.label+'}\n'
        dftextOut = dftextOut+'\\end{table}'
        self.dftextOut = dftextOut
        if self.printOut:
            print(self.dftextOut)
            print('\n-------------\n')
        if self.export:
            fn = os.path.join(self.exportFolder, self.label[4:]+'.tex')
            self.writeTextToFile(fn, self.dftextOut) # write import command to text file

    def createTable(self) -> None:
        '''create a single table'''
        self.df = pd.DataFrame([])
        
        if len(self.ss[self.yvar].unique())<2:
            self.checkYvar()
            return
        self.prepareSSI()  # get the independent variable list and the dataframe with those variables added
        
        # go through each variable and get sup, ink, product, ratio
        for s2 in self.varlist:
            self.addVariable(s2)

        # combine into table
        self.addHeaders()        
        self.getCaptions()

        if self.package=='tabular':
            self.tabularText()
        elif self.package=='pgfplot':
            self.pgfText()
        else:
            raise ValueError(f'Unexpected package {self.package}')
            
    def show(self) -> None:
        '''shows the formatted table'''
        print(self.dftextOut)


class regressionTables:
    '''for holding multiple tables of regression variables, where values were split into water-based and oil-based materials'''
    
    def __init__(self, ms:summaryMetric, ss:pd.DataFrame, yvar:str, **kwargs):
        self.ms = ms
        self.ss0 = ss.copy()
        self.yvar = yvar
        self.kwargs = kwargs
        self.run()
        
    def createSS(self):
        '''create separate dataframes for water-based and oil-based fluids'''
        self.ss0.dropna(subset=[self.yvar], inplace=True)
        self.ss0 = self.ss0[self.ss0.ink_days==1]
        self.ss0 = self.ss0.sort_values(by='sigma')
            
        # split into oil and water
        self.ssca1 = self.ss0.copy()
        self.ssca1 = self.ssca1[self.ssca1.sigma>0]
        self.sslap = self.ss0.copy()
        self.sslap = self.sslap[self.sslap.ink_base=='water']
        
    def createTable(self, ss:pd.DataFrame):
        '''create a single table'''
        if len(ss)>0:
            obj = regressionTable(self.ms, ss, self.yvar, **self.kwargs)
            self.objlist.append(obj)
            self.dflist.append(obj.df0)

    def run(self):
        '''create all tables and store values in dflist'''
        self.dflist = []
        self.objlist = []
        self.createSS()
        self.createTable(self.ssca1)
        self.createTable(self.sslap)
