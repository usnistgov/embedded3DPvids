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
import matplotlib.pyplot as plt

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
sys.path.append(os.path.dirname(os.path.dirname(currentdir)))
import tools.regression as rg
from tools.config import cfg
from m_summary.summary_metric import *
from m_stats import *
from p_xvarlines import xvarlines

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)

#-------------------------------------------------------------

class regressionTable:
    '''for creating a single table of regression metrics for a single y variable'''
    
    def __init__(self, ms:summaryMetric, ss:pd.DataFrame, yvar:str
                 , logx:bool=True
                 , logy:bool=True, printOut:bool=True, export:bool=False
                 , getLinReg:bool=True
                 , getSpearman:bool=True
                 , trimVariables:bool=False
                 , plot:bool=False
                 , exportFolder:str='', tag:str='', package:str='pgfplot', **kwargs):
        self.ms = ms
        self.ss = ss.copy()
        self.yvar = yvar
        self.logx = logx
        self.logy = logy
        self.defineY()
        self.printOut = printOut
        self.export = export
        self.getLinReg = getLinReg           # get linear regressions
        self.getSpearman = getSpearman       # get Spearman rank correlations
        self.trimVariables = trimVariables   # remove variables that aren't significant
        self.exportFolder = exportFolder
        self.tag = tag
        self.package = package
        self.kwargs = kwargs
        self.bestVars = {}
        self.smax = self.ss.sigma.max()
        self.hlineRows = []
        self.dffull = pd.DataFrame([])
        if export and not os.path.exists(exportFolder):
            logging.warning(f'exportFolder {exportFolder} does not exist. Defaulting to {cfg.path.fig}')
            exportFolder = cfg.path.fig
        self.createTable()
        if plot:
            self.plotBest(**kwargs)
        
    def defineY(self):
        # define y variables
        if self.logy:
            self.ss = self.ms.addLogs(self.ss, [self.yvar])
            self.ycol = f'{self.yvar}_log'
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
    
    def indepVars(self) -> list:
        '''a  list of the nondimensional variables for nonzero surface tension'''
        if self.smax>0:
            self.varlist = ['Ca', 'dnorm', 'We', 'Oh', 'Re', 'Bm', 'visc0']
        else:
            self.varlist = ['Re', 'Bm', 'visc0']
        self.ratioList = []
        self.constList = []

    def prepareSSI(self) -> Tuple[list, pd.DataFrame]:
        '''get the list of independent variables and add to the dataframe'''
        # define x variables
        self.indepVars()

        # add logs and ratios
        if self.logx:
            for i,s1 in enumerate(['sup', 'ink']):
                self.ss = self.ms.addLogs(varlist=[f'{s1}_{v}' for v in self.varlist], ss=self.ss)
        for i,s1 in enumerate(['Prod', 'Ratio']):
            self.ss = self.ms.addRatios(varlist=self.varlist, operator=s1, ss=self.ss)
            if self.logx:
                self.ss = self.ms.addLogs(varlist=[f'{v}{s1}' for v in self.varlist], ss=self.ss)
    
    def regRow(self, df:list, xcol:str, title:str) -> None:
        '''get regression and correlation info for a single x,y variable combo'''
        if len(self.ss[xcol].unique())<2:
            return
        if self.getLinReg:
            reg = rg.regPD(self.ss, [xcol], self.ycol)
            reg['coeff'] = reg.pop('b')
        else:
            reg = {}
        if self.getSpearman:
            spear = rg.spearman(self.ss, xcol, self.ycol)
            reg = {**reg, **spear}
        reg['title'] = title
        reg['var'] = xcol
        df.append(reg)
    
    def createVariableTable(self, scvar:str) -> pd.DataFrame:
        '''create a table of correlations for the scaling variable, in combos of ink, sup, ink*sup, and ink/sup'''
        df = []
        
        if scvar=='Ca':
            if self.logx:
                self.ss = self.ms.addLogs(ss=self.ss, varlist=['int_Ca'])
                self.regRow(df, 'int_Ca_log', '$Ca$')           
            else:
                self.regRow(df, 'int_Ca', '$Ca$')    
        
        # single variable correlation
        for prefix in ['ink_', 'sup_']:
            if self.logx:
                xcol = f'{prefix}{scvar}_log'
            else:
                xcol = f'{prefix}{scvar}'
            title = self.ms.varSymbol(f'{prefix}{scvar}', commas=False)
            self.regRow(df, xcol, title)

        # products and ratios
        for suffix in ['Prod', 'Ratio']:
            if self.logx:
                xcol = f'{scvar}{suffix}_log'
            else:
                xcol = f'{scvar}{suffix}'
            title = self.ms.varSymbol(f'{scvar}{suffix}', commas=False)
            self.regRow(df, xcol, title)      
        df = pd.DataFrame(df)
        return df
    
    def labelBestFit(self, df:pd.DataFrame) -> pd.DataFrame:
        '''find the best fits and bold them in the table'''
        # label best fit
        
        if not self.getSpearman:
            if self.getLinReg:
                crit = df[df.r2>0.9]
                goodfits = df[(abs(df.r2)==abs(df.r2).max())&(abs(df.r2)>0.9)]
                for i,row in goodfits.iterrows():
                    print(row)
                    self.bestVars[row['var']] = row['r2']
            else:
                return df
        else:
            crit = ((abs(df.spearman_corr)>0.9*abs(df.spearman_corr).max())&(df.spearman_p<0.05)&(abs(df.spearman_corr)>0.5))   # get good variables
            
            df['spearman_pf'] = ['{:0.1e}'.format(i) for i in df.spearman_p]         # format variables
            df['spearman_corrf'] = ['{:0.2f}'.format(i) for i in df.spearman_corr]
            goodfits = df[(abs(df.spearman_corr)==abs(df.spearman_corr).max())&(abs(df.spearman_corr)>0.5)]
            for i,row in goodfits.iterrows():
                self.bestVars[row['var']] = row['spearman_corrf']
        if self.getLinReg:
            df.r2 = ['{:0.2f}'.format(i) for i in df.r2]      
        
        self.dffull = pd.concat([self.dffull, df])
        
        if self.trimVariables:
            # only take the good fits
            df = df[crit]
        else:
            # bold rows that are best fit
            ll = ['title']
            if self.getLinReg:
                ll.append('r2')
            if self.getSpearman:
                ll = ll+['spearman_corrf', 'spearman_pf']
            for sname in ll:
                df.loc[crit,sname] = ['$\\bm{'+(i[1:-1] if i[0]=='$' else i)+'}$' for i in df.loc[crit,sname]]
                
        if len(df)>0:
            self.df = pd.concat([self.df, df])
            self.df.reset_index(drop=True, inplace=True)
            self.hlineRows.append(str(df.iloc[0]['title']))
    
    def addVariable(self, scvar:str) -> None:
        '''add the variable to the table'''
        df = self.createVariableTable(scvar)
        if len(df)>0:
            df = self.labelBestFit(df)       
            
        
    def addRatios(self) -> None:
        '''add the requested variables that are just ratios, one row per variable'''
        df = []
        self.ss = self.ms.addLogs(ss=self.ss, varlist=self.ratioList)
        for vlist in [self.constList, self.ratioList]:
            for var in vlist:
                if 'spacing' in var:
                    xvar = var
                else:
                    if self.logx and not var in ['zdepth']:
                        xvar = f'{var}_log'
                    else:
                        xvar = var
                title = self.ms.varSymbol(var, commas=False)
                self.regRow(df, xvar, title)  
        df = pd.DataFrame(df)
        if len(df)>0:
            df = self.labelBestFit(df)       
        
    def addHeaders(self) -> None:
        '''add headers to the table'''
        self.df0 = self.df.copy()
        header = ['title']
        cols = {'title':'variable'}
        if self.getLinReg:
            header = header+['r2', 'coeff', 'c']
            cols['r2'] = '$r^2$'
            cols['coeff'] = 'b'
        if self.getSpearman:
            header = header+['spearman_corrf', 'spearman_pf']
            cols['spearman_corrf'] = 'Spearman coeff'
            cols['spearman_pf'] = 'Spearman p'
        if len(self.bestVars)==0:
            # no good fits. find the best one
            best = self.dffull[abs(self.dffull.spearman_corr)==abs(self.dffull.spearman_corr).max()] 
            if len(self.df)==0:
                self.df = best
            self.bestVars[str(best.iloc[0]['var'])] = '{:0.2f}'.format(best.iloc[0]['spearman_corr'])
        self.df = self.df[header]
        self.df = self.df.rename(columns=cols)
        
    def writeTextToFile(self, fn:str, text:str) -> None:
        '''write the text to file'''
        file2 = open(fn ,'w')
        file2.write(text)
        file2.close()
        logging.info(f'Exported {fn}')
        if self.printOut:
            print('\n---------------------------\n\n')

    
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
        ctr = -1100
        for line in iter(dftext.splitlines()):
            if 'Spearman coeff' in line:
                ctr = -1
            if not self.trimVariables and ctr>=0 and ctr<len(self.df) and str(self.df.loc[ctr,'variable']) in self.hlineRows:
                dftextOut = f'{dftextOut}\t\t\\hline\n'     # add hline between sections
            dftextOut = f'{dftextOut}{line}\n'              # add the line
            ctr = ctr+1
            
        self.dftextOut = dftextOut
        if self.printOut:
            # print(self.dftextOut)
            display(self.df)
        if self.export:
            fn = os.path.join(self.exportFolder, 'regressionTables', f'{self.label[4:]}.tex')
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
            # print(dftext)
            display(self.df)
            # print('\n-------------\n')
        if self.export:
            fn = os.path.join(self.exportFolder, 'regressionTables', self.label[4:]+'Import.tex')
            self.writeTextToFile(fn, dftext)

        # displayed table
        dftextOut = '\\begin{table}\n\\centering\n\\caption['
        dftextOut = dftextOut+self.shortCaption+r']{'+self.longCaption+'}\n'
        dftextOut = dftextOut+'\\pgfplotstabletypeset[\n\tcol sep=comma,\n\tstring type,\n'
        header = ['variable']
        if self.getLinReg:
            header = header + ['$r^2$','b','c']
        if self.getSpearman:
            header = header+['Spearman coeff','Spearman p']
        for s in header:
            dftextOut = dftextOut+'\tcolumns/'+s+'/.style={column type=l},\n'
        dftextOut = dftextOut+'\tevery head row/.style={after row=\hline}'
        if not self.trimVariables:
            # add hlines between sections
            dftextOut = dftextOut+',\n\tevery nth row={4'
            if 'Ca' in self.df.iloc[0]['variable']:
                # skip another row before hline
                dftextOut = dftextOut+'[+1]'
            dftextOut = dftextOut+'}{before row=\\hline}'
        dftextOut = dftextOut+'\n]'+'\\'+self.label.replace(':','')+'\n'
        dftextOut = dftextOut+'\\label{'+self.label+'}\n'
        dftextOut = dftextOut+'\\end{table}'
        self.dftextOut = dftextOut
        # if self.printOut:
        #     print(self.dftextOut)
        #     print('\n-------------\n')
        if self.export:
            fn = os.path.join(self.exportFolder, 'regressionTables', self.label[4:]+'.tex')
            self.writeTextToFile(fn, self.dftextOut) # write import command to text file

    def createTable(self) -> None:
        '''create a single table'''
        self.df = pd.DataFrame([])
        
        if len(self.ss[self.yvar].unique())<2:
            self.checkYvar()
            return
        self.prepareSSI()  # get the independent variable list and the dataframe with those variables added
        
        # go through each variable and get sup, ink, product, ratio
        self.addRatios()
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
        
    def plotBest(self, **kwargs) -> None:
        '''plot the best correlations'''
        self.xvl = xvarlines(self.ms, self.ss, xvarList=list(self.bestVars.keys()), yvar=self.ycol, cvar='vRatio', mode='scatter', dx=0.1, **kwargs)
        for i,coeff in enumerate(self.bestVars.values()):
            self.xvl.labelAx(i, f's = {coeff}')
        if self.export:
            fn = os.path.join(self.exportFolder, 'plots', 'regression', f'{self.label[4:]}')
            
            self.xvl.export(fn)
        if not self.printOut:
            plt.close(self.xvl.fig)


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
        
    
#----------------------------------------------------------
        
class regressionTableSingle(regressionTable):
    '''regression table for single lines'''
       
    def __init__(self, ms:summaryMetric, ss:pd.DataFrame, yvar:str, **kwargs):
        super().__init__(ms, ss, yvar, **kwargs) 
    
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
       
    
#----------------------------------------------------------
    
class regressionTableSDT(regressionTable):
    '''for holding a single regression table for singledoubletriple prints'''
    
    def __init__(self, ms:summaryMetric, ss:pd.DataFrame, yvar:str, getLinReg:bool=False, trimVariables:bool=True, spacing:float=0, Camin:float=0, Camax:float=0, **kwargs):
        if spacing>0:
            ss0 = ss[ss.spacing==spacing]
        else:
            ss0 = ss
        self.spacing = spacing
        self.Camin = Camin
        self.Camax = Camax
        if Camin>0:
            ss0 = ss0[ss0.int_Ca>=Camin]
        if Camax>0:
            ss0 = ss0[ss0.int_Ca<=Camax]
        super().__init__(ms, ss0, yvar, getLinReg=getLinReg, trimVariables=trimVariables, **kwargs)
        
    def indepVars(self) -> list:
        '''a  list of the nondimensional variables for nonzero surface tension'''
        self.constList = ['spacing', 'spacing_adj']
        if 'zdepth' in self.ss:
            self.constList.append('zdepth')
        self.ratioList = ['GtaRatio', 'tGdRatio', 'GaRatio', 'GdRatio', 'tau0aRatio', 'tau0dRatio']
        l0 = ['Re', 'Bma', 'Bmd', 'visc0']
        
        if self.smax>0:
            self.varlist = ['Ca', 'dnorma', 'dnormd', 'We', 'Oh']+l0
        else:
            self.varlist = l0
            
    def addToCaptions(self, addition:str) -> None:
        '''add description to the short and long captions'''
        self.shortCaption = f'{self.shortCaption}{addition}'
        self.longCaption = f'{self.longCaption}{addition}'
            
    def getCaptions(self) -> None:
        '''get captions for a singledoubletriple table'''
        if 'nickname' in self.kwargs:
            self.nickname = self.kwargs['nickname']
        else:
            self.nickname = self.ms.varSymbol(self.yvar)
        self.shortCaption = f'Linear regressions for {self.nickname}'
        self.longCaption = r'Table of Spearman rank correlations for \\textbf{'+self.nickname+r'}'
        self.label = f'tab:{self.yvar}_{self.tag}'
        if self.spacing>0:
            addition = f' at a spacing of {self.spacing}$d_i$'
            self.addToCaptions(addition)
            self.label = f'{self.label}_{int(self.spacing*1000)}'
        if self.Camin>0:
            addition = f' at a Ca greater than or equal to {self.Camin}'
            self.addToCaptions(addition)
            self.label = f'{self.label}_Camin{self.Camin}'
        if self.Camax>0:
            addition = f' at a Ca less than or equal to {self.Camax}'
            self.addToCaptions(addition)
            self.label = f'{self.label}_Camax{self.Camax}'

        self.label = f'{self.label}_Reg'
        self.longCaption = self.longCaption + r'. A Spearman rank correlation coefficient of -1 or 1 indicates a strong correlation. Variables are defined in table \\ref{tab:variableDefs}.'
        