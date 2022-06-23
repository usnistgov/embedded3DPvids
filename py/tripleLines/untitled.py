class stitchSorterSingle:
    '''class that holds, sorts, and stitches lists of stills for singleLines prints'''
    
    def __init__(self, subFolder):
        '''folder should be a subfolder holding many sbp folders'''
        super(stitchSorterSingle,self).__init__(subFolder)
        self.pfd = fh.printFileDict(self.subFolder)
        # number of columns for each type. 
        
        self.labelPics()
        

        
        
    def resetFolders(self):
        '''reset all of the folder lists and dictionaries'''
        
        self.horizCols = 12
        self.vertCols = 4
        self.xsCols = 5
        
        # horiz, vert1, vert2, ... xs1, xs2...
        self.stlist = fh.singleLineSt()
        
        for st in self.stlist:
            setattr(self, f'{st}groups', [])  # list of stillGroup objects
        
        self.horizStitchFolders = {-1:self.subFolder}
        self.vertStitchFolders = dict([[i+1,self.subFolder] for i in range(self.vertCols)])
        self.xsStitchFolders = dict([[i+1,self.subFolder] for i in range(self.xsCols)])
        
        self.basStill = list(filter(lambda f: 'Basler' in f, self.pfd.still))  # get list of still images
        self.basStill.sort(key=lambda f:os.path.basename(f))   # sort by date

    
    def labelPics(self) -> None:
        '''find the stills in the subfolder and sort them'''
        self.resetFolders()
        self.detectNumCols()
        self.createGroups()
        
    def createGroups(self) -> None:
        '''create stillGroups'''
        kwargs = {}
        if self.horizCols==12:
            kwargs['dxcols'] = 274
        elif self.horizCols==6:
            kwargs['dxcols'] = 2*274
        elif self.horizCols==8:
            kwargs['dxcols'] = 424
        kwargs['dyrows'] = -280
        kwargs['scale'] = round(3/self.horizPerCol,3)
        self.setGroup('horiz', self.subFolder, self.still, self.horizPerCol, self.horizCols, 'c', offset=0, lastSkip=self.lastSkip, **kwargs)
        offset = self.horizPerCol*self.horizCols
        if self.lastSkip:
            offset = offset-1
            
        # vert
        kwargs = {}
        kwargs['dxcols'] = -1
        kwargs['dycols'] = -277
        kwargs['scale'] = round(3/self.vertPerCol,3)
        for i in range(self.vertCols):
            self.setGroup(f'vert{i+1}', self.subFolder, self.still, self.vertPerCol, 1, 'c', offset=offset, **kwargs)
        offset = offset + self.vertCols*self.vertPerCol    
        
        # xs
        kwargs = {}
        kwargs['dxcols'] = 4.24781116
        kwargs['dycols'] = -265.683797
        for i in range(self.xsCols):
            self.setGroup(f'xs{i+1}', self.subFolder, self.still, self.xsPerCol, 1, 'c', offset=offset, **kwargs)
        
        
        

#     def stitchGroup(self, st:str, archive:bool=True, scale:float=1, duplicate:bool=0, **kwargs) -> int:
#         '''stitch the group of files together and export. 
#         st must be horiz, vert1, vert2, vert3, vert4, xs1, xs2, xs3, xs4, or xs5. 
#         archive true to move raw images to raw folder
#         scale=1 to keep original resolution. lower to compress image.
#         put duplicate=1 in kwargs to add a new file if this stitch already exists. duplicate=0 to check if the file already exists, duplicate=2 to overwrite existing file
#         Returns 0 if stitched, 1 if failed, 2 if the file already exists.'''

#         if duplicate==0:
#             if len(getattr(self, st+'Stitch'))>0:
#                 return 2 # file already exists
        
#         if st=='horiz':
#             self.sortHorizCols() # sort stitched images into horiz folder
#         if st=='horizfull':
#             self.splitFiles()
#             self.detectHorizFiles()        
#         files, dirname, sample = self.getFiles(st)
            
#         if len(files)==0:
#             return 1
        
#         s = stitching.Stitch(files)
#         tag = f'{sample}_{st}'
        
#         if duplicate==0:
#             # check if the file exists
#             fn = s.newFN(duplicate=False, tag=tag)
#             fn2 = s.newFN(duplicate=False, tag=tag, scale=False)
#             if os.path.exists(fn) or os.path.exists(fn2):
#                 return 2 # file already exists
#             if 'horiz' in fn and not 'horiz_' in fn:
#                 # horiz1, horiz2, etc.
#                 fn2 = os.path.join(os.path.dirname(fn), 'raw', 'horiz', os.path.basename(fn))
#                 if os.path.exists(fn2):
#                     return 2 # file already exists
#                 else:
#                     fn2 = os.path.join(os.path.dirname(fn), 'raw', 'horizfull', os.path.basename(fn))
#                     if os.path.exists(fn2):
#                         return 2 # file already exists
#             elif 'horiz_' in fn:
#                 fn2 = os.path.join(os.path.dirname(fn), 'raw', 'horizfull', os.path.basename(fn))
#                 if os.path.exists(fn2):
#                     return 2 # file already exists
        
#         self.detectNumCols() # detect number of columns for scaling

#         # set default displacements
#         if 'xs' in st:
#             s.matcher.setDefaults(4.24781116*scale, -265.683797*scale)
#             s.matcher.resetLastH()
#         elif 'vert' in st:
#             if scale==1:
#                 scale=round(3/self.vertPerCol,3) # automatically scale horiz1... to 3/number of images
#             s.matcher.setDefaults(-1*scale, -277*scale)
#             s.matcher.resetLastH()
#         elif 'horiz'==st:
#             scaleOrig=float(fileScale(files[0])) # need to adopt scaling from source images
#             if self.horizCols==12:
#                 dx = 274
#             elif self.horizCols==6:
#                 dx = 2*274
#             elif self.horizCols==8:
# #                 dx = 424
#                 dx = 124
#             s.matcher.setDefaults(dx*scale*scaleOrig, 0*scale*scaleOrig)
#             s.matcher.resetLastH()
#         elif 'horizfull'==st:
#             # for adding the last column
#             scaleOrig=float(fileScale(files[0])) # need to adopt scaling from source images
#             if self.horizCols==12:
#                 dx = 274*(self.horizCols-1)
#             elif self.horizCols==6:
#                 dx = 2*274*(self.horizCols-1)
#             elif self.horizCols==8:
#                 dx = 422*(self.horizCols-1)
#             s.matcher.setDefaults(dx*scale*scaleOrig, 262*scale*scaleOrig)
#             s.matcher.resetLastH()
#         elif 'horiz' in st:
#             if scale==1:
#                 scale=round(3/self.horizPerCol,3) # automatically scale horiz1... to 3/number of images
#             s.matcher.setDefaults(0*scale, -280*scale)
# #             s.matcher.setDefaults(0*scale, -314*scale)
# #             s.matcher.resetLastH()
        
#         if not scale==1:
#             s.scaleImages(scale) # rescale images

#         try:
#             # stitch images and export
#             s.stitchTranslate(export=True, tag=tag, duplicate=duplicate, **kwargs)
# #             if st=='horiz':
# #                 self.stitchGroup('horizfull', archive=archive,scale=1,duplicate=duplicate,**kwargs)
#         except:
#             logging.warning('Stitching error')
#             traceback.print_exc()
#             return
#         else:
#             # archive
#             if archive:
#                 self.archiveGroup(st, files=files, **kwargs)
#             return 0
        
    def detectNumCols(self) -> None:
        '''determine how many columns and rows there are in each group, depending on which shopbot file created the pics'''
        self.lastSkip = False
#         logging.info(len(self.basStill))
        if len(self.basStill)>0:
            if len(self.basStill)==49 and 'singleLinesPics' in self.basStill[0]:
                # we used the shopbot script to generate these images
                self.horizCols=1
                self.horizPerCol=9
                self.vertPerCol=5
                self.xsPerCol=4
            elif len(self.basStill)==48 and 'singleLinesPics' in self.basStill[0]:
                # we used the shopbot script to generate these images
                self.horizCols=1
                self.horizPerCol=10
                self.vertPerCol=7
                self.xsPerCol=2
            elif len(self.basStill)==58 and 'singleLinesPics' in self.basStill[0]:
                # we used the shopbot script to generate these images
                self.horizCols=1
                self.horizPerCol=10
                self.vertPerCol=7
                self.xsPerCol=4
            elif len(self.basStill)==174 and 'singleLinesPics3' in self.basStill[0]:
                # we used the shopbot script to generate these images
                self.horizCols=12
                self.horizPerCol=11
                self.vertPerCol=7
                self.xsPerCol=3
                self.lastSkip = True
            elif len(self.basStill)==108 and 'singleLinesPics4' in self.basStill[0]:
                # we used the shopbot script to generate these images
                self.horizCols=6
                self.horizPerCol=11
                self.vertPerCol=7
                self.xsPerCol=3
                self.lastSkip = True
            elif len(self.basStill)==130 and 'singleLinesPics5' in self.basStill[0]:
                # we used the shopbot script to generate these images
                self.horizCols=8
                self.horizPerCol=11
                self.vertPerCol=7
                self.xsPerCol=3
                self.lastSkip = True
            elif len(self.basStill)==131 and 'singleLinesPics6' in self.basStill[0]:
                # we used the shopbot script to generate these images
                self.horizCols=8
                self.horizPerCol=10
                self.vertPerCol=9
                self.xsPerCol=3
            elif len(self.basStill)==121 and 'singleLinesPics7' in self.basStill[-1]:
                # we used the shopbot script to generate these images
                self.horizCols=8
                self.horizPerCol=10
                self.vertPerCol=9
                self.xsPerCol=1
            elif len(self.basStill)==131 and 'singleLinesPics8' in self.basStill[-1]:
                # we used the shopbot script to generate these images
                self.horizCols=8
                self.horizPerCol=10
                self.vertPerCol=9
                self.xsPerCol=3
            elif len(self.basStill)==136 and 'singleLinesPics9' in self.basStill[-1]:
                # we used the shopbot script to generate these images
                self.horizCols=8
                self.horizPerCol=10
                self.vertPerCol=9
                self.xsPerCol=4
            return 
        else:
            # unknown sorting: check folders
            for s in ['horiz', 'vert', 'xs']:
                numfiles = len(getattr(self, f'{s}1Still'))
                if numfiles>0:
                    setattr(self, f'{s}PerCol', numfiles)
                else:
                    return ValueError(f'{self.folder}: Cannot calculate files per column for {s}')
            return 
    
    
    
        
#     #-------
        
#     def resetSortedLists(self):
#         '''reset the sorted lists of files'''       
#         for s in ['Still', 'Stitch']:
#             for st in self.stlist(s2=s):
#                 setattr(self, st, [])
#         self.horizStill = []
#         self.horizStitch = []
#         for s in ['Still', 'Stitch']:
#             for s2 in ['horiz', 'vert', 'xs']:
#                 l = [getattr(self, f'{s2}{i}Still') for i in range(1,getattr(self, f'{s2}Cols')+1)]
#                 setattr(self, f'{s2}{s}Groups', l)
     
#     #-------
    
#     def addIf(self, l:list, val:Any) -> None:
#         '''add the val to the list if it isn't already there'''
#         if os.path.exists(val):
#             # this is a file
#             v2 = os.path.basename(val)
#             l2 = [os.path.basename(li) for li in l]
#         else:
#             v2 = val
#             l2 = l
#         if not v2 in l2:
#             l.append(val)
    
#     def putInStillList(self, f:str, f1:str) -> None:
#         '''Put a still in the list of stills. f is base name, f1 is full path to file'''
#         if len(f1)==0:
#             raise NameError('file name is empty')
#         if len(f)==0:
#             f = os.path.basename(f1)

#         if 'phoneCam' in f:
#             self.addIf(self.phoneStill, f1)
#         elif 'Fluigent' in f:
#             self.addIf(self.fluigent, f1)
#         elif 'Nozzle camera' in f:
#             if '.avi' in f:
#                 self.addIf(self.webcamVideo, f1)
#             elif '.png' in f:
#                 self.addIf(self.webcamStill, f1)
#         elif 'Basler camera' in f:
#             for st in self.stlist():
#                 if st==os.path.basename(os.path.dirname(f1)):
#                     # if the string is the folder name, this is a still
#                     self.addIf(getattr(self, f'{st}Still'),f1)
#                     self.addIf(self.basStill, f1)
#                     return
#             # if we make it through the loop, this has no label and goes into basStill or basVideo
#             if '.png' in f:
#                 self.addIf(self.basStill, f1)
#                 self.addIf(self.unstitchedStill, f1)
#             elif '.avi' in f:
#                 self.addIf(self.basVideo, f1)
#         else:
#             # no camera type: this is a stitch
#             for st in self.stlist():
#                 if st+'_' in f and not '_vid_' in f:
#                     # if the string is in the basename, this is a stitch
#                     self.addIf(getattr(self, st+'Stitch'), f1)
#                     return
    
#     #-------
    
#     def splitFiles(self, folder:str='') -> None:
#         '''sort the files in the folder into the lists'''
#         if not os.path.exists(folder):
#             folder = self.folder
#         for f in os.listdir(folder): # for each item in folder
#             f1 = os.path.join(folder, f)
#             if os.path.isdir(f1): # if the item is a folder, recurse
#                 self.splitFiles(f1)
#             else:
#                 self.putInStillList(f,f1) # if the item is a file, sort it by type (nozzle, bas, video, etc.)      
#         if not 'raw' in folder:
#             self.splitBasStill()
#             self.reduceLists()
# #             self.detectHorizFiles()
                
#     #-------
    
#     def reduceLists(self) -> None:
#         '''get rid of empty still lists'''
#         for s in ['Still']:
#             for s2 in ['horiz', 'vert', 'xs']:
#                 numcols = 0
#                 for i in range(1,getattr(self, s2+'Cols')+1):
#                     files = getattr(self, s2+str(i)+s)
#                     if len(files)==0:
# #                         delattr(self, s2+str(i)+s)
#                         pass
#                     else:
#                         numcols = numcols+1
#                 setattr(self, s2+'Cols', numcols) # adjust the number of cols
                
    


                
#     def splitBasStill(self) -> None:
#         '''sort the basStill files into horiz1Still, etc.'''
#         self.basStill.sort(key=fh.fileTime)
#         self.detectNumCols()
#         # put stills in lists
        
#         last=0
#         for s in ['horiz', 'vert', 'xs']:
#             for i in range(getattr(self, s+'Cols')):
#                 i0 = last+i*getattr(self, s+'PerCol')
#                 i1 = last+(i+1)*getattr(self, s+'PerCol')
#                 if self.lastSkip:
#                     if s=='horiz' and i==getattr(self, s+'Cols')-1:
#                         i1 = i1-1 # last column is missing 1 image, for some reason
#                 setattr(self, s+str(i+1)+'Still', self.basStill[i0:i1])
#             last=i1
            
#         return
    
    
    
#     def sortHorizCols(self) -> None:
#         '''take the stitched horiz images and put them in the horizStills folder'''
#         self.resetList()
#         self.splitFiles()
#         self.detectNumCols()
#         self.detectHorizFiles()
        
#     def addToHorizList(self, i:Union[int, str], scale:str) -> None:
#         '''add the horizontal still file name to the list of stills, and get the scale'''
#         stlist = getattr(self, f'horiz{i}Stitch')
#         if len(stlist)==0:
# #             raise NameError(f'Missing horiz stitch: {i}')
#             return
#         if i==1:
#             stfile = stlist[0]
#             scale=fileScale(stfile)
#         else:
#             j = 0
#             stfile = ''
#             while not scale in stfile and j<len(stlist):
#                 stfile = stlist[j]
#                 j+=1
#             if not scale in stfile:
# #                 raise NameError(f'Missing horiz stitch: {i} with scale {scale}')
#                 return scale

#         if self.lastSkip:
#             c = self.horizCols
#         else:
#             c = self.horizCols+1
#         if type(i) is int and i<c:
#             self.addIf(self.horizStill, stfile) # get first entry in each stitch list
#         else:
#             self.addIf(self.horizfullStill,stfile)
#         return scale
        
#     def detectHorizFiles(self) -> None:
#         '''sort the horiz stills into lists'''
#         if len(self.horizStill)>0:
#             if not 'horiz' in os.path.basename(self.horizStill[0]):
#                 # this is a single horiz column folder. don't look for stills
#                 return
#         self.horizStill = []
#         scale = ''
#         for i in list(range(1, self.horizCols+1))+['']:
#             scale = self.addToHorizList(i, scale)
#         self.horizfullStill.sort()
#         self.horizfullStill.reverse()
        
    
#     #-------
    
#     def printFiles(self, name:str) -> None:
#         '''print all of the files under that list name'''
#         files = getattr(self, name)
#         if 'Still' in name and not name=='horizStill' and not name=='horizfullStill':
#             times = [fh.fileTime(s) for s in files]
#         else:
#             times = [os.path.basename(s) for s in files]
#         logging.info(f'{name}, {times}')
            
#     def printGroups(self) -> None:
#         '''print file dates in all groups'''
#         if len(self.stlist())==0:
#             logging.info(f'No groups sorted. {len(self.basStill)} stills')
#         for s in ['Still', 'Stitch']:
#             for st in self.stlist(s2=s):
#                 self.printFiles(st)
    
#     #-------
    
#     def countFiles(self) -> dict:
#         '''count the types of files in the folder'''
#         c = [['folder',self.folder]]
#         for r in ['Still', 'Stitch']:
#             for st in self.stlist(s2=r):
#                 files = getattr(self, st)
#                 c.append([st, len(files)])
#         return dict(c)
    
#     def stitchDone(self) -> bool:
#         '''determine if the folder is done stitching'''
#         for st in self.stlist(s2='Stitch'):
#             files = getattr(self, st)
#             if len(files)==0:
#                 return False
#         else:
#             return True
            
#     def getFiles(self, st:str) -> Tuple[List, str, str]:
#         '''get file list and directory name. st is an image tag, e.g. xs1 '''
#         files = getattr(self, f'{st}Still')
#         if len(files)==0:
#             return [],'',''
#         dirname = files[0]
#         sample = ''
#         i = 0
#         while not ('I_' in sample and '_S_' in sample) and i<3:
#             # keep going up folders until you hit the sample subfolder
#             dirname = os.path.dirname(dirname)
#             sample = os.path.basename(dirname) # folder name
#             i+=1
#         return files, dirname, sample
            
    def archiveGroup(self, st:str, files:List[str]=[], debug:bool=False, **kwargs) -> None:
        '''put files in an archive folder'''
        if len(files)==0:
            files, dirname, _ = self.getFiles(st)
        else:
            dirname = os.path.dirname(files[0])
        if 'raw' in files[0]:
            # already archived
            return
        rawfolder = os.path.join(dirname, 'raw')
        if not os.path.exists(rawfolder):
            if not debug:
                os.mkdir(rawfolder)
        linefolder = os.path.join(rawfolder, st)
        if not os.path.exists(linefolder):
            if not debug:
                os.mkdir(linefolder)
        newnames = []
        for f in files:
            newname = os.path.join(linefolder, os.path.basename(f))
            if debug:
                logging.info(f'Old: {f}, New: {newname}')
            else:
                if os.path.exists(newname):
                    # remove existing file
                    os.remove(newname)
                os.rename(f, newname)
                newnames.append(newname)
        if not debug:
            setattr(self, st+"Still", newnames)
            
  

                
    def stitchGroups(self, archive:bool=True, **kwargs) -> None:
        '''stitch all groups and export'''
        if 'stlist' in kwargs:
            stlist = kwargs['stlist']
        else:
            stlist = self.stlist()
        for st in stlist:
#             self.printGroups()
            self.stitchGroup(st, archive=archive, **kwargs)