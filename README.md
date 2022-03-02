### NIST Repository:
Python tools for analysis of videos and iamges of filament shapes in embedded 3D printing
Version 1.0.0

### Authors:
- Leanne Friedrich
    - National Institute of Standards and Technology, MML
    - Leanne.Friedrich@nist.gov
- Jonathan E. Seppala
    - National Institute of Standards and Technology, MML
    - Jonathan.Seppala@nist.gov

### Contact:
- Leanne Friedrich
    - Leanne.Friedrich@nist.gov

### Description:

In embedded 3D printing, a nozzle is embedded into a support bath and extrudes filaments or droplets into the bath. This repository includes python code for analyzing and managing images and videos of the printing process.


--- 

# General Information

Version 1.0.0 was generated: April 2021-March 2022

---

# Data Use Notes


This code is publicly available according to the NIST statements of copyright,
fair use and licensing; see 
https://www.nist.gov/director/copyright-fair-use-and-licensing-statements-srd-data-and-software

You may cite the use of this code as follows:
Friedrich, L., & Seppala, J.E. (2021), Python tools for image analysis of embedded 3D printing, Version 1.0.0, National Institute of Standards and Technology, doi:[] (Accessed XXXX-XX-XX)

---

# References



---

# Data Overview

The files included in this publication use the following hierarchy:

- *README.md*

- *LICENSE*

- *fileHandling.py* 
    - Functions for handling files and folders

- *requirements.txt*
    - List of required packages, for use with virtual environments. This file can be used with virtualenv for easy import of dependencies.

- **configs/**
    - *config.yml*
        - set path variables
        
    - *logging.yml*
        - set logging variables
        
- **ipynb/**
    - *filehandling.ipynb*
        - Sort files into the correct folders
        
    - *singleLineMetrics.ipynb*
        - Extract data from stitched images
        
    - *stitching.ipynb*
        - Stitch raw images together
        
    - *vidanalysis.ipynb*
        - Extract and summarize video data for each folder
        
    - *vidplots.ipynb*
        - Plot stills together
        
    - *vidSummaries_horiz.ipynb*
        - Plot horizontal line still data
        
    - *vidSummaries_horizvid.ipynb*
        - Plot horizontal line video data
        
    - *vidSummaries_ST.ipynb*
        - Plot roughness of horizontal lines
        
    - *vidSummaries_stats.ipynb*
        - Statistical analysis of all metrics
        
    - *vidSummaries_vert.ipynb*
        - Plot vertical line metrics
        
    - *vidSummaries_xs.ipynb*
        - Plot cross-section metrics
        
    - *vidSummaries.ipynb*
        - Summarize extracted metrics into a table
     


- **logs/**
    - for holding logs
    
- **py/**
    - Functions

    - *config.py*
        - Loads configuration settings
    
    - *figureLabels.py*
        - Functions for putting figure labels on figures
        
    - *fileHandling.py*
        - Functions for handling files
        
    - *imshow.py*
        - Functions for displaying images in jupyter notebooks
        
    - *logs.py*
        - Tools for creating and handling logs
        
    - *metricPlots.py*
        - Functions for plotting still and video data
        
    - *metrics.py*
        - Functions for collecting data from stills of single lines

    - *plainIm.py*
        - Functions for importing a csv to a pandas dataframe and units dictionary
        
    - *printVals.py*
        - Functions for storing metadata and compiling data about print folders
        
    - *regression.py*
        - Tools for fitting regressions
        
    - *stitchBas.py*
        - Tools for stitching and sorting images from a Basler camera
        
    - *stitching.py*
        - Tools for stitching together any images
        
    - *vidCrop.py*
        - Tools for cropping images to an ROI
        
    - *vidMorph.py*
        - Morphological operations on images
        
    - *vidplots.py*
        - Tools for plotting data about videos
        
    - *vidTools.py*
        - Tools for collecting data from videos
        
- **tests/**
    - Unit tests
    
    - *test_nozDetect.py*
        - Go through files and test if nozzle is correctly detected

--- 

# Version History


3/2/22: v1.0.0

---

# METHODOLOGICAL INFORMATION


