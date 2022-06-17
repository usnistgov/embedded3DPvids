# Embedded 3D Printing video analysis

Python tools for analysis of videos and images of filament shapes in embedded 3D printing


## Authors
- Leanne M. Friedrich
    - National Institute of Standards and Technology, MML
    - Leanne.Friedrich@nist.gov
    - https://github.com/leanfried
    - ORCID: 0000-0002-0382-3980
- Jonathan E. Seppala
    - National Institute of Standards and Technology, MML
    - ORCID: 0000-0002-5937-8716

## Contact
- Leanne Friedrich
    - Leanne.Friedrich@nist.gov

## Description

In embedded 3D printing, a nozzle is embedded into a support bath and extrudes filaments or droplets into the bath. This repository includes python code for analyzing and managing images and videos of the printing process. Images should be .png format, and videos should be .avi format. Images and videos should depict a single filament, either a horizontal head-on filament, a horizontal filament from the side, or a vertical filament from the side, that is dark on a light background. Videos contain a dark nozzle and depict the printing process. All images and videos must have the same scale, reported in pixels per mm. 


## Change log

|version|Timeframe|Scope|
|-------|---------|-----|
|1.0.0  |April 2021 - March 2022|Single filaments: cross-sections, side view, vertical lines|


## Data Use Notes


This code is publicly available according to the NIST statements of copyright, fair use and licensing; see 
https://www.nist.gov/director/copyright-fair-use-and-licensing-statements-srd-data-and-software

You may cite the use of this code as follows:

> Friedrich, L.M., & Seppala, J.E. (2022), Python tools for image analysis of embedded 3D printing, Version 1.0.0, National Institute of Standards and Technology, doi:10.18434/mds2-2564 (Accessed XXXX-XX-XX)


## References

This code is described in the following paper:

> Friedrich, L.M., Gunther, R.T., & Seppala, J.E. (2022) Suppression of filament defects in embedded 3D printing, submitted for publication

The dataset analyzed by this code is stored at:

> Friedrich, L.M., & Seppala, J.E. (2022) Suppression of filament defects in embedded 3D printing, Version 1.0.0, National Institute of Standards and Technology, doi:10.18434/mds2-2566


## Usage

For full functionality, you will need to make the following files:

- Copy `configs/config_template.yml` and call it `configs/config.yml`. Change the path names to the paths to your data. 


## Data Overview

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





## Methodology


### Experiments

Use the GUI at https://github.com/usnistgov/ShopbotPyQt/ to generate files (examples in parentheses) during printing of single lines. For each print, the GUI exports:

- a video of the printing process from a color Basler camera (`singleLinesNoZig_Basler camera_I_2.25_S_3.50_210727_104032.avi`)
- a table of pressure over time (`singleLinesNoZig_Fluigent_I_2.25_S_3.50_210727_104032.csv`)
- a table of programmed speeds (`singleLinesNoZig_speeds_I_2.25_S_3.50_210727_104032.csv`)
- a group of images from the Basler camera, taken after the print is done (`singleLinesPics6_Basler camera_I_2.25_S_3.50_210727_104247.png`)
- (optional) other images


### Analysis

1. Use `fileHandling.ipynb` to sort and label the files into folders by sample name, run name
2. Use `stitching.ipynb` to stitch images together and archive the raw data
3. Use `singleLineMetrics.ipynb` to extract values from stitched images
4. Use `vidSummaries.ipynb` to summarize extracted values from all runs
5. Use `vidanalysis.ipynb` to extract and summarize values from videos
6. Use `vidplots.ipynb` to plot stills together
7. Use `vidSummaries_horiz.ipynb`, `vidSummaries_horizvid.ipynb`, `vidSummaries_ST.ipynb`, `vidSummaries_vert.ipynb`, and `vidSummaries_xs.ipynb` to plot summary data
8. Use `vidSummaries_stats.ipynb` to create tables of statistical tests


