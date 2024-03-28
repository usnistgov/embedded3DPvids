# Embedded 3D Printing video analysis

Python tools for analysis of videos and images of filament shapes in embedded 3D printing

## Authors
- Leanne M. Friedrich
    - National Institute of Standards and Technology, MML
    - Leanne.Friedrich@nist.gov
    - https://github.com/leanfried
    - ORCID: 0000-0002-0382-3980

## Contact
- Leanne Friedrich
    - Leanne.Friedrich@nist.gov

## Description

In embedded 3D printing, a nozzle is embedded into a support bath and extrudes filaments or droplets into the bath. This repository includes python code for analyzing and managing images and videos of the printing process. Images should be .png format, and videos should be .avi format. There are three datasets this was designed for: one that features single filaments, one that features single filaments that are later disturbed by the nozzle, and a third that contains single filaments, pairs of filaments, and trios of filaments that are later disturbed by the nozzle.

There are four types of datasets that this code was designed for. Version 1.0.0 was designed for single lines. Version 1.1.0 was designed for disturbed single lines, where the nozzle returned to run next to an existing filament, and triple lines, which included multiple morphologies including sets of lines and cross intersections. Version 1.2.0 was designed for single, double, and triple lines (SDT), some of which were disturbed, that followed a consistent format. If a file is labeled just as "triple lines," that refers to the unpublished dataset for version 1.1.0. If the file is labeled as "SDT" or "single double triple", that means the code was designed for the published data in version 1.2.0. Although the code from version 1.0.0 and 1.1.0 was designed to be backwards compatible with version 1.2.0, it may be unstable.


## Change log

|version|Timeframe|Scope|
|-------|---------|-----|
|1.0.0  |April 2021 - March 2022|Single filaments: cross-sections, side view, vertical lines|
|1.1.0  |March 2022 - January 2023|Single filaments that are disturbed by the nozzle|
|1.2.0  |February 2023 - March 2024|Single, double, and triple filaments|


## Data Use Notes

This code is publicly available according to the NIST statements of copyright, fair use and licensing; see 
https://www.nist.gov/director/copyright-fair-use-and-licensing-statements-srd-data-and-software

You may cite the use of this code as follows:

> Friedrich, L.M. (2024), Python tools for image analysis of embedded 3D printing, Version 1.2.0, National Institute of Standards and Technology, doi:10.18434/mds2-3128 (Accessed XXXX-XX-XX)


## References

This version of the code is described in the following papers:

> Friedrich, L.M., Woodcock, J.W. (2024) Inducing Inter-Filament Fusion during Embedded 3D Printing of Silicones, submitted for publication
> 
> Friedrich, L.M., Woodcock, J.W. (2024) Filament Disturbance and Fusion during Embedded 3D Printing of Silicones, submitted for publication


The dataset analyzed by this version of the code is stored at:

> Friedrich, L.M., & Woodcock, J.W. (2024) Videos of Single, Double, and Triple Filaments in Embedded 3D Printing, Version 1.0.0, National Institute of Standards and Technology, doi:10.18434/mds2-3195


## Usage

For full functionality, you will need to make the following files:

- Copy `configs/config_template.yml` and call it `configs/config.yml`. Change the path names to the paths to your data. 


## Data Overview

The files included in this publication use the following hierarchy:

- *README.md*

- *LICENSE*

- *requirements.txt*
    - List of required packages, for use with virtual environments. This file can be used with virtualenv for easy import of dependencies.

- **configs/**
    - *config.yml*
        - set path variables
        
    - *logging.yml*
        - set logging variables
        
- **ipynb/**
    - **singleDisturb/**
        - For analyzing data from disturbed single lines. Designed for version 1.1.0 and may be incompatible with current code.
          
        - *1_file_handling.ipynb*
            - Sort files into the correct folders
              
        - *2_vid_stills.ipynb*
            - Export stills from videos
              
        - *3_pic_stitch_plot.ipynb*
            - Plot stills from different prints together in the same plot
              
        - *4_nozzle_detect.ipynb*
            - Detect the nozzle for each print folder
              
        - *5_still_measure.ipynb*
            - Segment and measure stills
              
        - *6_summary_horiz.ipynb*
            - Summarize data from horizontal lines
              
        - *6_summary_vert.ipynb*
            - Summarize data from vertical lines
              
        - *6_summary_xs.ipynb*
            - Summarize data from cross-sections
         
    - **singleDoubleTriple/**
        - For analyzing data from single, double, and triple lines. This is designed for version 1.2.0 of the code.
         
        - *0_debugfile.ipynb*
            - Run full analysis for a single print folder and debug errors
            
        - *1_file_handling.ipynb*
            - Sort files into the correct folders
     
        - *2_debug.ipynb*
            - Run full analysis for all print folders in a folder and debug errors
              
        - *3_still_measure.ipynb*
            - Segment and measure stills
         
        - *4_testFailures.ipynb*
            - Go through measured folders and check segmentation, address errors
         
        - *5_measure_manual.ipynb*
            - Manually label morphologies
              
        - *6_summary_horiz.ipynb*
            - Summarize data from horizontal out of plane lines
         
        - *6_summary_under.ipynb*
            - Summarize data from horizontal in plane lines
              
        - *6_summary_vert.ipynb*
            - Summarize data from vertical lines
              
        - *6_summary_xsy.ipynb*
            - Summarize data from cross-sections of horizontal in plane lines
              
        - *6_summary_xsz.ipynb*
            - Summarize data from cross-sections of horizontal out of plane lines
              
        - *6_uncertainty.ipynb*
            - Get uncertainty and ranges for scaling variables

        - *7_pic_stitch_plot.ipynb*
            - Plot stills from different prints together in the same plot
 
        - *7_time_plot.ipynb*
            - Plot parameters over time
              
        - *7_vid_stills.ipynb*
            - Export stills from videos
              
        - *still_measure_horiz_MLtrain.ipynb*
            - Train and evaluate a machine learning model for segmenting horizontal out of plane lines (not used in papers)

        - *still_measure_horiz_test.ipynb*
            - Evaluate a traditional computer vision model for segmenting horizontal out of plane lines
       
        - *still_measure_vert_MLtrain.ipynb*
            - Train and evaluate a machine learning model for segmenting vertical lines (not used in papers)

        - *still_measure_vert_test.ipynb*
            - Evaluate a traditional computer vision model for segmenting vertical lines
         
        - *still_measure_xs_test.ipynb*
            - Evaluate a traditional computer vision model for segmenting cross-sections

    - **singleLines/**
        - For analyzing data from single lines. This is designed for version 1.0.0 of the code and may be incompatible with current python files.

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
    
- **py/**
    - Functions
 
    - *__init__.py*
        - Headers
     
    - *full_sequence.py*
        - Runs through the entire analysis process for all folders
     
    - **file/**
        - *f_tools.py*
            - Tools for handling files and strings
         
        - *file_handling.py*
            - Functions that find and sort files

        - *file_names.py*
            - Tools for working with file names
         
        - *folder_loop.py*
            - Class that loops through all print folders
         
        - *levels.py*
            - Tools for labeling where files lie in a file hierarchy
         
        - *print_file_dict.py*
            - Tools for finding files in a print folder
         
        - *print_folders.py*
            - Tools for finding all print folders in a larger folder
         
    - **im/**
        - *contour.py*
            - Tools for working with contours of segmented images
         
        - *crop.py*
            - Tools for cropping images
         
        - *gif.py*
            - Tools for generating moving gifs
         
        - *im_fill.py*
            - Tools for filling holes in segmented images
         
        - *imshow.py*
            - Tools for displaying images in jupyter notebooks
         
        - *morph.py*
            - Tools for morphological operations like opening and closing
         
        - *s_segmentCombiner*
            - Class that combines images segmented using the ML model and the conventional model
         
        - *s_segmenter.py*
            - Class that segments images
         
        - *s_segmenterDF.py*
            - Class that works with measured segments

        - *segmenterSingle.py*
            - Class that segments images of single filaments, from version 1.0.0
         
        - *segment.py*
            - Tools for segmenting images
    
    - **metrics/**
        - *crop_locs.py*
            - Class for holding the locations of cropped images
         
        - *m_disturb.py*
            - Tools to import for analyzing disturbed single lines (version 1.1.0)
         
        - *m_SDT.py*
            - Tools to import for analyzing single, double, and triple lines (version 1.2.0)
         
        - *m_single.py*
            - Tools to import for analyzing single lines (version 1.0.0)
         
        - *m_stats.py*
            - Statistical tools
         
        - *m_tools.py*
            - Tools for measuring images
         
        - *open_in_paint.py*
            - Tools for opening images in MS Paint
         
        - **m_file/**
            - *file_disturb.py*
                - Functions for collecting measurements from a single image of a disturbed line
             
            - *file_horiz_disturb.py*
                - Functions for collecting measurements from a single image of a disturbed horizontal line
             
            - *file_horiz_SDT.py*
                - Functions for collecting measurements from a single image of a single/double/triple horizontal line
             
            - *file_horiz_single.py*
                - Functions for collecting measurements from a single image of a single horizontal line
             
            - *file_horiz.py*
                - Functions for collecting measurements from a single image of a horizontal line
             
            - *file_metric.py*
                - All functions for collecting measurements from a single image
             
            - *file_ML.py*
                - Functions for working with machine learning models for segmenting images
             
            - *file_SDT.py*
                - Functions for collecting measurements from a single image of a single/double/triple line
             
            - *file_single.py*
                - Functions for collecting measurements from a single image of a single line
             
            - *file_under_SDT.py*
                - Functions for collecting measurements from a single image of a single/double/triple line viewed from under
             
            - *file_unit.py*
                - Functions for testing data collection on single images
             
            - *file_vert_disturb.py*
                - Functions for collecting measurements from a single image of a disturbed vertical line
             
            - *file_vert_SDT.py*
                - Functions for collecting measurements from a single image of a single/double/triple vertical line
             
            - *file_vert_single.py*
                - Functions for collecting measurements from a single image of a single vertical line
             
            - *file_vert.py*
                - Functions for collecting measurements from a single image of a vertical line
             
            - *file_xs_disturb.py*
                - Functions for collecting measurements from a single image of a disturbed cross-section
             
            - *file_xs_SDT.py*
                - Functions for collecting measurements from a single image of a single/double/triple cross-section
             
            - *file_xs_single.py*
                - Functions for collecting measurements from a single image of a single cross-section
             
            - *file_xs_triple.py*
                - Functions for collecting measurements from a single image of a triple line cross-section
             
            - *file_xs.py*
                - Functions for collecting measurements from a single image of a cross-section
             
        - **m_folder/**
            - *folder_disturb.py*
                - Functions for collecting measurements from a print folder of disturbed lines
             
            - *folder_horiz_disturb.py*
                - Functions for collecting measurements from a print folder of disturbed horizontal lines
             
            - *folder_horiz_SDT.py*
                - Functions for collecting measurements from a print folder of single/double/triple horizontal lines
             
            - *folder_manual.py*
                - Functions for manually labeling morphologies in a folder
             
            - *folder_metric_exporter.py*
                - Tools for exporting cropped images, segmenting images
             
            - *folder_metric.py*
                - Functions for analyzing print folders
             
            - *folder_SDT.py*
                - Functions for collecting measurements from a print folder of single/double/triple lines
             
            - *folder_single.py*
                - Functions for collecting measurements from a print folder of single lines
             
            - *folder_size_check.py*
                - Functions for making sure machine learning and conventional segmentation images are the same size
             
            - *folder_under_SDT.py*
                - Functions for collecting measurements from a print folder of single/double/triple lines viewed from under
             
            - *folder_vert_disturb.py*
                - Functions for collecting measurements from a print folder of disturbed vertical lines
             
            - *folder_vert_SDT.py*
                - Functions for collecting measurements from a print folder of single/double/triple vertical lines
             
            - *folder_xs_disturb.py*
                - Functions for collecting measurements from a print folder of disturbed cross-sections
             
            - *folder_xs_SDT.py*
                - Functions for collecting measurements from a print folder of single/double/triple cross-sections
             
        - **m_plot/**
            - *color.py*
                - Tools for handling plot color schemes
             
            - *legend.py*
                - Tools for handling plot legends
             
            - *m_plots.py*
                - Tools for plotting measurements
             
            - *markers.py*
                - Tools for handling plot markers
             
            - *p_contour.py*
                - Tools for making contour plots
             
            - *p_mesh.py*
                - Tools for making gradient plots
             
            - *p_metric.py*
                - Tools for making generic plots
             
            - *p_multi_specific.py*
                - Tools for plotting a single yvar measured on line 1, line 2, etc., across the same xvar and color variable
             
            - *p_multi.py*
                - Tools for plotting multiple axes in one figure
             
            - *p_qualityScatter.py*
                - Tools for making scatter plots of qualitative measurements
             
            - *p_scatter.py*
                - Tools for making scatter plots
             
            - *p_SDT.py*
                - All tools for making plots in single/double/triple line prints
             
            - *p_single.py*
                - All tools for making plots in single line prints
             
            - *p_SDT.py*
                - All tools for making plots in single/double/triple line prints
             
            - *p_xvarlines.py*
                - Tools for plotting a single yvar measured on line 1, line 2, etc., across the same xvar and color variable
             
            - *p_yvarlines.py*
                - Tools for plotting a single yvar measured on line 1, line 2, etc., across the same xvar and color variable
             
            - *p_sizes.py*
                - Tools for handling plot sizes
             
            - *p_stats_table.py*
                - Tools for creating tables of correlation strengths
             
        - **m_summarizer/**
            - *failureTest.py*
                - Tools for testing failed files
             
            - *summarizer_disturb.py*
                - Tools for collecting summaries for all folders into one table for disturbed single lines
             
            - *summarizer_SDT.py*
                - Tools for collecting summaries for all folders into one table for single/double/triple lines
             
            - *summarizer.py*
                - Tools for collecting summaries for all folders into one table 
             
        - **m_summary/**
            - *summary_disturb.py*
                - Tools for handling summary tables for disturbed single lines
             
            - *summary_ideals.py*
                - Ideal values for summarized metrics
    
            - *summary_metric.py*
                - Tools for handling summary tables
                  
            - *summary_SDT.py*
                - Tools for handling summary tables for single/double/triple lines
                  
            - *summary_single.py*
                - Tools for handling summary tables for single lines

    - **pic_plots/**
        - *p_comboPlot.py*
            - Class for plotting multiple images on the same plot
         
        - *p_disturb.py*
            - Class for plotting multiple images on the same plot for disturbed lines
         
        - *p_folderImages.py*
            - Tools for finding images to plot
         
        - *p_plots.py*
            - All tools for plotting images on the same plot
         
        - *p_SDT.py*
            - Class for plotting multiple images on the same plot for single/double/triple lines
         
        - *p_triple.py*
            - Class for plotting multiple images on the same plot for triple lines
         
    - **pic_stitch/**
        - *p_bas.py*
            - Functions for sorting bascam stills to be stitched
         
        - *p_stillGroup.py*
            - Functions for stitching a single group of bascam stills
         
        - *pic_stitch.py*
            - Functions for stitching images
         
    - **progDim/**
        - *pg_SDT.py*
            - Tools for finding programmed dimensions and timings of printed single/double/triple lines
         
        - *pg_singleDisturb.py*
            - Tools for finding programmed dimensions and timings of disturbed single lines
         
        - *pg_singleLine.py*
            - Tools for finding programmed dimensions and timings of single lines
         
        - *pg_tools.py*
            - Tools for finding programmed dimensions and timings
         
        - *pg_triple.py*
            - Tools for finding programmed dimensions and timings of triple lines
         
        - *prog_dim.py*
            - Class for finding programmed dimensions and timings
         
        - *progDimsChecker.py*
            - Class for checking that progDims make sense
         
        - *progDimsLabeler.py*
            - Class for labeling lines from progPos
         
        - *progPosChecker.py*
            - Class for checking that progPos make sense
         
        - *progPosSplitter.py*
            - Class for splitting time tables into programmed positions
         
        - *timeRewriteChecker.py*
            - Class for checking that the time table makes sense
         
    - **tools/**
        - *config.py*
            - Tools for importing config files

        - *figureLabels.py*
            - Tools for adding subfigure labels to plots
         
        - *logs.py*
            - Tools for handling status logs
         
        - *plainIm.py*
            - Tools for importing and exporting files
         
        - *regression.py*
            - Tools for measuring regression strengths
         
        - *timeCounter.py*
            - Tools for measuring code speeds
         
    - **val/**
        - *v_fluid.py*
            - Class for storing metadata about fluids
         
        - *v_geometry.py*
            - Class for storing metadata about the system geometry
         
        - *v_pressure.py*
            - Class for storing metadata about the extrusion pressure
         
        - *v_print.py*
            - Class for storing metadata about a print folder
         
        - *v_tables.py*
            - Class for extracting information from tables of metadata stored outside of print folders
         
    - **vid/**
        - *analysis_single.py*
            - Tools for analyzing videos of single lines (from version 1.0.0)
         
        - *background.py*
            - Class for exporting background images
         
        - *noz_detect.py*
            - Class for detecting nozzle
         
        - *noz_detector_side.py*
            - Class for detecting nozzle for side views
         
        - *noz_detector_tools.py*
            - Tools for detecting nozzle
         
        - *noz_detector_under.py*
            - Class for detecting nozzle for underneath views
         
        - *noz_detector.py*
            - Imports packages for nozzle detection
         
        - *noz_dims_side.py*
            - Class for storing nozzle dimensions for side views
         
        - *noz_dims_tools.py*
            - Class for storing nozzle dimensions
         
        - *noz_dims_under.py*
            - Class for storing nozzle dimensions for under views
         
        - *noz_dims.py*
            - Imports packages for nozzle dimension handling
         
        - *noz_frame.py*
            - Class for collecting frames from videos
         
        - *noz_plots.py*
            - Class for plotting detected nozzles on images
         
        - *v_tools.py*
            - Tools for working with videos

- **tests/**
    - Unit tests
    
    - *test_disturbHoriz.py*, ~.csv
        - Go through files and test if disturbHoriz files get correctly measured
    
    - *test_disturbVert.py*, ~.csv
        - Go through files and test if disturbVert files get correctly measured
    
    - *test_disturbXS.py*, ~.csv
        - Go through files and test if disturbXS files get correctly measured
    
    - *test_fileDateAndTime.py*, ~.csv
        - Script for testing that dates get correctly read
    
    - *test_fileDict.py*, ~.csv
        - Script for testing that files get correctly labeled
     
    - *test_fileLabel.py*, ~.csv
        - Script for testing that file hierarchy gets correctly labeled
    
    - *test_geometryVals.py*, ~.csv
        - Go through files and test if geometry metadata gets correctly labeled
    
    - *test_nozDetect.py*, ~.csv
        - Go through files and test if nozzle is correctly detected
    
    - *test_pressureVals.py*, ~.csv
        - Go through files and test if pressures get correctly calculated
    
    - *test_SDTHoriz.py*, ~.csv
        - Go through files and test if disturbHoriz single/double/triple files are correctly measured
    
    - *test_SDTVert.py*, ~.csv
        - Go through files and test if disturbVert single/double/triple files are correctly measured
    
    - *test_SDTXS.py*, ~.csv
        - Go through files and test if disturbXS single/double/triple files are correctly measured
    
    - *test_stitch_bas.py*, ~.csv
        - Go through files and test if files are labeled correctly for stitching

## Methodology


### Experiments

Use the GUI at https://github.com/usnistgov/ShopbotPyQt/ to generate files (examples in parentheses) during printing of single lines. For each print, the GUI exports:

- a video of the printing process from a color Basler camera (`disturbUnder_2_0.500_Basler camera_I_PDMSS7.5_S_3.50_230920_110040_0.avi`)
- a table of pressure over time (`disturbUnder_2_0.500_time_I_PDMSS7.5_S_3.50_230920_110040_0.csv`)
- a table of metadata (`disturbUnder_2_0.500_meta_I_PDMSS7.5_S_3.50_230920_110040_0`)
- a group of images from the Basler camera, taken after the print is done (`disturbUnder_2_0.500_Basler camera_I_PDMSS7.5_S_3.50_230920_110042_9.png`)


### Analysis
Using the jupyter notebooks in ipynb/singleDoubleTriple/:

1. Use `1_fileHandling.ipynb` to sort and label the print folders into folders by sample name, run name
2. Use `2_debug.ipynb` to analyze all folders
3. Use `3_still_measure.ipynb` to summarize extracted values from all runs
4. Use `4_testFailures.ipynb` to debug failed measurements
5. Use `5_measure_manual.ipynb` to extract and summarize values from videos
6. Use `6_summary_horiz.ipynb`, `6_summary_under.ipynb`, `6_summary_vert.ipynb`, `6_summary_xsy.ipynb`, `6_summary_xsz.ipynb` to plot measured values and generate statistical tables
7. Use `6_uncertainty.ipynb` to extract ranges and uncertainties of scaling variables
8. Use `7_pic_stitch_plot.ipynb`, `7_time_plot.ipynb`, `7_vid_stills.ipynb` to generate plots for individual folders and images


