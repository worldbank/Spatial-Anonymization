
# Introduction 
----
**spatial-anonymization** is a Python package developed by <span style="color:#3EACAD"> DECAT team at the World Bank Group</span>. The package has utilities to enable anonymization of GPS coordinates. For example, given highly confidential household GPS coordinates from a survey which need to be disseminated, **spatial-anonymization** enables one to perturb the raw coordinates and generate new anomymized coordinates with dramatically reduced risk of household re-identification. The anonymization techniques in this package have been adapted and inspired from the following works.
1. [**The DHS Program Geographic Displacement procedure.**](https://www.dhsprogram.com/pubs/pdf/SAR7/SAR7.pdf). Implementation of displacement in the ```SimpleDisplacer``` class is based on the procedure described in this manual.
2. [**UNSD guidelines on Spatial Anonymization.**](https://unstats.un.org/unsd/statcom/52nd-session/documents/BG-3l-Spatial_Anonymization-E.pdf). The UN Statistics Division(UNSD) in coordination with the World Bank Group and DHS Program developed guidelines on how to mask GPS coordinates from surveys. In addition, the guidelines also describes quantitative meausures of household re-identification (disclosure) after the anonymization process. The package implements some of these guidelines.

# Features
**spatial-anonymization** allows you to do the folowing:
1. Generate anonymized coordinates ready for publication/dissemination from raw confidential coordinates
2. Measure risk of re-identification of anonymized coordinates
3. Perform commonly required geospatial data processing related to GPS coordinates from household surveys

# Installation
This package is private (available to World Bank Group staff) but will later be made public. Please follow the steps below to install **spatial-anonymization**.
1. **Get added to the repo.** Contact any of the following people:
    - dmatekenya@worldbankgroup.org
    - bstewart@worldbankgroup.org 
2. **Clone the repo.**
3. **Install dependencies**. Although pip takes care of all other dependencies, please install **GDAL** prior to installing the package. The recommended approach is to use conda as shown below:
```
conda install -c conda-forge gdal
``` 
3. **Install the package** Navigate to the package directory and install like below.
```
python -m pip install .
``` 
# Usage
Refer to this [example usage notebook](https://github.com/worldbank/Spatial-Anonymization/blob/main/examples/point_displacement_usage_example.ipynb) for an example usage code. 

# Documentation
This package is being developed as part of the **Surveys Geo-enhacement** project within DECAT. See [Survey Geo-enhacement repo](https://github.com/worldbank/DECAT_HH_Geovariables) for details. 


