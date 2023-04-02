
# Introduction 
----
**spatial-anonymization** is a Python package deveoped by <span style="color:#3EACAD">DECAT team at the World Bank Group</span>. The package has utilities to enable anonymization of GPS coordinates. For example, given highly confidential household GPS coordinates from a Household survey which need to be disseminated, **spatial-anonymization** enables one to perturb the raw coordinates and generate new anomymized coordinates with dramaticall reduced risk of identification. The anonymization techniques in this package have been adapted and inspired from the following:
1. [** The DHS Program Geographic Displacement procedure.**](https://www.dhsprogram.com/pubs/pdf/SAR7/SAR7.pdf). Some of the techniques used in this package is an implementation of the procedure described in this manual.
2. [** UN Statistics Division guidelines on Spatial Anonymization.**](https://unstats.un.org/unsd/statcom/52nd-session/documents/BG-3l-Spatial_Anonymization-E.pdf). The UN Statistics Division(UNSD) in coordination with the World Bank and DHS Program developed guidelines on how to mask GPS coordinates from surveys as well as meausure re-identification (disclosure) risk. The package implements some of these guidelines.

# Features
**spatial-anonymization** allows you to do the folowing:
1. Generate anonymized coordinates ready for publication/dissemination from raw confidential coordinates
2. Measure risk of re-identification of anonymized coordinates
3. Perform commonly required geospatial data processing related to GPS coordinates from household surveys


# Installation
For now, this package is still private and only available to World Bank Group staff. Do the following to install the package.
1. **Get added to the repo.** Contact dmatekenya@worldbankgroup.org to get added
2. **Clone the repo.**
3. **Install dependencies**. Although pip takes care of all other dependencies, please install **GDAL** separately. The recommended approach is to use conda as shown below:
```
conda install -c conda-forge gdal
``` 
3. **Install the package** Navigate to the package directory and install like below.
```
python -m pip install .
``` 
# Usage
Please see this [example usage notebook](https://github.com/worldbank/Spatial-Anonymization/blob/main/examples/point_displacement_usage_example.ipynb)

## Documentation
This package is being developed as part of the **Surveys Geo-enhacement** project within DECAT. See [Survey Geo-enhacement repo](https://github.com/worldbank/DECAT_HH_Geovariables) for details. 


