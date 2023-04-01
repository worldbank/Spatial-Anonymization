
# Introduction 
----
**spatial-anonymization** is a Python package with utilities to enable anonymization of GPS coordinates. For example, given highly confidential household GPS coordinates from a Household survey which you need to disseminate, **spatial-anonymization** allows you to perturb the raw coordinates and generate new anomymized coordinates with dramaticall reduced risk of identification. 

# Features
**spatial-anonymization** allows you to do the folowing:
1. Generate anonymized coordinates ready for publication/dissemination from raw confidential coordinates
2. Measure risk of re-identification of anonymized coordinates
3. Perform commonly required geospatial data processing related to GPS coordinates from household surveys


# Installation
For now, this package is still private and only available to World Bank Group staff. Do the following to install the package.
1. **Get added to the repo.** Contact dmatekenya@worldbankgroup.org to get added
2. **Clone the repo.**
3. **Install the package** Navigate to the package directory and install like below.
```
python -m pip install .
``` 
# Usage
Please see this [example usage notebook](https://github.com/worldbank/Spatial-Anonymization/blob/main/notebooks/point_displacement_usage_example.ipynb)

## Documentation
This package is being developed as part of the **Surveys Geo-enhacement** project withing DECAT. See [Survey Geo-enhacement repo](https://github.com/worldbank/DECAT_HH_Geovariables) for details. 


