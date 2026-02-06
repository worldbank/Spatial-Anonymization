
# Background 
----
**spatial-anonymization** is a Python package developed by <span style="color:#3EACAD"> DECAT team at the World Bank Group</span>. The package has utilities to enable anonymization of GPS coordinates. For example, given highly confidential household GPS coordinates from a survey which need to be disseminated, **spatial-anonymization** enables one to perturb the raw coordinates and generate new anomymized coordinates with dramatically reduced risk of household re-identification. The anonymization techniques in this package have been adapted and inspired from the following works.
1. [**The DHS Program Geographic Displacement Procedure.**](https://www.dhsprogram.com/pubs/pdf/SAR7/SAR7.pdf). Implementation of displacement in the ```SimpleDisplacer``` class is based on the procedure described in this manual.
2. [**UNSD Guidelines on Spatial Anonymization.**](https://unstats.un.org/unsd/statcom/52nd-session/documents/BG-3l-Spatial_Anonymization-E.pdf). The UN Statistics Division(UNSD) in coordination with the World Bank Group and DHS Program developed guidelines on how to mask GPS coordinates from surveys. In addition, the guidelines also describes quantitative meausures of household re-identification (disclosure) after the anonymization process. The package implements some of these guidelines.

# Features
The **spatial-anonymization** package allows you to do the folowing:
1. Generate anonymized coordinates ready for publication/dissemination from raw confidential coordinates
2. Measure risk of re-identification of anonymized coordinates
3. Perform commonly required geospatial data processing related to GPS coordinates from household surveys

# Installation

## Prerequisites

This package relies heavily on geospatial libraries. Before installing, ensure you have the following system-level dependencies:

### GDAL Installation

**GDAL** (Geospatial Data Abstraction Library) is required for raster and vector data processing. We recommend installing it via conda:

```bash
# Using conda
conda install -c conda-forge gdal

# Or using mamba (faster)
mamba install -c conda-forge gdal
```

**macOS (Homebrew):**
```bash
brew install gdal
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install gdal-bin libgdal-dev
```

**Windows:**
Download and install from [OSGeo4W](https://trac.osgeo.org/osgeo4w/) or use conda (recommended).

---

## Installation Methods

### Option 1: Using `uv` (Recommended)

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable package management.

1. **Install uv** (if not already installed):
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # Or via pip
   pip install uv
   ```

2. **Install the package from GitHub**:
   ```bash
   uv pip install git+https://github.com/worldbank/Spatial-Anonymization.git
   ```

3. **For development installation** (editable mode):
   ```bash
   git clone https://github.com/worldbank/Spatial-Anonymization.git
   cd Spatial-Anonymization
   uv pip install -e .
   ```

4. **With optional dependencies** (notebook support):
   ```bash
   uv pip install "git+https://github.com/worldbank/Spatial-Anonymization.git#egg=spatial-anonymization[notebook]"
   ```

---

### Option 2: Using `pip`

```bash
# Direct installation from GitHub
pip install git+https://github.com/worldbank/Spatial-Anonymization.git

# For editable/development installation
git clone https://github.com/worldbank/Spatial-Anonymization.git
cd Spatial-Anonymization
pip install -e .

# With optional dependencies
pip install "git+https://github.com/worldbank/Spatial-Anonymization.git#egg=spatial-anonymization[notebook]"
```

---

## Key Dependencies

The package automatically installs these geospatial libraries:

- **GeoPandas** (≥0.14.0): Vector data manipulation
- **Rasterio** (≥1.3.0): Raster data I/O
- **Shapely** (≥2.0.0): Geometric operations
- **PyProj** (≥3.6.0): Coordinate reference systems
- **NumPy** (≥1.26.0) & **Pandas** (≥2.0.0): Data processing
- **rasterstats** (≥0.20.0): Zonal statistics
- **PyYAML** (≥6.0.3): Configuration management

---

## Verify Installation

```python
import spatial_anonymization
print(spatial_anonymization.__version__)

# Test basic import
from spatial_anonymization import anonymize_survey_data
print("✓ Installation successful!")
```

---

## Troubleshooting

**GDAL Import Error:**
If you encounter `ImportError: GDAL not found`, ensure GDAL is installed system-wide before installing the package.

**Rasterio/Fiona Build Errors:**
These depend on GDAL. Install GDAL first, then try again.

**macOS ARM64 (M1/M2) Issues:**
Use conda-forge channel for ARM-optimized builds:
```bash
conda install -c conda-forge gdal rasterio geopandas
```

---

## Python Version Support

- **Python 3.10+** required
- Tested on Python 3.10, 3.11, 3.12, 3.13 
# Usage
Refer to the comprehensive [usage examples notebook](https://github.com/worldbank/Spatial-Anonymization/blob/main/examples/usage_examples.ipynb) for detailed demonstrations of:
- Simple API usage with auto-downloaded data
- Config-based API for full control
- Urban/rural differentiation
- Spatial constraints (admin boundaries, water bodies)
- Population-adaptive anonymization
- Custom data sources 

# Documentation
This package is being developed as part of the **Surveys Geo-enhacement** project within DECAT. See [Survey Geo-enhacement repo](https://github.com/worldbank/DECAT_HH_Geovariables) for details. 


