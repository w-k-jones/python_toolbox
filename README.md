# python_toolbox

A mixed bag of tools for data analysis

To install from command line run:

git clone https://github.com/w-k-jones/python_toolbox.git
pip install python_toolbox/

Submodules:

abi_tools:
  A set of functions for working with ABI level 1 data, all using xarray datasets
amv_tools:
  A set of functions for loading SEVIRI atmospheric motion vectors
dataset_tools:
  General functions for working with xarray datasets and dataarrays
era_tools:
  Functions for working with ECMWF ERA and IFS data. Includes functions for calculating pressure and geopotential on model levels.
plotting_tools:
  Additional matplotlib plotting functions
seviri_tools:
  Functions for working with SEVIRI data
spectral:
  Functions for working with spectral coefficients data. Mostly wrappers for pyspharm and spherepack libraries
thermodynamics:
  Latent heat calculation function
tracking_tools:
  In development tools for detection and tracking of deep convection. To be included in future versions of tobac
