# InP_QD_Analysis

**gaussian_fitting_script_final.py**
This expects an Excel file that is made up of sheets, where each sheet is a single experiment measuring absorption over a time period. The first column should be wavelength (nm), second column is energy in eV, subsequent columns are absortion data. There can be as many extra columns as required. It will expect a shoulder in the 1.7-3 eV range, and fit a Gaussian to the lower energy part of the shoulder.
