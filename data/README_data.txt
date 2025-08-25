Sample Data Description
=======================

This folder contains example input and groundtruth data for Sample 1 only.
They are provided for demonstration and debugging purposes.

Folder Structure:
-----------------

data/
├── input/
│   └── 1.xlsx                     # Input spectra for Sample 1
├── groundtruth_d65/
│   └── Sample1_D65.xlsx          # Groundtruth matrix under D65 illumination
├── groundtruth_d75/
│   └── Sample1_D75.xlsx          # Groundtruth matrix under D75 illumination
├── groundtruth_A/
│   └── Sample1_A.xlsx            # Groundtruth matrix under A illumination
├── groundtruth_C/
│   └── Sample1_C.xlsx            # Groundtruth matrix under C illumination

Format Notes:
-------------

- `input/1.xlsx` is a 49×13 Excel file:
  - 49 rows represent wavelengths
  - 1st column is wavelength
  - Columns 2–13 are spectral values (12 bands)

- Each groundtruth file is a 41×49 matrix:
  - Rows: different emission wavelengths
  - Columns: excitation wavelengths
  - Some upper-triangle regions may be zero-padded

Usage:
------

You can test the model pipeline using this sample data by modifying `train.py`
to load only Sample 1.

For example:

```python
X = X[1:2]   # Select Sample 1
Y = Y[1:2]