# utils/data_loader.py

import os
import numpy as np
import pandas as pd

def load_combined_spectra(folder_path):
    """
    Load combined excitation input spectra from Excel files.
    Returns shape: [N, 49, D]
    """
    spectra_list = []
    for i in range(41):
        file_path = os.path.join(folder_path, f"{i}.xlsx")
        if os.path.exists(file_path):
            data = pd.read_excel(file_path).values
            spectra_list.append(data)
        else:
            print(f"⚠️ Skipped missing file: {file_path}")
    return np.array(spectra_list)

def load_groundtruth(folder_path, mode="D65"):
    """
    Load groundtruth emission matrix data.
    Returns shape: [N, 41, 49]
    """
    results = []
    for i in range(41):
        file_name = f"Sample{i}_{mode}.xlsx"
        path = os.path.join(folder_path, file_name)
        if os.path.exists(path):
            matrix = pd.read_excel(path, header=None).values
            matrix = zero_upper_triangle(matrix)
            results.append(matrix)
    return np.array(results)

def zero_upper_triangle(matrix):
    """
    Zero out upper-diagonal noise in matrix.
    """
    for i in range(35):
        matrix[i, (14 + i + 1):] = 0
    return matrix