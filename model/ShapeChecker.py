import os
import numpy as np

file_path = r"/home/jiyusss/gestura/model/Keypoint_Data_Selected/AND/82979256329312-AND.npy"  

a = np.load(file_path)
        
print(f"File: {file_path}")
print(f"Shape: {a.shape}")
print(f"Dimensions: {a.ndim}")
print(f"Size: {a.size}")
print(f"Data type: {a.dtype}")
        
