import os
import numpy as np

file_path = r"C:\Users\yus\Desktop\gestura\model\keypoint_data\ACCENT\1.npy"  

a = np.load(file_path)
        
print(f"File: {file_path}")
print(f"Shape: {a.shape}")
print(f"Dimensions: {a.ndim}")
print(f"Size: {a.size}")
print(f"Data type: {a.dtype}")
        
