"""
============================================================================================
                                 data_process_subset.py

Python file that creates a subset of the whole dataset with relatively even committor
distribution.
 ===========================================================================================
// H. Naguszewski. University of Warwick
"""

import torch
import numpy as np

print("Beginning subset process")

image_dir = "../training_data/image_data"
feature_dir = "../training_data/feature_data"
edge_dir = "../training_data/edge_data"
label_dir = "../training_data/label_data"
image_data = torch.load(image_dir)
feature_data = torch.load(feature_dir)
edge_data = torch.load(edge_dir)
label_data = torch.load(label_dir)[:,0]

# sort data based on committor
ordering = label_data.argsort()
image_data = image_data[ordering]
label_data = label_data[ordering]
edge_data = edge_data[ordering]
feature_data = feature_data[ordering]

# map features to [0,1]
feature_data -= feature_data.min()
feature_data /= feature_data.max()

tmp_idx = np.zeros(50000)
select = 20
select_count = 0
sample_space = np.linspace(0.0, 1.0, 101)
j = 0; k = 0

for i, sample in enumerate(label_data):
    value = sample_space[j]
    if abs(sample.item() - value) < 0.001 and select_count < select:
        select_count += 1
        tmp_idx[k] = i; k += 1
    elif abs(sample.item() - value) > 0.001:
        select_count = 0
        j += 1

tmp_idx = tmp_idx[:k]

print("Subset size = %d" % k)

torch.save(image_data[tmp_idx], "../training_data/image_data_subset")
torch.save(feature_data[tmp_idx], "../training_data/feature_data_subset")
torch.save(edge_data[tmp_idx], "../training_data/edge_data_subset")
torch.save(label_data[tmp_idx], "../training_data/label_data_subset")
