#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

# Normalize a sequence of 3D points using nearest neighbor averaging
def nearest_neighbor_point_normalization(points, n=5):
    """
    Normalize a sequence of 3D points using nearest neighbor averaging.

    Parameters:
    - points: 3D point sequence
    - n: Number of previous points to consider for averaging (default is 5)

    Returns:
    - Normalized sequence of 3D points
    """
    points_normalized = []

    for i in range(len(points)):
        # Determine the start index for averaging, ensuring it doesn't go below 0
        id_start = max(0, i - n + 1)

        # Extract the subset of points to be averaged
        subset = points[id_start:i + 1]

        # Compute the average point, considering only x, y, and z coordinates
        point_average = np.mean(subset, axis=0)

        # Append the averaged point to the normalized sequence
        points_normalized.append(point_average[:3])

    return np.array(points_normalized)

# Normalize a sequence of 3D points by subtracting the root point.
def root_point_normalization(points):
    """
    Normalize a sequence of 3D points by subtracting the root point.

    Parameters:
    - points: 3D point sequence

    Returns:
    - Normalized sequence of 3D points
    """
    # Get the root point (first point in the sequence)
    root_point = points[0]

    # Subtract the root point from all points in the sequence
    return points - root_point

# Pad a sequence with a specified padding value to reach a maximum length.
def pad_sequence(sequence, max_length=221, padding_value=0):
    """
    Pad a sequence with a specified padding value to reach a maximum length.

    Parameters:
    - sequence: Input sequence
    - max_length: Desired maximum length of the sequence
    - padding_value: Value to use for padding (default is 0)

    Returns:
    - Padded sequence
    """
    # Create a matrix filled with the padding value, with dimensions (max_length, num_features)
    padded_sequence = np.full((max_length, sequence.shape[1]), padding_value)

    # Copy the input sequence to the end of the padded sequence
    padded_sequence[-len(sequence):] = sequence

    return padded_sequence

