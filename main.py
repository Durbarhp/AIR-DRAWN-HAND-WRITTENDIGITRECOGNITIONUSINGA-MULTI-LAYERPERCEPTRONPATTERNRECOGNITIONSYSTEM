#!/usr/bin/env python
# coding: utf-8

# In[17]:

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd
import numpy as np
from MLPClassifier import MLP
from Preprocessing import nearest_neighbor_point_normalization, root_point_normalization, pad_sequence
import pickle

def main(testdata):
    """
    Classify digit(s) using the trained neural network model.

    Parameters:
    - testdata: Test data (DataFrame or NumPy array) containing 3D locations of finger tips

    Returns:
    - Predicted digit labels
    """
    X_test = []

    # Extract 3D location from the DataFrame or NumPy array
    if type(testdata) == pd.DataFrame:
        location = testdata.values[:, :3]
    elif type(testdata) == np.array or type(testdata) == np.ndarray:
        location = testdata
    else:
        raise ValueError("Input must be either a DataFrame or a NumPy array.")

    # Normalize the location using nearest neighbor averaging and root point normalization
    normalized_location = nearest_neighbor_point_normalization(location)
    normalized_location = root_point_normalization(normalized_location)

    # Pad the normalized location to the specified maximum length
    padded_location = pad_sequence(normalized_location)

    X_test.append(padded_location)
    X_test = np.array(X_test)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Load the trained model from the file
    neural_net = MLP(X_test.shape[1], 128, 10)
    neural_net.load_model('model.pkl')

    # Make predictions on the test data using the loaded model
    C = neural_net.predict(X_test)

    return C

if __name__ == "__main__":
    C = main(testdata)
# In[19]:


# testdata=pd.read_csv('training_data/stroke_7_0001.csv') # input path to file for test data
# # Classify sequence
# digit_classify(testdata)


# In[ ]:




