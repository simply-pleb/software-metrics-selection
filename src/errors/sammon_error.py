import numpy as np
from scipy.spatial.distance import pdist, squareform

def sammon_error(X, selected_features):
    """
    Calculate Sammon error (stress) between original data X and low-dimensional data X_low.
    
    Parameters:
    - X: Original data (n_samples, n_features)
    - selected_features: Binari vector representing the selected columns (n_features)
    
    Returns:
    - sammon_error: Sammon error (float)
    """
    # Ensure selected_features is a boolean or binary mask
    selected_features = np.array(selected_features, dtype=bool)
    
    X_low = X[:, selected_features]
    
    # Compute pairwise distances in the original data
    D_orig = pdist(X, metric='euclidean')
    D_orig = squareform(D_orig)  # Convert to a square matrix
    
    # Compute pairwise distances in the low-dimensional data (selected features)
    D_low = pdist(X_low, metric='euclidean')
    D_low = squareform(D_low)
    
    # Avoid division by zero by replacing zeros with a small number
    D_orig[D_orig == 0] = np.finfo(float).eps
    
    # Calculate the Sammon error (Sammon stress)
    delta = D_orig - D_low
    sammon_error = np.sum((delta**2) / D_orig)
    
    # Normalize the error by the sum of original distances
    sammon_error /= np.sum(D_orig)
    
    return sammon_error