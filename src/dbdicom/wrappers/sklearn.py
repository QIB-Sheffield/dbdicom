import numpy as np
from sklearn.cluster import KMeans
import dbdicom.wrappers.scipy as scipy


# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
def kmeans(features, mask=None, n_clusters=2, multiple_series=False):
    """
    Labels structures in an image
    
    Wrapper for sklearn.cluster.KMeans function. 

    Parameters
    ----------
    input: list of dbdicom series (one for each feature)
    mask: optional mask for clustering
    
    Returns
    -------
    clusters : list of dbdicom series, with labels per cluster.
    """

    # If a mask is provided, map it onto the reference feature and 
    # extract the indices of all pixels under the mask
    if mask is not None:
        mask.status.message('Reading mask array..')
        mask = scipy.map_to(mask, features[0], mask=True)
        mask_array, _ = mask.array(['SliceLocation', 'AcquisitionTime'], pixels_first=True)
        mask_array = np.ravel(mask_array)
        mask_indices = tuple(mask_array.nonzero())

    # Ensure all the features are in the same geometry as the reference feature
    msg = 'Mapping all features on the same geometry'
    mapped_features = [features[0]]
    for f, feature in enumerate(features[1:]):
        feature.status.progress(f+1, len(features)-1, msg)
        mapped = scipy.map_to(feature, features[0])
        mapped_features.append(mapped)
    features = mapped_features

    # Create array with shape (n_samples, n_features) and mask if needed.
    array = []
    for s, series in enumerate(features):
        series.status.progress(s+1, len(features), 'Reading features..')
        arr, headers = series.array(['SliceLocation', 'AcquisitionTime'], pixels_first=True)
        shape = arr.shape 
        arr = np.ravel(arr)
        if mask is not None:
            arr = arr[mask_indices]
        array.append(arr)
    array = np.vstack(array).T

    # Perform the K-Means clustering.
    series.status.message('Clustering. Please be patient - this is hard work..')
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=3, verbose=1).fit(array)
    
    # Create an output array for the labels
    if mask is not None:
        mask.status.message('Creating output array..')
        array = np.zeros(shape)
        array = np.ravel(array)
        array[mask_indices] = 1+kmeans.labels_ 
    else:
        array = 1+kmeans.labels_
    array = array.reshape(shape)

    # Save the results
    series.status.message('Saving clusters..')
    if multiple_series:
        # Save each cluster as a separate mask
        clusters = []
        for cluster in range(1,1+n_clusters):
            array_cluster = np.zeros(array.shape)
            array_cluster[array == cluster] = 1
            series_cluster = features[0].new_sibling(SeriesDescription = 'KMeans cluster ' + str(cluster))
            series_cluster.set_array(array_cluster, headers, pixels_first=True)
            _reset_window(series_cluster, array_cluster)
            clusters.append(series_cluster)
    else:
        # Save the label array in a single series
        clusters = features[0].new_sibling(SeriesDescription = 'KMeans')
        clusters.set_array(array, headers, pixels_first=True)
        _reset_window(clusters, array)

    return clusters



# Helper functions

def _reset_window(image, array):
    min = np.amin(array)
    max = np.amax(array)
    image.WindowCenter= (max+min)/2
    image.WindowWidth = 0.9*(max-min)