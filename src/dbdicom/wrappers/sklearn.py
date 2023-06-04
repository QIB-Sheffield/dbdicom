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
    features = scipy.overlay(features)

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



def sequential_kmeans(features, mask=None, n_clusters=2, multiple_series=False):
    """
    Labels structures in an image using sequential k-means clustering
    
    Sequential here means that the clustering is always performed on a single feature
    using the output of the previous iteration as a mask for the next.

    Parameters
    ----------
    input: list of dbdicom series (one for each feature)
    mask: optional mask for clustering
    
    Returns
    -------
    clusters : list of dbdicom series, with labels per cluster.
    """

    f = features[0]
    clusters = kmeans([f], mask=mask, n_clusters=n_clusters, multiple_series=True)
    for f in features[1:]:
        cf = []
        for mask in clusters:
            cf += kmeans([f], mask=mask, n_clusters=n_clusters, multiple_series=True)
            mask.remove()
        clusters = cf

    # Return clusters as a list
    if multiple_series:
        return clusters

    # Return a single label series
    label = masks_to_label(clusters)
    for c in clusters:
        c.remove()
    return label



def masks_to_label(masks):
    "Convert a list of masks into a single label series"

    # Ensure all the masks are in the same geometry
    masks = scipy.overlay(masks)

    # Create a single label array
    array, headers = masks[0].array(['SliceLocation', 'AcquisitionTime'], pixels_first=True)
    msg = 'Creating a single label image'
    for m, mask in enumerate(masks[1:]):
        mask.status.progress(m+1, len(masks)-1, msg)
        arr, _ = mask.array(['SliceLocation', 'AcquisitionTime'], pixels_first=True)
        ind = tuple(arr.nonzero())
        array[ind] = m+2

    # Save label array to disk
    desc = masks[0].SeriesDescription
    label = masks[0].new_sibling(SeriesDescription = desc + ' [Label]')
    label.set_array(array, headers, pixels_first=True)  

    return label    



# Helper functions

def _reset_window(image, array):
    min = np.amin(array)
    max = np.amax(array)
    image.WindowCenter= (max+min)/2
    image.WindowWidth = 0.9*(max-min)