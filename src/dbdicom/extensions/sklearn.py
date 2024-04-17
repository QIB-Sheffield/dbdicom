import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from dbdicom.extensions import scipy, vreg


# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
def kmeans(features, mask=None, n_clusters=2, multiple_series=False, normalize=True, return_features=False):
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
        mask.message('Reading mask array..')
        #mask = vreg.map_to(mask, features[0], mask=True)
        #mask_array, _ = mask.array(['SliceLocation', 'AcquisitionTime'], pixels_first=True)
        mask_array, _ = vreg.mask_array(mask, on=features[0], dim='AcquisitionTime')
        mask_array = np.ravel(mask_array)
        mask_indices = tuple(mask_array.nonzero())

    # Ensure all the features are in the same geometry as the reference feature
    features = scipy.overlay(features)

    # Create array with shape (n_samples, n_features) and mask if needed.
    array = []
    for s, series in enumerate(features):
        series.progress(s+1, len(features), 'Reading features..')
        arr, headers = series.array(['SliceLocation', 'AcquisitionTime'], pixels_first=True)
        shape = arr.shape 
        arr = np.ravel(arr)
        if mask is not None:
            arr = arr[mask_indices]
        #if normalize:
        #    arr = (arr-np.mean(arr))/np.std(arr)
        array.append(arr)
    array = np.vstack(array).T

    # Perform the K-Means clustering.
    series.message('Clustering. Please be patient - this is hard work..')
    if normalize:
        X = StandardScaler().fit_transform(array)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=3, verbose=1).fit(X)
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=3, verbose=1).fit(array)

    # Create an output array for the labels
    if mask is not None:
        mask.message('Creating output array..')
        output_array = np.zeros(shape)
        output_array = np.ravel(output_array)
        output_array[mask_indices] = 1+kmeans.labels_ 
    else:
        output_array = 1+kmeans.labels_
    output_array = output_array.reshape(shape)

    # Save the results in DICOM
    series.message('Saving clusters..')
    if multiple_series:
        # Save each cluster as a separate mask
        clusters = []
        for cluster in range(1,1+n_clusters):
            array_cluster = np.zeros(output_array.shape)
            array_cluster[output_array == cluster] = 1
            series_cluster = features[0].new_sibling(SeriesDescription = 'KMeans cluster ' + str(cluster))
            series_cluster.set_array(array_cluster, headers, pixels_first=True)
            _reset_window(series_cluster, array_cluster)
            clusters.append(series_cluster)
    else:
        # Save the label array in a single series
        clusters = features[0].new_sibling(SeriesDescription = 'KMeans')
        clusters.set_array(output_array, headers, pixels_first=True)
        _reset_window(clusters, output_array)

    # If requested, return features (mean values over clusters + size of cluster)
    if return_features: # move up
        cluster_features = []
        for cluster in range(1,1+n_clusters):
            vals = []
            #locs = (output_array.ravel() == cluster)
            locs = (1+kmeans.labels_ == cluster)
            for feature in range(array.shape[1]):
                val = np.mean(array[:,feature][locs])  
                vals.append(val)
            vals.append(np.sum(locs))
            cluster_features.append(vals) 
        return clusters, cluster_features   

    return clusters


def kmeans_4d(features, mask=None, n_clusters=2, multiple_series=False, normalize=True, return_features=False):

    # If a mask is provided, map it onto the reference feature and 
    # extract the indices of all pixels under the mask
    if mask is not None:
        mask.message('Reading mask array..')
        mask_array, _ = vreg.mask_array(mask, on=features[0], dim='AcquisitionTime')
        mask_array = np.ravel(mask_array)
        mask_indices = tuple(mask_array.nonzero())

    # Create array with shape (n_samples, n_features) and mask if needed.
    array, headers = features.array(['SliceLocation', 'AcquisitionTime'], pixels_first=True)
    shape = array.shape 
    array = array.reshape((-1, shape[-1]))
    if mask is not None:
        array = array[mask_indices, :]

    # Perform the K-Means clustering.
    features.message('Clustering. Please be patient - this is hard work..')
    if normalize:
        X = StandardScaler().fit_transform(array)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=3, verbose=1).fit(X)
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=3, verbose=1).fit(array)

    # Create an output array for the labels
    if mask is not None:
        mask.message('Creating output array..')
        output_array = np.zeros(shape)
        output_array = np.ravel(output_array)
        output_array[mask_indices] = 1+kmeans.labels_ 
    else:
        output_array = 1+kmeans.labels_
    output_array = output_array.reshape(shape)

    # Save the results in DICOM
    features.message('Saving clusters..')
    if multiple_series:
        # Save each cluster as a separate mask
        clusters = []
        for cluster in range(1,1+n_clusters):
            array_cluster = np.zeros(output_array.shape)
            array_cluster[output_array == cluster] = 1
            series_cluster = features[0].new_sibling(SeriesDescription = 'KMeans cluster ' + str(cluster))
            series_cluster.set_array(array_cluster, headers, pixels_first=True)
            _reset_window(series_cluster, array_cluster)
            clusters.append(series_cluster)
    else:
        # Save the label array in a single series
        clusters = features[0].new_sibling(SeriesDescription = 'KMeans')
        clusters.set_array(output_array, headers, pixels_first=True)
        _reset_window(clusters, output_array)

    # If requested, return features (mean values over clusters + size of cluster)
    if return_features: # move up
        cluster_features = []
        for cluster in range(1,1+n_clusters):
            vals = []
            #locs = (output_array.ravel() == cluster)
            locs = (1+kmeans.labels_ == cluster)
            for feature in range(array.shape[1]):
                val = np.mean(array[:,feature][locs])  
                vals.append(val)
            vals.append(np.sum(locs))
            cluster_features.append(vals) 
        return clusters, cluster_features   

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