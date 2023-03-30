"""
Functions for a new (sub)package dbimage inside dbdicom

A set of dbdicom wrappers for numpy
"""

import numpy as np


def mean_intensity_projection(series):
    """
    Segment by thresholding

    Parameters
    ----------
    series: dbdicom series (4D: slice + time)

    Returns
    -------
    mean : dbdicom series (3D)

    Example:

    mean_series = mean(series, axis=-1)
    """
    # Get numpy array with dimensions (slice, time, x, y)
    # array = series.array('SliceLocation')

    # Get numpy array with dimensions (x, y, slice, time)
    array, headers = series.array('SliceLocation', pixels_first=True)
    array = np.mean(array, axis=-1)
    desc = series.instance().SeriesDescription + ' [Mean Intensity Projection]'
    new_series = series.new_sibling(SeriesDescription=desc)
    new_series.set_array(array, headers[:,0], pixels_first=True)
    return new_series


def maximum_intensity_projection(series):
    """
    Segment by thresholding

    Parameters
    ----------
    series: dbdicom series (4D: slice + time)

    Returns
    -------
    mean : dbdicom series (3D)

    Example:

    mean_series = mean(series, axis=-1)
    """
    # Get numpy array with dimensions (slice, time, x, y)
    # array = series.array('SliceLocation')

    # Get numpy array with dimensions (x, y, slice, time)
    array, headers = series.array('SliceLocation', pixels_first=True)
    array = np.amax(array, axis=-1)
    desc = series.instance().SeriesDescription + ' [Maximum Intensity Projection]'
    new_series = series.new_sibling(SeriesDescription=desc)
    new_series.set_array(array, headers[:,0], pixels_first=True)
    return new_series

def norm_projection(series, ord=None):
    array, headers = series.array('SliceLocation', pixels_first=True)
    array = np.linalg.norm(array, ord=ord, axis=-1)
    desc = series.instance().SeriesDescription + ' [Norm projection]'
    new_series = series.new_sibling(SeriesDescription=desc)
    new_series.set_array(array, headers[:,0], pixels_first=True)
    return new_series
    


def threshold(input, low_threshold=0, high_threshold=1, method='absolute'):
    """
    Segment by thresholding

    Parameters
    ----------
    input: dbdicom series

    Returns
    -------
    filtered : dbdicom series
    """
    suffix = ' [Threshold segmentation]'
    desc = input.instance().SeriesDescription
    filtered = input.copy(SeriesDescription = desc+suffix)
    #images = filtered.instances()
    images = filtered.images()
    for i, image in enumerate(images):
        input.status.progress(i+1, len(images), 'Filtering ' + desc)
        image.read()
        array = image.array()
        if method == 'quantiles':
            range = np.quantile(array, [low_threshold, high_threshold])
        elif method == 'range':
            min, max = np.amin(array), np.amax(array)
            range = [min+low_threshold*(max-min), min+high_threshold*(max-min)]
        else:
            range = [low_threshold, high_threshold]
        array = np.logical_and(array > range[0],  array < range[1])
        image.set_array(array)
        array = array.astype(np.ubyte)
        _reset_window(image, array)
        image.clear()
    input.status.hide()
    return filtered



# Helper functions

def _reset_window(image, array):
    min = np.amin(array)
    max = np.amax(array)
    image.WindowCenter= (max+min)/2
    image.WindowWidth = 0.9*(max-min)
