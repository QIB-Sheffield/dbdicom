"""
Functions for a new (sub)package dbimage inside dbdicom

A set of dbdicom wrappers for numpy
"""

import numpy as np


def image_calculator(series1, series2, operation='Subtract'):
    pass


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
