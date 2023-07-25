"""
``dbdicom`` extensions calling numpy functions. These do not require additional packages to be installed.
"""

import numpy as np
import dbdicom as dbd


def mean_intensity_projection(series:dbd.Series, dims=('SliceLocation','InstanceNumber'), axis=-1) -> dbd.Series:
    """Create a mean intensity projection along a specified dimension.

    Args:
        series (dbdicom.Series): Original series.
        dims (tuple, optional): Dimensions of the array. Defaults to ('SliceLocation','InstanceNumber').
        axis (int, optional): axis along which the maximum is to be taken. Defaults to -1.
        
    Returns:
        dbicom.Series: mean intensity projection. 

    Example:

        Get the MIP function from the numpy extension to dbdicom:

        >>> from db.wrappers.numpy import mean_intensity_projection

        Create a zero-filled array, describing 8 MRI images each measured at 3 flip angles and 2 repetition times:

        >>> coords = {
        ...    'SliceLocation': np.arange(8),
        ...    'FlipAngle': [2, 15, 30],
        ...    'RepetitionTime': [2.5, 5.0],
        ... }
        >>> series = db.zeros((128,128,8,3,2), coords)

        Create a mean intensity projection on the slice locations and check the dimensions:

        >>> mip = mean_intensity_projection(series)
        >>> array = mip.ndarray(dims=('SliceLocation', 'ImageNumber'))
        >>> print(array.shape)
        (128, 128, 8, 1)

        Create a mean intensity projection along the Slice Location axis:

        >>> mip = mean_intensity_projection(series, dims=tuple(coords), axis=0)
        >>> array = mip.ndarray(dims=tuple(coords))
        >>> print(array.shape)
        (128, 128, 1, 3, 2)

        Create a mean intensity projection along the Flip Angle axis:

        >>> mip = mean_intensity_projection(series, dims=tuple(coords), axis=1)
        >>> array = mip.ndarray(dims=tuple(coords))
        >>> print(array.shape)
        (128, 128, 8, 1, 2)

        Create a mean intensity projection along the Repetition Time axis:

        >>> mip = mean_intensity_projection(series, dims=tuple(coords), axis=2)
        >>> array = mip.ndarray(dims=tuple(coords))
        >>> print(array.shape)
        (128, 128, 8, 3, 1)
    """
    array = series.ndarray(dims=dims)
    inds = {}
    for id, d in enumerate(dims):
        if d == dims[axis]:
            inds[d] = np.arange(1)
        else:
            inds[d] = np.arange(array.shape[2+id])
    if axis >= 0:
        axis += 2
    array = np.mean(array, axis=axis)
    array = np.expand_dims(array, axis=axis)
    result = series.slice(inds=inds, SeriesDescription='Mean Intensity Projection')
    result.set_ndarray(array, inds=inds)
    return result


def maximum_intensity_projection(series:dbd.Series, dims=('SliceLocation','InstanceNumber'), axis=-1) -> dbd.Series:
    """Create a maximum intensity projection along a specified dimension.

    Args:
        series (dbdicom.Series): Original series.
        dims (tuple, optional): Dimensions of the array. Defaults to ('SliceLocation','InstanceNumber').
        axis (int, optional): axis along which the maximum is to be taken. Defaults to -1.
        
    Returns:
        dbicom.Series: maximum intensity projection. 

    Example:

        Get the MIP function from the numpy extension to dbdicom:

        >>> from db.wrappers.numpy import maximum_intensity_projection

        Create a zero-filled array, describing 8 MRI images each measured at 3 flip angles and 2 repetition times:

        >>> coords = {
        ...    'SliceLocation': np.arange(8),
        ...    'FlipAngle': [2, 15, 30],
        ...    'RepetitionTime': [2.5, 5.0],
        ... }
        >>> series = db.zeros((128,128,8,3,2), coords)

        Create a maximum intensity projection on the slice locations and check the dimensions:

        >>> mip = maximum_intensity_projection(series)
        >>> array = mip.ndarray(dims=('SliceLocation', 'ImageNumber'))
        >>> print(array.shape)
        (128, 128, 8, 1)

        Create a maximum intensity projection along the Slice Location axis:

        >>> mip = maximum_intensity_projection(series, dims=tuple(coords), axis=0)
        >>> array = mip.ndarray(dims=tuple(coords))
        >>> print(array.shape)
        (128, 128, 1, 3, 2)

        Create a maximum intensity projection along the Flip Angle axis:

        >>> mip = maximum_intensity_projection(series, dims=tuple(coords), axis=1)
        >>> array = mip.ndarray(dims=tuple(coords))
        >>> print(array.shape)
        (128, 128, 8, 1, 2)

        Create a maximum intensity projection along the Repetition Time axis:

        >>> mip = maximum_intensity_projection(series, dims=tuple(coords), axis=2)
        >>> array = mip.ndarray(dims=tuple(coords))
        >>> print(array.shape)
        (128, 128, 8, 3, 1)
    """
    array = series.ndarray(dims=dims)
    inds = {}
    for id, d in enumerate(dims):
        if d == dims[axis]:
            inds[d] = np.arange(1)
        else:
            inds[d] = np.arange(array.shape[2+id])
    if axis >= 0:
        axis += 2
    array = np.amax(array, axis=axis)
    array = np.expand_dims(array, axis=axis)
    result = series.slice(inds=inds, SeriesDescription='Maximum Intensity Projection')
    result.set_ndarray(array, inds=inds)
    return result


def norm_projection(series:dbd.Series, dims=('SliceLocation','InstanceNumber'), axis=-1, ord=None) -> dbd.Series:
    """Projection along a specified dimension using the vector norm.

    This functions uses numpy.linalg.norm to calculate the projection, see: https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html

    Args:
        series (dbdicom.Series): Original series.
        dims (tuple, optional): Dimensions of the array. Defaults to ('SliceLocation','InstanceNumber').
        axis (int, optional): axis along which the maximum is to be taken. Defaults to -1.
        ord (int, optional): order of the norm - see documentation of numpy.linalg.norm for details
        
    Returns:
        dbicom.Series: maximum intensity projection. 

    Example:

        Get the function from the numpy extension to dbdicom:

        >>> from db.wrappers.numpy import norm_projection

        Create a zero-filled array, describing 8 MRI images each measured at 3 flip angles and 2 repetition times:

        >>> coords = {
        ...    'SliceLocation': np.arange(8),
        ...    'FlipAngle': [2, 15, 30],
        ...    'RepetitionTime': [2.5, 5.0],
        ... }
        >>> series = db.zeros((128,128,8,3,2), coords)

        Create a norm projection on the slice locations and check the dimensions:

        >>> mip = norm_projection(series)
        >>> array = mip.ndarray(dims=('SliceLocation', 'ImageNumber'))
        >>> print(array.shape)
        (128, 128, 8, 1)

        Create a norm projection along the Slice Location axis:

        >>> mip = norm_projection(series, dims=tuple(coords), axis=0)
        >>> array = mip.ndarray(dims=tuple(coords))
        >>> print(array.shape)
        (128, 128, 1, 3, 2)

        Create a norm projection along the Flip Angle axis:

        >>> mip = norm_projection(series, dims=tuple(coords), axis=1)
        >>> array = mip.ndarray(dims=tuple(coords))
        >>> print(array.shape)
        (128, 128, 8, 1, 2)

        Create a norm projection along the Repetition Time axis:

        >>> mip = norm_projection(series, dims=tuple(coords), axis=2)
        >>> array = mip.ndarray(dims=tuple(coords))
        >>> print(array.shape)
        (128, 128, 8, 3, 1)
    """
    array = series.ndarray(dims=dims)
    inds = {}
    for id, d in enumerate(dims):
        if d == dims[axis]:
            inds[d] = np.arange(1)
        else:
            inds[d] = np.arange(array.shape[2+id])
    if axis >= 0:
        axis += 2
    array = np.linalg.norm(array, ord=ord, axis=axis)
    array = np.expand_dims(array, axis=axis)
    result = series.slice(inds=inds, SeriesDescription='Norm Projection')
    result.set_ndarray(array, inds=inds)
    return result
    


def threshold(input:dbd.Series, low_threshold=0, high_threshold=1, method='absolute')-> dbd.Series:
    """Create a mask series by thresholding.

    Args:
        input (dbd.Series): original data to be masked
        low_threshold (int, optional): Lower threshold for masking. Defaults to 0.
        high_threshold (int, optional): Upper threshold for masking. Defaults to 1.
        method (str, optional): Type of thresholding, either 'absolute' (thresholds are absolute signal values), 'quantiles' (thresholds are quantiles), or 'range' (thresholds are between 0 and 1). Defaults to 'absolute'.

    Returns:
        dbd.Series: mask series with values = 1 inside and 0 outside.
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
