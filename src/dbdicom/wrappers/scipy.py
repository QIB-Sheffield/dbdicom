import numpy as np
import scipy
from scipy.ndimage import affine_transform
import nibabel as nib # unnecessary
import dbdicom
from dbdicom.record import merge



def _equal_geometry(affine1, affine2):
    # Check if both are the same, 
    # ignoring the order in the list
    if not isinstance(affine2, list):
        affine2 = [affine2]
    if not isinstance(affine1, list):
        affine1 = [affine1]
    if len(affine1) != len(affine2):
        return False
    unmatched = list(range(len(affine2)))
    for a1 in affine1:
        imatch = None
        for i in unmatched:
            if np.array_equal(a1[0], affine2[i][0]):
                imatch = i
                break
        if imatch is not None:
            unmatched.remove(imatch)
    return unmatched == []

# This suggestion from chatGPT should to the same thing - check
def _lists_have_equal_items(list1, list2):
    # Convert the lists to sets
    set1 = set([tuple(x) for x in list1])
    set2 = set([tuple(x) for x in list2])

    # Check if the sets are equal
    return set1 == set2


def map_to(source, target, **kwargs):
    """Map non-zero pixels onto another series"""

    # Get transformation matrix
    source.status.message('Loading transformation matrices..')
    affine_source = source.affine_matrix()
    affine_target = target.affine_matrix() 
    if _equal_geometry(affine_source, affine_target): 
        source.status.hide()
        return source

    if isinstance(affine_target, list):
        mapped_series = []
        for affine_slice_group in affine_target:
            #v = image_utils.dismantle_affine_matrix(affine_slice_group)
            #slice_group_target = target.subseries(ImageOrientationPatient = v['ImageOrientationPatient'])
            slice_group_target = target.new_sibling()
            slice_group_target.adopt(affine_slice_group[1])
            mapped = _map_series_to_slice_group(source, slice_group_target, affine_source, affine_slice_group[0], **kwargs)
            mapped_series.append(mapped)
            slice_group_target.remove()
        desc = source.instance().SeriesDescription 
        desc += ' mapped to ' + target.instance().SeriesDescription
        mapped_series = merge(mapped_series, inplace=True)
        mapped_series.SeriesDescription = desc
    else:
        mapped_series = _map_series_to_slice_group(source, target, affine_source, affine_target[0], **kwargs)
    return mapped_series


def _map_series_to_slice_group(source, target, affine_source, affine_target, **kwargs):

    if isinstance(affine_source, list):
        mapped_series = []
        for affine_slice_group in affine_source:
            slice_group_source = source.new_sibling()
            slice_group_source.adopt(affine_slice_group[1])
            mapped = _map_slice_group_to_slice_group(slice_group_source, target, affine_slice_group[0], affine_target, **kwargs)
            mapped_series.append(mapped)
            slice_group_source.remove()
        return merge(mapped_series, inplace=True)
    else:
        return _map_slice_group_to_slice_group(source, target, affine_source[0], affine_target, **kwargs)


def _map_slice_group_to_slice_group(source, target, affine_source, affine_target, mask=False):

    source_to_target = np.linalg.inv(affine_source).dot(affine_target)
    matrix, offset = nib.affines.to_matvec(source_to_target) 
    
    # Get arrays
    array_source, headers_source = source.array(['SliceLocation','AcquisitionTime'], pixels_first=True)
    array_target, headers_target = target.array(['SliceLocation','AcquisitionTime'], pixels_first=True)

    #Perform transformation
    source.status.message('Performing transformation..')
    output_shape = array_target.shape[:3]
    nt, nk = array_source.shape[3], array_source.shape[4]
    array_mapped = np.empty(output_shape + (nt, nk))
    cnt=0
    for t in range(nt):
        for k in range(nk):
            cnt+=1
            source.status.progress(cnt, nt*nk, 'Performing transformation..')
            array_mapped[:,:,:,t,k] = affine_transform(
                array_source[:,:,:,t,k],
                matrix = matrix,
                offset = offset,
                output_shape = output_shape)

    # If source is a mask array, set values to [0,1]
    if mask:
        array_mapped[array_mapped > 0.5] = 1
        array_mapped[array_mapped <= 0.5] = 0
    
    # If data needs to be saved, create new series
    source.status.message('Saving results..')
    desc = source.instance().SeriesDescription 
    desc += ' mapped to ' + target.instance().SeriesDescription
    mapped_series = source.new_sibling(SeriesDescription = desc)
    ns, nt, nk = headers_target.shape[0], headers_source.shape[1], headers_source.shape[2]
    cnt=0
    for t in range(nt):
        # Retain source acquisition times
        # Assign acquisition time of slice=0 to all slices
        acq_time = headers_source[0,t,0].AcquisitionTime
        for k in range(nk):
            for s in range(ns):
                cnt+=1
                source.status.progress(cnt, ns*nt*nk, 'Saving results..')
                image = headers_target[s,0,0].copy_to(mapped_series)
                image.AcquisitionTime = acq_time
                image.set_pixel_array(array_mapped[:,:,s,t,k])
    source.status.message('Finished mapping..')
    return mapped_series


def map_mask_to(source, target, **kwargs):
    """Map non-zero pixels onto another series"""

    # Get transformation matrix
    source.status.message('Loading transformation matrices..')
    affine_source = source.affine_matrix()
    affine_target = target.affine_matrix() 

    if isinstance(affine_target, list):
        mapped_arrays = []
        mapped_headers = []
        for affine_slice_group_target in affine_target:
            slice_group_target = target.new_sibling()
            slice_group_target.adopt(affine_slice_group_target[1])
            mapped, headers = _map_mask_series_to_slice_group(source, slice_group_target, affine_source, affine_slice_group_target[0], **kwargs)
            mapped_arrays.append(mapped)
            mapped_headers.append(headers)
            slice_group_target.remove()
    else:
        mapped_arrays, mapped_headers = _map_mask_series_to_slice_group(source, target, affine_source, affine_target[0], **kwargs)
    source.status.hide()
    return mapped_arrays, mapped_headers


def _map_mask_series_to_slice_group(source, target, affine_source, affine_target, **kwargs):

    if isinstance(affine_source, list):
        mapped_arrays = []
        for affine_slice_group in affine_source:
            slice_group_source = source.new_sibling()
            slice_group_source.adopt(affine_slice_group[1])
            mapped, headers = _map_mask_slice_group_to_slice_group(slice_group_source, target, affine_slice_group[0], affine_target, **kwargs)
            mapped_arrays.append(mapped)
            slice_group_source.remove()
        array = np.logical_or(mapped_arrays[:2])
        for a in mapped_arrays[2:]:
            array = np.logical_or(array, a)
        return array, headers
    else:
        return _map_mask_slice_group_to_slice_group(source, target, affine_source[0], affine_target, **kwargs)


def _map_mask_slice_group_to_slice_group(source, target, affine_source, affine_target):

    source_to_target = np.linalg.inv(affine_source).dot(affine_target)
    matrix, offset = nib.affines.to_matvec(source_to_target) 
    
    # Get arrays
    array_source, headers_source = source.array(['SliceLocation','AcquisitionTime'], pixels_first=True)
    array_target, headers_target = target.array(['SliceLocation','AcquisitionTime'], pixels_first=True)

    if np.array_equal(affine_source, affine_target):
        array_source[array_source > 0.5] = 1
        array_source[array_source <= 0.5] = 0
        return array_source[:,:,:,0,0], headers_source[:,0,0]
    
    #Perform transformation
    source.status.message('Performing transformation..')
    output_shape = array_target.shape[:3]
    nt, nk = array_source.shape[3], array_source.shape[4]
    array_mapped = np.empty(output_shape + (nt, nk))
    cnt=0
    for t in range(nt):
        for k in range(nk):
            cnt+=1
            source.status.progress(cnt, nt*nk, 'Performing transformation..')
            array_mapped[:,:,:,t,k] = affine_transform(
                array_source[:,:,:,t,k],
                matrix = matrix,
                offset = offset,
                output_shape = output_shape,
                order = 3)

    # If source is a mask array, set values to [0,1]
    array_mapped[array_mapped > 0.5] = 1
    array_mapped[array_mapped <= 0.5] = 0

    # If the array is all that is needed we are done
    source.status.message('Finished mapping..')
    return array_mapped[:,:,:,0,0], headers_target[:,0,0]


# SEGMENTATION


def label(input, **kwargs):
    """
    Labels structures in an image
    
    Wrapper for scipy.ndimage.label function. 

    Parameters
    ----------
    input: dbdicom series

    Returns
    -------
    filtered : dbdicom series
    """
    suffix = ' [labels]'
    desc = input.instance().SeriesDescription
    filtered = input.copy(SeriesDescription = desc+suffix)
    #images = filtered.instances() # setting sort=False should be faster - TEST!!!!!!!
    images = filtered.images()
    for i, image in enumerate(images):
        input.status.progress(i+1, len(images), 'Labelling ' + desc)
        image.read()
        array = image.array() 
        array, _ = scipy.ndimage.label(array, **kwargs)
        image.set_array(array)
        _reset_window(image, array)
        image.clear()
    input.status.hide()
    return filtered


def binary_fill_holes(input, **kwargs):
    """
    Fill holes in an existing segmentation.
    
    Wrapper for scipy.ndimage.binary_fill_holes function. 

    Parameters
    ----------
    input: dbdicom series

    Returns
    -------
    filtered : dbdicom series
    """
    suffix = ' [Fill holes]'
    desc = input.instance().SeriesDescription
    filtered = input.copy(SeriesDescription = desc+suffix)
    #images = filtered.instances()
    images = filtered.images()
    for i, image in enumerate(images):
        input.status.progress(i+1, len(images), 'Filling holes ' + desc)
        image.read()
        array = image.array() 
        array = scipy.ndimage.binary_fill_holes(array, **kwargs)
        image.set_array(array)
        #array = array.astype(np.ubyte)
        #_reset_window(image, array)
        image.clear()
    input.status.hide()
    return filtered



# FILTERS




# https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.fourier_uniform.html#scipy.ndimage.fourier_ellipsoid
def fourier_ellipsoid(input, size, **kwargs):
    """
    wrapper for scipy.ndimage.fourier_ellipsoid

    Parameters
    ----------
    input: dbdicom series

    Returns
    -------
    filtered : dbdicom series
    """
    suffix = ' [Fourier Ellipsoid x ' + str(size) + ' ]'
    desc = input.instance().SeriesDescription
    filtered = input.copy(SeriesDescription = desc + suffix)
    #images = filtered.instances()
    images = filtered.images()
    for i, image in enumerate(images):
        input.status.progress(i+1, len(images), 'Filtering ' + desc)
        image.read()
        array = image.array()
        array = np.fft.fft2(array)
        array = scipy.ndimage.fourier_ellipsoid(array, size, **kwargs)
        array = np.fft.ifft2(array).real
        image.set_array(array)
        image.clear()
    input.status.hide()
    return filtered


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.fourier_uniform.html#scipy.ndimage.fourier_uniform
def fourier_uniform(input, size, **kwargs):
    """
    wrapper for scipy.ndimage.fourier_uniform

    Parameters
    ----------
    input: dbdicom series

    Returns
    -------
    filtered : dbdicom series
    """
    suffix = ' [Fourier Uniform x ' + str(size) + ' ]'
    desc = input.instance().SeriesDescription
    filtered = input.copy(SeriesDescription = desc + suffix)
    #images = filtered.instances()
    images = filtered.images()
    for i, image in enumerate(images):
        input.status.progress(i+1, len(images), 'Filtering ' + desc)
        image.read()
        array = image.array()
        array = np.fft.fft2(array)
        array = scipy.ndimage.fourier_uniform(array, size, **kwargs)
        array = np.fft.ifft2(array).real
        image.set_array(array)
        image.clear()
    input.status.hide()
    return filtered


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.fourier_shift.html#scipy.ndimage.fourier_shift
def fourier_gaussian(input, sigma, **kwargs):
    """
    wrapper for scipy.ndimage.fourier_gaussian.

    Parameters
    ----------
    input: dbdicom series

    Returns
    -------
    filtered : dbdicom series
    """
    suffix = ' [Fourier Gaussian x ' + str(sigma) + ' ]'
    desc = input.instance().SeriesDescription
    filtered = input.copy(SeriesDescription = desc + suffix)
    #images = filtered.instances()
    images = filtered.images()
    for i, image in enumerate(images):
        input.status.progress(i+1, len(images), 'Filtering ' + desc)
        image.read()
        array = image.array()
        array = np.fft.fft2(array)
        array = scipy.ndimage.fourier_gaussian(array, sigma, **kwargs)
        array = np.fft.ifft2(array).real
        image.set_array(array)
        image.clear()
    input.status.hide()
    return filtered


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_gradient_magnitude.html#scipy.ndimage.gaussian_gradient_magnitude
def gaussian_gradient_magnitude(input, sigma, **kwargs):
    """
    wrapper for scipy.ndimage.gaussian_gradient_magnitude.

    Parameters
    ----------
    input: dbdicom series

    Returns
    -------
    filtered : dbdicom series
    """
    suffix = ' [Gaussian Gradient Magnitude x ' + str(sigma) + ' ]'
    desc = input.instance().SeriesDescription
    filtered = input.copy(SeriesDescription = desc + suffix)
    #images = filtered.instances()
    images = filtered.images()
    for i, image in enumerate(images):
        input.status.progress(i+1, len(images), 'Filtering ' + desc)
        image.read()
        array = image.array()
        array = scipy.ndimage.gaussian_gradient_magnitude(array, sigma, **kwargs)
        image.set_array(array)
        _reset_window(image, array)
        image.clear()
    input.status.hide()
    return filtered


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_laplace.html#scipy.ndimage.gaussian_laplace
def gaussian_laplace(input, sigma, **kwargs):
    """
    wrapper for scipy.ndimage.gaussian_laplace.

    Parameters
    ----------
    input: dbdicom series

    Returns
    -------
    filtered : dbdicom series
    """
    suffix = ' [Gaussian Laplace x ' + str(sigma) + ' ]'
    desc = input.instance().SeriesDescription
    filtered = input.copy(SeriesDescription = desc + suffix)
    #images = filtered.instances()
    images = filtered.images()
    for i, image in enumerate(images):
        input.status.progress(i+1, len(images), 'Filtering ' + desc)
        image.read()
        array = image.array()
        array = scipy.ndimage.gaussian_laplace(array, sigma, **kwargs)
        image.set_array(array)
        _reset_window(image, array)
        image.clear()
    input.status.hide()
    return filtered


#https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.laplace.html#scipy.ndimage.laplace
def laplace(input, **kwargs):
    """
    wrapper for scipy.ndimage.sobel.

    Parameters
    ----------
    input: dbdicom series

    Returns
    -------
    filtered : dbdicom series
    """
    suffix = ' [Laplace Filter]'
    desc = input.instance().SeriesDescription
    filtered = input.copy(SeriesDescription = desc + suffix)
    #images = filtered.instances()
    images = filtered.images()
    for i, image in enumerate(images):
        input.status.progress(i+1, len(images), 'Filtering ' + desc)
        image.read()
        array = image.array()
        array = scipy.ndimage.laplace(array, **kwargs)
        image.set_array(array)
        _reset_window(image, array)
        image.clear()
    input.status.hide()
    return filtered


#https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.sobel.html#scipy.ndimage.sobel
def sobel_filter(input, axis=-1, **kwargs):
    """
    wrapper for scipy.ndimage.sobel.

    Parameters
    ----------
    input: dbdicom series

    Returns
    -------
    filtered : dbdicom series
    """
    suffix = ' [Sobel Filter along axis ' + str(axis) + ' ]'
    desc = input.instance().SeriesDescription
    filtered = input.copy(SeriesDescription = desc + suffix)
    #images = filtered.instances()
    images = filtered.images()
    for i, image in enumerate(images):
        input.status.progress(i+1, len(images), 'Filtering ' + desc)
        image.read()
        array = image.array()
        array = scipy.ndimage.sobel(array, axis=axis, **kwargs)
        image.set_array(array)
        _reset_window(image, array)
        image.clear()
    input.status.hide()
    return filtered


#https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.prewitt.html#scipy.ndimage.prewitt
def prewitt_filter(input, axis=-1, **kwargs):
    """
    wrapper for scipy.ndimage.prewitt.

    Parameters
    ----------
    input: dbdicom series

    Returns
    -------
    filtered : dbdicom series
    """
    suffix = ' [Prewitt Filter along axis ' + str(axis) + ' ]'
    desc = input.instance().SeriesDescription
    filtered = input.copy(SeriesDescription = desc + suffix)
    #images = filtered.instances()
    images = filtered.images()
    for i, image in enumerate(images):
        input.status.progress(i+1, len(images), 'Filtering ' + desc)
        image.read()
        array = image.array()
        array = scipy.ndimage.prewitt(array, axis=axis, **kwargs)
        image.set_array(array)
        _reset_window(image, array)
        image.clear()
    input.status.hide()
    return filtered


#https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html#scipy.ndimage.median_filter
def median_filter(input, size=3, **kwargs):
    """
    wrapper for scipy.ndimage.median_filter.

    Parameters
    ----------
    input: dbdicom series

    Returns
    -------
    filtered : dbdicom series
    """
    suffix = ' [Median Filter with size ' + str(size) + ' ]'
    desc = input.instance().SeriesDescription
    filtered = input.copy(SeriesDescription = desc + suffix)
    #images = filtered.instances()
    images = filtered.images()
    for i, image in enumerate(images):
        input.status.progress(i+1, len(images), 'Filtering ' + desc)
        image.read()
        array = image.array()
        array = scipy.ndimage.median_filter(array, size=size, **kwargs)
        image.set_array(array)
        image.clear()
    input.status.hide()
    return filtered


#https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.percentile_filter.html#scipy.ndimage.percentile_filter
def percentile_filter(input, percentile, **kwargs):
    """
    wrapper for scipy.ndimage.percentile_filter.

    Parameters
    ----------
    input: dbdicom series

    Returns
    -------
    filtered : dbdicom series
    """
    suffix = ' [Percentile Filter x ' + str(percentile) + ' ]'
    desc = input.instance().SeriesDescription
    filtered = input.copy(SeriesDescription = desc + suffix)
    #images = filtered.instances()
    images = filtered.images()
    for i, image in enumerate(images):
        input.status.progress(i+1, len(images), 'Filtering ' + desc)
        image.read()
        array = image.array()
        array = scipy.ndimage.percentile_filter(array, percentile, **kwargs)
        image.set_array(array)
        image.clear()
    input.status.hide()
    return filtered


#https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rank_filter.html#scipy.ndimage.rank_filter
def rank_filter(input, rank, **kwargs):
    """
    wrapper for scipy.ndimage.rank_filter.

    Parameters
    ----------
    input: dbdicom series

    Returns
    -------
    filtered : dbdicom series
    """
    suffix = ' [Rank Filter x ' + str(rank) + ' ]'
    desc = input.instance().SeriesDescription
    filtered = input.copy(SeriesDescription = desc + suffix)
    #images = filtered.instances()
    images = filtered.images()
    for i, image in enumerate(images):
        input.status.progress(i+1, len(images), 'Filtering ' + desc)
        image.read()
        array = image.array()
        array = scipy.ndimage.rank_filter(array, rank, **kwargs)
        image.set_array(array)
        image.clear()
    input.status.hide()
    return filtered


#https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.minimum_filter.html#scipy.ndimage.maximum_filter
def maximum_filter(input, size=3, **kwargs):
    """
    wrapper for scipy.ndimage.maximum_filter.

    Parameters
    ----------
    input: dbdicom series

    Returns
    -------
    filtered : dbdicom series
    """
    suffix = ' [Maximum Filter x ' + str(size) + ' ]'
    desc = input.instance().SeriesDescription
    filtered = input.copy(SeriesDescription = desc + suffix)
    #images = filtered.instances()
    images = filtered.images()
    for i, image in enumerate(images):
        input.status.progress(i+1, len(images), 'Filtering ' + desc)
        image.read()
        array = image.array()
        array = scipy.ndimage.maximum_filter(array, size=size, **kwargs)
        image.set_array(array)
        image.clear()
    input.status.hide()
    return filtered


#https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.minimum_filter.html#scipy.ndimage.minimum_filter
def minimum_filter(input, size=3, **kwargs):
    """
    wrapper for scipy.ndimage.minimum_filter.

    Parameters
    ----------
    input: dbdicom series

    Returns
    -------
    filtered : dbdicom series
    """
    suffix = ' [Minimum Filter x ' + str(size) + ' ]'
    desc = input.instance().SeriesDescription
    filtered = input.copy(SeriesDescription = desc + suffix)
    #images = filtered.instances()
    images = filtered.images()
    for i, image in enumerate(images):
        input.status.progress(i+1, len(images), 'Filtering ' + desc)
        image.read()
        array = image.array()
        array = scipy.ndimage.minimum_filter(array, size=size, **kwargs)
        image.set_array(array)
        image.clear()
    input.status.hide()
    return filtered


#https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.uniform_filter.html#scipy.ndimage.uniform_filter
def uniform_filter(input, size=3, **kwargs):
    """
    wrapper for scipy.ndimage.uniform_filter.

    Parameters
    ----------
    input: dbdicom series

    Returns
    -------
    filtered : dbdicom series
    """
    suffix = ' [Uniform Filter x ' + str(size) + ' ]'
    desc = input.instance().SeriesDescription
    filtered = input.copy(SeriesDescription = desc + suffix)
    #images = filtered.instances()
    images = filtered.images()
    for i, image in enumerate(images):
        input.status.progress(i+1, len(images), 'Filtering ' + desc)
        image.read()
        array = image.array()
        array = scipy.ndimage.uniform_filter(array, size=size, **kwargs)
        image.set_array(array)
        image.clear()
    input.status.hide()
    return filtered
    

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html#scipy.ndimage.gaussian_filter
def gaussian_filter(input, sigma, **kwargs):
    """
    wrapper for scipy.ndimage.gaussian_filter.

    Parameters
    ----------
    input: dbdicom series

    Returns
    -------
    filtered : dbdicom series
    """
    suffix = ' [Gaussian Filter x ' + str(sigma) + ' ]'
    desc = input.instance().SeriesDescription
    filtered = input.copy(SeriesDescription = desc + suffix)
    #images = filtered.instances()
    images = filtered.images()
    for i, image in enumerate(images):
        input.status.progress(i+1, len(images), 'Filtering ' + desc)
        image.read()
        array = image.array()
        array = scipy.ndimage.gaussian_filter(array, sigma, **kwargs)
        image.set_array(array)
        if 'order' in kwargs:
            if kwargs['order'] > 0:
                _reset_window(image, array)
        image.clear()
    input.status.hide()
    return filtered


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.fourier_shift.html#scipy.ndimage.fourier_shift
def fourier_shift(input, shift, **kwargs):
    """
    wrapper for scipy.ndimage.fourier_shift.

    Parameters
    ----------
    input: dbdicom series

    Returns
    -------
    filtered : dbdicom series
    """
    suffix = ' [Fourier Shift]'
    desc = input.instance().SeriesDescription
    filtered = input.copy(SeriesDescription = desc + suffix)
    #images = filtered.instances()
    images = filtered.images()
    for i, image in enumerate(images):
        input.status.progress(i+1, len(images), 'Filtering ' + desc)
        image.read()
        array = image.array()
        array = np.fft.fft2(array)
        array = scipy.ndimage.fourier_shift(array, shift, **kwargs)
        array = np.fft.ifft2(array).real
        image.set_array(array)
        image.clear()
    input.status.hide()
    return filtered




# RESCALE AND RESLICE


def image_calculator(series1, series2, operation='series 1 - series 2'):

    desc1 = series1.instance().SeriesDescription
    result = series1.copy(SeriesDescription = desc1 + ' [' + operation + ']')
    images1 = result.images(sortby=['SliceLocation', 'AcquisitionTime'])
    images2 = series2.images(sortby=['SliceLocation', 'AcquisitionTime'])
    for i, img1 in enumerate(images1):
        series1.status.progress(i+1, len(images1), 'Calculating..')
        if i > len(images2)-1:
            break
        img1.read()
        img2 = images2[i]
        array1 = img1.array()
        array2 = img2.array()
        if array2.shape != array1.shape:
            zoom_factor = (
                array1.shape[0]/array2.shape[0], 
                array1.shape[1]/array2.shape[1])
            zoomed_series2, array2 = zoom_image_calculator(series2, zoom_factor)
        if operation == 'series 1 + series 2':
            array = array1 + array2
        elif operation == 'series 1 - series 2':
            array = array1 - array2
        elif operation == 'series 1 / series 2':
            array = array1 / array2
        elif operation == 'series 1 * series 2':
            array = array1 * array2
        elif operation == '(series 1 - series 2)/series 2':
            array = (array1 - array2)/array2
        elif operation == 'average(series 1, series 2)':
            array = (array1 + array2)/2
        array[~np.isfinite(array)] = 0
        img1.set_array(array)
        _reset_window(img1, array.astype(np.ubyte))
        img1.clear()
    series1.status.hide()
    return result




# https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html#scipy.ndimage.zoom
def zoom(input, zoom, **kwargs):
    """
    wrapper for scipy.ndimage.zoom.

    Parameters
    ----------
    input: dbdicom series

    Returns
    -------
    zoomed : dbdicom series
    """
    suffix = ' [Resize x ' + str(zoom) + ' ]'
    desc = input.instance().SeriesDescription
    zoomed = input.copy(SeriesDescription = desc + suffix)
    #images = zoomed.instances()
    images = zoomed.images()
    for i, image in enumerate(images):
        input.status.progress(i+1, len(images), 'Resizing ' + desc)
        image.read()
        array = image.array()
        array = scipy.ndimage.zoom(array, zoom, **kwargs)
        image.set_array(array)
        pixel_spacing = image.PixelSpacing
        if type(zoom) is tuple:
           image.PixelSpacing[0] = pixel_spacing[0]/zoom[0] 
           image.PixelSpacing[1] = pixel_spacing[1]/zoom[1]
        else:
           image.PixelSpacing = [p/zoom for p in pixel_spacing]
        image.clear()
    input.status.hide()
    return zoomed

def zoom_image_calculator(input, zoom, **kwargs):
    """
    wrapper for scipy.ndimage.zoom. for image calculator

    Parameters
    ----------
    input: dbdicom series

    Returns
    -------
    zoomed : dbdicom series
    array: numpy ndarray
    """
    suffix = ' [Resize x ' + str(zoom) + ' ]'
    desc = input.instance().SeriesDescription
    zoomed = input.copy(SeriesDescription = desc + suffix)
    #images = zoomed.instances()
    images = zoomed.images()
    for i, image in enumerate(images):
        input.status.progress(i+1, len(images), 'Resizing ' + desc)
        image.read()
        array = image.array()
        array = scipy.ndimage.zoom(array, zoom, **kwargs)
        image.set_array(array)
        pixel_spacing = image.PixelSpacing
        if type(zoom) is tuple:
           image.PixelSpacing[0] = pixel_spacing[0]/zoom[0] 
           image.PixelSpacing[1] = pixel_spacing[1]/zoom[1]
        else:
           image.PixelSpacing = [p/zoom for p in pixel_spacing]
        print("p/zoom")
        print(image.PixelSpacing)
        image.clear()
    input.status.hide()
    return zoomed, array

def resample(series, voxel_size=[1.0, 1.0, 1.0]):
    series.status.message('Reading transformations..')
    affine_source = series.affine_matrix()
    if affine_source is None:
        return
    if isinstance(affine_source, list):
        mapped_series = []
        for affine_slice_group in affine_source:
            mapped = _resample_slice_group(series, affine_slice_group[0], affine_slice_group[0], voxel_size=voxel_size)
            mapped_series.append(mapped)
        desc = series.instance().SeriesDescription + '[resampled]'
        mapped_series = dbdicom.merge(mapped_series, inplace=True)
        mapped_series.SeriesDescription = desc
    else:
        mapped_series = _resample_slice_group(series, affine_source[0], affine_source[1], voxel_size=voxel_size)
    return mapped_series


def reslice(series, orientation='axial'):

    # Define geometry of axial series (isotropic)
    series.status.message('Reading transformations..')
    affine_source = series.affine_matrix()
    if isinstance(affine_source, list):
        mapped_series = []
        for affine_slice_group in affine_source:
            mapped = _reslice_slice_group(series, affine_slice_group[0], affine_slice_group[0], orientation=orientation)
            mapped_series.append(mapped)
            #slice_group.remove()
        desc = series.instance().SeriesDescription + '['+orientation+']'
        mapped_series = dbdicom.merge(mapped_series, inplace=True)
        mapped_series.SeriesDescription = desc
    else:
        mapped_series = _reslice_slice_group(series, affine_source[0], affine_source[1], orientation=orientation)
    return mapped_series


def _resample_slice_group(series, affine_source, slice_group, voxel_size=[1.0, 1.0, 1.0]):

    # Create new resliced series
    desc = series.instance().SeriesDescription + '[resampled]'
    resliced_series = series.new_sibling(SeriesDescription = desc)

    # Work out the affine matrix of the new series
    p = dbdicom.utils.image.dismantle_affine_matrix(affine_source)
    affine_target = affine_source.copy()
    affine_target[:3, 0] = voxel_size[0] * np.array(p['ImageOrientationPatient'][:3])
    affine_target[:3, 1] = voxel_size[1] * np.array(p['ImageOrientationPatient'][3:]) 
    affine_target[:3, 2] = voxel_size[2] * np.array(p['slice_cosine'])

    # If the series already is in the right orientation, return a copy
    if np.array_equal(affine_source, affine_target):
        series.status.message('Series is already in the right orientation..')
        resliced_series.adopt(slice_group)
        return resliced_series

    # Perform transformation on the arrays to determine the output shape
    dim = [
        array.shape[0] * p['PixelSpacing'][1],
        array.shape[1] * p['PixelSpacing'][0],
        array.shape[2] * p['SliceThickness'],
    ]
    output_shape = [1 + round(dim[i]/voxel_size[i]) for i in range(3)]

    # Determine the transformation matrix and offset
    source_to_target = np.linalg.inv(affine_source).dot(affine_target)
    matrix, offset = nib.affines.to_matvec(source_to_target)
    
    # Get arrays
    array, headers = series.array(['SliceLocation','AcquisitionTime'], pixels_first=True)
    if array is None:
        return resliced_series

    # Perform the affine transformation
    cnt=0
    ns, nt, nk = output_shape[2], array.shape[-2], array.shape[-1]
    pos, loc = dbdicom.utils.image.image_position_patient(affine_target, ns)
    for t in range(nt):
        for k in range(nk):
            cnt+=1
            series.status.progress(cnt, nt*nk, 'Performing transformation..')
            resliced = scipy.ndimage.affine_transform(
                array[:,:,:,t,k],
                matrix = matrix,
                offset = offset,
                output_shape = output_shape,
            )
            resliced_series.set_array(resliced, 
                source = headers[0,t,k],
                pixels_first = True,
                affine_matrix = affine_target,
                ImagePositionPatient = pos,
                SliceLocation = loc,
            )
    series.status.message('Finished mapping..')
    return resliced_series


def _reslice_slice_group(series, affine_source, slice_group, orientation='axial'):

    # Create new resliced series
    desc = series.instance().SeriesDescription + '['+orientation+']'
    resliced_series = series.new_sibling(SeriesDescription = desc)

    # Work out the affine matrix of the new series
    p = dbdicom.utils.image.dismantle_affine_matrix(affine_source)
    image_positions = [s.ImagePositionPatient for s in slice_group]
    rows = slice_group[0].Rows
    columns = slice_group[0].Columns
    box = dbdicom.utils.image.bounding_box(
        p['ImageOrientationPatient'],  
        image_positions,   
        p['PixelSpacing'], 
        rows,
        columns)
    spacing = np.mean([p['PixelSpacing'][0], p['PixelSpacing'][1], p['SliceThickness']])
    affine_target = dbdicom.utils.image.standard_affine_matrix(
        box, 
        [spacing, spacing],
        spacing,
        orientation=orientation)

    # If the series already is in the right orientation, return a copy
    if np.array_equal(affine_source, affine_target):
        series.status.message('Series is already in the right orientation..')
        resliced_series.adopt(slice_group)
        return resliced_series

    #Perform transformation on the arrays to determine the output shape
    if orientation == 'axial':
        dim = [
            np.linalg.norm(np.array(box['RAF'])-np.array(box['LAF'])),
            np.linalg.norm(np.array(box['RAF'])-np.array(box['RPF'])),
            np.linalg.norm(np.array(box['RAF'])-np.array(box['RAH'])),
        ]
    elif orientation == 'coronal':
        dim = [
            np.linalg.norm(np.array(box['RAH'])-np.array(box['LAH'])),
            np.linalg.norm(np.array(box['RAH'])-np.array(box['RAF'])),
            np.linalg.norm(np.array(box['RAH'])-np.array(box['RPH'])),
        ]
    elif orientation == 'sagittal':
        dim = [
            np.linalg.norm(np.array(box['LAH'])-np.array(box['LPH'])),
            np.linalg.norm(np.array(box['LAH'])-np.array(box['LAF'])),
            np.linalg.norm(np.array(box['LAH'])-np.array(box['RAH'])),
        ]
    output_shape = [1 + round(d/spacing) for d in dim]

    # Determine the transformation matrix and offset
    source_to_target = np.linalg.inv(affine_source).dot(affine_target)
    matrix, offset = nib.affines.to_matvec(source_to_target)
    
    # Get arrays
    array, headers = series.array(['SliceLocation','AcquisitionTime'], pixels_first=True)

    # Perform the affine transformation and save results
    cnt=0
    ns, nt, nk = output_shape[2], array.shape[-2], array.shape[-1] 
    pos, loc = dbdicom.utils.image.image_position_patient(affine_target, ns)
    for t in range(nt):
        for k in range(nk):
            cnt+=1
            series.status.progress(cnt, nt*nk, 'Calculating..')
            resliced = scipy.ndimage.affine_transform(
                array[:,:,:,t,k],
                matrix = matrix,
                offset = offset,
                output_shape = output_shape,
            )
            # Saving results at each time to avoid memory problems.
            # Assign acquisition time of slice=0 to all slices
            resliced_series.set_array(resliced, 
                source = headers[0,t,k],
                pixels_first = True,
                affine_matrix = affine_target,
                ImagePositionPatient = pos,
                SliceLocation = loc,
            )
    series.status.message('Finished mapping..')
    return resliced_series



# Helper functions

def _reset_window(image, array):
    min = np.amin(array)
    max = np.amax(array)
    image.WindowCenter= (max+min)/2
    image.WindowWidth = 0.9*(max-min)
