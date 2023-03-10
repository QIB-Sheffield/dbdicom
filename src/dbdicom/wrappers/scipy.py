import numpy as np
import pandas as pd
import scipy
from scipy.ndimage import affine_transform
import nibabel as nib # unnecessary
import dbdicom



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
                # If a slice group with the same affine is found, 
                # check if the image dimensions are the same too.
                dim1 = a1[1][0].array().shape
                dim2 = affine2[i][1][0].array().shape
                if dim1 == dim2:
                    imatch = i
                    break
        if imatch is not None:
            unmatched.remove(imatch)
    return unmatched == []


# Better use set(tuple())
def _lists_have_equal_items(list1, list2):
    # Convert the lists to sets
    set1 = set([tuple(x) for x in list1])
    set2 = set([tuple(x) for x in list2])

    # Check if the sets are equal
    return set1 == set2

def mask_curve_3d(masks, images, **kwargs):
    if not isinstance(masks, list):
        masks = [masks]
    if not isinstance(images, list):
        images = [images]
    df_all = []
    for mask in masks:
        for img in images:
            df = _mask_curve_3d(mask, img, **kwargs)
            df_all.append(df) 
    return df_all


def _mask_curve_3d(mask, series, dim='InstanceNumber'):

    # Get 4D mask array overlaid on 4D series
    msk_arr, img_hdrs = mask_array(mask, on=series, dim=dim)
    
    # Define variables
    vars = ['PatientID', 'StudyDescription', 'SeriesDescription', 'Region of Interest']
    vars += [str(dim), 'Mean', 'Stdev', 'Max', 'Min', 'Median', '2.5 perc', '97.5 perc']

    # Read values
    img = series.instance()
    ids = [img.PatientID, img.StudyDescription, img.SeriesDescription, mask.instance().SeriesDescription]
    if isinstance(msk_arr, list):
        data = _mask_curve_3d_data(msk_arr, img_hdrs, ids, dim)
    else:
        data = _mask_curve_3d_data_slice_group(msk_arr, img_hdrs, ids, dim)

    # Return as dataframe
    return pd.DataFrame(data, columns=vars)


def _mask_curve_3d_data_slice_group(msk_arr, img_hdrs, ids, dim):
    data = []
    nt = msk_arr.shape[-1]
    for t in range(nt):
        img_hdrs[0,0].status.progress(t+1, nt, 'Extracting mask time curves..')
        arr = _mask_data(msk_arr[...,t], img_hdrs[...,t])
        vals = [
            img_hdrs[0,t][dim], # 3d-assuming all slice locations have the same time coordinate
            np.mean(arr),
            np.std(arr),
            np.amax(arr),
            np.amin(arr),
            np.percentile(arr, 50),
            np.percentile(arr, 2.5),
            np.percentile(arr, 97.5),
        ]
        data.append(ids + vals)
    return data


def _mask_curve_3d_data(msk_arr, img_hdrs, ids, dim):
    data = []
    nt = msk_arr[0].shape[-1]
    for t in range(nt):
        img_hdrs[0][0,0].status.progress(t+1, nt, 'Extracting mask time curves..')
        # Concatenate data at time t for each slice group
        arr = [_mask_data(arr_i[...,t], img_hdrs[i][...,t]) for i, arr_i in enumerate(msk_arr)]
        arr = [d for d in arr if d is not None]
        arr = np.concatenate(arr)
        # Get values
        vals = [
            img_hdrs[0][0,t][dim], # 3d-assuming all slice locations have the same time coordinate
            np.mean(arr),
            np.std(arr),
            np.amax(arr),
            np.amin(arr),
            np.percentile(arr, 50),
            np.percentile(arr, 2.5),
            np.percentile(arr, 97.5),
        ]
        data.append(ids + vals)
    return data


def mask_statistics(masks, images):
    if not isinstance(masks, list):
        masks = [masks]
    if not isinstance(images, list):
        images = [images]
    df_all_masks = None
    for mask in masks:
        df_mask = None
        for img in images:
            df_img = _mask_statistics(mask, img)
            if df_mask is None:
                df_mask = df_img
            else:
                df_mask = pd.concat([df_mask, df_img], ignore_index=True)
        if df_all_masks is None:
            df_all_masks = df_mask
        else:
            df_all_masks = pd.concat([df_all_masks, df_mask], ignore_index=True)
    return df_all_masks


def _mask_statistics(mask, image):

    # Get mask array
    msk_arr, img_hdrs = mask_array(mask, on=image)
    data = _mask_data_slice_groups(msk_arr, img_hdrs)
    props = _summary_stats(data)
    instance = image.instance()
    columns = ['PatientID', 'StudyDescription', 'SeriesDescription', 'Region of Interest', 'Parameter', 'Value', 'Unit']
    ids = [instance.PatientID, instance.StudyDescription, instance.SeriesDescription, mask.instance().SeriesDescription]
    data = []
    for par, val in props.items():
        row = ids + [par, val, '']
        data.append(row)
    return pd.DataFrame(data, columns=columns)


def _mask_data_slice_groups(msk_arr, img_hdrs):
    if isinstance(msk_arr, list): 
        # Loop over slice groups
        data = [_mask_data(arr, img_hdrs[m]) for m, arr in enumerate(msk_arr)]
        data = [d for d in data if d is not None]
        if data == []:
            data = None
        else:
            data = np.concatenate(data)
    else: 
        # single slice group
        data = _mask_data(msk_arr, img_hdrs)
    return data


def _mask_data(msk_arr, imgs):
    data = []
    for i, image in np.ndenumerate(imgs):
        if image is not None:
            if len(i) == 1:
                mask = msk_arr[:,:,i[0]]
            elif len(i) == 2:
                mask = msk_arr[:,:,i[0],i[1]]
            if np.count_nonzero(mask) > 0:
                array = image.array()
                array = array[mask > 0.5]  
                data.append(array.ravel())
    if data == []:
        return None
    else:
        return np.concatenate(data)
    

def _summary_stats(data):
    if data is None:
        return {}
    return {
        'Mean': np.mean(data),
        'Standard deviation': np.std(data),
        'Maximum': np.amax(data),
        'Minimum': np.amin(data),
        '2.5% percentile': np.percentile(data, 2.5),
        '5% percentile': np.percentile(data, 5),
        '10% percentile': np.percentile(data, 10),
        '25% percentile': np.percentile(data, 25),
        'Median': np.percentile(data, 50),
        '75% percentile': np.percentile(data, 75),
        '90% percentile': np.percentile(data, 90),
        '95% percentile': np.percentile(data, 95),
        '97.5% percentile': np.percentile(data, 97.5),
        'Range': np.amax(data) - np.amin(data),
        'Interquartile range':np.percentile(data, 75) - np.percentile(data, 25),
        '90 percent range': np.percentile(data, 95) - np.percentile(data, 5),
        'Coefficient of variation': np.std(data)/np.mean(data),
        'Heterogeneity': (np.percentile(data, 95) - np.percentile(data, 5))/np.percentile(data, 50),
        'Kurtosis': scipy.stats.kurtosis(data),
        'Skewness': scipy.stats.skew(data),
    }
        


def overlay(features):
    """ Ensure all the features are in the same geometry as the reference feature"""

    msg = 'Mapping all features on the same geometry'
    mapped_features = [features[0]]
    for f, feature in enumerate(features[1:]):
        feature.status.progress(f+1, len(features)-1, msg)
        mapped = map_to(feature, features[0])
        mapped_features.append(mapped)
    return mapped_features
    

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
            slice_group_target = target.new_sibling()
            slice_group_target.adopt(affine_slice_group[1])
            mapped = _map_series_to_slice_group(source, slice_group_target, affine_source, affine_slice_group[0], **kwargs)
            mapped_series.append(mapped)
            slice_group_target.remove()
        desc = source.instance().SeriesDescription 
        desc += ' mapped to ' + target.instance().SeriesDescription
        mapped_series = dbdicom.merge(mapped_series, inplace=True)
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
        return dbdicom.merge(mapped_series, inplace=True)
    else:
        return _map_slice_group_to_slice_group(source, target, affine_source[0], affine_target, **kwargs)


def _map_slice_group_to_slice_group(source, target, affine_source, affine_target, mask=False, label=False, cval=0):

    # Get names for status updates
    source_desc = source.instance().SeriesDescription
    target_desc = target.instance().SeriesDescription

    # Get transformation matrix
    source_to_target = np.linalg.inv(affine_source).dot(affine_target)
    source_to_target = np.around(source_to_target, 3) # remove round-off errors in the inversion
    matrix, offset = nib.affines.to_matvec(source_to_target) 
    
    # Get arrays
    array_source, headers_source = source.array(['SliceLocation','AcquisitionTime'], pixels_first=True)
    array_target, headers_target = target.array(['SliceLocation','AcquisitionTime'], pixels_first=True)

    #Perform transformation
    message = 'Mapping ' + source_desc + ' onto ' + target_desc
    source.status.message(message)
    output_shape = array_target.shape[:3]
    nt, nk = array_source.shape[3], array_source.shape[4]
    array_mapped = np.empty(output_shape + (nt, nk))
    if mask:
        order = 0 
    else:
        order = 3
    cnt=0
    for t in range(nt):
        for k in range(nk):
            cnt+=1
            source.status.progress(cnt, nt*nk, message)
            array_mapped[:,:,:,t,k] = affine_transform(
                array_source[:,:,:,t,k],
                matrix = matrix,
                offset = offset,
                output_shape = output_shape,
                cval = cval,
                order = order)

    # If source is a mask array, set values to [0,1]
    if mask:
        array_mapped[array_mapped > 0.5] = 1
        array_mapped[array_mapped <= 0.5] = 0
    elif label:
        array_mapped = np.around(array_mapped)
    
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


def mask_array(mask, on=None, dim='InstanceNumber'):
    """Map non-zero pixels onto another series"""

    if on is None:
        arr, hdr = dbdicom.array(mask, sortby=['SliceLocation', dim], mask=True, pixels_first=True)
        return arr[...,0], hdr[...,0]

    # Get transformation matrix
    mask.status.message('Loading transformation matrices..')
    affine_source = mask.affine_matrix()
    affine_target = on.affine_matrix() 

    if isinstance(affine_target, list):
        mapped_arrays = []
        mapped_headers = []
        for affine_slice_group_target in affine_target:
            mapped, headers = _map_mask_series_to_slice_group(
                mask, 
                affine_slice_group_target[1], 
                affine_source, 
                affine_slice_group_target[0],
                dim=dim,
            )
            mapped_arrays.append(mapped)
            mapped_headers.append(headers)
    else:
        mapped_arrays, mapped_headers = _map_mask_series_to_slice_group(
            mask, on, affine_source, affine_target[0], dim=dim)
    mask.status.hide()
    return mapped_arrays, mapped_headers


def _map_mask_series_to_slice_group(source, target, affine_source, affine_target, **kwargs):

    if isinstance(affine_source, list):
        mapped_arrays = []
        for affine_slice_group in affine_source:
            mapped, headers = _map_mask_slice_group_to_slice_group(
                affine_slice_group[1], 
                target, 
                affine_slice_group[0], 
                affine_target,
                **kwargs,
            )
            mapped_arrays.append(mapped)
        array = np.logical_or(mapped_arrays[0], mapped_arrays[1])
        for a in mapped_arrays[2:]:
            array = np.logical_or(array, a)
        return array, headers
    else:
        return _map_mask_slice_group_to_slice_group(source, target, affine_source[0], affine_target, **kwargs)


def _map_mask_slice_group_to_slice_group(source, target, affine_source, affine_target, dim='InstanceNumber'):

    if isinstance(source, list):
        status = source[0].status
        dialog = source[0].dialog
    else:
        status = source.status
        dialog = source.dialog

    # Get arrays
    array_source, _ = dbdicom.array(source, sortby=['SliceLocation',dim], pixels_first=True)
    array_target, headers_target = dbdicom.array(target, sortby=['SliceLocation', dim], pixels_first=True)

    # Ignore spurious dimension
    array_source = array_source[...,0]
    array_target = array_target[...,0]
    headers_target = headers_target[...,0]

    # For mapping mask onto series, the time dimensions must be the same.
    # If they are not, the mask is extruded on to the series time dimensions.
    nk = array_target.shape[3]
    if array_source.shape[3] != nk:
        status.message('Extruding ROI on time series..')
        array_source = np.amax(array_source, axis=-1)
        array_source = np.repeat(array_source[:,:,:,np.newaxis], nk, axis=3)
    
    # If the dimensions and affines are equal there is nothing to do
    if np.array_equal(affine_source, affine_target):
        if array_source.shape == array_target.shape:
            # Make sure the result is a mask
            array_source[array_source > 0.5] = 1
            array_source[array_source <= 0.5] = 0
            return array_source, headers_target
    
    # Get transformation matrix
    source_to_target = np.linalg.inv(affine_source).dot(affine_target)
    source_to_target = np.around(source_to_target, 3) # to avoid round-off error in the inversion
    matrix, offset = nib.affines.to_matvec(source_to_target)
    
    #Perform transformation
    output_shape = array_target.shape[:3]
    for k in range(nk):
        status.progress(k+1, nk, 'Calculating overlay..')
        array_target[:,:,:,k] = affine_transform(
            array_source[:,:,:,k],
            matrix = matrix,
            offset = offset,
            output_shape = output_shape,
            order = 0)
        
    # Make sure the result is a mask
    status.message('Converting to binary..')
    array_target[array_target > 0.5] = 1
    array_target[array_target <= 0.5] = 0 

    # Return without spurious dimensions
    return array_target, headers_target
        



# SEGMENTATION


def label_2d(input, **kwargs):
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
    suffix = ' [label 2D]'
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


def label_3d(input, **kwargs):
    """
    Labels structures in a 3D volume
    
    Wrapper for scipy.ndimage.label function. 

    Parameters
    ----------
    input: dbdicom series

    Returns
    -------
    filtered : dbdicom series
    """
    desc = input.instance().SeriesDescription + ' [label 3D]'
    transform = input.new_sibling(SeriesDescription = desc)
    array, headers = input.array('SliceLocation', pixels_first=True)
    if array is None:
        return transform
    for t in range(array.shape[3]):
        input.status.progress(t, array.shape[3], 'Calculating ' + desc)
        array[:,:,:,t], _ = scipy.ndimage.label(array[:,:,:,t], **kwargs)
        transform.set_array(array[:,:,:,t], headers[:,t], pixels_first=True)
    _reset_window(transform, array)
    input.status.hide()
    return transform


def extract_largest_cluster_3d(input, **kwargs):
    """
    Label structures in 3D and then extract the largest cluster, return as a mask. 

    Parameters
    ----------
    input: dbdicom series

    Returns
    -------
    dbdicom series
    """
    desc = input.instance().SeriesDescription + ' [Largest cluster 3D]'
    transform = input.new_sibling(SeriesDescription = desc)
    array, headers = input.array('SliceLocation', pixels_first=True)
    if array is None:
        return transform
    for t in range(array.shape[3]):
        input.status.progress(t, array.shape[3], 'Calculating ' + desc)
        label_img, cnt = scipy.ndimage.label(array[:,:,:,t], **kwargs)
        # Find the label of the largest feature
        labels = range(1,cnt+1)
        size = [np.count_nonzero(label_img==l) for l in labels]
        max_label = labels[size.index(np.amax(size))]
        # Create a mask corresponding to the largest feature
        label_img = label_img==max_label
        #label_img = label_img[label_img==max_label]
        #label_img /= max_label
        transform.set_array(label_img, headers[:,t], pixels_first=True)
    _reset_window(transform, array)
    input.status.hide()
    return transform


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


def distance_transform_edt_3d(input, **kwargs):
    """
    Euclidian distance transform in 3D
    
    Wrapper for scipy.ndimage.distance_transform_edt function. 

    Parameters
    ----------
    input: dbdicom series
    markers: dbdicom series of the same dimensions as series

    Returns
    -------
    filtered : dbdicom series
    """
    desc = input.instance().SeriesDescription + ' [distance transform 3D]'
    #transform = input.copy(SeriesDescription = desc)
    transform = input.new_sibling(SeriesDescription = desc)
    array, headers = input.array('SliceLocation', pixels_first=True)
    if array is None:
        return transform
    for t in range(array.shape[3]):
        if array.shape[3] > 1:
            input.status.progress(t, array.shape[3], 'Calculating ' + desc)
        else:
            input.status.message('Calculating ' + desc + '. Please bear with me..')
        array[:,:,:,t] = scipy.ndimage.distance_transform_edt(array[:,:,:,t], **kwargs)
        transform.set_array(array[:,:,:,t], headers[:,t], pixels_first=True)
    _reset_window(transform, array)
    input.status.hide()
    return transform



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



def series_calculator(series, operation='1 - series'):

    desc = series.instance().SeriesDescription
    result = series.copy(SeriesDescription = desc + ' [' + operation + ']')
    images = result.images()
    for i, img in enumerate(images):
        series.status.progress(i+1, len(images), 'Calculating..')
        img.read()
        array = img.array()
        if operation == '1 - series':
            array = 1 - array
        elif operation == '- series':
            array = -array
        elif operation == '1 / series':
            array = 1 / array
        elif operation == 'exp(- series)':
            array = np.exp(-array)
        elif operation == 'exp(+ series)':
            array = np.exp(array)
        elif operation == 'integer(series)':
            array = np.around(array)
        array[~np.isfinite(array)] = 0
        img.set_array(array)
        _reset_window(img, array)
        img.clear()
    series.status.hide()
    return result


def image_calculator(series1, series2, operation='series 1 - series 2', integer=False):

    result = map_to(series2, series1)
    if result == series2: # same geometry
        result = series2.copy()
    images1 = series1.images(sortby=['SliceLocation', 'AcquisitionTime'])
    images2 = result.images(sortby=['SliceLocation', 'AcquisitionTime'])
    for i, img1 in enumerate(images1):
        series1.status.progress(i+1, len(images1), 'Calculating..')
        if i > len(images2)-1:
            break
        img2 = images2[i]
        img2.read()
        array1 = img1.array()
        array2 = img2.array()
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
        if integer:
            array = np.around(array)
        img2.set_array(array)
        _reset_window(img2, array.astype(np.ubyte))
        img2.clear()
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
            image.PixelSpacing = [pixel_spacing[i]/zoom[i] for i in range(2)]
        else:
            image.PixelSpacing = [p/zoom for p in pixel_spacing]
        image.clear()
    input.status.hide()
    return zoomed


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

    # Get arrays
    array, headers = series.array(['SliceLocation','AcquisitionTime'], pixels_first=True)
    if array is None:
        return resliced_series

    # Perform transformation on the arrays to determine the output shape
    dim = [
        array.shape[0] * p['PixelSpacing'][1],
        array.shape[1] * p['PixelSpacing'][0],
        array.shape[2] * p['SpacingBetweenSlices'],
    ]
    output_shape = [1 + round(dim[i]/voxel_size[i]) for i in range(3)]

    # Determine the transformation matrix and offset
    source_to_target = np.linalg.inv(affine_source).dot(affine_target)
    matrix, offset = nib.affines.to_matvec(source_to_target)
    
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
    spacing = np.mean([p['PixelSpacing'][0], p['PixelSpacing'][1], p['SpacingBetweenSlices']])
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
