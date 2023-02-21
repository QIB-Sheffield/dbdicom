import numpy as np
import skimage
import scipy.ndimage as ndi
from dbdicom.wrappers import scipy


def volume_features(series):

    affine = series.affine_matrix()
    if isinstance(affine, list):
        series.dialog.information('This series contains multiple volumes')
        return
    else:
        affine = affine[0]

    # Get array sorted by slice location
    arr, _ = series.array('SliceLocation', pixels_first=True)

    # If there are multiple volumes, show only the first one
    arr = arr[...,0]

    series.status.message('Preprocessing mask...')

    # Scale in the range [0,1] so it can be treated as mask
    max = np.amax(arr)
    min = np.amin(arr)
    arr -= min
    arr /= max-min

    # add zeropadding at the boundary slices
    shape = list(arr.shape)
    shape[-1] = shape[-1] + 2*4
    array = np.zeros(shape)
    array[:,:,4:-4] = arr

    series.status.message('Extracting surface...')


    # Get voxel dimensions (assumed this is in mm)
    column_spacing = np.linalg.norm(affine[:3, 0])
    row_spacing = np.linalg.norm(affine[:3, 1])
    slice_spacing = np.linalg.norm(affine[:3, 2])
    spacing = (column_spacing, row_spacing, slice_spacing)  #mm
    voxel_volume = column_spacing*row_spacing*slice_spacing
    nr_of_voxels = np.count_nonzero(array)
    volume = nr_of_voxels * voxel_volume


    # Surface properties (Area only for now)
    smooth_array = ndi.gaussian_filter(array, 1.5)
    verts, faces, _, _ = skimage.measure.marching_cubes(smooth_array, spacing=spacing, level=0.5, step_size=1.0)
    #cloud = pv.PolyData(verts, faces)
    #surf = cloud.reconstruct_surface()
    surface_area = skimage.measure.mesh_surface_area(verts, faces)


    # Volume properties
    # TODO: If the volume is not isotropic, resample to isotropic before getting the properties
    isotropic_voxel_volume = voxel_volume # for now
    isotropic_spacing = np.mean(np.array(spacing)) # for now
    region_props = skimage.measure.regionprops(np.round(array).astype(np.int16))
    region_props_3D = {}
    for prop in region_props[0]:
        try: 
            region_props_3D[prop] = region_props[0][prop]
        except:
            pass # not supported in 3D

    surface_props = {
        'Surface area (cm^2)': surface_area/100,
        'Volume (mL)': volume/1000,
        'Volume (mL) - check': region_props_3D['area']*isotropic_voxel_volume/1000,
        'Number of connected components': region_props_3D['euler_number'],
        'Bounding box volume (mL)': region_props_3D['area_bbox']*isotropic_voxel_volume/1000,
        'Percentage of bounding box filled (%)': region_props_3D['extent']*100,
        'Convex hull volume (mL)': region_props_3D['area_convex']*isotropic_voxel_volume/1000,
        'Percentage of convex hull filled (%)': region_props_3D['solidity']*100,
        'Volume of holes (mL)': (region_props_3D['area_filled']-region_props_3D['area'])*isotropic_voxel_volume/1000,
        'Long axis length (cm)': region_props_3D['axis_major_length']*isotropic_spacing/10,
        'Short axis length (cm)': region_props_3D['axis_minor_length']*isotropic_spacing/10,
        'Equivalent diameter (cm)': region_props_3D['equivalent_diameter_area']*isotropic_spacing/10,
        'Longest distance inside (cm)': region_props_3D['feret_diameter_max']*isotropic_spacing/10,
        'Primary moment of inertia (cm^2)': region_props_3D['inertia_tensor_eigvals'][0]*isotropic_spacing**2/100,
        'Second moment of inertia (cm^2)': region_props_3D['inertia_tensor_eigvals'][1]*isotropic_spacing**2/100,
        'Third moment of inertia (cm^2)': region_props_3D['inertia_tensor_eigvals'][2]*isotropic_spacing**2/100,
        # From eigenvectors of intertia tensor: Include orientation info with respect to LPH coordinate system (tilt, roll, yaw)
    }
    print(surface_props)

    return surface_props



def area_opening_2d(input, **kwargs):
    """
    Return grayscale area opening of an image.
    
    Wrapper for skimage.morphology.area_opening. 

    Parameters
    ----------
    input: dbdicom series

    Returns
    -------
    output : dbdicom series
    """
    desc = input.instance().SeriesDescription + ' [area opening 2D]'
    result = input.copy(SeriesDescription = desc)
    images = result.images()
    for i, image in enumerate(images):
        input.status.progress(i+1, len(images), 'Calculating ' + desc)
        image.read()
        array = image.array()
        array = skimage.morphology.area_opening(array, **kwargs)
        image.set_array(array)
        _reset_window(image, array)
        image.clear()
    input.status.hide()
    return result


def area_opening_3d(input, **kwargs):
    """
    Return grayscale area opening of an image.
    
    Wrapper for skimage.morphology.area_opening. 

    Parameters
    ----------
    input: dbdicom series

    Returns
    -------
    output : dbdicom series
    """
    array, headers = input.array('SliceLocation', pixels_first=True)
    if array is None:
        return input
    desc = input.instance().SeriesDescription + ' [area opening 3D]'
    result = input.new_sibling(SeriesDescription = desc)
    for t in range(array.shape[3]):
        input.status.progress(t, array.shape[3], 'Calculating ' + desc)
        array[...,t] = skimage.morphology.area_opening(array[...,t], **kwargs)
        result.set_array(array[...,t], headers[:,t], pixels_first=True)
    _reset_window(result, array)
    input.status.hide()
    return result


def area_closing_2d(input, **kwargs):
    """
    Return grayscale area closing of an image.
    
    Wrapper for skimage.morphology.area_closing. 

    Parameters
    ----------
    input: dbdicom series

    Returns
    -------
    output : dbdicom series
    """
    desc = input.instance().SeriesDescription + ' [area closing 2D]'
    result = input.copy(SeriesDescription = desc)
    images = result.images()
    for i, image in enumerate(images):
        input.status.progress(i+1, len(images), 'Calculating ' + desc)
        image.read()
        array = image.array()
        array = skimage.morphology.area_closing(array, **kwargs)
        image.set_array(array)
        _reset_window(image, array)
        image.clear()
    input.status.hide()
    return result


def area_closing_3d(input, **kwargs):
    """
    Return grayscale area closing of an image.
    
    Wrapper for skimage.morphology.area_closing. 

    Parameters
    ----------
    input: dbdicom series

    Returns
    -------
    output : dbdicom series
    """
    array, headers = input.array('SliceLocation', pixels_first=True)
    if array is None:
        return input
    desc = input.instance().SeriesDescription + ' [area closing 3D]'
    result = input.new_sibling(SeriesDescription = desc)
    for t in range(array.shape[3]):
        input.status.progress(t, array.shape[3], 'Calculating ' + desc)
        array[...,t] = skimage.morphology.area_closing(array[...,t], **kwargs)
        result.set_array(array[...,t], headers[:,t], pixels_first=True)
    _reset_window(result, array)
    input.status.hide()
    return result


def opening_2d(input, **kwargs):
    """
    Return grayscale morphological opening of an image.
    
    Wrapper for skimage.morphology.opening. 

    Parameters
    ----------
    input: dbdicom series

    Returns
    -------
    output : dbdicom series
    """
    desc = input.instance().SeriesDescription + ' [opening 2D]'
    result = input.copy(SeriesDescription = desc)
    images = result.images()
    for i, image in enumerate(images):
        input.status.progress(i+1, len(images), 'Calculating ' + desc)
        image.read()
        array = image.array()
        array = skimage.morphology.opening(array, **kwargs)
        image.set_array(array)
        _reset_window(image, array)
        image.clear()
    input.status.hide()
    return result


def opening_3d(input, **kwargs):
    """
    Return grayscale morphological opening of an image.
    
    Wrapper for skimage.morphology.opening. 

    Parameters
    ----------
    input: dbdicom series

    Returns
    -------
    output : dbdicom series
    """
    array, headers = input.array('SliceLocation', pixels_first=True)
    if array is None:
        return input
    desc = input.instance().SeriesDescription + ' [opening 3D]'
    result = input.new_sibling(SeriesDescription = desc)
    for t in range(array.shape[3]):
        input.status.progress(t, array.shape[3], 'Calculating ' + desc)
        array[...,t] = skimage.morphology.opening(array[...,t], **kwargs)
        result.set_array(array[...,t], headers[:,t], pixels_first=True)
    _reset_window(result, array)
    input.status.hide()
    return result


def closing_2d(input, **kwargs):
    """
    Return grayscale morphological closing of an image.
    
    Wrapper for skimage.morphology.closing. 

    Parameters
    ----------
    input: dbdicom series

    Returns
    -------
    output : dbdicom series
    """
    desc = input.instance().SeriesDescription + ' [closing 2D]'
    result = input.copy(SeriesDescription = desc)
    images = result.images()
    for i, image in enumerate(images):
        input.status.progress(i+1, len(images), 'Calculating ' + desc)
        image.read()
        array = image.array()
        array = skimage.morphology.closing(array, **kwargs)
        image.set_array(array)
        _reset_window(image, array)
        image.clear()
    input.status.hide()
    return result


def closing_3d(input, **kwargs):
    """
    Return grayscale morphological closing of an image.
    
    Wrapper for skimage.morphology.closing. 

    Parameters
    ----------
    input: dbdicom series

    Returns
    -------
    output : dbdicom series
    """
    array, headers = input.array('SliceLocation', pixels_first=True)
    if array is None:
        return input
    desc = input.instance().SeriesDescription + ' [closing 3D]'
    result = input.new_sibling(SeriesDescription = desc)
    for t in range(array.shape[3]):
        input.status.progress(t, array.shape[3], 'Calculating ' + desc)
        array[...,t] = skimage.morphology.closing(array[...,t], **kwargs)
        result.set_array(array[...,t], headers[:,t], pixels_first=True)
    _reset_window(result, array)
    input.status.hide()
    return result


def remove_small_holes_2d(input, **kwargs):
    """
    Remove contiguous holes smaller than the specified size.
    
    Wrapper for skimage.morphology.remove_small_holes. 

    Parameters
    ----------
    input: dbdicom series

    Returns
    -------
    output : dbdicom series
    """
    desc = input.instance().SeriesDescription + ' [remove small holes 2D]'
    result = input.copy(SeriesDescription = desc)
    images = result.images()
    for i, image in enumerate(images):
        input.status.progress(i+1, len(images), 'Calculating ' + desc)
        image.read()
        array = image.array().astype(np.int16)
        array = skimage.morphology.remove_small_holes(array, **kwargs)
        image.set_array(array)
        _reset_window(image, array)
        image.clear()
    input.status.hide()
    return result


def remove_small_holes_3d(input, **kwargs):
    """
    Remove contiguous holes smaller than the specified size.
    
    Wrapper for skimage.morphology.remove_small_holes. 

    Parameters
    ----------
    input: dbdicom series

    Returns
    -------
    output : dbdicom series
    """
    array, headers = input.array('SliceLocation', pixels_first=True)
    if array is None:
        return input
    else:
        array = array.astype(np.int16)
    desc = input.instance().SeriesDescription + ' [remove holes 3D]'
    result = input.new_sibling(SeriesDescription = desc)
    for t in range(array.shape[3]):
        input.status.progress(t, array.shape[3], 'Calculating ' + desc)
        array[...,t] = skimage.morphology.remove_small_holes(array[...,t], **kwargs)
        result.set_array(array[...,t], headers[:,t], pixels_first=True)
    _reset_window(result, array)
    input.status.hide()
    return result


def watershed_2d(input, markers=None, mask=None, **kwargs):
    """
    Labels structures in an image
    
    Wrapper for skimage.segmentation.watershed function. 

    Parameters
    ----------
    input: dbdicom series
    markers: dbdicom series of the same dimensions as series

    Returns
    -------
    filtered : dbdicom series
    """
    desc = input.instance().SeriesDescription + ' [watershed 2D]'
    result = input.copy(SeriesDescription = desc)
    sortby = ['SliceLocation', 'AcquisitionTime']
    images = result.images(sortby=sortby)
    if markers is not None:
        markers = scipy.map_to(markers, input, label=True)
        markers = markers.images(sortby=sortby)
    if mask is not None:
        mask = scipy.map_to(mask, input, mask=True)
        mask = mask.images(sortby=sortby)
    for i, image in enumerate(images):
        input.status.progress(i+1, len(images), 'Calculating ' + desc)
        image.read()
        array = image.array()
        if markers is None:
            mrk = None
        else:
            mrk = markers[i].array().astype(np.uint)
        if mask is None:
            msk = None
        else:
            msk = mask[i].array().astype(np.bool8)
        array = skimage.segmentation.watershed(array, markers=mrk, mask=msk, **kwargs)
        array.astype(np.float32) # unnecessary - test
        image.set_array(array)
        _reset_window(image, array)
        image.clear()
    input.status.hide()
    return result


def watershed_3d(input, markers=None, mask=None, **kwargs):
    """
    Determine watershed in 3D
    
    Wrapper for skimage.segmentation.watershed function.

    Parameters
    ----------
    input: dbdicom series

    Returns
    -------
    filtered : dbdicom series
    """
    array, headers = input.array('SliceLocation', pixels_first=True)
    if array is None:
        return input
    if markers is not None:
        markers = scipy.map_to(markers, input)
        markers, _ = markers.array('SliceLocation', pixels_first=True)
        markers = markers.astype(np.uint)
    if mask is not None:
        mask = scipy.map_to(mask, input)
        mask, _ = mask.array('SliceLocation', pixels_first=True)
        mask = mask.astype(np.bool8)
    desc = input.instance().SeriesDescription + ' [watershed 3D]'
    result = input.new_sibling(SeriesDescription = desc)
    for t in range(array.shape[3]):
        input.status.progress(t, array.shape[3], 'Calculating ' + desc)
        array[...,t] = skimage.segmentation.watershed(
            array[...,t], 
            markers = None if markers is None else markers[...,t], 
            mask = None if mask is None else mask[...,t], 
            **kwargs)
        result.set_array(array[...,t], headers[:,t], pixels_first=True)
    _reset_window(result, array)
    input.status.hide()
    return result


def skeletonize(input, **kwargs):
    """
    Labels structures in an image
    
    Wrapper for skimage.segmentation.watershed function. 

    Parameters
    ----------
    input: dbdicom series
    markers: dbdicom series of the same dimensions as series

    Returns
    -------
    filtered : dbdicom series
    """
    desc = input.instance().SeriesDescription + ' [2d skeleton]'
    filtered = input.copy(SeriesDescription = desc)
    #images = filtered.instances() #sort=False should be faster - check
    images = filtered.images()
    for i, image in enumerate(images):
        input.status.progress(i+1, len(images), 'Calculating ' + desc)
        image.read()
        array = image.array()
        array = skimage.morphology.skeletonize(array, **kwargs)
        #array.astype(np.float32)
        image.set_array(array)
        _reset_window(image, array)
        image.clear()
    input.status.hide()
    return filtered


def skeletonize_3d(input, **kwargs):
    """
    Labels structures in an image
    
    Wrapper for skimage.segmentation.watershed function. 

    Parameters
    ----------
    input: dbdicom series
    markers: dbdicom series of the same dimensions as series

    Returns
    -------
    filtered : dbdicom series
    """
    desc = input.instance().SeriesDescription + ' [skeleton 3D]'
    filtered = input.copy(SeriesDescription = desc)
    array, headers = filtered.array('SliceLocation', pixels_first=True)
    if array is None:
        return filtered
    for t in range(array.shape[3]):
        if array.shape[3] > 1:
            input.status.progress(t, array.shape[3], 'Calculating ' + desc)
        else:
            input.status.message('Calculating ' + desc + '. Please bear with me..')
        array[:,:,:,t] = skimage.morphology.skeletonize_3d(array[:,:,:,t], **kwargs)
        filtered.set_array(array, headers[:,t], pixels_first=True)
    _reset_window(filtered, array)
    input.status.hide()
    return filtered


def peak_local_max_3d(input, labels=None, **kwargs):
    """
    Determine local maxima
    
    Wrapper for skimage.feature.peak_local_max function. 
    # https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.peak_local_max

    Parameters
    ----------
    input: dbdicom series

    Returns
    -------
    filtered : dbdicom series
    """
    array, headers = input.array('SliceLocation', pixels_first=True)
    if array is None:
        return filtered
    if labels is not None:
        labels = scipy.map_to(labels, input)
        labels_array, _ = labels.array('SliceLocation', pixels_first=True)
    desc = input.instance().SeriesDescription + ' [peak local max 3D]'
    filtered = input.new_sibling(SeriesDescription = desc)
    for t in range(array.shape[3]):
        input.status.progress(t, array.shape[3], 'Calculating ' + desc)
        coords = skimage.feature.peak_local_max(
            array[:,:,:,t], 
            labels = labels_array[:,:,:,t].astype(np.int16), 
            **kwargs)
        mask = np.zeros(array.shape[:3], dtype=bool)
        mask[tuple(coords.T)] = True
        filtered.set_array(mask, headers[:,t], pixels_first=True)
    _reset_window(filtered, array)
    input.status.hide()
    return filtered


def canny(input, sigma=1.0, **kwargs):
    """
    wrapper for skimage.feature.canny
    # https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.canny

    Parameters
    ----------
    input: dbdicom series

    Returns
    -------
    filtered : dbdicom series
    """
    suffix = ' [Canny filter x ' + str(sigma) + ' ]'
    desc = input.instance().SeriesDescription
    filtered = input.copy(SeriesDescription = desc+suffix)
    #images = filtered.instances()
    images = filtered.images()
    for i, image in enumerate(images):
        input.status.progress(i+1, len(images), 'Filtering ' + desc)
        image.read()
        array = image.array()
        array = skimage.feature.canny(array, sigma=sigma, **kwargs)
        image.set_array(array)
        array = array.astype(np.ubyte)
        _reset_window(image, array)
        image.clear()
    input.status.hide()
    return filtered


def convex_hull_image(series, **kwargs):
    """
    wrapper for skimage.feature.canny

    Parameters
    ----------
    input: dbdicom series

    Returns
    -------
    filtered : dbdicom series
    """
    suffix = ' [Convex hull]'
    desc = series.instance().SeriesDescription 
    chull = series.copy(SeriesDescription = desc+suffix)
    images = chull.images()
    for i, image in enumerate(images):
        series.status.progress(i+1, len(images), 'Calculating convex hull for ' + desc)
        image.read()
        array = image.array()
        array = skimage.morphology.convex_hull_image(array, **kwargs)
        image.set_array(array)
        array = array.astype(np.ubyte)
        _reset_window(image, array)
        image.clear()
    series.status.hide()
    return chull


def coregister(moving, fixed, return_array=False, attachment=1):
    # https://scikit-image.org/docs/stable/api/skimage.registration.html#skimage.registration.optical_flow_tvl1

    #fixed = fixed.map_to(moving)
    fixed = scipy.map_to(fixed, moving)

    # Get arrays for fixed and moving series
    array_fixed, _ = fixed.array('SliceLocation', pixels_first=True)
    array_moving, headers_moving = moving.array('SliceLocation', pixels_first=True)
    if array_fixed is None or array_moving is None:
        return fixed

    # Coregister fixed and moving slice-by-slice
    row_coords, col_coords = np.meshgrid( 
        np.arange(array_moving.shape[0]), 
        np.arange(array_moving.shape[1]),
        indexing='ij')
    for z in range(array_moving.shape[2]):
        moving.status.progress(z+1, array_moving.shape[2], 'Performing coregistration..')
        image0 = array_fixed[:,:,z,0]
        image1 = array_moving[:,:,z,0]
        v, u = skimage.registration.optical_flow_tvl1(image0, image1, attachment=attachment)
        array_moving[:,:,z,0] = skimage.transform.warp(
            image1, 
            np.array([row_coords + v, col_coords + u]),
            mode='edge')

    # Return array or new series
    if return_array:
        moving.status.message('Finished coregistration..')
        return array_moving, headers_moving[:,0]
    else:
        moving.status.message('Writing coregistered series to database..')
        desc = moving.instance().SeriesDescription 
        desc += ' registered to ' + fixed.instance().SeriesDescription
        registered_series = moving.new_sibling(SeriesDescription = desc)
        registered_series.set_array(array_moving, headers_moving, pixels_first=True)
        moving.status.message('Finished coregistration..')
        return registered_series
    

def coregister_series(series, attachment=1):

    # Get arrays for fixed and moving series
    array, headers = series.array(['SliceLocation','AcquisitionTime'], pixels_first=True)
    if array is None:
        return
    array, headers = array[:,:,:,:,0], headers[:,:,0]

    
    # Coregister fixed and moving slice-by-slice
    row_coords, col_coords = np.meshgrid(
        np.arange(array.shape[0]), 
        np.arange(array.shape[1]),
        indexing='ij')
    target = np.mean(array, axis=3)
    cnt=0
    for t in range(array.shape[3]):
        for z in range(array.shape[2]):
            cnt+=1
            series.status.progress(cnt, array.shape[2]*array.shape[3], 'Performing coregistration..')
            fixed = target[:,:,z]
            moving = array[:,:,z,t]
            v, u = skimage.registration.optical_flow_tvl1(fixed, moving, attachment=attachment)
            array[:,:,z,t] = skimage.transform.warp(
                moving, 
                np.array([row_coords + v, col_coords + u]),
                mode='edge')

    # Return array or new series
    series.status.message('Writing coregistered series to database..')
    desc = series.instance().SeriesDescription + ' [coregistered]'
    registered_series = series.new_sibling(SeriesDescription=desc)
    registered_series.set_array(array, headers, pixels_first=True)
    series.status.message('Finished coregistration..')
    return registered_series


def mdreg_constant_2d(series, attachment=1, max_improvement=1, max_iter=5):

    # Get arrays for fixed and moving series
    array, headers = series.array(['SliceLocation','AcquisitionTime'], pixels_first=True)
    if array is None:
        return
    array, headers = array[:,:,:,:,0], headers[:,:,0]

    
    # Coregister fixed and moving slice-by-slice
    row_coords, col_coords = np.meshgrid(
        np.arange(array.shape[0]), 
        np.arange(array.shape[1]),
        indexing='ij')
    v, u = np.zeros(array.shape), np.zeros(array.shape)
    coreg = array.copy()
    for it in range(max_iter):
        target = np.mean(coreg, axis=3) # constant model
        cnt=0
        improvement = 0 # pixel sizes
        for t in range(array.shape[3]):
            for z in range(array.shape[2]):
                cnt+=1
                msg = 'Performing iteration ' + str(it) + ' < ' + str(max_iter)
                msg += ' (best improvement so far = ' + str(round(improvement,2)) + ' pixels)'
                series.status.progress(cnt, array.shape[2]*array.shape[3], msg)
                v_zt, u_zt = skimage.registration.optical_flow_tvl1(
                    target[:,:,z], 
                    array[:,:,z,t], 
                    attachment=attachment)
                coreg[:,:,z,t] = skimage.transform.warp(
                    array[:,:,z,t], 
                    np.array([row_coords + v_zt, col_coords + u_zt]),
                    mode='edge')
                improvement_zt = np.amax(np.sqrt(np.square(v_zt-v[:,:,z,t]) + np.square(u_zt-u[:,:,z,t])))
                if improvement_zt > improvement:
                    improvement = improvement_zt
                v[:,:,z,t], u[:,:,z,t] = v_zt, u_zt
        if improvement < max_improvement:
            break
    
    series.status.message('Writing coregistered series to database..')
    desc = series.instance().SeriesDescription + ' [coregistered]'
    registered_series = series.new_sibling(SeriesDescription=desc)
    registered_series.set_array(coreg, headers, pixels_first=True)
    series.status.message('Finished coregistration..')
    return registered_series


def mdreg_constant_3d(series, attachment=1, max_improvement=1, max_iter=5):

    # Get arrays for fixed and moving series
    array, headers = series.array(['SliceLocation','AcquisitionTime'], pixels_first=True)
    if array is None:
        return
    array, headers = array[:,:,:,:,0], headers[:,:,0]
    
    # Coregister fixed and moving slice-by-slice
    row_coords, col_coords, slice_coords = np.meshgrid(
        np.arange(array.shape[0]), 
        np.arange(array.shape[1]),
        np.arange(array.shape[2]),
        indexing='ij')
    v, u, w = np.zeros(array.shape), np.zeros(array.shape), np.zeros(array.shape)
    coreg = array.copy()
    for it in range(max_iter):
        target = np.mean(coreg, axis=3) # constant model
        cnt=0
        improvement = 0 # pixel sizes
        for t in range(array.shape[3]):
            cnt+=1
            msg = 'Performing iteration ' + str(it) + ' < ' + str(max_iter)
            msg += ' (best improvement so far = ' + str(round(improvement,2)) + ' pixels)'
            series.status.progress(cnt, array.shape[3], msg)
            v_t, u_t, w_t = skimage.registration.optical_flow_tvl1(
                target, 
                array[:,:,:,t], 
                attachment=attachment)
            coreg[:,:,:,t] = skimage.transform.warp(
                array[:,:,:,t], 
                np.array([row_coords + v_t, col_coords + u_t, slice_coords + w_t]),
                mode='edge')
            improvement_t = np.amax(np.sqrt(np.square(v_t-v[:,:,:,t]) + np.square(u_t-u[:,:,:,t]) + np.square(w_t-w[:,:,:,t])))
            if improvement_t > improvement:
                improvement = improvement_t
            v[:,:,:,t], u[:,:,:,t], w[:,:,:,t] = v_t, u_t, w_t
        if improvement < max_improvement:
            break
    
    series.status.message('Writing coregistered series to database..')
    desc = series.instance().SeriesDescription + ' [coregistered]'
    registered_series = series.new_sibling(SeriesDescription=desc)
    registered_series.set_array(coreg, headers, pixels_first=True)
    series.status.message('Finished coregistration..')
    return registered_series

    

# Helper functions

def _reset_window(image, array):
    arr = array.astype(np.float32)
    min = np.amin(arr)
    max = np.amax(arr)
    image.WindowCenter= (max+min)/2
    if min==max:
        if min == 0:
            image.WindowWidth = 1
        else:
            image.WindowWidth = min
    else:
        image.WindowWidth = 0.9*(max-min)