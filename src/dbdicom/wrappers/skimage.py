import numpy as np
import skimage
from dbdicom.wrappers import scipy

def watershed_3d(input, markers=250, mask=None, **kwargs):
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
    desc = input.instance().SeriesDescription + ' [watershed 3D]'
    filtered = input.copy(SeriesDescription = desc)
    array, headers = filtered.array('SliceLocation', pixels_first=True)
    if array is None:
        return filtered
    if mask is not None: 
        mask, _ = mask.array('SliceLocation', pixels_first=True) 
        if mask.shape != array.shape:
            mask = None
    for t in range(array.shape[3]):
        input.status.progress(t, array.shape[3], 'Calculating ' + desc)
        if mask is None:
            mask_t = None
        else:
            mask_t = mask[:,:,:,t]
        array = skimage.segmentation.watershed(array[:,:,:,t], markers=markers, mask=mask_t, **kwargs)
        filtered.set_array(array, headers[:,t], pixels_first=True)
    input.status.hide()
    return filtered


def watershed_2d(input, markers=5, **kwargs):
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
    desc = input.instance().SeriesDescription + ' [watershed]'
    filtered = input.copy(SeriesDescription = desc)
    #images = filtered.instances() #sort=False should be faster - check
    images = filtered.images()
    for i, image in enumerate(images):
        input.status.progress(i+1, len(images), 'Calculating ' + desc)
        image.read()
        array = image.array()
        array = skimage.segmentation.watershed(array, markers=markers, **kwargs)
        #array.astype(np.float32)
        image.set_array(array)
        _reset_window(image, array)
        image.clear()
    input.status.hide()
    return filtered


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
        array = skimage.morphology.skeletonize_3d(array[:,:,:,t], **kwargs)
        filtered.set_array(array, headers[:,t], pixels_first=True)
    _reset_window(filtered, array)
    input.status.hide()
    return filtered


def watershed_2d_labels(input, markers=None, **kwargs):
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
    suffix = ' [watershed]'
    desc = input.instance().SeriesDescription
    filtered = input.copy(SeriesDescription = desc+suffix)
    #images = filtered.instances() #sort=False should be faster - check
    images = filtered.images()
    for i, image in enumerate(images):
        input.status.progress(i+1, len(images), 'Calculating watershed for ' + desc)
        image.read()
        array = image.array()
        if markers is None:
            mrk = None
        else:
            mrk = markers.instances( 
                SliceLocation = image.SliceLocation,
                AcquisitionTime = image.AcquisitionTime)
            if mrk == []:
                image.clear()
                image.remove()
                continue
            else:
                # If there are multiple, use the first one
                mrk = mrk[0].array()
                mrk = np.rint(mrk).astype(np.uint)
                if array.shape != mrk.shape:
                    mrk = None
        array = skimage.segmentation.watershed(array, markers=mrk, **kwargs)
        array.astype(np.float32) # unnecessary - test
        image.set_array(array)
        _reset_window(image, array)
        image.clear()
    input.status.hide()
    return filtered


# https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.canny
def canny(input, sigma=1.0, **kwargs):
    """
    wrapper for skimage.feature.canny

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


# https://scikit-image.org/docs/stable/api/skimage.registration.html#skimage.registration.optical_flow_tvl1
def coregister(moving, fixed, return_array=False, attachment=1):

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