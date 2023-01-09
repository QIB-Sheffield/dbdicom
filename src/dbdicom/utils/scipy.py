import numpy as np
from scipy.ndimage import affine_transform
import nibabel as nib # unnecessary

import dbdicom.utils.image as image_utils
from dbdicom.record import merge, move_to


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
            if np.array_equal(a1, affine2[i]):
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


def map_to(source, target):
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
            v = image_utils.dismantle_affine_matrix(affine_slice_group)
            slice_group_target = target.subseries(ImageOrientationPatient = v['ImageOrientationPatient'])
            mapped = _map_series_to_slice_group(source, slice_group_target, affine_source, affine_slice_group)
            mapped_series.append(mapped)
            slice_group_target.remove()
        desc = source.instance().SeriesDescription 
        desc += ' mapped to ' + target.instance().SeriesDescription
        mapped_series = merge(mapped_series, inplace=True)
        mapped_series.SeriesDescription = desc
    else:
        mapped_series = _map_series_to_slice_group(source, target, affine_source, affine_target)
    return mapped_series


def _map_series_to_slice_group(source, target, affine_source, affine_target):

    if isinstance(affine_source, list):
        mapped_series = []
        for affine_slice_group in affine_source:
            v = image_utils.dismantle_affine_matrix(affine_slice_group)
            slice_group_source = source.subseries(ImageOrientationPatient = v['ImageOrientationPatient'])
            mapped = _map_slice_group_to_slice_group(slice_group_source, target, affine_slice_group, affine_target)
            mapped_series.append(mapped)
            slice_group_source.remove()
        return merge(mapped_series, inplace=True)
    else:
        return _map_slice_group_to_slice_group(source, target, affine_source, affine_target)


def _map_slice_group_to_slice_group(source, target, affine_source, affine_target):

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


def map_mask_to(source, target):
    """Map non-zero pixels onto another series"""

    # Get transformation matrix
    source.status.message('Loading transformation matrices..')
    affine_source = source.affine_matrix()
    affine_target = target.affine_matrix() 

    if isinstance(affine_target, list):
        mapped_arrays = []
        mapped_headers = []
        for affine_slice_group_target in affine_target:
            v = image_utils.dismantle_affine_matrix(affine_slice_group_target)
            slice_group_target = target.subseries(move=True, ImageOrientationPatient = v['ImageOrientationPatient'])
            mapped, headers = _map_mask_series_to_slice_group(source, slice_group_target, affine_source, affine_slice_group_target)
            mapped_arrays.append(mapped)
            mapped_headers.append(headers)
            source.status.message('Cleaning up..')
            move_to(slice_group_target, target)
            slice_group_target.remove()
    else:
        mapped_arrays, mapped_headers = _map_mask_series_to_slice_group(source, target, affine_source, affine_target)
    source.status.hide()
    return mapped_arrays, mapped_headers


def _map_mask_series_to_slice_group(source, target, affine_source, affine_target):

    if isinstance(affine_source, list):
        mapped_arrays = []
        for affine_slice_group in affine_source:
            v = image_utils.dismantle_affine_matrix(affine_slice_group)
            slice_group_source = source.subseries(ImageOrientationPatient = v['ImageOrientationPatient'])
            mapped, headers = _map_mask_slice_group_to_slice_group(slice_group_source, target, affine_slice_group, affine_target)
            mapped_arrays.append(mapped)
            slice_group_source.remove()
        array = np.logical_or(mapped_arrays[:2])
        for a in mapped_arrays[2:]:
            array = np.logical_or(array, a)
        return array, headers
    else:
        return _map_mask_slice_group_to_slice_group(source, target, affine_source, affine_target)


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
