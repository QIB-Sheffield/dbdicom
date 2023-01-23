import numpy as np
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric, EMMetric, SSDMetric
from dipy.segment.mask import median_otsu as median_otsu_np
import dbdicom.wrappers.scipy as scipy


def median_otsu(series, **kwargs):

    # Get arrays for fixed and moving series
    array, headers = series.array('SliceLocation', pixels_first=True)

    # Apply Otsu
    mask = np.empty(array.shape)
    cnt=0
    for z in range(array.shape[2]):
        for k in range(array.shape[3]):
            cnt+=1
            series.status.progress(cnt, array.shape[2]*array.shape[3], 'Applying Otsu segmentation..')
            image = np.squeeze(array[:,:,z,k])
            array[:,:,z,k], mask[:,:,z,k] = median_otsu_np(image, **kwargs)

    # Return new series
    series.status.message('Saving results..')
    desc = series.instance().SeriesDescription  
    masked_series = series.new_sibling(SeriesDescription = desc +' [masked]')
    masked_series.set_array(array, headers, pixels_first=True)
    otsu_mask = series.new_sibling(SeriesDescription = desc + ' [otsu mask]')
    otsu_mask.set_array(mask, headers, pixels_first=True)
    return masked_series, otsu_mask


def coregister(moving, fixed, **kwargs):

    fixed_map = scipy.map_to(fixed, moving)

    # Get arrays for fixed and moving series
    array_fixed, _ = fixed_map.array('SliceLocation', pixels_first=True)
    array_moving, headers_moving = moving.array('SliceLocation', pixels_first=True)
    if array_fixed is None or array_moving is None:
        return fixed_map

    # Get coregistration settings
    # TBC

    # Coregister fixed and moving slice-by-slice
    for z in range(array_moving.shape[2]):
        moving.status.progress(z+1, array_moving.shape[2], 'Performing coregistration..')
        image0 = array_fixed[:,:,z,0]
        image1 = array_moving[:,:,z,0]
        coreg, _ = _coregister_2D_arrays(image0, image1, **kwargs)
        array_moving[:,:,z,0] = coreg

    # Return new series
    moving.status.message('Saving results..')
    #fixed_map.remove()
    desc = moving.instance().SeriesDescription 
    desc += ' registered to ' + fixed.instance().SeriesDescription
    registered_series = moving.new_sibling(SeriesDescription = desc)
    registered_series.set_array(array_moving, headers_moving, pixels_first=True)
    moving.status.message('Finished coregistration..')
    return registered_series



def _coregister_2D_arrays(fixed, moving, transformation='Symmetric Diffeomorphic', metric="Cross-Correlation"):
    """
    Coregister two arrays and return coregistered + deformation field 
    """

    # Define the metric
    dim = fixed.ndim
    if metric == "Cross-Correlation":
        sigma_diff = 3.0    # Gaussian Kernel
        radius = 4          # Window for local CC
        metric = CCMetric(dim, sigma_diff, radius)
    elif metric == 'Expectation-Maximization':
        metric = EMMetric(dim, smooth=1.0)
    elif metric == 'Sum of Squared Differences':
        metric = SSDMetric(dim, smooth=4.0)
    else:
        msg = 'The metric ' + metric + ' is currently not implemented.'
        raise RuntimeError(msg) 

    # Define the deformation model
    if transformation == 'Symmetric Diffeomorphic':
        level_iters = [200, 100, 50, 25]
        sdr = SymmetricDiffeomorphicRegistration(metric, level_iters, inv_iter=50)
    else:
        msg = 'The transform ' + transformation + ' is currently not implemented.'
        raise RuntimeError(msg) 

    # Perform the optimization
    mapping = sdr.optimize(fixed, moving)

    # Warp the moving image
    warped_moving = mapping.transform(moving, 'linear')

    return warped_moving, mapping # mapping is not the deformation field array yet
