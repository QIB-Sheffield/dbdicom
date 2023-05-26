import numpy as np
from dbdicom.utils import vreg

def print_current(vk):
    print(vk)

def translation__sum_of_squares(moving, static, tolerance=0.1):

    # Get affine matrices and check that there is a single value
    affine_moving = moving.affine_matrix()
    if isinstance(affine_moving, list):
        msg = 'This function only works for series with a single slice group. \n'
        msg += 'Multiple slice groups detected in moving series - please split the series first.'
        raise ValueError(msg)
    else:
        affine_moving = affine_moving[0]
    affine_static = static.affine_matrix()
    if isinstance(affine_static, list):
        msg = 'This function only works for series with a single slice group. \n'
        msg += 'Multiple slice groups detected in static series - please split the series first.'
        raise ValueError(msg)
    else:
        affine_static = affine_static[0]

    # Get arrays for static and moving series
    array_static, headers = static.array('SliceLocation', pixels_first=True, first_volume=True)
    if array_static is None:
        msg = 'Fixed series is empty - cannot perform alignment.'
        raise ValueError(msg)
    array_moving, _ = moving.array('SliceLocation', pixels_first=True, first_volume=True)
    if array_moving is None:
        msg = 'Moving series is empty - cannot perform alignment.'
        raise ValueError(msg)

    moving.status.message('Performing coregistration. Please be patient. Its hard work and I need to concentrate..')

    # Define initial values and optimization
    _, _, static_pixel_spacing = vreg.affine_components(affine_static)
    optimization = {
        'method': 'GD', 
        'options': {'gradient step': static_pixel_spacing, 'tolerance': tolerance}, 
        'callback': lambda vk: moving.status.message('Current parameter: ' + str(vk)),
    }
    
    # Align volumes
    try:
        translation_estimate = vreg.align(
            moving = array_moving, 
            moving_affine = affine_moving, 
            static = array_static, 
            static_affine = affine_static, 
            parameters = np.array([0, 0, 0], dtype=np.float32), 
            resolutions = [4,2,1], 
            transformation =  vreg.translate,
            metric = vreg.sum_of_squares,
            optimization = optimization,
        )
    except:
        print('Failed to align volumes..')
        return None

    coregistered = vreg.translate(array_moving, affine_moving, array_static.shape, affine_static, translation_estimate)
    
    moving.status.message('Writing coregistered series to database..')
    
    # Save results as new dicom series
    desc = moving.instance().SeriesDescription
    coreg = moving.new_sibling(SeriesDescription = desc + ' [translation]')
    coreg.set_array(coregistered, headers, pixels_first=True)
    return coreg