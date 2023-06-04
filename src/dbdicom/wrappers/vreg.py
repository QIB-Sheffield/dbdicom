import numpy as np
from dbdicom.utils import vreg

def print_current(vk):
    print(vk)


def _get_input(moving, static):

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
    array_static, headers_static = static.array('SliceLocation', pixels_first=True, first_volume=True)
    if array_static is None:
        msg = 'Fixed series is empty - cannot perform alignment.'
        raise ValueError(msg)
    array_moving, headers_moving = moving.array('SliceLocation', pixels_first=True, first_volume=True)
    if array_moving is None:
        msg = 'Moving series is empty - cannot perform alignment.'
        raise ValueError(msg)
    
    moving.message('Performing coregistration. Please be patient. Its hard work and I need to concentrate..')
    
    return array_static, affine_static, array_moving, affine_moving, headers_static, headers_moving





def translation_sos(moving, static, tolerance=0.1):

    array_static, affine_static, array_moving, affine_moving, headers, _ = _get_input(moving, static)

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
    
    # Save results as new dicom series
    moving.message('Writing coregistered series to database..')
    desc = moving.instance().SeriesDescription
    coreg = moving.new_sibling(SeriesDescription = desc + ' [translation]')
    coreg.set_array(coregistered, headers, pixels_first=True)
    return coreg


def rigid_sos(moving, static, tolerance=0.1):

    array_static, affine_static, array_moving, affine_moving, headers, _ = _get_input(moving, static)

    # Define initial values and optimization
    _, _, static_pixel_spacing = vreg.affine_components(affine_static)
    rot_gradient_step, translation_gradient_step, _ = vreg.affine_resolution(array_static.shape, static_pixel_spacing)
    gradient_step = np.concatenate((1.0*rot_gradient_step, 0.5*translation_gradient_step))
    optimization = {
        'method': 'GD', 
        'options': {'gradient step': gradient_step, 'tolerance': tolerance}, 
        'callback': lambda vk: moving.message('Current parameter: ' + str(vk)),
    }
    
    # Align volumes
    try:
        rigid_estimate = vreg.align(
            moving = array_moving, 
            moving_affine = affine_moving, 
            static = array_static, 
            static_affine = affine_static, 
            parameters = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32), 
            resolutions = [4,2,1], 
            transformation =  vreg.rigid,
            metric = vreg.sum_of_squares,
            optimization = optimization,
        )
    except:
        print('Failed to align volumes..')
        return None

    coregistered = vreg.rigid(array_moving, affine_moving, array_static.shape, affine_static, rigid_estimate)
    
    moving.message('Writing coregistered series to database..')
    
    # Save results as new dicom series
    desc = moving.instance().SeriesDescription
    coreg = moving.new_sibling(SeriesDescription = desc + ' [rigid]')
    coreg.set_array(coregistered, headers, pixels_first=True)
    return coreg



def sbs_translation_sos(moving, static, tolerance=0.1):

    array_static, affine_static, array_moving, affine_moving, headers, headers_moving = _get_input(moving, static)
    slice_thickness = [headers_moving[z].SliceThickness for z in range(headers_moving.size)]

    # Define initial values and optimization
    _, _, static_pixel_spacing = vreg.affine_components(affine_static)
    optimization = {
        'method': 'GD', 
        'options': {'gradient step': static_pixel_spacing, 'tolerance': tolerance}, 
    #    'callback': lambda vk: moving.message('Current parameter: ' + str(vk)),
    }
    
    # Perform coregistration
    estimate = vreg.align_slice_by_slice(
        moving = array_moving, 
        static = array_static, 
        parameters =  np.array([0, 0, 0], dtype=np.float32), 
        moving_affine = affine_moving, 
        static_affine = affine_static, 
        transformation = vreg.translate,
        metric = vreg.sum_of_squares,
        optimization = optimization,
        resolutions = [4,2,1],
        slice_thickness = slice_thickness,
        progress = lambda z, nz: moving.progress(z, nz, 'Coregistering slice-by-slice using translations'),
    )
    
    # Calculate coregistered
    coregistered = vreg.transform_slice_by_slice(array_moving, affine_moving, array_static.shape, affine_static, estimate, vreg.translate, slice_thickness)
    # coregistered = vreg.transform_slice_by_slice(
    #     moving = array_moving, 
    #     moving_affine = affine_moving, 
    #     static_shape = array_static.shape, 
    #     static_affine = affine_static, 
    #     parameters = estimate, 
    #     transformation = vreg.translate, 
    #     slice_thickness = slice_thickness)

    # Save results as new dicom series
    moving.message('Writing coregistered series to database..')
    desc = moving.instance().SeriesDescription
    coreg = moving.new_sibling(SeriesDescription = desc + ' [sbs translation]')
    coreg.set_array(coregistered, headers, pixels_first=True)

    return coreg


def sbs_rigid_around_com_sos(moving, static, tolerance=0.1):

    array_static, affine_static, array_moving, affine_moving, headers, headers_moving = _get_input(moving, static)
    slice_thickness = [headers_moving[z].SliceThickness for z in range(headers_moving.size)]

    # Define initial values and optimization
    _, _, static_pixel_spacing = vreg.affine_components(affine_static)
    rot_gradient_step, translation_gradient_step, _ = vreg.affine_resolution(array_static.shape, static_pixel_spacing)
    gradient_step = np.concatenate((1.0*rot_gradient_step, 0.5*translation_gradient_step))
    optimization = {
        'method': 'GD', 
        'options': {'gradient step': gradient_step, 'tolerance': tolerance}, 
        #'callback': lambda vk: moving.message('Current parameter: ' + str(vk)),
    }

    # Perform coregistration
    estimate = vreg.align_slice_by_slice(
        moving = array_moving, 
        static = array_static, 
        parameters = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32), 
        moving_affine = affine_moving, 
        static_affine = affine_static, 
        transformation = vreg.rigid_around_com,
        metric = vreg.sum_of_squares,
        optimization = optimization,
        resolutions = [4,2,1],
        slice_thickness = slice_thickness,
        progress = lambda z, nz: moving.progress(z, nz, 'Coregistering slice-by-slice using rigid transformations'),
    )

    # The generic slice-by-slice transform does not work for center of mass rotations. 
    # Calculate rotation center and use rigid rotation around given center instead.
    estimate_center = [] 
    for z in range(len(estimate)):
        array_moving_z, affine_moving_z = vreg.extract_slice(array_moving, affine_moving, z, slice_thickness)
        center = estimate[z][3:] + vreg.center_of_mass(vreg.to_3d(array_moving_z), affine_moving_z)
        pars = np.concatenate((estimate[z][:3], center, estimate[z][3:]))
        estimate_center.append(pars)
    
    # Calculate coregistered (using rigid around known center)
    coregistered = vreg.transform_slice_by_slice(array_moving, affine_moving, array_static.shape, affine_static, estimate_center, vreg.rigid_around, slice_thickness)
    # coregistered = vreg.transform_slice_by_slice(
    #     moving = array_moving, 
    #     moving_affine = affine_moving, 
    #     static_shape = array_static.shape, 
    #     static_affine = affine_static, 
    #     parameters = estimate_center, 
    #     transformation = vreg.rigid_around, 
    #     slice_thickness = slice_thickness)

    # Save results as new dicom series
    moving.message('Writing coregistered series to database..')
    desc = moving.instance().SeriesDescription
    coreg = moving.new_sibling(SeriesDescription = desc + ' [sbs rigid com]')
    coreg.set_array(coregistered, headers, pixels_first=True)

    return coreg



def rigid_around_com_sos(moving, static, tolerance=0.1):

    array_static, affine_static, array_moving, affine_moving, headers, _ = _get_input(moving, static)

    # Define initial values and optimization
    _, _, static_pixel_spacing = vreg.affine_components(affine_static)
    rot_gradient_step, translation_gradient_step, _ = vreg.affine_resolution(array_static.shape, static_pixel_spacing)
    gradient_step = np.concatenate((1.0*rot_gradient_step, 0.5*translation_gradient_step))
    optimization = {
        'method': 'GD', 
        'options': {'gradient step': gradient_step, 'tolerance': tolerance}, 
        'callback': lambda vk: moving.message('Current parameter: ' + str(vk)),
    }
    
    # Align volumes
    try:
        rigid_estimate = vreg.align(
            moving = array_moving, 
            moving_affine = affine_moving, 
            static = array_static, 
            static_affine = affine_static, 
            parameters = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32), 
            resolutions = [4,2,1], 
            transformation =  vreg.rigid_around_com,
            metric = vreg.sum_of_squares,
            optimization = optimization,
        )
    except:
        print('Failed to align volumes..')
        return None

    coregistered = vreg.rigid_around_com(array_moving, affine_moving, array_static.shape, affine_static, rigid_estimate)
    
    moving.message('Writing coregistered series to database..')
    
    # Save results as new dicom series
    desc = moving.instance().SeriesDescription
    coreg = moving.new_sibling(SeriesDescription = desc + ' [rigid com]')
    coreg.set_array(coregistered, headers, pixels_first=True)
    return coreg