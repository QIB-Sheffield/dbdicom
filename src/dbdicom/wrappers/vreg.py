import numpy as np
from dbdicom.utils import vreg
from dbdicom import Series

def print_current(vk):
    print(vk)


def _get_input_volume(series):
    if series is None:
        return None, None, None
    desc = series.instance().SeriesDescription
    affine = series.affine_matrix()
    if isinstance(affine, list):
        msg = 'This function only works for series with a single slice group. \n'
        msg += 'Multiple slice groups detected in ' + desc + ' - please split the series first.'
        raise ValueError(msg)
    else:
        affine = affine[0]
    array, headers = series.array('SliceLocation', pixels_first=True, first_volume=True)
    if array is None:
        msg = 'Series ' + desc + ' is empty - cannot perform alignment.'
        raise ValueError(msg)  
    return array, affine, headers


def _get_input(moving, static, region=None, margin=0):

    array_moving, affine_moving, headers_moving = _get_input_volume(moving)
    array_static, affine_static, headers_static = _get_input_volume(static)
    
    moving.message('Performing coregistration. Please be patient. Its hard work and I need to concentrate..')
    
    # If a region is provided, use it extract a bounding box around the static array
    if region is not None:
        array_region, affine_region, _ = _get_input_volume(region)
        array_static, affine_static = vreg.mask_volume(array_static, affine_static, array_region, affine_region, margin)
    
    return array_static, affine_static, array_moving, affine_moving, headers_static, headers_moving




def find_translation(moving:Series, static:Series, tolerance=0.1, metric='mutual information', region:Series=None, margin:float=0)->np.ndarray:
    """Find the translation that maps a moving volume onto a static volume.

    Args:
        moving (dbdicom.Series): Series with the moving volume.
        static (dbdicom.Series): Series with the static volume
        tolerance (float, optional): Positive tolerance parameter to decide convergence of the gradient descent. A smaller value means a more accurate solution but also a lomger computation time. Defaults to 0.1.
        metric (str, option): Determines which metric to use in the optimization. Current options are 'mutual information' (default) or 'sum of squares'.
        region (dbdicom.Series, optional): Series with region of interest to restrict the alignment. The translation will be chosen based on the goodness of the alignment in the bounding box of this region. If none is provided, the entire volume is used. Defaults to None.
        margin (float, optional): in case a region is provided, this specifies a margin (in mm) to take around the region. Default is 0 (no margin)

    Returns:
        np.ndarray: 3-element numpy array with values of the translation that maps the moving volume on to the static volume.
    """

    array_static, affine_static, array_moving, affine_moving, _, _ = _get_input(moving, static, region=region, margin=margin)

    # Define initial values and optimization
    _, _, static_pixel_spacing = vreg.affine_components(affine_static)
    optimization = {
        'method': 'GD', 
        'options': {'gradient step': static_pixel_spacing, 'tolerance': tolerance},
        'callback': lambda vk: moving.status.message('Current parameter: ' + str(vk)),
    }
    func = {
        'sum of squares': vreg.sum_of_squares,
        'mutual information': vreg.mutual_information,
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
            metric = func[metric],
            optimization = optimization,
        )
    except:
        print('Failed to align volumes..')
        translation_estimate = None

    return translation_estimate


def apply_translation(series_moving:Series, parameters:np.ndarray, target:Series=None, description:str=None)->Series:
    """Apply active translation of an image volume.

    Args:
        series_moving (dbdicom.Series): Series containing the volune to be moved.
        parameters (np.ndarray): three-element numpy array with coordinates of the translation in the absolute reference frame (mm).
        target (dbdicom.Series, optional): If provided, the result is mapped onto the geometry of this series. If none is provided, the result has the same geometry of the moving series. Defaults to None.

    Raises:
        ValueError: If the moving series contains multiple slice groups with different orientations. 
        ValueError: If the array to be moved is empty.

    Returns:
        dbdicom.Series: Sibling dbdicom series in the same study, containing the translated volume.
    """

    desc_moving = series_moving.instance().SeriesDescription
    affine_moving = series_moving.affine()
    if len(affine_moving) > 1:
        msg = 'Multiple slice groups detected in ' + desc_moving + '\n'
        msg += 'This function only works for series with a single slice group. \n'
        msg += 'Please split the series first.'
        raise ValueError(msg)
    else:
        affine_moving = affine_moving[0]

    array_moving = series_moving.ndarray(dims=('SliceLocation',))
    if array_moving.size == 0:
        msg = desc_moving + ' is empty - cannot perform alignment.'
        raise ValueError(msg)
    
    if target is None:
        shape_moved = array_moving.shape
        affine_moved = affine_moving
    else:
        array_moved = target.ndarray(dims=('SliceLocation',))
        shape_moved = array_moved.shape
        affine_moved = target.affine()[0]

    series_moving.message('Applying translation..')   
    if description is None:
        description = desc_moving + ' [translation]'
    array_moved = vreg.translate(array_moving, affine_moving, shape_moved, affine_moved, parameters)
    series_moved = series_moving.new_sibling(SeriesDescription=description)
    series_moved.set_ndarray(array_moved, coords={'SliceLocation': np.arange(array_moved.shape[-1])})
    series_moved.set_affine(affine_moved)
    return series_moved


def find_rigid_transformation(moving:Series, static:Series, tolerance=0.1, metric='mutual information', region:Series=None, margin:float=0)->np.ndarray:
    """Find the rigid transform that maps a moving volume onto a static volume.

    Args:
        moving (dbdicom.Series): Series with the moving volume.
        static (dbdicom.Series): Series with the static volume
        tolerance (float, optional): Positive tolerance parameter to decide convergence of the gradient descent. A smaller value means a more accurate solution but also a lomger computation time. Defaults to 0.1.
        metric (str, option): Determines which metric to use in the optimization. Current options are 'mutual information' (default) or 'sum of squares'.
        region (dbdicom.Series, optional): Series with region of interest to restrict the alignment. The rigid transform will be chosen based on the goodness of the alignment in the bounding box of this region. If none is provided, the entire volume is used. Defaults to None.
        margin (float, optional): in case a region is provided, this specifies a margin (in mm) to take around the region. Default is 0 (no margin).

    Returns:
        np.ndarray: 6-element numpy array with values of the translation (first 3 elements) and rotation vector (last 3 elements) that map the moving volume on to the static volume. The vectors are defined in an absolute reference frame in units of mm.
    """

    array_static, affine_static, array_moving, affine_moving, _, _ = _get_input(moving, static, region=region, margin=margin)

    # Define initial values and optimization
    _, _, static_pixel_spacing = vreg.affine_components(affine_static)
    rot_gradient_step, translation_gradient_step, _ = vreg.affine_resolution(array_static.shape, static_pixel_spacing)
    gradient_step = np.concatenate((1.0*rot_gradient_step, 0.5*translation_gradient_step))
    optimization = {
        'method': 'GD', 
        'options': {'gradient step': gradient_step, 'tolerance': tolerance}, 
        'callback': lambda vk: moving.message('Current parameter: ' + str(vk)),
    }
    func = {
        'sum of squares': vreg.sum_of_squares,
        'mutual information': vreg.mutual_information,
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
            metric = func[metric],
            optimization = optimization,
        )
    except:
        print('Failed to align volumes..')
        rigid_estimate = None

    return rigid_estimate


def apply_rigid_transformation(series_moving:Series, parameters:np.ndarray,  target:Series=None, description:str=None)->Series:
    """Apply rigid transformation of an image volume.

    Args:
        series_moving (dbdicom.Series): Series containing the volune to be moved.
        parameters (np.ndarray): 6-element numpy array with values of the translation (first 3 elements) and rotation vector (last 3 elements) that map the moving volume on to the static volume. The vectors are defined in an absolute reference frame in units of mm.
        target (dbdicom.Series, optional): If provided, the result is mapped onto the geometry of this series. If none is provided, the result has the same geometry of the moving series. Defaults to None.

    Raises:
        ValueError: If the moving series contains multiple slice groups with different orentations. 
        ValueError: If the array to be moved is empty.

    Returns:
        dbdicom.Series: Sibling dbdicom series in the same study, containing the transformed volume.
    """
    # TODO: target is not the right word. geometry?
    desc_moving = series_moving.instance().SeriesDescription
    affine_moving = series_moving.affine()
    if len(affine_moving) > 1:
        msg = 'Multiple slice groups detected in ' + desc_moving + '\n'
        msg += 'This function only works for series with a single slice group. \n'
        msg += 'Please split the series first.'
        raise ValueError(msg)
    else:
        affine_moving = affine_moving[0]

    array_moving = series_moving.ndarray(dims=('SliceLocation',))
    if array_moving.size == 0:
        msg = desc_moving + ' is empty - cannot perform alignment.'
        raise ValueError(msg)
    
    if target is None:
        shape_moved = array_moving.shape
        affine_moved = affine_moving
    else:
        array_moved = target.ndarray(dims=('SliceLocation',))
        shape_moved = array_moved.shape
        affine_moved = target.affine()[0]

    series_moving.message('Applying rigid transformation..')
    array_moved = vreg.rigid(array_moving, affine_moving, shape_moved, affine_moved, parameters)
    if description is None:
        description = desc_moving + ' [rigid]'
    series_moved = series_moving.new_sibling(SeriesDescription = description)
    series_moved.set_ndarray(array_moved, coords={'SliceLocation': np.arange(array_moved.shape[-1])})
    series_moved.set_affine(affine_moved)
    return series_moved



def find_sbs_translation(moving:Series, static:Series, tolerance=0.1, metric='mutual information', region:Series=None, margin:float=0)->np.ndarray:
    """Find the slice-by-slice translation that maps a moving volume onto a static volume.

    Args:
        moving (dbdicom.Series): Series with the moving volume.
        static (dbdicom.Series): Series with the static volume
        tolerance (float, optional): Positive tolerance parameter to decide convergence of the gradient descent. A smaller value means a more accurate solution but also a lomger computation time. Defaults to 0.1.
        metric (str, option): Determines which metric to use in the optimization. Current options are 'mutual information' (default) or 'sum of squares'.
        region (dbdicom.Series, optional): Series with region of interest to restrict the alignment. The translation will be chosen based on the goodness of the alignment in the bounding box of this region. If none is provided, the entire volume is used. Defaults to None.
        margin (float, optional): in case a region is provided, this specifies a margin (in mm) to take around the region. Default is 0 (no margin).

    Returns:
        np.ndarray: list of 3-element numpy arrays with values of the translation that maps the moving volume onto the static volume. The list has one entry per slice of the volume.
    """

    array_static, affine_static, array_moving, affine_moving, _, headers_moving = _get_input(moving, static, region=region, margin=margin)
    slice_thickness = [headers_moving[z].SliceThickness for z in range(headers_moving.size)]

    # Define initial values and optimization
    _, _, static_pixel_spacing = vreg.affine_components(affine_static)
    optimization = {
        'method': 'GD', 
        'options': {'gradient step': static_pixel_spacing, 'tolerance': tolerance}, 
    #    'callback': lambda vk: moving.message('Current parameter: ' + str(vk)),
    }
    func = {
        'sum of squares': vreg.sum_of_squares,
        'mutual information': vreg.mutual_information,
    }
    
    # Perform coregistration
    try:
        translation_estimate = vreg.align_slice_by_slice(
            moving = array_moving, 
            moving_affine = affine_moving, 
            static = array_static, 
            static_affine = affine_static, 
            parameters =  np.array([0, 0, 0], dtype=np.float32), 
            resolutions = [4,2,1],
            transformation = vreg.translate,
            metric = func[metric],
            optimization = optimization,
            slice_thickness = slice_thickness,
            progress = lambda z, nz: moving.progress(z+1, nz, 'Coregistering slice-by-slice using translations'),
        )
    except:
        print('Failed to align volumes..')
        translation_estimate = None

    return translation_estimate


def apply_sbs_translation(series_moving:Series, parameters:np.ndarray, target:Series=None)->Series:
    """Apply slice-by-slice translation of an image volume.

    Args:
        series_moving (dbdicom.Series): Series containing the volune to be moved.
        parameters (np.ndarray): list of 3-element numpy arrays with values of the translation that maps the moving volume onto the static volume. The list has one entry per slice of the volume.
        target (dbdicom.Series, optional): If provided, the result is mapped onto the geometry of this series. If none is provided, the result has the same geometry of the moving series. Defaults to None.

    Raises:
        ValueError: If the moving series contains multiple slice groups with different orientations. 
        ValueError: If the array to be moved is empty.

    Returns:
        dbdicom.Series: Sibling dbdicom series in the same study, containing the translated volume.
    """
    desc_moving = series_moving.instance().SeriesDescription
    affine_moving = series_moving.affine()
    if len(affine_moving) > 1:
        msg = 'Multiple slice groups detected in ' + desc_moving + '\n'
        msg += 'This function only works for series with a single slice group. \n'
        msg += 'Please split the series first.'
        raise ValueError(msg)
    else:
        affine_moving = affine_moving[0]

    array_moving, headers_moving = series_moving.array('SliceLocation', pixels_first=True, first_volume=True)
    if array_moving.size == 0:
        msg = desc_moving + ' is empty - cannot perform alignment.'
        raise ValueError(msg)
    slice_thickness = [headers_moving[z].SliceThickness for z in range(headers_moving.size)]

    if target is None:
        shape_moved = array_moving.shape
        affine_moved = affine_moving
    else:
        array_moved = target.ndarray(dims=('SliceLocation',))
        shape_moved = array_moved.shape
        affine_moved = target.affine()[0]

    series_moving.message('Applying slice-by-slice translation..')
    array_moved = vreg.transform_slice_by_slice(array_moving, affine_moving, shape_moved, affine_moved, parameters, vreg.translate, slice_thickness)
    series_moved = series_moving.new_sibling(SeriesDescription = desc_moving + ' [sbs translation]')
    series_moved.set_ndarray(array_moved, coords={'SliceLocation': np.arange(array_moved.shape[-1])})
    series_moved.set_affine(affine_moved)
    return series_moved


def find_sbs_rigid_transformation(moving:Series, static:Series, tolerance=0.1, metric='mutual information', region:Series=None, margin:float=0, moving_mask:Series=None, static_mask:Series=None, resolutions=[4,2,1])->np.ndarray:
    """Find the slice-by-slice rigid transformation that maps a moving volume onto a static volume.

    Args:
        moving (dbdicom.Series): Series with the moving volume.
        static (dbdicom.Series): Series with the static volume
        tolerance (float, optional): Positive tolerance parameter to decide convergence of the gradient descent. A smaller value means a more accurate solution but also a lomger computation time. Defaults to 0.1.
        metric (str, option): Determines which metric to use in the optimization. Current options are 'mutual information' (default) or 'sum of squares'.
        region (dbdicom.Series, optional): Series with region of interest to restrict the alignment. The translation will be chosen based on the goodness of the alignment in the bounding box of this region. If none is provided, the entire volume is used. Defaults to None.
        margin (float, optional): in case a region is provided, this specifies a margin (in mm) to take around the region. Default is 0 (no margin).
        moving_mask (dbdicom.Series): Series for masking the moving volume.
        static_mask (dbdicom.Series): Series for masking the static volume.

    Returns:
        np.ndarray: list of 6-element numpy arrays with values of the translation (first 3 elements) and rotation vector (last 3 elements) that map the moving volume on to the static volume. The list contains one entry per slice, ordered by slice location. The vectors are defined in an absolute reference frame in units of mm.
    """

    array_static, affine_static, array_moving, affine_moving, _, headers_moving = _get_input(moving, static, region=region, margin=margin)
    #array_static_mask, affine_static_mask, array_moving_mask, affine_moving_mask, _, _ = _get_input_volumes(moving_mask, static_mask)
    array_moving_mask, affine_moving_mask, _ = _get_input_volume(moving_mask)
    array_static_mask, affine_static_mask, _ = _get_input_volume(static_mask)

    slice_thickness = [headers_moving[z].SliceThickness for z in range(headers_moving.size)]

    # Define initial values and optimization
    _, _, static_pixel_spacing = vreg.affine_components(affine_static)
    rot_gradient_step, translation_gradient_step, _ = vreg.affine_resolution(array_static.shape, static_pixel_spacing)
    gradient_step = np.concatenate((1.0*rot_gradient_step, 0.5*translation_gradient_step))
    optimization = {
        'method': 'GD', 
        'options': {'gradient step': gradient_step, 'tolerance': tolerance}, 
    }
    func = {
        'sum of squares': vreg.sum_of_squares,
        'mutual information': vreg.mutual_information,
    }
    
    # Perform coregistration
    try:
        parameters = vreg.align_slice_by_slice(
            moving = array_moving, 
            moving_affine = affine_moving, 
            static = array_static, 
            static_affine = affine_static, 
            parameters = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32), 
            resolutions = resolutions,
            transformation = vreg.rigid,
            metric = func[metric],
            optimization = optimization,
            slice_thickness = slice_thickness,
            progress = lambda z, nz: moving.progress(z+1, nz, 'Coregistering slice-by-slice using rigid transformations'),
            static_mask = array_static_mask,
            static_mask_affine = affine_static_mask,
            moving_mask = array_moving_mask,
            moving_mask_affine = affine_moving_mask,
        )
    except:
        print('Failed to align volumes..')
        parameters = None

    return parameters


def apply_sbs_passive_rigid_transformation(series_moving:Series, parameters:np.ndarray)->Series:
    """Apply slice-by-slice passive rigid transformation of an image volume.

    Passive in this context means that the coordinates are transformed rather than the image array itself.

    Args:
        series_moving (dbdicom.Series): Series containing the volune to be moved.
        parameters (np.ndarray): 6-element numpy array with values of the translation (first 3 elements) and rotation vector (last 3 elements) that map the moving volume on to the static volume. The list contains one entry per slice, ordered by slice location. The vectors are defined in an absolute reference frame in units of mm.

    Raises:
        ValueError: If the moving series contains multiple slice groups with different orientations. 
        ValueError: If the array to be moved is empty.

    Returns:
        dbdicom.Series: Sibling dbdicom series in the same study, containing the translated volume.
    """
    desc_moving = series_moving.instance().SeriesDescription
    affine_moving = series_moving.affine()
    if len(affine_moving) > 1:
        msg = 'Multiple slice groups detected in ' + desc_moving + '\n'
        msg += 'This function only works for series with a single slice group. \n'
        msg += 'Please split the series first.'
        raise ValueError(msg)
    else:
        affine_moving = affine_moving[0]

    _, headers_moving = series_moving.array('SliceLocation', pixels_first=True, first_volume=True)

    series_moving.message('Applying slice-by-slice passive rigid transformation..')
    output_affine = vreg.passive_rigid_transform_slice_by_slice(affine_moving, parameters)
    series_moved = series_moving.new_sibling(SeriesDescription = desc_moving + ' [sbs passive rigid]')
    for z in range(headers_moving.size):
        imz = headers_moving[z].copy_to(series_moved)
        affine_z = output_affine[z]
        affine_z[:3,2] *= imz.SliceThickness/np.linalg.norm(affine_z[:3,2])
        imz.set_affine(affine_z)
    return series_moved


def apply_sbs_rigid_transformation(series_moving:Series, parameters:np.ndarray, target:Series=None)->Series:
    """Apply slice-by-slice rigid transformation of an image volume.

    Args:
        series_moving (dbdicom.Series): Series containing the volune to be moved.
        parameters (np.ndarray): 6-element numpy array with values of the translation (first 3 elements) and rotation vector (last 3 elements) that map the moving volume on to the static volume. The list contains one entry per slice, ordered by slice location. The vectors are defined in an absolute reference frame in units of mm.
        target (dbdicom.Series, optional): If provided, the result is mapped onto the geometry of this series. If none is provided, the result has the same geometry as the moving series. Defaults to None.

    Raises:
        ValueError: If the moving series contains multiple slice groups with different orientations. 
        ValueError: If the array to be moved is empty.

    Returns:
        dbdicom.Series: Sibling dbdicom series in the same study, containing the translated volume.
    """
    desc_moving = series_moving.instance().SeriesDescription
    affine_moving = series_moving.affine()
    if len(affine_moving) > 1:
        msg = 'Multiple slice groups detected in ' + desc_moving + '\n'
        msg += 'This function only works for series with a single slice group. \n'
        msg += 'Please split the series first.'
        raise ValueError(msg)
    else:
        affine_moving = affine_moving[0]

    array_moving, headers_moving = series_moving.array('SliceLocation', pixels_first=True, first_volume=True)
    if array_moving.size == 0:
        msg = desc_moving + ' is empty - cannot perform alignment.'
        raise ValueError(msg)
    slice_thickness = [headers_moving[z].SliceThickness for z in range(headers_moving.size)]

    if target is None:
        shape_moved = array_moving.shape
        affine_moved = affine_moving
    else:
        array_moved = target.ndarray(dims=('SliceLocation',))
        shape_moved = array_moved.shape
        affine_moved = target.affine()[0]

    series_moving.message('Applying slice-by-slice rigid transformation..')
    array_moved = vreg.transform_slice_by_slice(array_moving, affine_moving, shape_moved, affine_moved, parameters, vreg.rigid, slice_thickness)
    series_moved = series_moving.new_sibling(SeriesDescription = desc_moving + ' [sbs rigid]')
    series_moved.set_ndarray(array_moved, coords={'SliceLocation': np.arange(array_moved.shape[-1])})
    series_moved.set_affine(affine_moved)
    return series_moved






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

    # Save results as new dicom series
    moving.message('Writing coregistered series to database..')
    desc = moving.instance().SeriesDescription
    coreg = moving.new_sibling(SeriesDescription = desc + ' [sbs rigid com]')
    coreg.set_array(coregistered, headers, pixels_first=True)

    return coreg