# pip install SimpleITK-SimpleElastix

import numpy as np
import SimpleITK as sitk
import dbdicom.wrappers.scipy as scipy


def invert_deformation_field(deformation_field, **kwargs):

    # Get arrays for deformation_field
    deform, headers = deformation_field.array('SliceLocation', pixels_first=True)

    # Raise an error if the array is empty
    if deform is None:
        msg = 'The deformation field is an empty series. \n'
        msg += 'Please select a valid series and try again.'
        raise ValueError(msg)
   
    # Calculate the inverse
    deformation_field.status.message('Calculating inverse..')
    deform_inv = _invert_deformation_field(deform, **kwargs)

    # Return as new series
    inv = deformation_field.new_sibling(suffix='inverse')
    inv.set_array(deform_inv, headers, pixels_first=True)
    return inv


def coregister_3d_to_3d(moving, fixed, 
        transformation = 'Affine',
        metric = "NormalizedMutualInformation",
        final_grid_spacing = 1.0,
        _ignore_empty_slices = False, # Do not use for now
    ):

    nan = 2**16-1
    fixed_map = scipy.map_to(fixed, moving, cval=nan)

    # Get arrays for fixed and moving series
    array_fixed, _ = fixed_map.array('SliceLocation', pixels_first=True, first_volume=True)
    array_moving, headers_moving  = moving.array('SliceLocation', pixels_first=True, first_volume=True)
    
    # If one of the datasets is empty do nothing
    if array_fixed is None or array_moving is None:
        return fixed_map
    
    # Remove temporary overlay
    if fixed_map != fixed:
        fixed_map.remove()

    # # If time series consider only first time point
    # array_fixed = array_fixed[...,0]
    # array_moving = array_moving[...,0]
    # headers_moving = headers_moving[...,0]

    # Apply coregistration settings
    if transformation == 'Rigid':
        pars = _default_rigid()  
    elif transformation == 'Affine':
        pars = _default_affine() 
    else:
        pars = _default_bspline()

    pars["Metric"] = [metric]
    pars["FinalGridSpacingInPhysicalUnits"] = [str(final_grid_spacing)]
    pars["FixedImageDimension"] = ['3']
    pars["MovingImageDimension"] = ['3']

    # Coregister fixed and moving slice-by-slice
    moving.status.message('Performing coregistration..')
    #deformation = np.empty(array_moving.shape + (2,))
    ind_fixed = np.where(array_fixed==nan)
    array_fixed[ind_fixed] = 0
    array_moving[ind_fixed] = 0

    # Don't use for now
    # This may lead to an error of too many samples outside moving image buffer
    if _ignore_empty_slices:
        fixed_mask = _coregistration_mask_3d(array_fixed)
        moving_mask = _coregistration_mask_3d(array_moving)
    else:
        fixed_mask = None
        moving_mask = None

    # get this from series affine instead - more robust
    slice_spacing = headers_moving[0].SpacingBetweenSlices # needs a custom keyword slice_spacing
    if slice_spacing is None:
        slice_spacing = headers_moving[0].SliceThickness
    pixel_spacing = headers_moving[0].PixelSpacing + [slice_spacing]

    coregistered, deformation = _coregister_arrays(array_fixed, array_moving, pars, pixel_spacing, pixel_spacing, fixed_mask=fixed_mask, moving_mask=moving_mask)

    # Return new series
    coreg = moving.new_sibling(suffix='coregistered')
    deform = moving.new_sibling(suffix='deformation field')

    coreg.set_array(coregistered, headers_moving, pixels_first=True)
    for dim in range(deformation.shape[-1]):
        deform.set_array(deformation[...,dim], headers_moving, pixels_first=True)
    # deform_size = moving.new_sibling(SeriesDescription = desc + ' [deformation]')
    # deform_size.set_array( np.linalg.norm(deformation, axis=-1), headers_moving, pixels_first=True)
    moving.status.message('Finished coregistration..')
    return coreg, deform


def _coregistration_mask_3d(array):
    mask = np.zeros(array.shape, np.uint8)
    for z in range(array.shape[2]):
        if np.count_nonzero(array[:,:,z]) > 0:
            mask[:,:,z] = 1
    return mask


def coregister_2d_to_2d(moving, fixed, 
        transformation = 'Affine',
        metric = "NormalizedMutualInformation",
        final_grid_spacing = 1.0,
    ):

    background = 2**16-1
    fixed_map = scipy.map_to(fixed, moving, cval=background)

    # Get arrays for fixed and moving series
    array_fixed, _ = fixed_map.array('SliceLocation', pixels_first=True, first_volume=True)
    array_moving, headers_moving  = moving.array('SliceLocation', pixels_first=True, first_volume=True)
    
    # Raise an error if one of the datasets is empty
    if array_fixed is None or array_moving is None:
        msg = 'One of the series is empty. \n'
        msg += 'Please select a non-empty series and try again.'
        raise ValueError(msg)
    
    # Remove temporary overlay
    if fixed_map != fixed:
        fixed_map.remove()

    # If time series consider only first time point
    # array_fixed = array_fixed[...,0]
    # array_moving = array_moving[...,0]
    # headers_moving = headers_moving[...,0]

    # Set background pixels to zero for both images
    idx = np.where(array_fixed==background)
    array_fixed[idx] = 0
    array_moving[idx] = 0

    # Get coregistration settings
    if transformation == 'Rigid':
        pars = _default_rigid()  
    elif transformation == 'Affine':
        pars = _default_affine() 
    else:
        pars = _default_bspline()
    pars["Metric"] = [metric]
    pars["FinalGridSpacingInPhysicalUnits"] = [str(final_grid_spacing)]
    pars["FixedImageDimension"] = ['2']
    pars["MovingImageDimension"] = ['2']
   
    # Coregister fixed and moving slice-by-slice
    deformation = np.empty(array_moving.shape + (2,))
    pixel_spacing = headers_moving[0].PixelSpacing
    for z in range(array_moving.shape[2]):
        moving.status.progress(z+1, array_moving.shape[2], 'Performing coregistration..')
        coreg, deform = _coregister_arrays(array_fixed[:,:,z], array_moving[:,:,z], pars, pixel_spacing, pixel_spacing)
        deformation[:,:,z,:] = deform
        array_moving[:,:,z] = coreg

    # Create new series
    coreg = moving.new_sibling(suffix='coregistered')
    deform = moving.new_sibling(suffix='deformation field')

    # Save data
    coreg.set_array(array_moving, headers_moving, pixels_first=True)
    for dim in range(deformation.shape[-1]):
        deform.set_array(deformation[...,dim], headers_moving, pixels_first=True)
    deform[['WindowCenter', 'WindowWidth']] = [0, 10]

    # Return coregistered image and deformation field
    return coreg, deform


# ONLY TESTED FOR RIGID TRANSFORMATION
# AFFINE AND DEFORMABLE DOES NOT WORK
def coregister_3d_to_2d(moving_3d, fixed_2d,  # moving=3D, fixed=2D
        transformation = 'Affine',
        metric = "NormalizedMutualInformation",
        final_grid_spacing = 1.0,
    ):

    nan = 2**16-1
    moving_map = scipy.map_to(moving_3d, fixed_2d, cval=nan)

    # Get arrays for fixed and moving series
    array_fixed, headers_fixed = fixed_2d.array('SliceLocation', pixels_first=True, first_volume=True)
    array_moving, _ = moving_map.array('SliceLocation', pixels_first=True, first_volume=True)
    
    # If one of the datasets is empty do nothing
    if array_fixed is None or array_moving is None:
        return moving_map
    
    # Remove temporary overlay
    if moving_map != moving_3d:
        moving_map.remove()

    # # If time series consider only first time point
    # array_fixed = array_fixed[...,0]
    # array_moving = array_moving[...,0]
    # headers_fixed = headers_fixed[...,0]

    # Get coregistration settings
    if transformation == 'Rigid':
        pars = _default_rigid()  
    elif transformation == 'Affine':
        pars = _default_affine() 
    else:
        pars = _default_bspline()

    pars["Metric"] = [metric]
    pars["FinalGridSpacingInPhysicalUnits"] = [str(final_grid_spacing)]
    pars["FixedImageDimension"] = ['3'] # 2D image must be entered as 3D with 3d dimension = 1
    pars["MovingImageDimension"] = ['3']
   
    # Coregister fixed and moving slice-by-slice
    deformation = np.empty(array_moving.shape + (3,))
    ind_nan = np.where(array_fixed==nan)
    array_fixed[ind_nan] = 0
    array_moving[ind_nan] = 0
    pixel_spacing = headers_fixed[0].PixelSpacing
    slice_spacing = headers_fixed[0].SpacingBetweenSlices # needs a custom keyword slice_spacing
    if slice_spacing is None:
        slice_spacing = headers_fixed[0].SliceThickness
    spacing = pixel_spacing + [slice_spacing]
    for z in range(array_fixed.shape[2]):
        moving_3d.status.progress(z+1, array_fixed.shape[2], 'Performing coregistration..')
        fixed = array_fixed[:,:,z].reshape(array_fixed.shape[:2]+(1,)) # enter as 3d with 3d dim=1
        coreg, deform = _coregister_arrays(fixed, array_moving, pars, spacing, spacing)
        deformation[:,:,z,:] = np.squeeze(deform) # remove z-dimension of 1 again 
        array_fixed[:,:,z] = np.squeeze(coreg)

    # Return new series
    coreg = moving_3d.new_sibling(suffix='coregistered')
    deform = moving_3d.new_sibling(suffix='deformation field')

    coreg.set_array(array_fixed, headers_fixed, pixels_first=True)
    for dim in range(deformation.shape[-1]):
        deform.set_array(deformation[...,dim], headers_fixed, pixels_first=True)
    return coreg, deform


# THIS DOES NOT WORK
def coregister_2d_to_3d(moving_2d, fixed_3d,  # moving=2D, fixed=3D
        transformation = 'Affine',
        metric = "NormalizedMutualInformation",
        final_grid_spacing = 1.0,
    ):

    nan = 2**16-1
    fixed_map = scipy.map_to(fixed_3d, moving_2d, cval=nan)

    # Get arrays for fixed and moving series
    array_fixed, _ = fixed_map.array('SliceLocation', pixels_first=True, first_volume=True)
    array_moving, headers_moving = moving_2d.array('SliceLocation', pixels_first=True, first_volume=True)
    
    # If one of the datasets is empty do nothing
    if array_fixed is None or array_moving is None:
        return fixed_map
    
    # Remove temporary overlay
    if fixed_map != fixed_3d:
        fixed_map.remove()

    # # If time series consider only first time point
    # array_fixed = array_fixed[...,0]
    # array_moving = array_moving[...,0]
    # headers_moving = headers_moving[...,0]

    # Get coregistration settings
    if transformation == 'Rigid':
        #pars = _default_rigid() 
        pars = _default_2d_to_3d() 
    elif transformation == 'Affine':
        #pars = _default_affine() 
        pars = _default_2d_to_3d()
    else:
        #pars = _default_bspline()
        pars = _default_2d_to_3d()

    pars["Metric"] = [metric]
    pars["FinalGridSpacingInPhysicalUnits"] = [str(final_grid_spacing)]
    pars["FixedImageDimension"] = ['3'] 
    pars["MovingImageDimension"] = ['3'] # 2D image must be entered as 3D with 3d dimension = 1
   
    # Coregister fixed and moving slice-by-slice
    deformation = np.empty(array_moving.shape + (3,))
    ind_nan = np.where(array_fixed==nan)
    array_fixed[ind_nan] = 0
    array_moving[ind_nan] = 0
    pixel_spacing = headers_moving[0].PixelSpacing
    slice_spacing = headers_moving[0].SpacingBetweenSlices 
    if slice_spacing is None:
        slice_spacing = headers_moving[0].SliceThickness
    spacing = pixel_spacing + [slice_spacing]
    for z in range(array_moving.shape[2]):
        moving_2d.status.progress(z+1, array_moving.shape[2], 'Performing coregistration..')
        moving = array_moving[:,:,z].reshape(array_moving.shape[:2]+(1,)) # enter as 3d with 3d dim=1
        coreg, deform = _coregister_arrays(array_fixed, moving, pars, spacing, spacing)
        deformation[:,:,z,:] = np.squeeze(deform) # remove z-dimension of 1 again 
        array_fixed[:,:,z] = np.squeeze(coreg)

    # Create new series
    coreg = moving_2d.new_sibling(suffix='coregistered')
    deform = moving_2d.new_sibling(suffix='deformation field')

    # Set arrays
    coreg.set_array(array_fixed, headers_moving, pixels_first=True)
    for dim in range(deformation.shape[-1]):
        deform.set_array(deformation[...,dim], headers_moving, pixels_first=True)
    
    # return coregistered image and deformation field
    return coreg, deform


def warp(image, deformation_field):

    # Get arrays for image and deformation field
    array, headers = image.array('SliceLocation', pixels_first=True, first_volume=True)
    array_deform, _ = deformation_field.array('SliceLocation', pixels_first=True)
    
    # Warp array with deformation field
    image.message('Warping image with deformation field..')
    array = _warp_arrays(array, array_deform)

    # Return as dbdicom series
    warped = image.new_sibling(suffix='warped')
    warped.set_array(array, headers, pixels_first=True)
    return warped





# BELOW HERE NON-DICOM FUNCTIONALITY


def _invert_deformation_field(deform: np.ndarray, smooth=False) -> np.ndarray:
    # deform must have shape (x,y,z,ndim) with ndim=2 or 3
    ndim = deform.shape[-1]
    if ndim==3:
        return _invert_deformation_field_volume(deform, smooth=smooth)
    elif ndim==2:
        nslices = deform.shape[2]
        for z in range(nslices):
            deform[:,:,z,:] = _invert_deformation_field_volume(deform[:,:,z,:], smooth=smooth)
        return deform
    else:
        msg = 'The deformation field must have either 2 or 3 dimensions'
        raise ValueError(msg)
    

def _invert_deformation_field_volume(deform: np.ndarray, smooth=False) -> np.ndarray:
    deform = sitk.GetImageFromArray(deform, isVector=True)
    if smooth:
        filter = sitk.InvertDisplacementFieldImageFilter()
        filter.EnforceBoundaryConditionOn()
        deform_inv = filter.Execute(deform)
    else:
        deform_inv = sitk.InverseDisplacementField(deform,
            size = deform.GetSize(),
            outputOrigin = (0.0, 0.0),
            outputSpacing = (1.0, 1.0))
    return sitk.GetArrayFromImage(deform_inv)


def _warp_arrays(array, deformation_field):

    # For this function, image and deformation field must be aligned
    if array.shape != deformation_field.shape[:-1]:
        msg = 'The dimensions of image and deformation field are not matching up. \n'
        msg += 'Please select two series with matching dimensions.'
        raise ValueError(msg)

    ndim = deformation_field.shape[-1]
    if ndim == 3:
        return _warp_volume(array, deformation_field)

    # if the deformation field is 2D, then loop over the slices
    elif ndim == 2:
        nslices = deformation_field.shape[2]
        for z in range(nslices):
            array[:,:,z] = _warp_volume(array[:,:,z], deformation_field[:,:,z,:])
        return array

    # Raise an error if the deformation field does not have 2 or 3 components
    else:
        msg = 'The deformation field must have either 2 or 3 dimensions'
        raise ValueError(msg)



def _warp_volume(volume, displacement):
    # Create an image from the volume numpy array
    volume_sitk = sitk.GetImageFromArray(volume)

    # Create a displacement field image from the displacement numpy array
    displacement_sitk = sitk.GetImageFromArray(displacement, isVector=True)
    displacement_sitk = sitk.Cast(displacement_sitk, sitk.sitkVectorFloat64)

    # Set up the transformation object
    displacement_transform = sitk.DisplacementFieldTransform(displacement_sitk)

    # Warp the volume using the displacement field transform
    warped_sitk = sitk.Resample(volume_sitk, displacement_transform)

    # Convert the warped image to a numpy array
    warped = sitk.GetArrayFromImage(warped_sitk)

    return warped


# Would prefer to use Transformix for warping but this does not work
def _wip_warp_volume(array, deformation_field):

    # Convert the numpy arrays to a SimpleITK images
    input_image = sitk.GetImageFromArray(array)
    displacement_sitk = sitk.GetImageFromArray(deformation_field)

    # Define transform parameter map for transformation with displacement field
    parameter_map = _default_bspline()
    parameter_map['Transform'] = ['DisplacementFieldTransform']
    parameter_map['DisplacementField'] = [displacement_sitk]
    parameter_map['ResampleInterpolator'] = ['FinalBSplineInterpolator']
    parameter_map['FinalBSplineInterpolationOrder'] = ['3']
    parameter_map['ResultImagePixelType'] = ['float']

    # Create the Transformix image filter object
    transformix = sitk.TransformixImageFilter()
    transformix.SetTransformParameterMap(parameter_map)

    # Set the input and deformation images for the Transformix filter
    transformix.SetMovingImage(input_image)

    warped = transformix.Execute()

    # Get the warped image as a SimpleITK image
    #warped = transformix.GetResultImage()

    # Convert the warped image to a numpy array
    warped = sitk.GetArrayFromImage(warped)

    return warped



def _coregister_arrays(fixed, moving, params, fixed_spacing, moving_spacing, fixed_mask=None, moving_mask=None):
    """
    Coregister two arrays and return coregistered + deformation field 
    """

    # Convert numpy arrays to sitk images
    moving = sitk.GetImageFromArray(moving) 
    moving.SetSpacing(moving_spacing)
    fixed = sitk.GetImageFromArray(fixed)
    fixed.SetSpacing(fixed_spacing)
    if moving_mask is not None:
        moving_mask = sitk.GetImageFromArray(moving_mask)
        moving_mask.SetSpacing(moving_spacing)
        #moving_mask.__SetPixelAsUInt8__  
    if fixed_mask is not None:
        fixed_mask = sitk.GetImageFromArray(fixed_mask)
        fixed_mask.SetSpacing(fixed_spacing)  
        #fixed_mask.__SetPixelAsUInt8__      

    # Perform registration
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.LogToConsoleOn() # turn on for debugging
    #elastixImageFilter.LogToConsoleOff()
    elastixImageFilter.SetFixedImage(fixed)
    elastixImageFilter.SetMovingImage(moving)
    elastixImageFilter.SetParameterMap(params)
    if fixed_mask is not None:
        elastixImageFilter.SetFixedMask(fixed_mask)
    if moving_mask is not None:
        elastixImageFilter.SetMovingMask(moving_mask)
    elastixImageFilter.Execute()

    # Calculate deformation field
    transformParameterMap = elastixImageFilter.GetTransformParameterMap()
    transformixImageFilter = sitk.TransformixImageFilter()
    #transformixImageFilter.LogToConsoleOn() # turn on for debugging
    transformixImageFilter.LogToConsoleOff()
    transformixImageFilter.SetTransformParameterMap(transformParameterMap)
    # transformixImageFilter.UpdateLargestPossibleRegion()
    transformixImageFilter.ComputeDeformationFieldOn()
    transformixImageFilter.SetMovingImage(moving) 
    transformixImageFilter.Execute()

    # Collect return values
    # coregistered = transformixImageFilter.GetResultImage() # same result
    coregistered = elastixImageFilter.GetResultImage()
    deformation_field = transformixImageFilter.GetDeformationField()

    # Convert sitk Images back to numpy arrays
    coregistered = sitk.GetArrayFromImage(coregistered)
    deformation_field = sitk.GetArrayFromImage(deformation_field)

    return coregistered, deformation_field




def _default_bspline():
   
    p = sitk.GetDefaultParameterMap("bspline")

    # *********************
    # * ImageTypes
    # *********************
    p["FixedInternalImagePixelType"] = ["float"]
    p["MovingInternalImagePixelType"] = ["float"]
    ## selection based on 3D or 2D image data: newest elastix version does not require input image dimension
    p["FixedImageDimension"] = ["2"] 
    p["MovingImageDimension"] = ["2"] 
    p["UseDirectionCosines"] = ["true"]
    # *********************
    # * Components
    # *********************
    p["Registration"] = ["MultiResolutionRegistration"]
    # Image intensities are sampled using an ImageSampler, Interpolator and ResampleInterpolator.
    # Image sampler is responsible for selecting points in the image to sample. 
    # The RandomCoordinate simply selects random positions.
    p["ImageSampler"] = ["RandomCoordinate"]
    # Interpolator is responsible for interpolating off-grid positions during optimization. 
    # The BSplineInterpolator with BSplineInterpolationOrder = 1 used here is very fast and uses very little memory
    p["Interpolator"] = ["BSplineInterpolator"]
    # ResampleInterpolator here chosen to be FinalBSplineInterpolator with FinalBSplineInterpolationOrder = 1
    # is used to resample the result image from the moving image once the final transformation has been found.
    # This is a one-time step so the additional computational complexity is worth the trade-off for higher image quality.
    p["ResampleInterpolator"] = ["FinalBSplineInterpolator"]
    p["Resampler"] = ["DefaultResampler"]
    # Order of B-Spline interpolation used during registration/optimisation.
    # It may improve accuracy if you set this to 3. Never use 0.
    # An order of 1 gives linear interpolation. This is in most 
    # applications a good choice.
    p["BSplineInterpolationOrder"] = ["1"]
    # Order of B-Spline interpolation used for applying the final
    # deformation.
    # 3 gives good accuracy; recommended in most cases.
    # 1 gives worse accuracy (linear interpolation)
    # 0 gives worst accuracy, but is appropriate for binary images
    # (masks, segmentations); equivalent to nearest neighbor interpolation.
    p["FinalBSplineInterpolationOrder"] = ["3"]
    # Pyramids found in Elastix:
    # 1)	Smoothing -> Smoothing: YES, Downsampling: NO
    # 2)	Recursive -> Smoothing: YES, Downsampling: YES
    #      If Recursive is chosen and only # of resolutions is given 
    #      then downsamlping by a factor of 2 (default)
    # 3)	Shrinking -> Smoothing: NO, Downsampling: YES
    # p["FixedImagePyramid"] = ["FixedSmoothingImagePyramid"] # Smoothing requires 3d dimension at least 4 pixels
    # p["MovingImagePyramid"] = ["MovingSmoothingImagePyramid"]
    p["FixedImagePyramid"] = ["FixedRecursiveImagePyramid"]
    p["MovingImagePyramid"] = ["MovingRecursiveImagePyramid"]
    p["Optimizer"] = ["AdaptiveStochasticGradientDescent"]
    # Whether transforms are combined by composition or by addition.
    # In generally, Compose is the best option in most cases.
    # It does not influence the results very much.
    p["HowToCombineTransforms"] = ["Compose"]
    p["Transform"] = ["BSplineTransform"]
    # Metric
    # p["Metric"] = ["AdvancedMeanSquares"]
    p["Metric"] = ["NormalizedMutualInformation"]
    # Number of grey level bins in each resolution level,
    # for the mutual information. 16 or 32 usually works fine.
    # You could also employ a hierarchical strategy:
    #(NumberOfHistogramBins 16 32 64)
    p["NumberOfHistogramBins"] = ["32"]
    # *********************
    # * Transformation
    # *********************
    # The control point spacing of the bspline transformation in 
    # the finest resolution level. Can be specified for each 
    # dimension differently. Unit: mm.
    # The lower this value, the more flexible the deformation.
    # Low values may improve the accuracy, but may also cause
    # unrealistic deformations.
    # By default the grid spacing is halved after every resolution,
    # such that the final grid spacing is obtained in the last 
    # resolution level.
    # The grid spacing here is specified in voxel units.
    #(FinalGridSpacingInPhysicalUnits 10.0 10.0)
    #(FinalGridSpacingInVoxels 8)
    #p["FinalGridSpacingInPhysicalUnits"] = ["50.0"]
    p["FinalGridSpacingInPhysicalUnits"] = ["25.0", "25.0"]
    # *********************
    # * Optimizer settings
    # *********************
    # The number of resolutions. 1 Is only enough if the expected
    # deformations are small. 3 or 4 mostly works fine. For large
    # images and large deformations, 5 or 6 may even be useful.
    p["NumberOfResolutions"] = ["4"]
    p["AutomaticParameterEstimation"] = ["true"]
    p["ASGDParameterEstimationMethod"] = ["Original"]
    p["MaximumNumberOfIterations"] = ["500"]
    # The step size of the optimizer, in mm. By default the voxel size is used.
    # which usually works well. In case of unusual high-resolution images
    # (eg histology) it is necessary to increase this value a bit, to the size
    # of the "smallest visible structure" in the image:
    # p["MaximumStepLength"] = ["1.0"] 
    p["MaximumStepLength"] = ["0.1"]
    # *********************
    # * Pyramid settings
    # *********************
    # The downsampling/blurring factors for the image pyramids.
    # By default, the images are downsampled by a factor of 2
    # compared to the next resolution.
    #p["ImagePyramidSchedule"] = ["8 8  4 4  2 2  1 1"]
    # *********************
    # * Sampler parameters
    # *********************
    # Number of spatial samples used to compute the mutual
    # information (and its derivative) in each iteration.
    # With an AdaptiveStochasticGradientDescent optimizer,
    # in combination with the two options below, around 2000
    # samples may already suffice.
    p["NumberOfSpatialSamples"] = ["2048"]
    # Refresh these spatial samples in every iteration, and select
    # them randomly. See the manual for information on other sampling
    # strategies.
    p["NewSamplesEveryIteration"] = ["true"]
    p["CheckNumberOfSamples"] = ["true"]
    # *********************
    # * Mask settings
    # *********************
    # If you use a mask, this option is important. 
    # If the mask serves as region of interest, set it to false.
    # If the mask indicates which pixels are valid, then set it to true.
    # If you do not use a mask, the option doesn't matter.
    p["ErodeMask"] = ["false"]
    p["ErodeFixedMask"] = ["false"]
    # *********************
    # * Output settings
    # *********************
    #Default pixel value for pixels that come from outside the picture:
    p["DefaultPixelValue"] = ["0"]
    # Choose whether to generate the deformed moving image.
    # You can save some time by setting this to false, if you are
    # not interested in the final deformed moving image, but only
    # want to analyze the deformation field for example.
    p["WriteResultImage"] = ["true"]
    # The pixel type and format of the resulting deformed moving image
    p["ResultImagePixelType"] = ["float"]
    p["ResultImageFormat"] = ["nii"]

    return p



def _default_affine():

    p = sitk.GetDefaultParameterMap("affine")

    # ImageTypes
    p["FixedInternalImagePixelType"] = ["float"]
    p["MovingInternalImagePixelType"] = ["float"]
    p["FixedImageDimension"] = ['2'] 
    p["MovingImageDimension"] = ['2'] 
    p["UseDirectionCosines"] = ["true"]

    # Components
    p["Registration"] = ["MultiResolutionRegistration"]
    p["ImageSampler"] = ["Random"]
    p["Interpolator"] = ["BSplineInterpolator"]
    p["ResampleInterpolator"] = ["FinalBSplineInterpolator"]
    p["Resampler"] = ["DefaultResampler"]
    p["BSplineInterpolationOrder"] = ["3"]
    p["FinalBSplineInterpolationOrder"] = ["3"]
    # p["FixedImagePyramid"] = ["FixedSmoothingImagePyramid"] # Smoothing requires 3d dimension at least 4 pixels
    # p["MovingImagePyramid"] = ["MovingSmoothingImagePyramid"]
    p["FixedImagePyramid"] = ["FixedRecursiveImagePyramid"]
    p["MovingImagePyramid"] = ["MovingRecursiveImagePyramid"]
    
    # p["Metric"] = ["AdvancedMeanSquares"]
    p["Metric"] = ["AdvancedMattesMutualInformation"]
    p["NumberOfHistogramBins"] = ["32"]

    # Transformation
    p["Transform"] = ["AffineTransform"]
    p["HowToCombineTransforms"] = ["Compose"]
    p["AutomaticTransformInitialization"] = ["false"]
    p["FinalGridSpacingInPhysicalUnits"] = ["25.0", "25.0"]

    # Optimizer
    p["Optimizer"] = ["AdaptiveStochasticGradientDescent"]
    p["NumberOfResolutions"] = ["4"]
    p["AutomaticParameterEstimation"] = ["true"]
    p["ASGDParameterEstimationMethod"] = ["Original"]
    p["MaximumNumberOfIterations"] = ["500"]
    p["MaximumStepLength"] = ["1.0"] 

    # Pyramid settings
    #p["ImagePyramidSchedule"] = ["8 8  4 4  2 2  1 1"]

    # Sampler parameters
    p["NumberOfSpatialSamples"] = ["2048"]
    p["NewSamplesEveryIteration"] = ["true"]
    p["CheckNumberOfSamples"] = ["true"]

    # Mask settings
    p["ErodeMask"] = ["true"]
    p["ErodeFixedMask"] = ["false"]

    # Output settings
    p["DefaultPixelValue"] = ["0"]
    p["WriteResultImage"] = ["true"]
    p["ResultImagePixelType"] = ["float"]
    p["ResultImageFormat"] = ["nii"]
    
    return p



def _default_rigid():
    # https://github.com/SuperElastix/ElastixModelZoo/tree/master/models/Par0064

    p = sitk.GetDefaultParameterMap("rigid")

    # ImageTypes
    p["FixedInternalImagePixelType"] = ["float"]
    p["MovingInternalImagePixelType"] = ["float"]
    p["FixedImageDimension"] = ['2'] 
    p["MovingImageDimension"] = ['2'] 
    p["UseDirectionCosines"] = ["true"]

    # Components
    p["Registration"] = ["MultiResolutionRegistration"]
    p["ImageSampler"] = ["Random"]
    p["Interpolator"] = ["BSplineInterpolator"]
    p["ResampleInterpolator"] = ["FinalBSplineInterpolator"]
    p["Resampler"] = ["DefaultResampler"]
    p["BSplineInterpolationOrder"] = ["1"]
    p["FinalBSplineInterpolationOrder"] = ["3"]
    # p["FixedImagePyramid"] = ["FixedSmoothingImagePyramid"] # Smoothing requires 3d dimension at least 4 pixels
    # p["MovingImagePyramid"] = ["MovingSmoothingImagePyramid"]
    p["FixedImagePyramid"] = ["FixedRecursiveImagePyramid"]
    p["MovingImagePyramid"] = ["MovingRecursiveImagePyramid"]

    # Pyramid settings
    #p["ImagePyramidSchedule"] = ["8 8 4 4 4 2 2 2 1"]
    p["Optimizer"] = ["AdaptiveStochasticGradientDescent"]

    # Transformation
    p["HowToCombineTransforms"] = ["Compose"]
    p["Transform"] = ["EulerTransform"]
    p["AutomaticTransformInitialization"] = ["true"]
    p["AutomaticTransformInitializationMethod"] = ["GeometricalCenter"]
    p["AutomaticScalesEstimation"] = ["true"]
    p["Metric"] = ["AdvancedMattesMutualInformation"]
    p["NumberOfHistogramBins"] = ["32"]

    # Optimizer settings
    p["NumberOfResolutions"] = ["3"]
    p["AutomaticParameterEstimation"] = ["true"]
    # p["ASGDParameterEstimationMethod"] = ["Original"]
    p["MaximumNumberOfIterations"] = ["500"]
    p["MaximumStepLength"] = ["1.0"] 

     # Sampler parameters
    p["NumberOfSpatialSamples"] = ["2048"]
    p["NewSamplesEveryIteration"] = ["true"]
    p["NumberOfSamplesForExactGradient"] = ["1024"]
    p["MaximumNumberOfSamplingAttempts"] = ["15"]
    p["CheckNumberOfSamples"] = ["true"]

    # Mask settings
    p["ErodeMask"] = ["true"]
    # p["ErodeFixedMask"] = ["false"]

    # Output settings
    p["DefaultPixelValue"] = ["0"]
    p["WriteResultImage"] = ["true"]
    p["ResultImagePixelType"] = ["float"]
    p["ResultImageFormat"] = ["nii"]
    
    return p


def _default_2d_to_3d():
    # modified from:
    # https://github.com/SuperElastix/ElastixModelZoo/tree/master/models/Par0013

    p = sitk.GetDefaultParameterMap("rigid")

    p['FixedInternalImagePixelType'] = ["float"]
    p['MovingInternalImagePixelType'] = ["float"]
    p['FixedImageDimension'] = ['3']
    p['MovingImageDimension'] = ['3']
    p['UseDirectionCosines"'] = ["false"]

    # **************** Main Components **************************

    p['Registration'] = ["MultiResolutionRegistration"]
    #p['Interpolator'] = ["RayCastInterpolator"]
    #p['ResampleInterpolator'] = ["FinalRayCastInterpolator"]
    p['Resampler'] = ["DefaultResampler"]

    p['FixedImagePyramid'] = ["FixedRecursiveImagePyramid"]
    p['MovingImagePyramid'] = ["MovingRecursiveImagePyramid"]

    #p['Optimizer'] = ["Powell"]
    p['Transform'] = ["EulerTransform"]
    p['Metric'] = ["GradientDifference"]

    # ***************** Transformation **************************

    #p['Scales'] = ['57.3']
    #p['AutomaticTransformInitialization'] = ["false"]
    p['AutomaticTransformInitialization'] = ["true"]
    p["AutomaticTransformInitializationMethod"] = ["GeometricalCenter"]
    p["AutomaticScalesEstimation"] = ["true"]
    p['HowToCombineTransforms'] = ["Compose"]
    p['CenterOfRotationPoint'] = ['0.0 0.0 0.0']

    # ******************* Similarity measure *********************

    p['UseNormalization'] = ["true"]

    # ******************** Multiresolution **********************

    p['NumberOfResolutions'] = ['1']

    # ******************* Optimizer ****************************

    p['MaximumNumberOfIterations'] = ['10']
    p['MaximumStepLength'] = ['1.0']
    p['StepTolerance'] = ['0.0001']
    p['ValueTolerance'] = ['0.000001']

    # **************** Image sampling **********************

    #p['ImageSampler'] = ["Full"]
    p['NewSamplesEveryIteration'] = ["false"]

    # ************* Interpolation and Resampling ****************

    #p['Origin'] = ['-145.498 -146.889 381.766']
    #p['Interpolator0PreParameters'] = ['-0.009475 -0.006807 -0.030067 0.0 0.0 0.0']
    #p['ResampleInterpolator0PreParameters'] = ['-0.009475 -0.006807 -0.030067 0.0 0.0 0.0']
    #p['Interpolator0FocalPoint'] = ['0.54 -0.85 -813.234']
    #p['ResampleInterpolator0FocalPoint'] = ['0.54 -0.85 -813.234']
    p['Threshold'] = ['1000']
    p['DefaultPixelValue'] = ['0']
    p['WriteResultImage'] = ["true"]
    #p['WriteTransformParametersEachIteration'] = ["true"]
    p['ResultImagePixelType'] = ["float"]

    return p
