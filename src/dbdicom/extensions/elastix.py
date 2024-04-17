# pip install SimpleITK-SimpleElastix

import numpy as np
import itk
from skimage.measure import block_reduce
from dbdicom.extensions import vreg


def coregister_3d_to_3d(moving, fixed, 
        transformation = 'Affine',
        metric = "NormalizedMutualInformation",
        final_grid_spacing = 1.0,
        apply_to = []):

    background = 2**16-1
    fixed_map = vreg.map_to(fixed, moving, cval=background)

    # Get arrays for fixed and moving series
    array_fixed, _ = fixed_map.array('SliceLocation', pixels_first=True, first_volume=True)
    array_moving, headers_moving  = moving.array('SliceLocation', pixels_first=True, first_volume=True)
    
    # Remove temporary overlay
    if fixed_map != fixed:
        fixed_map.remove()

    # Set background pixels to zero for both images
    idx = np.where(array_fixed==background)
    array_fixed[idx] = 0
    array_moving[idx] = 0

    # Apply coregistration settings
    moving.message('Preparing coregistration..')
    if transformation == 'Rigid':
        pars = _default_rigid('3')  
    elif transformation == 'Affine':
        pars = _default_affine('3') 
    else:
        pars = _default_bspline('3')
    pars.SetParameter("Metric", metric)
    pars.SetParameter("FinalGridSpacingInPhysicalUnits", str(final_grid_spacing))

    # Get slice spacing
    slice_spacing = headers_moving[0].SpacingBetweenSlices
    if slice_spacing is None:
        slice_spacing = headers_moving[0].SliceThickness
    pixel_spacing = headers_moving[0].PixelSpacing + [slice_spacing]

    # Coregister and save as DICOM
    coregistered, params = _coregister_arrays_3d(
        array_fixed, array_moving, 
        pars, pixel_spacing, 
        return_deformation_field=False)
    coreg = moving.new_sibling(suffix='coreg')
    coreg.set_array(coregistered, headers_moving, pixels_first=True)

    # Apply the transformation to other series
    apply_to_coreg = []
    for s, series in enumerate(apply_to):
        moving.progress(s+1, len(apply_to), 'Applying transformation..')
        array, headers = series.array('SliceLocation', pixels_first=True, first_volume=True)
        if array.shape != array_moving.shape:
            msg = 'Cannot apply the same transformation to other series with a different shape'
            raise ValueError(msg)
        array = _apply_3d_transformation(array, params, pixel_spacing)
        new = series.new_sibling(suffix='coreg')
        new.set_array(array, headers, pixels_first=True)
        apply_to_coreg.append(new)

    return coreg, apply_to_coreg




def coregister_2d_to_2d(moving, fixed, 
        transformation = 'Affine',
        metric = "NormalizedMutualInformation",
        final_grid_spacing = 1.0,
    ):

    background = 2**16-1
    fixed_map = vreg.map_to(fixed, moving, cval=background)

    # Get arrays for fixed and moving series
    array_fixed, _ = fixed_map.array('SliceLocation', pixels_first=True, first_volume=True)
    array_moving, headers_moving  = moving.array('SliceLocation', pixels_first=True, first_volume=True)
    
    # Remove temporary overlay
    if fixed_map != fixed:
        fixed_map.remove()

    # Set background pixels to zero for both images
    idx = np.where(array_fixed==background)
    array_fixed[idx] = 0
    array_moving[idx] = 0

    # Get coregistration settings
    moving.message('Setting up coregistration..')
    if transformation == 'Rigid':
        pars = _default_rigid('2')  
    elif transformation == 'Affine':
        pars = _default_affine('2') 
    else:
        pars = _default_bspline('2')
    pars.SetParameter("Metric", metric)
    pars.SetParameter("FinalGridSpacingInPhysicalUnits", str(final_grid_spacing))
   
    # Coregister fixed and moving slice-by-slice
    pixel_spacing = headers_moving[0].PixelSpacing
    for z in range(array_moving.shape[2]):
        moving.status.progress(z+1, array_moving.shape[2], 'Performing coregistration..')
        coreg = _coregister_arrays(array_fixed[:,:,z], array_moving[:,:,z], pars, pixel_spacing, return_deformation_field=False)
        array_moving[:,:,z] = coreg

    # Save as DICOM
    coreg = moving.new_sibling(suffix='coregistered')
    coreg.set_array(array_moving, headers_moving, pixels_first=True)
    return coreg





# BELOW HERE NON-DICOM FUNCTIONALITY


    

# 3D version does not have the downsampling option
# some bug involving the image shaping
def _coregister_arrays_3d(
        target, source, 
        elastix_model_parameters, 
        spacing, 
        log=False,
        return_deformation_field=False):
    
    # Convert to itk images
    source = itk.GetImageFromArray(np.array(source, np.float32)) 
    target = itk.GetImageFromArray(np.array(target, np.float32))
    source.SetSpacing(spacing)
    target.SetSpacing(spacing)

    # Coregister source to target
    coreg, result_transform_parameters = itk.elastix_registration_method(
        target, source,
        parameter_object=elastix_model_parameters, 
        log_to_console=log)
    coreg = itk.GetArrayFromImage(coreg)

    if not return_deformation_field:
        return coreg, result_transform_parameters

    # Get deformation field
    deformation_field = itk.transformix_deformation_field(
        target, result_transform_parameters, 
        log_to_console=log)
    deformation_field = itk.GetArrayFromImage(deformation_field)
    return coreg, result_transform_parameters, deformation_field


def _apply_3d_transformation(
        source, result_transform_parameters,
        spacing, log=False):
    
    source = itk.GetImageFromArray(np.array(source, np.float32)) 
    source.SetSpacing(spacing)
    coreg = itk.transformix_filter(
        source, result_transform_parameters,
        log_to_console=log)
    return itk.GetArrayFromImage(coreg)




def _coregister_arrays(
        target_large, 
        source_large, 
        elastix_model_parameters, 
        spacing_large, 
        log=False, 
        downsample:int=1, 
        return_deformation_field=False):

    # Downsample source and target
    # The origin of an image is the center of the voxel in the lower left corner
    # The origin of the large image is (0,0).
    # The original of the small image is therefore: 
    #   spacing_large/2 + (spacing_small/2 - spacing_large)
    #   = (spacing_small - spacing_large)/2
    target_small = block_reduce(target_large, block_size=downsample, func=np.mean)
    source_small = block_reduce(source_large, block_size=downsample, func=np.mean)
    spacing_small = [spacing*downsample for spacing in spacing_large]
    origin_large = [0] * len(spacing_large)
    origin_small = [(spacing_small[i] - spacing_large[i])/2 for i in range(len(spacing_small))]

    # Coregister downsampled source to target
    source_small = itk.GetImageFromArray(np.array(source_small, np.float32)) 
    target_small = itk.GetImageFromArray(np.array(target_small, np.float32))
    source_small.SetSpacing(spacing_small)
    target_small.SetSpacing(spacing_small)
    source_small.SetOrigin(origin_small)
    target_small.SetOrigin(origin_small)
    coreg_small, result_transform_parameters = itk.elastix_registration_method(
        target_small, source_small,
        parameter_object=elastix_model_parameters, 
        log_to_console=log)
    
    # Get coregistered image at original size
    result_transform_parameters.SetParameter(0, "Size", [str(dim) for dim in source_large.shape])
    result_transform_parameters.SetParameter(0, "Spacing", [str(dim) for dim in spacing_large])
    source_large = itk.GetImageFromArray(np.array(source_large, np.float32))
    source_large.SetSpacing(spacing_large)
    source_large.SetOrigin(origin_large)
    coreg_large = itk.transformix_filter(
        source_large,
        result_transform_parameters,
        log_to_console=log)
    coreg_large = itk.GetArrayFromImage(coreg_large)
    if coreg_large.ndim==3:
        coreg_large = np.transpose(coreg_large, (1,2,0))

    if not return_deformation_field:
        return coreg_large
    
    # Get deformation field at original size
    target_large = itk.GetImageFromArray(np.array(target_large, np.float32))
    target_large.SetSpacing(spacing_large)
    target_large.SetOrigin(origin_large)
    deformation_field = itk.transformix_deformation_field(
        target_large, 
        result_transform_parameters, 
        log_to_console=log)
    deformation_field = itk.GetArrayFromImage(deformation_field)
    if deformation_field.ndim==4:
        deformation_field = np.transpose(deformation_field, (1,2,0,3))
    else:
        deformation_field = np.reshape(deformation_field, target_large.shape + (len(target_large.shape), ))
    return coreg_large, deformation_field


def _default_bspline(d):
    param_obj = itk.ParameterObject.New()
    parameter_map_bspline = param_obj.GetDefaultParameterMap('bspline')
    ## add parameter map file to the parameter object: required in itk-elastix
    param_obj.AddParameterMap(parameter_map_bspline) 
    #OPTIONAL: Write the default parameter file to output file
    # param_obj.WriteParameterFile(parameter_map_bspline, "bspline.default.txt")
    # *********************
    # * ImageTypes
    # *********************
    param_obj.SetParameter("FixedInternalImagePixelType", "float")
    param_obj.SetParameter("MovingInternalImagePixelType", "float")
    ## selection based on 3D or 2D image data: newest elastix version does not require input image dimension
    param_obj.SetParameter("FixedImageDimension", d) 
    param_obj.SetParameter("MovingImageDimension", d) 
    param_obj.SetParameter("UseDirectionCosines", "true")
    # *********************
    # * Components
    # *********************
    param_obj.SetParameter("Registration", "MultiResolutionRegistration")
    # Image intensities are sampled using an ImageSampler, Interpolator and ResampleInterpolator.
    # Image sampler is responsible for selecting points in the image to sample. 
    # The RandomCoordinate simply selects random positions.
    param_obj.SetParameter("ImageSampler", "RandomCoordinate")
    # Interpolator is responsible for interpolating off-grid positions during optimization. 
    # The BSplineInterpolator with BSplineInterpolationOrder = 1 used here is very fast and uses very little memory
    param_obj.SetParameter("Interpolator", "BSplineInterpolator")
    # ResampleInterpolator here chosen to be FinalBSplineInterpolator with FinalBSplineInterpolationOrder = 1
    # is used to resample the result image from the moving image once the final transformation has been found.
    # This is a one-time step so the additional computational complexity is worth the trade-off for higher image quality.
    param_obj.SetParameter("ResampleInterpolator", "FinalBSplineInterpolator")
    param_obj.SetParameter("Resampler", "DefaultResampler")
    # Order of B-Spline interpolation used during registration/optimisation.
    # It may improve accuracy if you set this to 3. Never use 0.
    # An order of 1 gives linear interpolation. This is in most 
    # applications a good choice.
    param_obj.SetParameter("BSplineInterpolationOrder", "1")
    # Order of B-Spline interpolation used for applying the final
    # deformation.
    # 3 gives good accuracy; recommended in most cases.
    # 1 gives worse accuracy (linear interpolation)
    # 0 gives worst accuracy, but is appropriate for binary images
    # (masks, segmentations); equivalent to nearest neighbor interpolation.
    param_obj.SetParameter("FinalBSplineInterpolationOrder", "3")
    # Pyramids found in Elastix:
    # 1)	Smoothing -> Smoothing: YES, Downsampling: NO
    # 2)	Recursive -> Smoothing: YES, Downsampling: YES
    #      If Recursive is chosen and only # of resolutions is given 
    #      then downsamlping by a factor of 2 (default)
    # 3)	Shrinking -> Smoothing: NO, Downsampling: YES
    param_obj.SetParameter("FixedImagePyramid", "FixedSmoothingImagePyramid")
    param_obj.SetParameter("MovingImagePyramid", "MovingSmoothingImagePyramid")
    param_obj.SetParameter("Optimizer", "AdaptiveStochasticGradientDescent")
    # Whether transforms are combined by composition or by addition.
    # In generally, Compose is the best option in most cases.
    # It does not influence the results very much.
    param_obj.SetParameter("HowToCombineTransforms", "Compose")
    param_obj.SetParameter("Transform", "BSplineTransform")
    # Metric
    param_obj.SetParameter("Metric", "NormalizedMutualInformation")
    # Number of grey level bins in each resolution level,
    # for the mutual information. 16 or 32 usually works fine.
    # You could also employ a hierarchical strategy:
    #(NumberOfHistogramBins 16 32 64)
    param_obj.SetParameter("NumberOfHistogramBins", "32")
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
    #param_obj.SetParameter("FinalGridSpacingInPhysicalUnits", ["50.0", "50.0"])
    param_obj.SetParameter("FinalGridSpacingInPhysicalUnits", "25.0")
    # *********************
    # * Optimizer settings
    # *********************
    # The number of resolutions. 1 Is only enough if the expected
    # deformations are small. 3 or 4 mostly works fine. For large
    # images and large deformations, 5 or 6 may even be useful.
    param_obj.SetParameter("NumberOfResolutions", "4")
    param_obj.SetParameter("AutomaticParameterEstimation", "true")
    param_obj.SetParameter("ASGDParameterEstimationMethod", "Original")
    param_obj.SetParameter("MaximumNumberOfIterations", "500")
    # The step size of the optimizer, in mm. By default the voxel size is used.
    # which usually works well. In case of unusual high-resolution images
    # (eg histology) it is necessary to increase this value a bit, to the size
    # of the "smallest visible structure" in the image:
    param_obj.SetParameter("MaximumStepLength", "0.1") 
    # *********************
    # * Pyramid settings
    # *********************
    # The downsampling/blurring factors for the image pyramids.
    # By default, the images are downsampled by a factor of 2
    # compared to the next resolution.
    #param_obj.SetParameter("ImagePyramidSchedule", "8 8  4 4  2 2  1 1")
    # *********************
    # * Sampler parameters
    # *********************
    # Number of spatial samples used to compute the mutual
    # information (and its derivative) in each iteration.
    # With an AdaptiveStochasticGradientDescent optimizer,
    # in combination with the two options below, around 2000
    # samples may already suffice.
    param_obj.SetParameter("NumberOfSpatialSamples", "2048")
    # Refresh these spatial samples in every iteration, and select
    # them randomly. See the manual for information on other sampling
    # strategies.
    param_obj.SetParameter("NewSamplesEveryIteration", "true")
    param_obj.SetParameter("CheckNumberOfSamples", "true")
    # *********************
    # * Mask settings
    # *********************
    # If you use a mask, this option is important. 
    # If the mask serves as region of interest, set it to false.
    # If the mask indicates which pixels are valid, then set it to true.
    # If you do not use a mask, the option doesn't matter.
    param_obj.SetParameter("ErodeMask", "false")
    param_obj.SetParameter("ErodeFixedMask", "false")
    # *********************
    # * Output settings
    # *********************
    #Default pixel value for pixels that come from outside the picture:
    param_obj.SetParameter("DefaultPixelValue", "0")
    # Choose whether to generate the deformed moving image.
    # You can save some time by setting this to false, if you are
    # not interested in the final deformed moving image, but only
    # want to analyze the deformation field for example.
    param_obj.SetParameter("WriteResultImage", "true")
    # The pixel type and format of the resulting deformed moving image
    param_obj.SetParameter("ResultImagePixelType", "float")
    param_obj.SetParameter("ResultImageFormat", "nii")
    
    return param_obj


def _default_affine(d):
    param_obj = itk.ParameterObject.New()
    parameter_map_bspline = param_obj.GetDefaultParameterMap('affine')
    param_obj.AddParameterMap(parameter_map_bspline) 

    # ImageTypes
    param_obj.SetParameter("FixedInternalImagePixelType", "float")
    param_obj.SetParameter("MovingInternalImagePixelType", "float")
    param_obj.SetParameter("FixedImageDimension", d) 
    param_obj.SetParameter("MovingImageDimension", d) 
    param_obj.SetParameter("UseDirectionCosines", "true")

    # Components
    param_obj.SetParameter("Registration", "MultiResolutionRegistration")
    param_obj.SetParameter("ImageSampler", "Random")
    param_obj.SetParameter("Interpolator", "BSplineInterpolator")
    param_obj.SetParameter("ResampleInterpolator", "FinalBSplineInterpolator")
    param_obj.SetParameter("Resampler", "DefaultResampler")
    param_obj.SetParameter("BSplineInterpolationOrder", "3")
    param_obj.SetParameter("FinalBSplineInterpolationOrder", "3")
    param_obj.SetParameter("FixedImagePyramid", "FixedRecursiveImagePyramid")
    param_obj.SetParameter("MovingImagePyramid", "MovingRecursiveImagePyramid")

    # Metric
    param_obj.SetParameter("Metric", "AdvancedMattesMutualInformation")
    param_obj.SetParameter("NumberOfHistogramBins", "32")

    # Transformation
    param_obj.SetParameter("Transform", "AffineTransform")
    param_obj.SetParameter("HowToCombineTransforms", "Compose")
    param_obj.SetParameter("AutomaticTransformInitialization", "false")
    param_obj.SetParameter("FinalGridSpacingInPhysicalUnits", "25.0")

    # Optimizer
    param_obj.SetParameter("Optimizer", "AdaptiveStochasticGradientDescent")
    param_obj.SetParameter("NumberOfResolutions", "4")
    param_obj.SetParameter("AutomaticParameterEstimation", "true")
    param_obj.SetParameter("ASGDParameterEstimationMethod", "Original")
    param_obj.SetParameter("MaximumNumberOfIterations", "500")
    param_obj.SetParameter("MaximumStepLength", "1.0") 

    # Pyramid settings
    param_obj.SetParameter("NumberOfSpatialSamples", "2048")
    param_obj.SetParameter("NewSamplesEveryIteration", "true")
    param_obj.SetParameter("CheckNumberOfSamples", "true")

    # Mask settings
    param_obj.SetParameter("ErodeMask", "false")
    param_obj.SetParameter("ErodeFixedMask", "false")

    # Output settings
    param_obj.SetParameter("DefaultPixelValue", "0")
    param_obj.SetParameter("WriteResultImage", "true")
    param_obj.SetParameter("ResultImagePixelType", "float")
    param_obj.SetParameter("ResultImageFormat", "nii")
    
    return param_obj


def _default_rigid(d):
    param_obj = itk.ParameterObject.New()
    parameter_map_bspline = param_obj.GetDefaultParameterMap('rigid')
    param_obj.AddParameterMap(parameter_map_bspline) 

    # ImageTypes
    param_obj.SetParameter("FixedInternalImagePixelType", "float")
    param_obj.SetParameter("MovingInternalImagePixelType", "float")
    param_obj.SetParameter("FixedImageDimension", d) 
    param_obj.SetParameter("MovingImageDimension", d) 
    param_obj.SetParameter("UseDirectionCosines", "true")
   
    # Components
    param_obj.SetParameter("Registration", "MultiResolutionRegistration")
    param_obj.SetParameter("ImageSampler", "Random")
    param_obj.SetParameter("Interpolator", "BSplineInterpolator")
    param_obj.SetParameter("ResampleInterpolator", "FinalBSplineInterpolator")
    param_obj.SetParameter("Resampler", "DefaultResampler")
    param_obj.SetParameter("BSplineInterpolationOrder", "1")
    param_obj.SetParameter("FinalBSplineInterpolationOrder", "3")
    param_obj.SetParameter("FixedImagePyramid", "FixedRecursiveImagePyramid")
    param_obj.SetParameter("MovingImagePyramid", "MovingRecursiveImagePyramid")
    param_obj.SetParameter("Optimizer", "AdaptiveStochasticGradientDescent")

    # Transformation
    param_obj.SetParameter("HowToCombineTransforms", "Compose")
    param_obj.SetParameter("Transform", "EulerTransform")
    param_obj.SetParameter("AutomaticTransformInitialization", "true")
    param_obj.SetParameter("AutomaticTransformInitializationMethod", "GeometricalCenter")
    param_obj.SetParameter("AutomaticScalesEstimation", "true")

    # Metric
    param_obj.SetParameter("Metric", "AdvancedMattesMutualInformation")
    param_obj.SetParameter("NumberOfHistogramBins", "32")

    # Optimizer settings
    param_obj.SetParameter("NumberOfResolutions", "3")
    param_obj.SetParameter("AutomaticParameterEstimation", "true")
    #param_obj.SetParameter("ASGDParameterEstimationMethod", "Original")
    param_obj.SetParameter("MaximumNumberOfIterations", "500")
    param_obj.SetParameter("MaximumStepLength", "1.0") 

    # Sampler parameters
    param_obj.SetParameter("NumberOfSpatialSamples", "2048")
    param_obj.SetParameter("NewSamplesEveryIteration", "true")
    param_obj.SetParameter("NumberOfSamplesForExactGradient", "1024")
    param_obj.SetParameter("MaximumNumberOfSamplingAttempts", "15")
    param_obj.SetParameter("CheckNumberOfSamples", "true")
    
    # Mask settings
    param_obj.SetParameter("ErodeMask", "false")
    param_obj.SetParameter("ErodeFixedMask", "false")
   
    # Output settings
    param_obj.SetParameter("DefaultPixelValue", "0")
    param_obj.SetParameter("WriteResultImage", "true")
    param_obj.SetParameter("ResultImagePixelType", "float")
    param_obj.SetParameter("ResultImageFormat", "nii")
    
    return param_obj
