import numpy as np
import itk
import dbdicom.wrappers.scipy as scipy


def coregister(moving, fixed, 
        transformation = 'Affine',
        metric = "NormalizedMutualInformation",
        final_grid_spacing = 1.0,
    ):

    nan = 2**16-1
    fixed_map = scipy.map_to(fixed, moving, cval=nan)

    # Get arrays for fixed and moving series
    array_fixed, _ = fixed_map.array('SliceLocation', pixels_first=True)

    array_moving, headers_moving  = moving.array('SliceLocation', pixels_first=True)
    
    if array_fixed is None or array_moving is None:
        return fixed_map

    pixel_spacing = headers_moving[0,0].PixelSpacing
   
    # Get coregistration settings
    if transformation == 'Rigid':
        pars = _default_rigid('2')
        
    elif transformation == 'Affine':
        pars = _default_affine('2')
       
    else:
        pars = _default_bspline('2')
       
    pars.SetParameter("Metric", metric)
    pars.SetParameter("FinalGridSpacingInPhysicalUnits", str(final_grid_spacing))
   
    # Coregister fixed and moving slice-by-slice
    for z in range(array_moving.shape[2]):
        moving.status.progress(z+1, array_moving.shape[2], 'Performing coregistration..')
        image0 = array_fixed[:,:,z,0]
        image1 = array_moving[:,:,z,0]
        ind_img0 = np.where(image0==nan)
        image1[ind_img0] = 0
        image0[ind_img0] = 0
        coreg, _ = _coregister_2D_arrays(image0, image1, pars, pixel_spacing)
        array_moving[:,:,z,0] = coreg

    # Return new series
    moving.status.message('Saving results..')
    fixed_map.remove()
    desc = moving.instance().SeriesDescription 
    desc += ' registered to ' + fixed.instance().SeriesDescription
    registered_series = moving.new_sibling(SeriesDescription = desc)
    registered_series.set_array(array_moving, headers_moving, pixels_first=True)
    moving.status.message('Finished coregistration..')
    return registered_series



def _coregister_2D_arrays(target, source, elastix_model_parameters, spacing):
    """
    Coregister two arrays and return coregistered + deformation field 
    """
    shape_source = np.shape(source)
    source = itk.GetImageFromArray(np.array(source, np.float32)) 
    source.SetSpacing(spacing)
    target = itk.GetImageFromArray(np.array(target, np.float32))
    target.SetSpacing(spacing)
    
    # Perform coregistration
    coregistered, result_transform_parameters = itk.elastix_registration_method(
        target, source,
        parameter_object=elastix_model_parameters)
    coregistered = itk.GetArrayFromImage(coregistered)

    # Compute the deformation field
    transformix_object = itk.TransformixFilter.New(target)
    transformix_object.SetTransformParameterObject(result_transform_parameters)
    transformix_object.UpdateLargestPossibleRegion()
    transformix_object.ComputeDeformationFieldOn() 
    deformation_field = itk.GetArrayFromImage(transformix_object.GetOutputDeformationField()).flatten()
    if len(shape_source) == 2: # 2D
        deformation_field = np.reshape(deformation_field,(shape_source[0], shape_source[1], 2)) 
    else: #3D 
        deformation_field = np.reshape(deformation_field,(shape_source[0], shape_source[1], shape_source[2], 3)) 

    return coregistered, deformation_field


def _default_bspline(d):
   
    param_obj = itk.ParameterObject.New()
    parameter_map_bspline = param_obj.GetDefaultParameterMap('bspline')
    ## add parameter map file to the parameter object: required in itk-elastix
    param_obj.AddParameterMap(parameter_map_bspline) 
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
    # param_obj.SetParameter("Metric", "AdvancedMeanSquares")
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
    #param_obj.SetParameter("FinalGridSpacingInPhysicalUnits", "50.0")
    param_obj.SetParameter("FinalGridSpacingInPhysicalUnits", ["25.0", "25.0"])
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
    # param_obj.SetParameter("MaximumStepLength", "1.0") 
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
    param_obj.SetParameter("FixedImagePyramid", "FixedSmoothingImagePyramid")
    param_obj.SetParameter("MovingImagePyramid", "MovingSmoothingImagePyramid")
    
    # param_obj.SetParameter("Metric", "AdvancedMeanSquares")
    param_obj.SetParameter("Metric", "AdvancedMattesMutualInformation")
    param_obj.SetParameter("NumberOfHistogramBins", "32")

    # Transformation
    param_obj.SetParameter("Transform", "AffineTransform")
    param_obj.SetParameter("HowToCombineTransforms", "Compose")
    param_obj.SetParameter("AutomaticTransformInitialization", "false")
    param_obj.SetParameter("FinalGridSpacingInPhysicalUnits", ["25.0", "25.0"])

    # Optimizer
    param_obj.SetParameter("Optimizer", "AdaptiveStochasticGradientDescent")
    param_obj.SetParameter("NumberOfResolutions", "4")
    param_obj.SetParameter("AutomaticParameterEstimation", "true")
    param_obj.SetParameter("ASGDParameterEstimationMethod", "Original")
    param_obj.SetParameter("MaximumNumberOfIterations", "500")
    param_obj.SetParameter("MaximumStepLength", "1.0") 

    # Pyramid settings
    #param_obj.SetParameter("ImagePyramidSchedule", "8 8  4 4  2 2  1 1")

    # Sampler parameters
    param_obj.SetParameter("NumberOfSpatialSamples", "2048")
    param_obj.SetParameter("NewSamplesEveryIteration", "true")
    param_obj.SetParameter("CheckNumberOfSamples", "true")

    # Mask settings
    param_obj.SetParameter("ErodeMask", "true")
    param_obj.SetParameter("ErodeFixedMask", "false")

    # Output settings
    param_obj.SetParameter("DefaultPixelValue", "0")
    param_obj.SetParameter("WriteResultImage", "true")
    param_obj.SetParameter("ResultImagePixelType", "float")
    param_obj.SetParameter("ResultImageFormat", "nii")
    
    return param_obj



def _default_rigid(d):
    # https://github.com/SuperElastix/ElastixModelZoo/tree/master/models/Par0064
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

    # Pyramid settings
    #param_obj.SetParameter("ImagePyramidSchedule", "8 8 4 4 4 2 2 2 1")
    param_obj.SetParameter("Optimizer", "AdaptiveStochasticGradientDescent")

    # Transformation
    param_obj.SetParameter("HowToCombineTransforms", "Compose")
    param_obj.SetParameter("Transform", "EulerTransform")
    param_obj.SetParameter("AutomaticTransformInitialization", "true")
    param_obj.SetParameter("AutomaticTransformInitializationMethod", "GeometricalCenter")
    param_obj.SetParameter("AutomaticScalesEstimation", "true")
    param_obj.SetParameter("Metric", "AdvancedMattesMutualInformation")
    param_obj.SetParameter("NumberOfHistogramBins", "32")

    # Optimizer settings
    param_obj.SetParameter("NumberOfResolutions", "3")
    param_obj.SetParameter("AutomaticParameterEstimation", "true")
    # param_obj.SetParameter("ASGDParameterEstimationMethod", "Original")
    param_obj.SetParameter("MaximumNumberOfIterations", "500")
    param_obj.SetParameter("MaximumStepLength", "1.0") 

     # Sampler parameters
    param_obj.SetParameter("NumberOfSpatialSamples", "2048")
    param_obj.SetParameter("NewSamplesEveryIteration", "true")
    param_obj.SetParameter("NumberOfSamplesForExactGradient", "1024")
    param_obj.SetParameter("MaximumNumberOfSamplingAttempts", "15")
    param_obj.SetParameter("CheckNumberOfSamples", "true")

    # Mask settings
    param_obj.SetParameter("ErodeMask", "true")
    # param_obj.SetParameter("ErodeFixedMask", "false")

    # Output settings
    param_obj.SetParameter("DefaultPixelValue", "0")
    param_obj.SetParameter("WriteResultImage", "fals")
    param_obj.SetParameter("ResultImagePixelType", "float")
    param_obj.SetParameter("ResultImageFormat", "nii")
    
    return param_obj


# Chat GPT suggestion

# def _chat_gpt_coreg(target, source, elastix_model_parameters, spacing):
#     import SimpleITK as sitk
#     from elastix import Elastix, ParameterFile

#     # Convert the numpy arrays to SimpleITK images
#     image1_sitk = sitk.GetImageFromArray(source)
#     image2_sitk = sitk.GetImageFromArray(target)

#     # Initialize the Elastix object
#     elastix = Elastix()

#     # Create a parameter file for the registration
#     parameter_file = ParameterFile()
#     parameter_file.ReadParameterFile('default.txt')

#     # Set the fixed and moving images
#     elastix.SetFixedImage(image1_sitk)
#     elastix.SetMovingImage(image2_sitk)

#     # Set the parameter file
#     elastix.SetParameterMap(parameter_file.GetParameterMap())

#     # Execute the registration
#     elastix.LogToConsoleOn()
#     elastix.Execute()

#     # Get the result image
#     result_image = elastix.GetResultImage()

#     # Get the deformation field
#     deformation_field = elastix.GetDeformationField()

#     # Convert the result image and deformation field to numpy arrays
#     result_array = sitk.GetArrayFromImage(result_image)
#     deformation_field_array = sitk.GetArrayFromImage(deformation_field)

#     return result_array, deformation_field_array
