import numpy as np
import mdreg

def fit_DTI(series):

    array, header = series.array(['SliceLocation', 'AcquisitionTime'], pixels_first=True)
    signal_model = mdreg.models.DTI
    
    # PARAMETER VARIABLES INITIALIZATION
    model_fit = np.empty(array.shape)
    pars = np.empty(array.shape[:3] + (len(parameters),) )

    # LOOP THROUGH SLICES
    for i, slice in enumerate(range(array.shape[2])):

        series.status.progress(i+1, array.shape[2], 'Fitting DTI model..')

        #extracting DTI relevant parameters from DICOM headers                                              
        b_values = [float(hdr[(0x19, 0x100c)]) for hdr in header[slice,:,0]]
        b_vectors = [hdr[(0x19, 0x100e)] for hdr in header[slice,:,0]]
        orientation = [hdr.ImageOrientationPatient for hdr in header[slice,:,0]] 

        # Perform the model fit using mdreg
        mdr = mdreg.MDReg()
        mdr.signal_parameters = [b_values, b_vectors, orientation]
        mdr.set_array(array[:,:,slice,:,0])    
        mdr.pixel_spacing = header[slice,0,0].PixelSpacing
        mdr.signal_model = signal_model
        mdr.model_fit()

        # Store results
        model_fit[:,:,slice,:,0] = mdr.model_fit
        pars[:,:,slice,:] = mdr.pars

    #EXPORT RESULTS
    study = series.new_pibling(StudyDescription = 'DTI')
    parameters = signal_model.pars()
    series_par = []
    for p in range(len(parameters)):
        par = series.SeriesDescription + '_DTI_' + parameters[p]
        par = study.new_series(SeriesDescription=par)
        par.set_array(pars[...,p], header[:,0], pixels_first=True)
        series_par.append(par)
    fit = series.SeriesDescription + '_DTI_fit'
    fit = study.new_series(SeriesDescription=fit)
    fit.set_array(model_fit, header, pixels_first=True)
    return fit, series_par


def DTI(series):

    array, header = series.array(['SliceLocation', 'AcquisitionTime'], pixels_first=True)
    signal_model = mdreg.models.DTI
    #elastix_file = 'BSplines_DTI.txt'
    
    # PARAMETER VARIABLES INITIALIZATION
    model_fit = np.empty(array.shape)
    pars = np.empty(array.shape[:3] + (len(parameters),) )
    coreg = np.empty(array.shape)

    # LOOP THROUGH SLICES
    for i, slice in enumerate(range(array.shape[2])):
        series.status.progress(i+1, array.shape[2], 'Fitting DTI model..')

        #extracting DTI relevant parameters from DICOM headers                                              
        b_values = [float(hdr[(0x19, 0x100c)]) for hdr in header[slice,:,0]]
        b_vectors = [hdr[(0x19, 0x100e)] for hdr in header[slice,:,0]]
        orientation = [hdr.ImageOrientationPatient for hdr in header[slice,:,0]] 

        # Perform the model fit using mdreg
        mdr = mdreg.MDReg()
        mdr.signal_parameters = [b_values, b_vectors, orientation]
        mdr.set_array(array[:,:,slice,:,0])    
        mdr.pixel_spacing = header[slice,0,0].PixelSpacing
        mdr.signal_model = signal_model
        #mdr.read_elastix(os.path.join(elastix_pars, elastix_file))
        # SET ELASTIX PARAMETERS PROGRAMMATICALLY
        mdr.fit()

        # Store results
        model_fit[:,:,slice,:,0] = mdr.model_fit
        coreg[:,:,slice,:,0] = mdr.coreg
        pars[:,:,slice,:] = mdr.pars

    #EXPORT RESULTS
    study = series.new_pibling(StudyDescription = 'DTI')
    parameters = signal_model.pars()
    series_par = []
    for p in range(len(parameters)):
        par = series.SeriesDescription + '_DTI_' + parameters[p]
        par = study.new_series(SeriesDescription=par)
        par.set_array(pars[...,p], header[:,0], pixels_first=True)
        series_par.append(par)
    fit = series.SeriesDescription + '_DTI_fit'
    fit = study.new_series(SeriesDescription=fit)
    fit.set_array(model_fit, header, pixels_first=True)
    mdr = series.SeriesDescription + '_DTI_mdr'
    mdr = study.new_series(SeriesDescription = mdr)
    mdr.set_array(coreg, header, pixels_first=True)

    return mdr, fit, series_par


