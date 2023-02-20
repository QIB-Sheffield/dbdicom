import numpy as np
import mdreg
from mdreg.models import DTI, T1_simple, T2_simple, T2star_simple, DWI_simple, constant, DCE_2CFM
import actions.autoaif

def fit_DTI(series):

    array, header = series.array(['SliceLocation', 'AcquisitionTime'], pixels_first=True)
    signal_model = DTI #models.DTI#mdreg.models.DTI
    
    # PARAMETER VARIABLES INITIALIZATION
    model_fit = np.empty(array.shape)
    parameters = signal_model.pars()
    pars = np.empty(array.shape[:3] + (len(parameters),) )
    print("shape pars")
    print(np.shape(pars)) # (172, 172, 30, 2)
    print("array.shape[:3] ")
    print(array.shape[:3]) # (172, 172, 30)
    print("len(parameters)")
    print(len(parameters)) # 2
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
        mdr.fit_signal()#mdr.model_fit()

        # Store results
        model_fit[:,:,slice,:,0] = mdr.model_fit
        pars[:,:,slice,:] = mdr.pars

    #EXPORT RESULTS
    study = series.new_pibling(StudyDescription = 'DTI')
    
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


def process_DTI(series):

    array, header = series.array(['SliceLocation', 'AcquisitionTime'], pixels_first=True)
    signal_model = DTI#mdreg.models.DTI
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

def fit_T1(series):

    array, header = series.array(['SliceLocation', 'AcquisitionTime'], pixels_first=True)
    signal_model = T1_simple 
    
    # PARAMETER VARIABLES INITIALIZATION
    model_fit = np.empty(array.shape)
    parameters = signal_model.pars()
    pars = np.empty(array.shape[:3] + (len(parameters),) )
    print("shape pars")
    print(np.shape(pars)) # (172, 172, 30, 2)
    print("array.shape[:3] ")
    print(array.shape[:3]) # (172, 172, 30)
    print("len(parameters)")
    print(len(parameters)) # 2
    # LOOP THROUGH SLICES
    for i, slice in enumerate(range(array.shape[2])):

        series.status.progress(i+1, array.shape[2], 'Fitting T1 model..')
        
        array, header = series.array(['SliceLocation', 'InversionTime'], pixels_first=True)
   
        # Perform the model fit using mdreg
        mdr = mdreg.MDReg()
        sort_by='InversionTime'
        mdr.signal_parameters = [hdr[sort_by] for hdr in (header[slice,:,0])]  
        mdr.set_array(array[:,:,slice,:,0])    
        mdr.pixel_spacing = header[slice,0,0].PixelSpacing
        mdr.signal_model = signal_model
        mdr.fit_signal()

        # Store results
        model_fit[:,:,slice,:,0] = mdr.model_fit
        pars[:,:,slice,:] = mdr.pars

    #EXPORT RESULTS
    study = series.new_pibling(StudyDescription = 'T1')
    
    series_par = []
    for p in range(len(parameters)):
        par = series.SeriesDescription + '_T1_' + parameters[p]
        par = study.new_series(SeriesDescription=par)
        par.set_array(pars[...,p], header[:,0], pixels_first=True)
        series_par.append(par)
    fit = series.SeriesDescription + '_T1_fit'
    fit = study.new_series(SeriesDescription=fit)
    fit.set_array(model_fit, header, pixels_first=True)
    return fit, series_par


def fit_T2(series):

    array, header = series.array(['SliceLocation', 'AcquisitionTime'], pixels_first=True)
    signal_model = T2_simple 
    
    # PARAMETER VARIABLES INITIALIZATION
    model_fit = np.empty(array.shape)
    parameters = signal_model.pars()
    pars = np.empty(array.shape[:3] + (len(parameters),) )
    print("shape pars")
    print(np.shape(pars)) # (172, 172, 30, 2)
    print("array.shape[:3] ")
    print(array.shape[:3]) # (172, 172, 30)
    print("len(parameters)")
    print(len(parameters)) # 2
    # LOOP THROUGH SLICES
    for i, slice in enumerate(range(array.shape[2])):

        series.status.progress(i+1, array.shape[2], 'Fitting T2 model..')
        
        array, header = series.array(['SliceLocation', 'AcquisitionTime'], pixels_first=True)
   
        # Perform the model fit using mdreg
        mdr = mdreg.MDReg()
        
        mdr.signal_parameters = [0,30,40,50,60,70,80,90,100,110,120]  
        mdr.set_array(array[:,:,slice,:,0])    
        mdr.pixel_spacing = header[slice,0,0].PixelSpacing
        mdr.signal_model = signal_model
        mdr.fit_signal()

        # Store results
        model_fit[:,:,slice,:,0] = mdr.model_fit
        pars[:,:,slice,:] = mdr.pars

    #EXPORT RESULTS
    study = series.new_pibling(StudyDescription = 'T2')
    
    series_par = []
    for p in range(len(parameters)):
        par = series.SeriesDescription + '_T2_' + parameters[p]
        par = study.new_series(SeriesDescription=par)
        par.set_array(pars[...,p], header[:,0], pixels_first=True)
        series_par.append(par)
    fit = series.SeriesDescription + '_T2star_fit'
    fit = study.new_series(SeriesDescription=fit)
    fit.set_array(model_fit, header, pixels_first=True)
    return fit, series_par

def fit_T2star(series):

    array, header = series.array(['SliceLocation', 'AcquisitionTime'], pixels_first=True)
    signal_model = T2star_simple 
    
    # PARAMETER VARIABLES INITIALIZATION
    model_fit = np.empty(array.shape)
    parameters = signal_model.pars()
    pars = np.empty(array.shape[:3] + (len(parameters),) )
    print("shape pars")
    print(np.shape(pars)) # 
    print("array.shape[:3] ")
    print(array.shape[:3]) # 
    print("len(parameters)")
    print(len(parameters)) # 
    # LOOP THROUGH SLICES
    for i, slice in enumerate(range(array.shape[2])):

        series.status.progress(i+1, array.shape[2], 'Fitting T2star model..')
        
        array, header = series.array(['SliceLocation', 'EchoTime'], pixels_first=True)
   
        # Perform the model fit using mdreg
        mdr = mdreg.MDReg()
        sort_by='EchoTime'
        mdr.signal_parameters = [hdr[sort_by] for hdr in (header[slice,:,0])] 
        mdr.set_array(array[:,:,slice,:,0])    
        mdr.pixel_spacing = header[slice,0,0].PixelSpacing
        mdr.signal_model = signal_model
        mdr.fit_signal()

        # Store results
        model_fit[:,:,slice,:,0] = mdr.model_fit
        pars[:,:,slice,:] = mdr.pars

    #EXPORT RESULTS
    study = series.new_pibling(StudyDescription = 'T2star')
    
    series_par = []
    for p in range(len(parameters)):
        par = series.SeriesDescription + '_T2star_' + parameters[p]
        par = study.new_series(SeriesDescription=par)
        par.set_array(pars[...,p], header[:,0], pixels_first=True)
        series_par.append(par)
    fit = series.SeriesDescription + '_T2star_fit'
    fit = study.new_series(SeriesDescription=fit)
    fit.set_array(model_fit, header, pixels_first=True)
    return fit, series_par

def fit_IVIM(series):

    array, header = series.array(['SliceLocation', 'AcquisitionTime'], pixels_first=True)
    signal_model = DWI_simple # TBC
    
    # PARAMETER VARIABLES INITIALIZATION
    model_fit = np.empty(array.shape)
    parameters = signal_model.pars()
    pars = np.empty(array.shape[:3] + (len(parameters),) )
    print("shape pars")
    print(np.shape(pars)) # (172, 172, 30, 2)
    print("array.shape[:3] ")
    print(array.shape[:3]) # 
    print("len(parameters)")
    print(len(parameters)) # 
    # LOOP THROUGH SLICES
    for i, slice in enumerate(range(array.shape[2])):

        series.status.progress(i+1, array.shape[2], 'Fitting DWI model..')

        # Perform the model fit using mdreg
        mdr = mdreg.MDReg()
        mdr.signal_parameters = [0,10.000086, 19.99908294, 30.00085926, 50.00168544, 80.007135, 100.0008375, 199.9998135, 300.0027313, 600.0]
        mdr.set_array(array[:,:,slice,:,0])    
        mdr.pixel_spacing = header[slice,0,0].PixelSpacing
        mdr.signal_model = signal_model
        mdr.fit_signal()#mdr.model_fit()

        # Store results
        model_fit[:,:,slice,:,0] = mdr.model_fit
        pars[:,:,slice,:] = mdr.pars

    #EXPORT RESULTS
    study = series.new_pibling(StudyDescription = 'IVIM')
    
    series_par = []
    for p in range(len(parameters)):
        par = series.SeriesDescription + '_IVIM_' + parameters[p]
        par = study.new_series(SeriesDescription=par)
        par.set_array(pars[...,p], header[:,0], pixels_first=True)
        series_par.append(par)
    fit = series.SeriesDescription + '_IVIM_fit'
    fit = study.new_series(SeriesDescription=fit)
    fit.set_array(model_fit, header, pixels_first=True)
    return fit, series_par

def fit_MT(series): #TBC # HOW TO DECIDE WHICH SERIES IS SELECTED FIRST? MT OFF OR MT ON

    #array, header = series.array(['SliceLocation', 'AcquisitionTime'], pixels_first=True)
    array_off, header_off = mt_off.array(['SliceLocation', 'AcquisitionTime'], pixels_first=True)
    array_on, header_on = mt_on.array(['SliceLocation', 'AcquisitionTime'], pixels_first=True)
    
    #array_off = np.reshape(array_off,np.shape(array_off)+(1,))
    #array_on = np.reshape(array_on,np.shape(array_on)+(1,))
    array = np.concatenate((array_off, array_on), axis=3)
    header = np.concatenate((header_off, header_on), axis=1)
    
    array = np.reshape(array,np.shape(array)+(1,))

    signal_model = constant # TBC
    
    # PARAMETER VARIABLES INITIALIZATION
    model_fit = np.empty(array.shape)
    parameters = signal_model.pars()
    pars = np.empty(array.shape[:3] + (len(parameters),) )
    print("shape pars")
    print(np.shape(pars)) #
    print("array.shape[:3] ")
    print(array.shape[:3]) # 
    print("len(parameters)")
    print(len(parameters)) # 
    # LOOP THROUGH SLICES
    for i, slice in enumerate(range(array.shape[2])):

        series.status.progress(i+1, array.shape[2], 'Fitting constant model..')

        # Perform the model fit using mdreg
        mdr = mdreg.MDReg()
        mdr.signal_parameters = []
        mdr.set_array(array[:,:,slice,:,0])    
        mdr.pixel_spacing = header[slice,0,0].PixelSpacing
        mdr.signal_model = signal_model
        mdr.fit_signal()#mdr.model_fit()

        # Store results
        model_fit[:,:,slice,:,0] = mdr.model_fit
        pars[:,:,slice,:] = mdr.pars

    #EXPORT RESULTS
    study = series.new_pibling(StudyDescription = 'MT')
    
    series_par = []
    for p in range(len(parameters)):
        par = series.SeriesDescription + '_MT_' + parameters[p]
        par = study.new_series(SeriesDescription=par)
        par.set_array(pars[...,p], header[:,0], pixels_first=True)
        series_par.append(par)
    fit = series.SeriesDescription + '_MT_fit'
    fit = study.new_series(SeriesDescription=fit)
    fit.set_array(model_fit, header, pixels_first=True)
    return fit, series_par


def fit_DCE(series): #

    array, header = series.array(['SliceLocation', 'AcquisitionTime'], pixels_first=True)
    
    signal_model = DCE_2CFM # TBC
    
    # PARAMETER VARIABLES INITIALIZATION
    model_fit = np.empty(array.shape)
    parameters = signal_model.pars()
    pars = np.empty(array.shape[:3] + (len(parameters),) )
    print("shape pars")
    print(np.shape(pars)) # (384, 384, 9, 4)
    print("array.shape[:3] ")
    print(array.shape[:3]) # (384, 384, 9)  
    print("len(parameters)")
    print(len(parameters)) # 4
    # LOOP THROUGH SLICES
    for i, slice in enumerate(range(array.shape[2])):

        series.status.progress(i+1, array.shape[2], 'Fitting DCE_2CFM model..')

        # Perform the model fit using mdreg
        mdr = mdreg.MDReg()

         # GET AIF
        cutRatio=0.25             #create a window around the center of the image where the aorta is
        filter_kernel=(15,15)     #gaussian kernel for smoothing the image to destroy any noisy single high intensity filter
        regGrow_threshold = 2     #threshold for the region growing algorithm

        for i in range(header.shape[0]):
            if (header[i,0,0]["ImageOrientationPatient"]== [1, 0, 0, 0, 1, 0]):
                aortaslice = int(i + 1)
                print("aortaslice")
                print(aortaslice)
                break
            else:
                aortaslice = 9

        aif = actions.autoaif.DCEautoAIF(array, header, series, aortaslice, cutRatio, filter_kernel, regGrow_threshold)

        time = np.zeros(header.shape[1])
        for i in range(header.shape[1]):
            tempTime = header[slice,i,0]['AcquisitionTime']
            tempH = int(tempTime[0:2])
            tempM = int(tempTime[2:4])
            tempS = int(tempTime[4:6])
            tempRest = float("0." + tempTime[7:])
            time[i] = tempH*3600+tempM*60+tempS+tempRest
        time -=time[0]

        baseline = 15
        hematocrit = 0.45
        signal_pars = [aif, time, baseline, hematocrit]
        mdr.signal_parameters = signal_pars
        print("DCE parameters were calculated ")

        mdr.set_array(array[:,:,slice,:,0])    
        mdr.pixel_spacing = header[slice,0,0].PixelSpacing
        mdr.signal_model = signal_model
        mdr.fit_signal()#mdr.model_fit()
        
        # Store results
        model_fit[:,:,slice,:,0] = mdr.model_fit
        pars[:,:,slice,:] = mdr.pars

    #EXPORT RESULTS
    study = series.new_pibling(StudyDescription = 'DCE')
    
    series_par = []
    for p in range(len(parameters)):
        par = series.SeriesDescription + '_DCE_' + parameters[p]
        par = study.new_series(SeriesDescription=par)
        par.set_array(pars[...,p], header[:,0], pixels_first=True)
        series_par.append(par)
    fit = series.SeriesDescription + '_DCE_fit'
    fit = study.new_series(SeriesDescription=fit)
    fit.set_array(model_fit, header, pixels_first=True)
    return fit, series_par