import os
import struct
from datetime import datetime
import numpy as np
import pandas as pd
from matplotlib import cm
import pydicom
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
import dbdicom.utils.image as image


def module_patient():

    return [
        'ReferencedPatientSequence',
        'PatientName',
        'PatientID',
        'IssuerOfPatientID',
        'TypeOfPatientID',
        'IssuerOfPatientIDQualifiersSequence',
        'SourcePatientGroupIdentificationSequence',
        'GroupOfPatientsIdentificationSequence',
        'PatientBirthDate',
        'PatientBirthTime',
        'PatientBirthDateInAlternativeCalendar',
        'PatientDeathDateInAlternativeCalendar',
        'PatientAlternativeCalendar',
        'PatientSex',
        'QualityControlSubject',
        'StrainDescription',
        'StrainNomenclature',
        'StrainStockSequence',
        'StrainAdditionalInformation',
        'StrainCodeSequence',
        'GeneticModificationsSequence',
        'OtherPatientNames',
        'OtherPatientIDsSequence',
        'ReferencedPatientPhotoSequence',
        'EthnicGroup',
        'PatientSpeciesDescription',
        'PatientSpeciesCodeSequence',
        'PatientBreedDescription',
        'PatientBreedCodeSequence',
        'BreedRegistrationSequence',
        'ResponsiblePerson',
        'ResponsiblePersonRole',
        'ResponsibleOrganization',
        'PatientComments',
        'PatientIdentityRemoved',
        'DeidentificationMethod',
        'DeidentificationMethodCodeSequence',
        'ClinicalTrialSponsorName',
        'ClinicalTrialProtocolID',
        'ClinicalTrialProtocolName',
        'ClinicalTrialSiteID',
        'ClinicalTrialSiteName',
        'ClinicalTrialSubjectID',
        'ClinicalTrialSubjectReadingID',
        'ClinicalTrialProtocolEthicsCommitteeName',
        'ClinicalTrialProtocolEthicsCommitteeApprovalNumber',
    ]

def module_study():

    return [
        'StudyDate',
        'StudyTime',
        'AccessionNumber',
        'IssuerOfAccessionNumberSequence',
        'ReferringPhysicianName',
        'ReferringPhysicianIdentificationSequence',
        'ConsultingPhysicianName',
        'ConsultingPhysicianIdentificationSequence',
        'StudyDescription',
        'ProcedureCodeSequence',
        'PhysiciansOfRecord',
        'PhysiciansOfRecordIdentificationSequence',
        'NameOfPhysiciansReadingStudy',
        'PhysiciansReadingStudyIdentificationSequence',
        'ReferencedStudySequence',
        'StudyInstanceUID',
        'StudyID',
        'RequestingService',
        'RequestingServiceCodeSequence',
        'ReasonForPerformedProcedureCodeSequence',
        'AdmittingDiagnosesDescription',
        'AdmittingDiagnosesCodeSequence',
        'PatientAge',
        'PatientSize',
        'PatientSizeCodeSequence',
        'PatientBodyMassIndex',
        'MeasuredAPDimension',
        'MeasuredLateralDimension',
        'PatientWeight',
        'MedicalAlerts',
        'Allergies',
        'Occupation',
        'SmokingStatus',
        'AdditionalPatientHistory',
        'PregnancyStatus',
        'LastMenstrualDate',
        'PatientSexNeutered',
        'ReasonForVisit',
        'ReasonForVisitCodeSequence',
        'AdmissionID',
        'IssuerOfAdmissionIDSequence',
        'ServiceEpisodeID',
        'ServiceEpisodeDescription',
        'IssuerOfServiceEpisodeIDSequence',
        'PatientState',
        'ClinicalTrialTimePointID',
        'ClinicalTrialTimePointDescription',
        'LongitudinalTemporalOffsetFromEvent',
        'LongitudinalTemporalEventType',
        'ConsentForClinicalTrialUseSequence',
    ]   

def module_series():

    return [
        'SeriesDate',
        'SeriesTime',
        'Modality',
        'SeriesDescription',
        'SeriesDescriptionCodeSequence',
        'PerformingPhysicianName',
        'PerformingPhysicianIdentificationSequence',
        'OperatorsName',
        'OperatorIdentificationSequence',
        'ReferencedPerformedProcedureStepSequence',
        'RelatedSeriesSequence',
        'AnatomicalOrientationType',
        'BodyPartExamined',
        'ProtocolName',
        'PatientPosition',
        'ReferencedDefinedProtocolSequence',
        'ReferencedPerformedProtocolSequence',
        'SeriesInstanceUID',
        'SeriesNumber',
        'Laterality',
        'SmallestPixelValueInSeries',
        'LargestPixelValueInSeries',
        'PerformedProcedureStepStartDate',
        'PerformedProcedureStepStartTime',
        'PerformedProcedureStepEndDate',
        'PerformedProcedureStepEndTime',
        'PerformedProcedureStepID',
        'PerformedProcedureStepDescription',
        'PerformedProtocolCodeSequence',
        'RequestAttributesSequence',
        'CommentsOnThePerformedProcedureStep',
        'ClinicalTrialCoordinatingCenterName',
        'ClinicalTrialSeriesID',
        'ClinicalTrialSeriesDescription',
    ]

def SOPClass(SOPClassUID):

    if SOPClassUID == '1.2.840.10008.5.1.4.1.1.4':
        return 'MRImage'
    if SOPClassUID == '1.2.840.10008.5.1.4.1.1.4.1':
        return 'EnhancedMRImage'
    if SOPClassUID == '1.2.840.10008.5.1.4.1.1.7':
        return 'SecondaryCaptureImage'
    return 'Instance'

def read(file, dialog=None):

    try:
        return pydicom.dcmread(file)
    except:
        message = "Failed to read " + file
        message += "\n The file may be opened or deleted by another application."
        message += "\n Please close the file and try again."
        if dialog is not None:
            dialog.information(message) 
        else:
            print(message)  

def write(ds, file, dialog=None): # ds is a pydicom dataset

    try:
        # check if directory exists and create it if not
        dir = os.path.dirname(file)
        if not os.path.exists(dir):
            os.makedirs(dir)
        ds.save_as(file) 
    except:
        message = "Failed to write to " + file
        message += "\n The file may be open in another application, or is being synchronised by a cloud service."
        message += "\n Please close the file or pause the synchronisation and try again."
        if dialog is not None:
            dialog.information(message) 
        else:
            print(message)  

def to_set_type(value):
    """
    Convert pydicom datatypes to the python datatypes used to set the parameter.
    """

    if value.__class__.__name__ == 'PersonName':
        return str(value)
    if value.__class__.__name__ == 'Sequence':
        return [ds for ds in value]
    if value.__class__.__name__ == 'TM': 
        return str(value) 
    if value.__class__.__name__ == 'UID': 
        return str(value) 
    if value.__class__.__name__ == 'IS': 
        return int(value)
    if value.__class__.__name__ == 'DT': 
        return str(value)
    if value.__class__.__name__ == 'DA': 
        return str(value)
    if value.__class__.__name__ == 'DSfloat': 
        return float(value)
    if value.__class__.__name__ == 'DSdecimal': 
        return int(value)
    else:
        return value

def get_values(ds, tags):
    """Helper function - return a list of values for a DbObject"""

    # https://pydicom.github.io/pydicom/stable/guides/element_value_types.html
    if not isinstance(tags, list): 
        if tags not in ds:
            return None
        else:
        #    return ds[tags].value
            return to_set_type(ds[tags].value)
            
    row = []  
    for tag in tags:
        if tag not in ds:
            value = None
        else:
        #    value = ds[tag].value
            value = to_set_type(ds[tag].value)
        row.append(value)
    return row

def set_values(ds, tags, values):
    """Sets DICOM tags in the pydicom dataset in memory"""

    # TODO: Automatically convert datatypes to the correct ones required by pydicom for setting (above)

    if not isinstance(tags, list): 
        tags = [tags]
        values = [values]
    for i, tag in enumerate(tags):
        if values[i] is None:
            if tag in ds:
                del ds[tag]
        else:
            if tag in ds:
                ds[tag].value = values[i]
            else:
                if not isinstance(tag, pydicom.tag.BaseTag):
                    tag = pydicom.tag.Tag(tag)
                if not tag.is_private: # Add a new data element
                    VR = pydicom.datadict.dictionary_VR(tag)
                    ds.add_new(tag, VR, values[i])
                else:
                    pass # for now
    return ds

def read_dataframe(path, files, tags, status=None, message='Reading DICOM folder..'):
    """Reads a list of tags in a list of files.

    Arguments
    ---------
    files : str or list
        A filepath or a list of filepaths
    tags : str or list 
        A DICOM tag or a list of DICOM tags
    status : StatusBar

    Creates
    -------
    dataframe : pandas.DataFrame
        A Pandas dataframe with one row per file
        The index is the file path 
        Each column corresponds to a Tag in the list of Tags
        The returned dataframe is sorted by the given tags.
    """
    if not isinstance(files, list):
        files = [files]
    if not isinstance(tags, list):
        tags = [tags]
    array = []
    dicom_files = []
    for i, file in enumerate(files):
        ds = pydicom.dcmread(file, force=True)
        if isinstance(ds, pydicom.dataset.FileDataset):
            if 'TransferSyntaxUID' in ds.file_meta:
                row = get_values(ds, tags)
                array.append(row)
                relpath = os.path.relpath(file, path)
                dicom_files.append(relpath) 
        if status is not None: 
            status.progress(i+1, len(files), message)
    if status is not None: 
        status.hide()
    return pd.DataFrame(array, index = dicom_files, columns = tags)


def get_dataframe(datasets, tags):
    """Reads a list of tags in a list of datasets.

    Arguments
    ---------
    files : str or list
        A filepath or a list of filepaths
    tags : str or list 
        A DICOM tag or a list of DICOM tags
    status : StatusBar

    Creates
    -------
    dataframe : pandas.DataFrame
        A Pandas dataframe with one row per file
        The index is the file path 
        Each column corresponds to a Tag in the list of Tags
        The returned dataframe is sorted by the given tags.
    """
    if not isinstance(datasets, list):
        datasets = [datasets]
    if not isinstance(tags, list):
        tags = [tags]
    array = []
    indices = []
    for ds in datasets:
        if isinstance(ds, pydicom.dataset.FileDataset):
            if 'TransferSyntaxUID' in ds.file_meta:
                row = get_values(ds, tags)
                uid = get_values(ds, 'SOPInstanceUID')
                array.append(row)
                indices.append(uid) 
    return pd.DataFrame(array, index=indices, columns=tags)






def new_uid(n=1):
    
    if n == 1:
        return pydicom.uid.generate_uid()
    uid = []
    for _ in range(n):
        uid.append(pydicom.uid.generate_uid())
    return uid




def _image_array(ds):
    """Read the pixel array from an image"""

    if SOPClass(ds) == 'MRImage':
        return _mri_image_array(ds)
    if SOPClass(ds) == 'EnhancedMRImage':
        return _enhanced_mri_image_array(ds)

    array = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, 'RescaleSlope', 1)) 
    intercept = float(getattr(ds, 'RescaleIntercept', 0)) 
    #array = array * slope + intercept
    array *= slope
    array += intercept
    array = np.transpose(array)
    
    return array

def _mri_image_array(ds):
    """Read the pixel array from an MR image"""

    array = ds.pixel_array.astype(np.float32)
    if [0x2005, 0x100E] in ds: # 'Philips Rescale Slope'
        slope = ds[(0x2005, 0x100E)].value
        intercept = ds[(0x2005, 0x100D)].value
        array -= intercept
        array /= slope
    else:
        slope = float(getattr(ds, 'RescaleSlope', 1)) 
        intercept = float(getattr(ds, 'RescaleIntercept', 0)) 
        #array = array * slope + intercept
        array *= slope
        array += intercept
    
    return np.transpose(array)

def _enhanced_mri_image_array(ds):

    pixelArray = ds.pixel_array.astype(np.float32)
    frames = ds.PerFrameFunctionalGroupsSequence
    for index, frame in enumerate(frames):
        slice = np.squeeze(pixelArray[index, ...])
        if [0x2005, 0x100E] in ds: # 'Philips Rescale Slope'
            slope = ds[(0x2005, 0x100E)].value
            intercept = ds[(0x2005, 0x100D)].value
            slice = (slice - intercept) / slope
        else:
            transform = frame.PixelValueTransformationSequence[0]
            slope = float(getattr(transform, 'RescaleSlope', 1)) 
            intercept = float(getattr(transform, 'RescaleIntercept', 0)) 
            slice = slice * slope + intercept
        pixelArray[index, ...] = np.transpose(slice)
    
    return pixelArray

def _set_image_array(ds, array, value_range=None):

    if SOPClass(ds) == 'MRImage':
        return _set_mri_image_array(ds, array, value_range=value_range)
    if SOPClass(ds) == 'EnhancedMRImage':
        return _set_enhanced_mri_image_array(ds, array, value_range=value_range)
    
    if array.ndim >= 3: # remove spurious dimensions of 1
        array = np.squeeze(array) 
    array = image.clip(array, value_range=value_range)
    array, slope, intercept = image.scale_to_range(array, ds.BitsAllocated)
    array = np.transpose(array)

    maximum = np.amax(array)
    minimum = np.amin(array)
    shape = np.shape(array)

    ds.PixelRepresentation = 0
    ds.SmallestImagePixelValue = int(maximum)
    ds.LargestImagePixelValue = int(minimum)
    ds.RescaleSlope = 1 / slope
    ds.RescaleIntercept = - intercept / slope
#        ds.WindowCenter = (maximum + minimum) / 2
#        ds.WindowWidth = maximum - minimum
    ds.Rows = shape[0]
    ds.Columns = shape[1]
    ds.PixelData = array.tobytes()

def _set_mri_image_array(ds, array, value_range=None):

    if ds is None:
        # TODO: Handle this by creating new dataset from scratch
        raise RuntimeError('Cannot set array: no dataset defined on disk or in memory')
    if (0x2005, 0x100E) in ds: del ds[0x2005, 0x100E]  # Delete 'Philips Rescale Slope'
    if (0x2005, 0x100D) in ds: del ds[0x2005, 0x100D]
    
    if array.ndim >= 3: # remove spurious dimensions of 1
        array = np.squeeze(array) 
    array = image.clip(array, value_range=value_range)
    array, slope, intercept = image.scale_to_range(array, ds.BitsAllocated)
    array = np.transpose(array)

    maximum = np.amax(array)
    minimum = np.amin(array)
    shape = np.shape(array)

    ds.PixelRepresentation = 0
    ds.SmallestImagePixelValue = int(maximum)
    ds.LargestImagePixelValue = int(minimum)
    ds.RescaleSlope = 1 / slope
    ds.RescaleIntercept = - intercept / slope
#        ds.WindowCenter = (maximum + minimum) / 2
#        ds.WindowWidth = maximum - minimum
    ds.Rows = shape[0]
    ds.Columns = shape[1]
    ds.PixelData = array.tobytes()

def _set_enhanced_mri_image_array(ds, pixelArray, value_range=None):

    pixelArray = image.clip(pixelArray, value_range=value_range)
    pixelArray, slope, intercept = image.scale_to_range(pixelArray, ds.BitsAllocated)
    pixelArray = np.transpose(pixelArray, (0, 2, 1))

    maximum = np.amax(pixelArray)
    minimum = np.amin(pixelArray)
    shape = np.shape(pixelArray)

    ds.NumberOfFrames = np.shape(pixelArray)[0]
    del ds.PerFrameFunctionalGroupsSequence[ds.NumberOfFrames:]

    del ds[0x2005, 0x100E]  # Delete 'Philips Rescale Slope'
    del ds[0x2005, 0x100D]
    ds.PixelRepresentation = 0
    ds.SmallestImagePixelValue = int(maximum)
    ds.LargestImagePixelValue = int(minimum)
    ds.RescaleSlope = 1 / slope
    ds.RescaleIntercept = - intercept / slope
    ds.WindowCenter = (maximum + minimum) / 2
    ds.WindowWidth = maximum - minimum
    ds.Rows = shape[0]
    ds.Columns = shape[1]
    ds.PixelData = pixelArray.tobytes()



def _initialize(ds, UID=None, ref=None): # ds is pydicom dataset

    # Date and Time of Creation
    dt = datetime.now()
    timeStr = dt.strftime('%H%M%S')  # long format with micro seconds

    ds.ContentDate = dt.strftime('%Y%m%d')
    ds.ContentTime = timeStr
    ds.AcquisitionDate = dt.strftime('%Y%m%d')
    ds.AcquisitionTime = timeStr
    ds.SeriesDate = dt.strftime('%Y%m%d')
    ds.SeriesTime = timeStr
    ds.InstanceCreationDate = dt.strftime('%Y%m%d')
    ds.InstanceCreationTime = timeStr

    if UID is not None:

        # overwrite UIDs
        ds.PatientID = UID[0]
        ds.StudyInstanceUID = UID[1]
        ds.SeriesInstanceUID = UID[2]
        ds.SOPInstanceUID = UID[3]

    if ref is not None: 

        # Series, Instance and Class for Reference
        refd_instance = Dataset()
        refd_instance.ReferencedSOPClassUID = ref.SOPClassUID
        refd_instance.ReferencedSOPInstanceUID = ref.SOPInstanceUID
        refd_instance_sequence = Sequence()
        refd_instance_sequence.append(refd_instance)

        refd_series = Dataset()
        refd_series.ReferencedInstanceSequence = refd_instance_sequence
        refd_series.SeriesInstanceUID = ds.SeriesInstanceUID
        refd_series_sequence = Sequence()
        refd_series_sequence.append(refd_series)

        ds.ReferencedSeriesSequence = refd_series_sequence

    return ds







def mr_image_type(ds):
    """Determine if an image is Magnitude, Phase, Real or Imaginary image or None"""

    if (0x0043, 0x102f) in ds:
        private_ge = ds[0x0043, 0x102f]
        try: value = struct.unpack('h', private_ge.value)[0]
        except: value = private_ge.value
        if value == 0: return 'MAGNITUDE'
        if value == 1: return 'PHASE'
        if value == 2: return 'REAL'
        if value == 3: return 'IMAGINARY'
    if 'ImageType' in ds:
        type = set(ds.ImageType)
        if set(['M', 'MAGNITUDE']).intersection(type):
            return 'MAGNITUDE'
        if set(['P', 'PHASE']).intersection(type):
            return 'PHASE'
        if set(['R', 'REAL']).intersection(type):
            return 'REAL'
        if set(['I', 'IMAGINARY']).intersection(type):
            return 'IMAGINARY'
    if 'ComplexImageComponent' in ds:
        return ds.ComplexImageComponent

def enhanced_mr_image_type(ds):
    """Determine if a dataset is Magnitude, Phase, Real or Imaginary"""

    flagMagnitude = []
    flagPhase = []
    flagReal = []
    flagImaginary = []
    for index, singleSlice in enumerate(ds.PerFrameFunctionalGroupsSequence):
        sequence = singleSlice.MRImageFrameTypeSequence[0]
        if hasattr(sequence, 'FrameType'):
            type = set(sequence.FrameType)
            if set(['M', 'MAGNITUDE']).intersection(type):
                flagMagnitude.append(index)
                continue
            elif set(['P', 'PHASE']).intersection(type):
                flagPhase.append(index)
                continue
            elif set(['R', 'REAL']).intersection(type):
                flagReal.append(index)
                continue
            elif set(['I', 'IMAGINARY']).intersection(type):
                flagImaginary.append(index)
                continue
        if hasattr(sequence, 'ComplexImageComponent'):
            type = set(sequence.ComplexImageComponent)
            if set(['M', 'MAGNITUDE']).intersection(type):
                flagMagnitude.append(index)
                continue
            elif set(['P', 'PHASE']).intersection(type):
                flagPhase.append(index)
                continue
            elif set(['R', 'REAL']).intersection(type):
                flagReal.append(index)
                continue
            elif set(['I', 'IMAGINARY']).intersection(type):
                flagImaginary.append(index)
                continue

    return flagMagnitude, flagPhase, flagReal, flagImaginary


def mr_image_signal_type(ds):
    """Determine if an image is Water, Fat, In-Phase, Out-phase image or None"""

    flagWater = False
    flagFat = False
    flagInPhase = False
    flagOutPhase = False
    if hasattr(ds, 'ImageType'):
        type = set(ds.ImageType)
        if set(['W', 'WATER']).intersection(type):
            flagWater = True
        elif set(['F', 'FAT']).intersection(type):# or ('B0' in dataset.ImageType) or ('FIELD_MAP' in dataset.ImageType):
            flagFat = True
        elif set(['IP', 'IN_PHASE']).intersection(type):
            flagInPhase = True
        elif set(['OP', 'OUT_PHASE']).intersection(type):
            flagOutPhase = True
    
    return flagWater, flagFat, flagInPhase, flagOutPhase

def enhanced_mr_image_signal_type(ds):
    """Determine if an image is Water, Fat, In-Phase, Out-phase image or None"""

    flagWater = False
    flagFat = False
    flagInPhase = False
    flagOutPhase = False
    type = ds.MRImageFrameTypeSequence[0]
    if hasattr(type, 'FrameType'):
        type = set(type.FrameType)
    elif hasattr(type, 'ComplexImageComponent'):
        type = set(type.ComplexImageComponent)
    else:
        return flagWater, flagFat, flagInPhase, flagOutPhase
    if set(['W', 'WATER']).intersection(type):
        flagWater = True
    elif set(['F', 'FAT']).intersection(type):
        flagFat = True
    elif set(['IP', 'IN_PHASE']).intersection(type):
        flagInPhase = True
    elif set(['OP', 'OUT_PHASE']).intersection(type):
        flagOutPhase = True

    return flagWater, flagFat, flagInPhase, flagOutPhase


def lut(ds):

    redColour = list(ds.RedPaletteColorLookupTableData)
    greenColour = list(ds.GreenPaletteColorLookupTableData)
    blueColour = list(ds.BluePaletteColorLookupTableData)
    redLut = list(struct.unpack('<' + ('H' * ds.RedPaletteColorLookupTableDescriptor[0]), bytearray(redColour)))
    greenLut = list(struct.unpack('<' + ('H' * ds.GreenPaletteColorLookupTableDescriptor[0]), bytearray(greenColour)))
    blueLut = list(struct.unpack('<' + ('H' * ds.BluePaletteColorLookupTableDescriptor[0]), bytearray(blueColour)))
    colours = np.transpose([redLut, greenLut, blueLut])
    normaliseFactor = int(np.power(2, ds.RedPaletteColorLookupTableDescriptor[2]))
    # Fast ColourTable loading
    colourTable = np.around(colours/normaliseFactor, decimals = 2)
    indexes = np.unique(colourTable, axis=0, return_index=True)[1]
    lut = [colourTable[index].tolist() for index in sorted(indexes)]
    # Full / Complete Colourmap - takes 20 seconds to load each image
    # lut = (colours/normaliseFactor).tolist()   
    
    return lut 


def colormap(ds):
    """Returns the colormap if there is any."""

    lookuptable = None
    if hasattr(ds, 'ContentLabel'):
        if ds.PhotometricInterpretation == 'PALETTE COLOR':
            colormap = ds.ContentLabel
        elif 'MONOCHROME' in ds.PhotometricInterpretation:
            colormap = 'gray'
    elif len(ds.dir("PaletteColor"))>=3 and ds.PhotometricInterpretation == 'PALETTE COLOR':
        colormap = 'custom'
        lookuptable = lut(ds)
    else:
        colormap = 'gray' # default

    return colormap, lookuptable


def set_colormap(ds, colormap=None, levels=None):

    #and (colormap != 'gray') removed from If statement below, so as to save gray colour tables
    if (colormap == 'gray'):
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.ContentLabel = ''
        if hasattr(ds, 'RedPaletteColorLookupTableData'):
            del (ds.RGBLUTTransferFunction, ds.RedPaletteColorLookupTableData,
                ds.GreenPaletteColorLookupTableData, ds.BluePaletteColorLookupTableData,
                ds.RedPaletteColorLookupTableDescriptor, ds.GreenPaletteColorLookupTableDescriptor,
                ds.BluePaletteColorLookupTableDescriptor)
    if ((colormap is not None)  and (colormap != 'custom') and (colormap != 'gray') 
        and (colormap != 'default') and isinstance(colormap, str)):
        ds.PhotometricInterpretation = 'PALETTE COLOR'
        ds.RGBLUTTransferFunction = 'TABLE'
        ds.ContentLabel = colormap
        stringType = 'US' # ('SS' if minValue < 0 else 'US')
        ds.PixelRepresentation = 0 # (1 if minValue < 0 else 0)
        pixelArray = ds.pixel_array
        minValue = int(np.amin(pixelArray))
        maxValue = int(np.amax(pixelArray))
        numberOfValues = int(maxValue - minValue)
        arrayForRGB = np.arange(0, numberOfValues)
        colorsList = cm.ScalarMappable(cmap=colormap).to_rgba(np.array(arrayForRGB), bytes=False)
        totalBytes = ds.BitsAllocated
        ds.add_new('0x00281101', stringType, [numberOfValues, minValue, totalBytes])
        ds.add_new('0x00281102', stringType, [numberOfValues, minValue, totalBytes])
        ds.add_new('0x00281103', stringType, [numberOfValues, minValue, totalBytes])
        ds.RedPaletteColorLookupTableData = bytes(np.array([int((np.power(
            2, totalBytes) - 1) * value) for value in colorsList[:, 0].flatten()]).astype('uint'+str(totalBytes)))
        ds.GreenPaletteColorLookupTableData = bytes(np.array([int((np.power(
            2, totalBytes) - 1) * value) for value in colorsList[:, 1].flatten()]).astype('uint'+str(totalBytes)))
        ds.BluePaletteColorLookupTableData = bytes(np.array([int((np.power(
            2, totalBytes) - 1) * value) for value in colorsList[:, 2].flatten()]).astype('uint'+str(totalBytes)))
    if levels is not None:
        ds.WindowCenter = levels[0]
        ds.WindowWidth = levels[1]