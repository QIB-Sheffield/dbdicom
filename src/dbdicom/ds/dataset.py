"""A colections of tools to extend functionality of pydicom datasets."""

import os
import struct
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import cm

import pydicom
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.util.codify import code_file

import dbdicom.utils.image as image


class DbDataset(Dataset):

    def __init__(self, dataset=None):
        super().__init__()

        if dataset is not None:
            self.__dict__ = dataset.__dict__

    def write(self, file, dialog=None):
        write(self, file, dialog=dialog)

    def get_values(self, tags):
        return get_values(self, tags)

    def set_values(self, tags, values):
        return set_values(self, tags, values)

    def get_lut(self):
        return get_lut(self)

    def get_colormap(self):
        return get_colormap(self)

    def set_colormap(*args, **kwargs):
        set_colormap(*args, **kwargs)

    def get_pixel_array(self):
        return get_pixel_array(self)

    def set_pixel_array(self, array, value_range=None):
        set_pixel_array(self, array, value_range=value_range)

    def affine_matrix(self):
        return affine_matrix(self)

    def get_window(self):
        return get_window(self)


def get_window(ds):
    """Centre and width of the pixel data after applying rescale slope and intercept"""

    if 'WindowCenter' in ds: 
        centre = ds.WindowCenter
    if 'WindowWidth' in ds: 
        width = ds.WindowWidth
    if centre is None or width is None:
        array = ds.get_pixel_array()
    if centre is None: 
        centre = np.median(array)
    if width is None: 
        p = np.percentile(array, [25, 75])
        width = p[1] - p[0]
    return centre, width


def read(file, dialog=None):

    try:
        ds = pydicom.dcmread(file)
        return DbDataset(ds)
    except:
        message = "Failed to read " + file
        if dialog is not None:
            dialog.information(message)  
        raise FileNotFoundError(message)


def write(ds, file, dialog=None): 

    try:
        # check if directory exists and create it if not
        dir = os.path.dirname(file)
        if not os.path.exists(dir):
            os.makedirs(dir)
        ds.save_as(file, write_like_original=False) 
    except Exception as message:
        if dialog is not None:
            dialog.information(message) 
        else:
            print(message) 

def codify(source_file, save_file, **kwargs):
    
    str = code_file(source_file, **kwargs)
    file = open(save_file, "w")
    file.write(str)
    file.close()

def read_dataframe(files, tags, status=None, path=None, message='Reading DICOM folder..'):
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
                if path is None:
                    index = file
                else:
                    index = os.path.relpath(file, path)
                dicom_files.append(index) 
        if status is not None: 
            status.progress(i+1, len(files), message)
    return pd.DataFrame(array, index = dicom_files, columns = tags)

 

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


def get_values(ds, tags):
    """Return a list of values for a dataset"""

    # https://pydicom.github.io/pydicom/stable/guides/element_value_types.html
    if not isinstance(tags, list): 
        if tags not in ds:
            value = None
            if isinstance(tags, str):
                if hasattr(ds, tags):
                    value = getattr(ds, tags)()
            return value
        else:
        #    return ds[tags].value
            return to_set_type(ds[tags].value)
            
    row = []  
    for tag in tags:
        if tag not in ds:
            value = None
            if isinstance(tag, str):
                if hasattr(ds, tag):
                    value = getattr(ds, tag)()
        else:
        #    value = ds[tag].value
            value = to_set_type(ds[tag].value)
        row.append(value)
    return row


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

def new_uid(n=None):
    
    if n is None:
        return pydicom.uid.generate_uid()
    else:
        return [pydicom.uid.generate_uid() for _ in range(n)]


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
        # if isinstance(ds, pydicom.dataset.FileDataset):
        #     if 'TransferSyntaxUID' in ds.file_meta:
        row = get_values(ds, tags)
        uid = get_values(ds, 'SOPInstanceUID')
        array.append(row)
        indices.append(uid) 
    return pd.DataFrame(array, index=indices, columns=tags)


def affine_matrix(ds):
    """Affine transformation matrix for a DICOM image"""

    return image.affine_matrix(
        ds.ImageOrientationPatient, 
        ds.ImagePositionPatient, 
        ds.PixelSpacing, 
        ds.SliceThickness)


def get_lut(ds):

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


def get_colormap(ds):
    """Returns the colormap if there is any."""

    lookuptable = None
    if hasattr(ds, 'ContentLabel'):
        if ds.PhotometricInterpretation == 'PALETTE COLOR':
            colormap = ds.ContentLabel
        elif 'MONOCHROME' in ds.PhotometricInterpretation:
            colormap = 'gray'
    elif len(ds.dir("PaletteColor"))>=3 and ds.PhotometricInterpretation == 'PALETTE COLOR':
        colormap = 'custom'
        lookuptable = get_lut(ds)
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


def get_pixel_array(ds):
    """Read the pixel array from an image"""

    array = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, 'RescaleSlope', 1)) 
    intercept = float(getattr(ds, 'RescaleIntercept', 0)) 
    array *= slope
    array += intercept
    
    return np.transpose(array)


def set_pixel_array(ds, array, value_range=None):
    
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

        refd_series = Dataset()
        refd_series.ReferencedInstanceSequence = Sequence([refd_instance])
        refd_series.SeriesInstanceUID = ds.SeriesInstanceUID

        ds.ReferencedSeriesSequence = Sequence([refd_series])

    return ds
