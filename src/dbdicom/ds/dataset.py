"""A colections of tools to extend functionality of pydicom datasets."""

import os
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import cm
import nibabel as nib
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

    def write(self, file, status=None):
        write(self, file, status=status)

    def get_values(self, tags):
        return get_values(self, tags)

    def set_values(self, tags, values):
        return set_values(self, tags, values)

    def get_lut(self): 
        return get_lut(self)

    def set_lut(*args, **kwargs): 
        set_lut(*args, **kwargs)

    def get_colormap(self):
        return get_colormap(self)

    def set_colormap(*args, **kwargs):
        set_colormap(*args, **kwargs)

    def get_attribute_lut(self): # use _get_attribute to encode these
        return get_lut(self)

    def set_attribute_lut(*args, **kwargs): # use _set_attribute to encode these
        set_lut(*args, **kwargs)

    def get_attribute_colormap(self):
        return get_colormap(self)

    def set_attribute_colormap(*args, **kwargs):
        set_colormap(*args, **kwargs)

    # Should be just pixel_array to fit in with logic 
    # of custom attributes but conflicts with pydicom definition
    # go back to just array?
    def get_pixel_array(self):
        return get_pixel_array(self)

    def set_pixel_array(self, array, value_range=None):
        set_pixel_array(self, array, value_range=value_range)

    def get_attribute_affine_matrix(self):
        return affine_matrix(self)

    def affine_matrix(self):
        return affine_matrix(self)

    def get_attribute_window(self):
        return get_window(self)

    def map_mask_to(self, ds_target):
        return map_mask_to(self, ds_target)


def get_window(ds):
    """Centre and width of the pixel data after applying rescale slope and intercept"""

    if 'WindowCenter' in ds: 
        centre = ds.WindowCenter
    if 'WindowWidth' in ds: 
        width = ds.WindowWidth
    if centre is None or width is None:
        array = ds.get_pixel_array()
        #p = np.percentile(array, [25, 50, 75])
        min = np.min(array)
        max = np.max(array)
    if centre is None: 
        centre = (max+min)/2
        #centre = p[1]
    if width is None: 
        width = 0.9*(max-min)
        #width = p[2] - p[0]
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


def write(ds, file, status=None):
    try:
        # check if directory exists and create it if not
        dir = os.path.dirname(file)
        if not os.path.exists(dir):
            os.makedirs(dir)
        ds.save_as(file, write_like_original=False)
    except:
        msg = 'Cannot write to file \n' + file
        if status is not None:
            status.message(msg)
        else:
            print(msg)


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
        try:
            ds = pydicom.dcmread(file, force=True, specific_tags=tags)
        except:
            pass
        else:
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
            status.progress(i+1, len(files))
    df = pd.DataFrame(array, index = dicom_files, columns = tags)
    return df


def set_values(ds, tags, values):
    """Sets DICOM tags in the pydicom dataset in memory"""

    if not isinstance(tags, list): 
        tags = [tags]
        values = [values]
    for i, tag in enumerate(tags):
        if values[i] is None:
            if isinstance(tag, str):
                if hasattr(ds, tag):
                    # Setting standard DICOM attribute to None
                    del ds[tag]
                else:
                    # Setting custom attribute to None
                    if hasattr(ds, 'set_attribute_' + tag):
                        getattr(ds, 'set_attribute_' + tag)(values[i])  
            else: # hexadecimal tuple
                if tag in ds:
                    del ds[tag]
        else:
            if isinstance(tag, str):
                if hasattr(ds, tag):
                #if tag in ds:
                    ds[tag].value = values[i]
                else:
                    if hasattr(ds, 'set_attribute_' + tag):
                        getattr(ds, 'set_attribute_' + tag)(values[i])
                        continue
                    if not isinstance(tag, pydicom.tag.BaseTag):
                        tag = pydicom.tag.Tag(tag)
                    if not tag.is_private: # Add a new data element
                        VR = pydicom.datadict.dictionary_VR(tag)
                        ds.add_new(tag, VR, values[i])
                    else:
                        pass # for now
            else: # hexadecimal tuple
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
        return get_values(ds, [tags])[0]
            
    row = []  
    for tag in tags:
        value = None
        if isinstance(tag, str):
            if not hasattr(ds, tag):
                if hasattr(ds, 'get_attribute_' + tag):
                    value = getattr(ds, 'get_attribute_' + tag)()
            else:
                value = to_set_type(ds[tag].value)
        else: # tuple of hexadecimal values
            if tag in ds:
                value = to_set_type(ds[tag].value)
        row.append(value)
    return row


def to_set_type(value):
    """
    Convert pydicom datatypes to the python datatypes used to set the parameter.
    """

    if value.__class__.__name__ == 'MultiValue':
        return [to_set_type(v) for v in value]
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


def affine_matrix(ds):
    """Affine transformation matrix for a DICOM image"""

    return image.affine_matrix(
        ds.ImageOrientationPatient, 
        ds.ImagePositionPatient, 
        ds.PixelSpacing, 
        ds.SliceThickness)

def map_mask_to(ds_source, ds_target):
    """Map non-zero image pixels onto a target image.
    
    Overwrite pixel values in the target"""

    # Create a coordinate array of non-zero pixels
    coords = np.transpose(np.where(ds_source.get_pixel_array() != 0)) 
    coords = [[coord[0], coord[1], 0] for coord in coords] 
    coords = np.array(coords)

    # Determine coordinate transformation matrix
    affine_source = ds_source.affine_matrix()
    affine_target = ds_target.affine_matrix()
    source_to_target = np.linalg.inv(affine_target).dot(affine_source)

    # Apply coordinate transformation and interpolate (nearest neighbour)
    coords = nib.affines.apply_affine(source_to_target, coords)
    coords = np.round(coords).astype(int)
    # x = y = []
    # for r in coords:
    #     if r[2] == 0:
    #         if (0 <= r[0]) & (r[0] < ds_target.Columns):
    #             if (0 <= r[1]) & (r[1] < ds_target.Rows):
    #                 x.append(r[0])
    #                 y.append(r[1])
    # x = tuple(x)
    # y = tuple(y)
    x = tuple([c[0] for c in coords if (c[2] == 0) & (0 <= c[0]) & (c[0] < ds_target.Columns) & (0 <= c[1]) & (c[1] < ds_target.Rows)])
    y = tuple([c[1] for c in coords if (c[2] == 0) & (0 <= c[0]) & (c[0] < ds_target.Columns) & (0 <= c[1]) & (c[1] < ds_target.Rows)])
    # x = tuple([c[0] for c in coords if c[2] == 0])
    # y = tuple([c[1] for c in coords if c[2] == 0])

    # Set values in the target image
    # array = np.zeros((record.Rows, record.Columns))
    array = np.zeros((ds_target.Columns, ds_target.Rows))
    array[(x, y)] = 1.0

    return array

# List of all supported (matplotlib) colormaps

COLORMAPS =  ['cividis',  'magma', 'plasma', 'viridis', 
    'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
    'binary', 'gist_yarg', 'gist_gray', 'bone', 'pink',
    'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
    'hot', 'afmhot', 'gist_heat', 'copper',
    'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
    'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
    'twilight', 'twilight_shifted', 'hsv',
    'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
    'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'turbo',
    'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']

# Include support for DICOM natiove colormaps (see pydicom guide on working with pixel data)

def get_colormap(ds):
    """Returns the colormap if there is any."""

    # Hijacking this free text field to store the colormap
    # This should use ContentDescription instead (0070, 0081)
    # 
    if 'WindowCenterWidthExplanation' in ds:
        if ds.WindowCenterWidthExplanation in COLORMAPS:
            return ds.WindowCenterWidthExplanation


def set_colormap(ds, colormap=None):

    if colormap is None:
        ds.PhotometricInterpretation = 'MONOCHROME2'
        if hasattr(ds, 'WindowCenterWidthExplanation'):
            del ds.WindowCenterWidthExplanation
        if hasattr(ds, 'RGBLUTTransferFunction'):
            del ds.RGBLUTTransferFunction
        if hasattr(ds, 'GreenPaletteColorLookupTableData'):
            del ds.GreenPaletteColorLookupTableData
        if hasattr(ds, 'RedPaletteColorLookupTableData'):
            del ds.RedPaletteColorLookupTableData
        if hasattr(ds, 'BluePaletteColorLookupTableData'):
            del ds.BluePaletteColorLookupTableData
        if hasattr(ds, 'RedPaletteColorLookupTableDescriptor'):
            del ds.RedPaletteColorLookupTableDescriptor
        if hasattr(ds, 'GreenPaletteColorLookupTableDescriptor'):
            del ds.GreenPaletteColorLookupTableDescriptor
        if hasattr(ds, 'BluePaletteColorLookupTableDescriptor'):
            del ds.BluePaletteColorLookupTableDescriptor
    else:
        ds.WindowCenterWidthExplanation = colormap
        # Get a LUT as float numpy array with values in the range [0,1]
        RGBA = cm.ScalarMappable(cmap=colormap).to_rgba(np.arange(256))
        set_lut(ds, RGBA[:,:3])


def set_lut(ds, RGB):
    """Set RGB as float with values in range [0,1]"""

    ds.PhotometricInterpretation = 'PALETTE COLOR'

    RGB *= (np.power(2, ds.BitsAllocated) - 1)

    if ds.BitsAllocated == 8:
        RGB = RGB.astype(np.ubyte)
    elif ds.BitsAllocated == 16:
        RGB = RGB.astype(np.uint16)

    # Define the properties of the LUT
    ds.add_new('0x00281101', 'US', [255, 0, ds.BitsAllocated])
    ds.add_new('0x00281102', 'US', [255, 0, ds.BitsAllocated])
    ds.add_new('0x00281103', 'US', [255, 0, ds.BitsAllocated])

    # Scale the colorsList to the available range
    ds.RedPaletteColorLookupTableData = bytes(RGB[:,0])
    ds.GreenPaletteColorLookupTableData = bytes(RGB[:,1])
    ds.BluePaletteColorLookupTableData = bytes(RGB[:,2])


def get_lut(ds):
    """Return RGB as float with values in [0,1]"""

    if 'PhotometricInterpretation' not in ds:
        return None
    if ds.PhotometricInterpretation != 'PALETTE COLOR':
        return None

    if ds.BitsAllocated == 8:
        dtype = np.ubyte
    elif ds.BitsAllocated == 16:
        dtype = np.uint16
    
    R = ds.RedPaletteColorLookupTableData
    G = ds.GreenPaletteColorLookupTableData
    B = ds.BluePaletteColorLookupTableData

    R = np.frombuffer(R, dtype=dtype)
    G = np.frombuffer(G, dtype=dtype)
    B = np.frombuffer(B, dtype=dtype)

    R = R.astype(np.float32)
    G = G.astype(np.float32)
    B = B.astype(np.float32)

    R *= 1.0/(np.power(2, ds.RedPaletteColorLookupTableDescriptor[2]) - 1)
    G *= 1.0/(np.power(2, ds.GreenPaletteColorLookupTableDescriptor[2]) - 1)
    B *= 1.0/(np.power(2, ds.BluePaletteColorLookupTableDescriptor[2]) - 1)
    
    return np.transpose([R, G, B])


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
    array = image.clip(array.astype(np.float32), value_range=value_range)
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


# def _initialize(ds, UID=None, ref=None): # ds is pydicom dataset

#     # Date and Time of Creation
#     dt = datetime.now()
#     timeStr = dt.strftime('%H%M%S')  # long format with micro seconds

#     ds.ContentDate = dt.strftime('%Y%m%d')
#     ds.ContentTime = timeStr
#     ds.AcquisitionDate = dt.strftime('%Y%m%d')
#     ds.AcquisitionTime = timeStr
#     ds.SeriesDate = dt.strftime('%Y%m%d')
#     ds.SeriesTime = timeStr
#     ds.InstanceCreationDate = dt.strftime('%Y%m%d')
#     ds.InstanceCreationTime = timeStr

#     if UID is not None:

#         # overwrite UIDs
#         ds.PatientID = UID[0]
#         ds.StudyInstanceUID = UID[1]
#         ds.SeriesInstanceUID = UID[2]
#         ds.SOPInstanceUID = UID[3]

#     if ref is not None: 

#         # Series, Instance and Class for Reference
#         refd_instance = Dataset()
#         refd_instance.ReferencedSOPClassUID = ref.SOPClassUID
#         refd_instance.ReferencedSOPInstanceUID = ref.SOPInstanceUID

#         refd_series = Dataset()
#         refd_series.ReferencedInstanceSequence = Sequence([refd_instance])
#         refd_series.SeriesInstanceUID = ds.SeriesInstanceUID

#         ds.ReferencedSeriesSequence = Sequence([refd_series])

#     return ds
