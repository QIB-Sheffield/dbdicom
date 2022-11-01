# Coded version of DICOM file
# 'RIDER Neuro MRI-3369019796\03-21-1904-BRAINRESEARCH-00598\14.000000-sag 3d gre c-04769\1-010.dcm'
# Produced by pydicom codify utility script
import struct
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence

from dbdicom.ds.dataset import DbDataset
import dbdicom.utils.image as image


class MRImage(DbDataset):

    def __init__(self, dataset=None, template=None):
        super().__init__()

        if (dataset is None) and (template is None):
            template = 'RIDER'

        if dataset is not None:
            self.__dict__ = dataset.__dict__

        if template == 'RIDER':
            rider(self)

    def get_pixel_array(self):
        return get_pixel_array(self)

    def set_pixel_array(self, array):
        set_pixel_array(self, array)

    def get_attribute_image_type(self):
        return get_attribute_image_type(self)

    def set_attribute_image_type(self, value):
        set_attribute_image_type(self, value)

    def get_attribute_signal_type(self):
        return get_attribute_signal_type(self)

    def set_attribute_signal_type(self, value):
        set_attribute_signal_type(self, value)


def rider(ds):  # required only - check

    # File meta info data elements
    ds.file_meta = FileMetaDataset()
    ds.file_meta.FileMetaInformationGroupLength = 190
    ds.file_meta.FileMetaInformationVersion = b'\x00\x01'
    ds.file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.4'
    ds.file_meta.MediaStorageSOPInstanceUID = '1.3.6.1.4.1.9328.50.16.175333593952805976694548436931998383940'
    ds.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2'
    ds.file_meta.ImplementationClassUID = '1.2.40.0.13.1.1'
    ds.file_meta.ImplementationVersionName = 'dcm4che-1.4.27'

    ds.is_implicit_VR = True
    ds.is_little_endian = True

    # Main data elements
    ds.SpecificCharacterSet = 'ISO_IR 100'
    ds.ImageType = ['ORIGINAL', 'PRIMARY', 'M', 'ND', 'NORM']
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.4'
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.StudyDate = '19040321'
    ds.ContentDate = '19040321'
    ds.StudyTime = ''
    ds.AcquisitionTime = '075649.057496'
    ds.ContentTime = ''
    ds.AccessionNumber = '2819497684894126'
    ds.Modality = 'MR'
    ds.Manufacturer = 'SIEMENS'
    ds.ReferringPhysicianName = ''
    ds.StationName = ''
    ds.StudyDescription = 'BRAIN^RESEARCH'
    ds.SeriesDescription = 'sag 3d gre +c'
    ds.ManufacturerModelName = ''
    ds.ReferencedSOPClassUID = '1.3.6.1.4.1.9328.50.16.295504506656781074046411123909869020125'
    ds.ReferencedSOPInstanceUID = '1.3.6.1.4.1.9328.50.16.303143938897288157958328401346374476407'
    ds.PatientName = '281949'
    ds.PatientID = pydicom.uid.generate_uid()
    ds.PatientBirthDate = ''
    ds.PatientSex = ''
    ds.PatientIdentityRemoved = 'YES'
    ds.DeidentificationMethod = 'CTP:NBIA Default w/ extra date removal:20100323:172722'
    ds.ContrastBolusAgent = 'Magnevist'
    ds.BodyPartExamined = 'FAKE'
    ds.ScanningSequence = 'GR'
    ds.SequenceVariant = 'SP'
    ds.ScanOptions = ''
    ds.MRAcquisitionType = '3D'
    ds.SequenceName = '*fl3d1'
    ds.AngioFlag = 'N'
    ds.SliceThickness = '1.0'
    ds.RepetitionTime = '8.6'
    ds.EchoTime = '4.11'
    ds.NumberOfAverages = '1.0'
    ds.ImagingFrequency = '63.676701'
    ds.ImagedNucleus = '1H'
    ds.EchoNumbers = '0'
    ds.MagneticFieldStrength = '1.4939999580383'
    ds.NumberOfPhaseEncodingSteps = '224'
    ds.EchoTrainLength = '1'
    ds.PercentSampling = '100.0'
    ds.PercentPhaseFieldOfView = '100.0'
    ds.PixelBandwidth = '150.0'
    ds.DeviceSerialNumber = '25445'
    ds.SoftwareVersions = 'syngo MR 2004V 4VB11D'
    ds.ProtocolName = 'sag 3d gre +c'
    ds.ContrastBolusVolume = '20.0'
    ds.DateOfLastCalibration = '19031229'
    ds.TimeOfLastCalibration = '155156.000000'
    ds.TransmitCoilName = 'Body'
    ds.InPlanePhaseEncodingDirection = 'ROW'
    ds.FlipAngle = '20.0'
    ds.VariableFlipAngleFlag = 'N'
    ds.SAR = '0.09494107961655'
    ds.dBdt = '0.0'
    ds.PatientPosition = 'HFS'
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.StudyID = ''
    ds.SeriesNumber = '14'
    ds.AcquisitionNumber = '1'
    ds.InstanceNumber = '1'
    ds.ImagePositionPatient = [
        75.561665058136, -163.6216506958, 118.50172901154]
    ds.ImageOrientationPatient = [0, 1, 0, 0, 0, -1]
    ds.FrameOfReferenceUID = '1.3.6.1.4.1.9328.50.16.22344679587635360510174487884943834158'
    ds.PositionReferenceIndicator = ''
    ds.SliceLocation = '75.561665058136'
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.Rows = 64
    ds.Columns = 64
    ds.PixelSpacing = [1, 1]
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SmallestImagePixelValue = 0
    ds.LargestImagePixelValue = 913
    ds.WindowCenter = '136.0'
    ds.WindowWidth = '380.0'
    ds.RescaleIntercept = '0.0'
    ds.RescaleSlope = '1.0'
    ds.RescaleType = 'PIXELVALUE'
    ds.WindowCenterWidthExplanation = 'Algo1'
    ds.RequestedProcedureDescription = 'MRI BRAIN W/WO ENHANCEMENT'
    ds.ScheduledProcedureStepDescription = 'MRI BRAIN W/WO ENHANCEMENT'
    ds.ScheduledProcedureStepID = '5133240'
    ds.PerformedProcedureStepStartDate = '19040611'
    ds.PerformedProcedureStepDescription = 'MRI BRAIN W/WO ENHANCEMENT'
    ds.RequestAttributesSequence = Sequence()
    ds.RequestedProcedureID = '5133240'
    ds.StorageMediaFileSetUID = '1.3.6.1.4.1.9328.50.16.162890465625511526068665093825399871205'
    ds.PixelData = np.arange(ds.Rows * ds.Columns, dtype=np.uint16) * \
        ds.LargestImagePixelValue / (ds.Rows * ds.Columns)

    return ds


def get_pixel_array(ds):
    """Read the pixel array from an MR image"""

    #array = ds.pixel_array.astype(np.float64)
    #array = ds.pixel_array
    #array = np.frombuffer(ds.PixelData, dtype=np.uint16).reshape(ds.Rows, ds.Columns)
    #array = array.astype(np.float32)

    array = ds.pixel_array.astype(np.float32)
    if [0x2005, 0x100E] in ds:  # 'Philips Rescale Slope'
        slope = ds[(0x2005, 0x100E)].value
        intercept = ds[(0x2005, 0x100D)].value
        array -= intercept
        array /= slope
    else:
        slope = float(getattr(ds, 'RescaleSlope', 1))
        intercept = float(getattr(ds, 'RescaleIntercept', 0))
        array *= slope
        array += intercept
    return np.transpose(array)


def set_pixel_array(ds, array):

    if (0x2005, 0x100E) in ds:
        del ds[0x2005, 0x100E]  # Delete 'Philips Rescale Slope'
    if (0x2005, 0x100D) in ds:
        del ds[0x2005, 0x100D]

    if array.ndim >= 3:  # remove spurious dimensions of 1
        array = np.squeeze(array)

    #ds.BitsAllocated = 32
    #ds.BitsStored = 32
    #ds.HighBit = 31

    # room for speed up
    # clipping may slow down a lot
    # max/min are calculated multiple times
    array = image.clip(array)
    maximum = np.amax(array)
    minimum = np.amin(array)
    array, slope, intercept = image.scale_to_range(array, ds.BitsAllocated)
    array = np.transpose(array)

    ds.PixelRepresentation = 0
    ds.SmallestImagePixelValue = int(maximum)
    ds.LargestImagePixelValue = int(minimum)
    ds.RescaleSlope = 1 / slope
    ds.RescaleIntercept = - intercept / slope
#        ds.WindowCenter = (maximum + minimum) / 2
#        ds.WindowWidth = maximum - minimum
    ds.Rows = array.shape[0]
    ds.Columns = array.shape[1]
    ds.PixelData = array.tobytes()


def get_attribute_image_type(ds):
    """Determine if an image is Magnitude, Phase, Real or Imaginary image or None"""

    if (0x0043, 0x102f) in ds:
        private_ge = ds[0x0043, 0x102f]
        try:
            value = struct.unpack('h', private_ge.value)[0]
        except BaseException:
            value = private_ge.value
        if value == 0:
            return 'MAGNITUDE'
        if value == 1:
            return 'PHASE'
        if value == 2:
            return 'REAL'
        if value == 3:
            return 'IMAGINARY'

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

    return 'UNKNOWN'


def set_attribute_image_type(ds, value):
    ds.ImageType = value


def get_attribute_signal_type(ds):
    """Determine if an image is Water, Fat, In-Phase, Out-phase image or None"""

    if hasattr(ds, 'ImageType'):
        type = set(ds.ImageType)
        if set(['W', 'WATER']).intersection(type):
            return 'WATER'
        elif set(['F', 'FAT']).intersection(type):
            return 'FAT'
        elif set(['IP', 'IN_PHASE']).intersection(type):
            return 'IN_PHASE'
        elif set(['OP', 'OUT_PHASE']).intersection(type):
            return 'OP_PHASE'
    return 'UNKNOWN'


def set_attribute_signal_type(ds, value):
    ds.ImageType = value
