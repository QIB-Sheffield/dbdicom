# Coded version of DICOM file 
# 'RIDER Neuro MRI-3369019796\03-21-1904-BRAINRESEARCH-00598\14.000000-sag 3d gre c-04769\1-010.dcm'
# Produced by pydicom codify utility script
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence

def rider():

    ds = Dataset()

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
    ds.ImagePositionPatient = [75.561665058136, -163.6216506958, 118.50172901154]
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
    ds.PixelData = np.arange(ds.Rows*ds.Columns, dtype=np.uint16)*ds.LargestImagePixelValue/(ds.Rows*ds.Columns)

    return ds