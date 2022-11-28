import pydicom

from pydicom.dataset import Dataset
from dbdicom.ds.dataset import DbDataset
from dbdicom.ds.types.xray_angiographic_image import XrayAngiographicImage
from dbdicom.ds.types.ct_image import CTImage
from dbdicom.ds.types.mr_image import MRImage
from dbdicom.ds.types.enhanced_mr_image import EnhancedMRImage
from dbdicom.ds.types.ultrasound_multiframe_image import UltrasoundMultiFrameImage

def SOPClass(SOPClassUID):

    if SOPClassUID == '1.2.840.10008.5.1.4.1.1.4':
        return 'MRImage'
    elif SOPClassUID == '1.2.840.10008.5.1.4.1.1.4.1':
        return 'EnhancedMRImage'
    elif SOPClassUID == '1.2.840.10008.5.1.4.1.1.2':
        return 'CTImage'
    elif SOPClassUID == '1.2.840.10008.5.1.4.1.1.12.2':
        return 'XrayAngiographicImage'
    elif SOPClassUID == '1.2.840.10008.5.1.4.1.1.3.1':
        return 'UltrasoundMultiFrameImage'
    else:
        return 'Instance'

def read_dataset(file, dialog=None):

    try:
        ds = pydicom.dcmread(file)
        # ds = pydicom.dcmread(file, force=True) # more robust but hides corrupted data
    except Exception as message:
        if dialog is not None:
            dialog.information(message)  
        raise FileNotFoundError(message)
    
    type = SOPClass(ds.SOPClassUID)
    if type == 'MRImage':
        return MRImage(ds)
    elif type == 'EnhancedMRImage':
        return EnhancedMRImage(ds)
    elif type == 'CTImage':
        return CTImage(ds)
    elif type == 'XrayAngiographicImage':
        return XrayAngiographicImage(ds)
    elif type == 'UltrasoundMultiFrameImage':
        return UltrasoundMultiFrameImage(ds)
    else:
        return DbDataset(ds)

def new_dataset(type):

    if type == 'MRImage':
        return MRImage()
    elif type == 'EnhancedMRImage':
        return EnhancedMRImage()
    elif type == 'CTImage':
        return CTImage()
    elif type == 'XrayAngiographicImage':
        return XrayAngiographicImage()
    elif type == 'UltrasoundMultiFrameImage':
        return UltrasoundMultiFrameImage()
    else:
        return Dataset()