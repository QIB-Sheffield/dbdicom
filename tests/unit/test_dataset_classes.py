import os
import pydicom

import dbdicom.utils.files as filetools
from dbdicom.dataset_classes.create import read_dataset, new_dataset
from dbdicom.dataset_classes.mr_image import MRImage
from dbdicom.dataset_classes.enhanced_mr_image import EnhancedMRImage

datapath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
onefile = os.path.join(datapath, 'ONEFILE')
multiframe = os.path.join(datapath, 'MULTIFRAME')


def test_MRImage():
    
    # create from template
    ds = MRImage()
    assert ds.BodyPartExamined == 'FAKE'

    # create from file
    files = filetools.all_files(onefile)
    ds = pydicom.dcmread(files[0])
    ds = MRImage(ds)
    assert ds.BodyPartExamined == 'BRAIN'


def test_EnhancedMRImage():
    
    # create from template
    ds = EnhancedMRImage()
    assert ds.file_meta.ImplementationVersionName == 'Philips MR 56.1'
    assert ds.SharedFunctionalGroupsSequence[0].ReferencedImageSequence[0].PurposeOfReferenceCodeSequence[0].ContextUID == '1.2.840.10008.6.1.508'
    assert ds.PerFrameFunctionalGroupsSequence[4].PlanePositionSequence[0].ImagePositionPatient[2] == -49.157058715820
    assert ds.ReferencedPerformedProcedureStepSequence[0].ReferencedSOPInstanceUID == '1.3.46.670589.11.71459.5.0.16828.2021061610573963005'

    # create from file
    files = filetools.all_files(multiframe)
    ds = pydicom.dcmread(files[0])
    ds = EnhancedMRImage(ds)
    assert ds.file_meta.ImplementationVersionName == 'Philips MR 56.1'
    assert ds.SharedFunctionalGroupsSequence[0].ReferencedImageSequence[0].PurposeOfReferenceCodeSequence[0].ContextUID == '1.2.840.10008.6.1.508'
    assert ds.PerFrameFunctionalGroupsSequence[4].PlanePositionSequence[0].ImagePositionPatient[2] == -49.157058715820
    assert ds.ReferencedPerformedProcedureStepSequence[0].ReferencedSOPInstanceUID == '1.3.46.670589.11.71459.5.0.16828.2021061610573963005'


def test_read_dataset():

    files = filetools.all_files(onefile)
    ds = read_dataset(files[0])
    assert ds.BodyPartExamined == 'BRAIN'

    files = filetools.all_files(multiframe)
    ds = read_dataset(files[0])
    assert ds.file_meta.ImplementationVersionName == 'Philips MR 56.1'
    assert ds.SharedFunctionalGroupsSequence[0].ReferencedImageSequence[0].PurposeOfReferenceCodeSequence[0].ContextUID == '1.2.840.10008.6.1.508'
    assert ds.PerFrameFunctionalGroupsSequence[4].PlanePositionSequence[0].ImagePositionPatient[2] == -49.157058715820
    assert ds.ReferencedPerformedProcedureStepSequence[0].ReferencedSOPInstanceUID == '1.3.46.670589.11.71459.5.0.16828.2021061610573963005'


def test_create_dataset():
    
    ds = new_dataset('MRImage')


if __name__ == "__main__":

    test_MRImage()
    test_EnhancedMRImage()
    test_read_dataset()
    test_create_dataset()

    print('-------------------------')
    print('objects passed all tests!')
    print('-------------------------')

