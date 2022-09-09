import dbdicom.templates.MRImage as MRImage
import dbdicom.templates.EnhancedMRImage as EnhancedMRImage


def test_MRImage():
    
    ds = MRImage.rider()
    assert ds.BodyPartExamined == 'FAKE'

def test_EnhancedMRImage():
    
    ds = EnhancedMRImage.ukrin_maps()
    assert ds.file_meta.ImplementationVersionName == 'Philips MR 56.1'
    assert ds.SharedFunctionalGroupsSequence[0].ReferencedImageSequence[0].PurposeOfReferenceCodeSequence[0].ContextUID == '1.2.840.10008.6.1.508'
    assert ds.PerFrameFunctionalGroupsSequence[4].PlanePositionSequence[0].ImagePositionPatient[2] == -49.157058715820
    assert ds.ReferencedPerformedProcedureStepSequence[0].ReferencedSOPInstanceUID == '1.3.46.670589.11.71459.5.0.16828.2021061610573963005'


if __name__ == "__main__":

    test_MRImage()
    test_EnhancedMRImage()


    print('---------------------------')
    print('templates passed all tests!')
    print('---------------------------')

