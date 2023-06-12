import os
import shutil
import pydicom

from dbdicom.ds import MRImage, EnhancedMRImage, read_dataset, new_dataset
import dbdicom.ds.dataset as dbdataset
import dbdicom.utils.files as filetools

#
# data
#

# top = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# templates = os.path.join(os.path.join(os.path.join(top, 'src'), 'dbdicom'), 'templates')
datapath = os.path.join(os.path.dirname(__file__), 'data')
twofiles = os.path.join(datapath, 'TWOFILES')
onefile = os.path.join(datapath, 'ONEFILE')
rider = os.path.join(datapath, 'RIDER')
zipped = os.path.join(datapath, 'ZIP')
multiframe = os.path.join(datapath, 'MULTIFRAME')



#
# Helper functions
#

def create_tmp_database(path):
    tmp = os.path.join(os.path.dirname(__file__), 'tmp')
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    shutil.copytree(path, tmp)
    return tmp

def remove_tmp_database(tmp):
    shutil.rmtree(tmp)

#
# Tests
#

def test_read_dataframe():

    path = twofiles
    tmp = create_tmp_database(path)
    files = [os.path.join(tmp, f) for f in os.listdir(tmp) if os.path.isfile(os.path.join(tmp, f))]

    tags = ['InstanceNumber', 'PatientName', 'AcquisitionTime']

    ds0 = dbdataset.read(files[0])
    ds1 = dbdataset.read(files[1])
    v0 = ds0.get_values(tags)
    v1 = ds1.get_values(tags)

    df = dbdataset.read_dataframe(files, tags, path=tmp)
    v0_df = df.iloc[0].values.tolist()
    v1_df = df.iloc[1].values.tolist()

    remove_tmp_database(tmp)

    assert v0 == v0_df
    assert v1 == v1_df


def test_codify():
    
    output_folder = os.path.join(os.path.dirname(__file__), 'tmp')
    if os.path.isdir(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    source = filetools.all_files(rider)
    output_file = os.path.join(output_folder, 'MRImage_test.py')
    dbdataset.codify(source[0], output_file, exclude_size=100)

    source = filetools.all_files(multiframe)
    output_file = os.path.join(output_folder, 'EnhancedMRImage_test.py')
    dbdataset.codify(source[0], output_file, exclude_size=100)

    path = os.path.join(datapath, 'VPH-Pelvis-CT')
    source = filetools.all_files(path)
    output_file = os.path.join(output_folder, 'CTImage_test.py')
    dbdataset.codify(source[0], output_file, exclude_size=100)

    path = os.path.join(datapath, 'XRayAngioUncompressed-dicom_viewer_0015')
    source = filetools.all_files(path)
    output_file = os.path.join(output_folder, 'XRayImage_test.py')
    dbdataset.codify(source[0], output_file, exclude_size=100)

    path = os.path.join(datapath, 'UltrasoundPaletteColor-dicom_viewer_0020')
    source = filetools.all_files(path)
    output_file = os.path.join(output_folder, 'USImage_test.py')
    dbdataset.codify(source[0], output_file, exclude_size=100)

    shutil.rmtree(output_folder)

def test_read_write():

    path = onefile
    tmp = create_tmp_database(path)
    files = filetools.all_files(tmp)
    ds = dbdataset.read(files[0])
    copy = os.path.join(tmp, 'copy')
    ds.write(copy)
    ds_copy = dbdataset.read(copy)
    remove_tmp_database(tmp)

    assert ds.SOPInstanceUID == ds_copy.SOPInstanceUID

def test_to_set_type():

    path = onefile
    tmp = create_tmp_database(path)
    files = filetools.all_files(tmp)

    ds = dbdataset.read(files[0])

    patient_name = dbdataset.to_set_type(ds.PatientName, pydicom.datadict.dictionary_VR('PatientName'))
    acq_time = dbdataset.to_set_type(ds.AcquisitionTime, pydicom.datadict.dictionary_VR('AcquisitionTime'))
    
    assert isinstance(patient_name, str)   # PN
    assert isinstance(acq_time, float)   # TM

    remove_tmp_database(tmp)

def test_modules():

    path = rider
    tmp = create_tmp_database(path)
    files = filetools.all_files(tmp)

    tags = dbdataset.module_patient()

    ds0 = dbdataset.read(files[0])
    ds1 = dbdataset.read(files[-1])
    v0 = ds0.get_values(tags)
    v1 = ds1.get_values(tags)

    assert v0[2] == 'RIDER Neuro MRI-3369019796'
    assert v1[2] == 'RIDER Neuro MRI-5244517593'


def test_get_values():

    path = onefile
    tmp = create_tmp_database(path)
    files = filetools.all_files(tmp)

    ds = dbdataset.read(files[0])
    values = ds.get_values(['ReferencedPatientSequence','PatientName', 'AcquisitionTime'])

    remove_tmp_database(tmp)

    assert values[0] is None
    assert values[1] == '281949'
    assert values[2] == 28609.057496

def test_set_values():

    path = onefile
    tmp = create_tmp_database(path)
    files = filetools.all_files(tmp)

    # set new values
    ds = dbdataset.read(files[0])
    values = ['Anonymous', 0.0]
    ds.set_values(['PatientName', 'AcquisitionTime'], values)

    # check if new values are preserved after writing
    copy = os.path.join(tmp, 'copy')
    ds.write(copy)
    ds_copy = dbdataset.read(copy)

    v = ds.get_values(['PatientName', 'AcquisitionTime'])
    v_copy = ds.get_values(['PatientName', 'AcquisitionTime'])

    assert v[0] == values[0]
    assert v[1] == values[1]
    assert v_copy[0] == values[0]
    assert v_copy[1] == values[1]

    # Check setting of None values (equivalent to deleting the data element)

    # set new values
    ds = dbdataset.read(files[0])
    values = [None, 0.0]
    ds.set_values(['PatientName', 'AcquisitionTime'], values)

    values = ds.get_values(['PatientName', 'AcquisitionTime'])
    assert None is values[0]

    # check if new values are preserved after writing
    copy = os.path.join(tmp, 'copy')
    ds.write(copy)
    ds_copy = dbdataset.read(copy)
    values = ds_copy.get_values(['PatientName', 'AcquisitionTime'])

    assert values[0] is None
    assert values[1] == 0.0

    remove_tmp_database(tmp)

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
    files = [f for f in files if os.path.basename(f) == 'IM_0010']
    ds = pydicom.dcmread(files[0])
    ds = EnhancedMRImage(ds)
    assert ds.file_meta.ImplementationVersionName == 'Philips MR 56.1'
    assert ds.SharedFunctionalGroupsSequence[0].ReferencedImageSequence[0].PurposeOfReferenceCodeSequence[0].ContextUID == '1.2.840.10008.6.1.508'
    assert ds.PerFrameFunctionalGroupsSequence[4].PlanePositionSequence[0].ImagePositionPatient[2] == -49.157058715820
    assert ds.ReferencedPerformedProcedureStepSequence[0].ReferencedSOPInstanceUID == '1.3.46.670589.11.71459.5.0.16828.2021061610573963005'


def test_read_dataset():

    files = filetools.all_files(onefile)
    files = [f for f in files if os.path.basename(f) == '1-011.dcm']
    ds = read_dataset(files[0])
    assert ds.BodyPartExamined == 'BRAIN'

    files = filetools.all_files(multiframe)
    files = [f for f in files if os.path.basename(f) == 'IM_0010']
    ds = read_dataset(files[0])
    assert ds.file_meta.ImplementationVersionName == 'Philips MR 56.1'
    assert ds.SharedFunctionalGroupsSequence[0].ReferencedImageSequence[0].PurposeOfReferenceCodeSequence[0].ContextUID == '1.2.840.10008.6.1.508'
    assert ds.PerFrameFunctionalGroupsSequence[4].PlanePositionSequence[0].ImagePositionPatient[2] == -49.157058715820
    assert ds.ReferencedPerformedProcedureStepSequence[0].ReferencedSOPInstanceUID == '1.3.46.670589.11.71459.5.0.16828.2021061610573963005'


def test_create_dataset():
    
    ds = new_dataset('MRImage')

def test_get_colormap():
    
    tmp = create_tmp_database(onefile)
    files = filetools.all_files(tmp)
    ds = read_dataset(files[0])
    assert None == ds.get_attribute_lut()
    assert None == ds.get_attribute_colormap()

if __name__ == "__main__":

    # test_read_write()
    # test_to_set_type()
    # test_get_values()
    # test_set_values()
    # test_modules()
    # test_read_dataframe()
    # test_codify()
    test_MRImage()
    # test_EnhancedMRImage()
    # test_read_dataset()
    # test_create_dataset()
    # test_get_colormap()
    
    print('-------------------------')
    print('dataset passed all tests!')
    print('-------------------------')
