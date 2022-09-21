import os
import shutil

import dbdicom.dataset as dbdataset
import dbdicom.utils.files as filetools

#
# data
#

top = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
templates = os.path.join(os.path.join(os.path.join(top, 'src'), 'dbdicom'), 'templates')
datapath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
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

    assert isinstance(dbdataset.to_set_type(ds.PatientName), str)   # PN
    assert isinstance(dbdataset.to_set_type(ds.AcquisitionTime), str)   # TM

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
    assert values[2] == '075649.057496'

def test_set_values():

    path = onefile
    tmp = create_tmp_database(path)
    files = filetools.all_files(tmp)

    # set new values
    ds = dbdataset.read(files[0])
    values = ['Anonymous', '000000.00']
    ds.set_values(['PatientName', 'AcquisitionTime'], values)

    # check if new values are preserved after writing
    copy = os.path.join(tmp, 'copy')
    ds.write(copy)
    ds_copy = dbdataset.read(copy)

    assert ds.PatientName == values[0]
    assert ds.AcquisitionTime == values[1]
    assert ds_copy.PatientName == values[0]
    assert ds_copy.AcquisitionTime == values[1]

    # Check setting of None values (equivalent to deleting the data element)

    # set new values
    ds = dbdataset.read(files[0])
    values = [None, '000000.00']
    ds.set_values(['PatientName', 'AcquisitionTime'], values)

    values = ds.get_values(['PatientName', 'AcquisitionTime'])
    assert None is values[0]

    # check if new values are preserved after writing
    copy = os.path.join(tmp, 'copy')
    ds.write(copy)
    ds_copy = dbdataset.read(copy)
    values = ds_copy.get_values(['PatientName', 'AcquisitionTime'])

    assert values[0] is None
    assert values[1] == '000000.00'

    remove_tmp_database(tmp)

if __name__ == "__main__":

    test_read_write()
    test_to_set_type()
    test_get_values()
    test_set_values()
    test_modules()
    test_read_dataframe()
    test_codify()
    
    print('-------------------------')
    print('dataset passed all tests!')
    print('-------------------------')
