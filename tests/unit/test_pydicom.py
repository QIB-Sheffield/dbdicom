import os
import shutil
import numpy as np

import dbdicom.utils.pydicom as pydcm

#
# data
#

datapath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fixtures')
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

def test_rw():

    path = onefile
    tmp = create_tmp_database(path)
    files = [os.path.join(tmp, f) for f in os.listdir(tmp) if os.path.isfile(os.path.join(tmp, f))]
    ds = pydcm.read(files[0])
    copy = os.path.join(tmp, 'copy')
    pydcm.write(ds, copy)
    ds_copy = pydcm.read(copy)
    remove_tmp_database(tmp)

    assert ds.SOPInstanceUID == ds_copy.SOPInstanceUID


def test_to_set_type():

    path = onefile
    tmp = create_tmp_database(path)
    files = [os.path.join(tmp, f) for f in os.listdir(tmp) if os.path.isfile(os.path.join(tmp, f))]

    ds = pydcm.read(files[0])

    assert isinstance(pydcm.to_set_type(ds.PatientName), str)   # PN
    assert isinstance(pydcm.to_set_type(ds.AcquisitionTime), str)   # TM

    remove_tmp_database(tmp)


def test_get_values():

    path = onefile
    tmp = create_tmp_database(path)
    files = [os.path.join(tmp, f) for f in os.listdir(tmp) if os.path.isfile(os.path.join(tmp, f))]

    ds = pydcm.read(files[0])
    values = pydcm.get_values(ds, ['PatientName', 'AcquisitionTime'])

    remove_tmp_database(tmp)

    assert values[0] == '281949'
    assert values[1] == '075649.057496'

    
def test_set_values():

    path = onefile
    tmp = create_tmp_database(path)
    files = [os.path.join(tmp, f) for f in os.listdir(tmp) if os.path.isfile(os.path.join(tmp, f))]

    # set new values
    ds = pydcm.read(files[0])
    values = ['Anonymous', '000000.00']
    pydcm.set_values(ds, ['PatientName', 'AcquisitionTime'], values)

    # check if new values are preserved after writing
    copy = os.path.join(tmp, 'copy')
    pydcm.write(ds, copy)
    ds_copy = pydcm.read(copy)

    remove_tmp_database(tmp)

    assert ds.PatientName == values[0]
    assert ds.AcquisitionTime == values[1]
    assert ds_copy.PatientName == values[0]
    assert ds_copy.AcquisitionTime == values[1]


def test_read_dataframe():

    path = twofiles
    tmp = create_tmp_database(path)
    files = [os.path.join(tmp, f) for f in os.listdir(tmp) if os.path.isfile(os.path.join(tmp, f))]

    tags = ['InstanceNumber', 'PatientName', 'AcquisitionTime']

    ds0 = pydcm.read(files[0])
    ds1 = pydcm.read(files[1])
    v0 = pydcm.get_values(ds0, tags)
    v1 = pydcm.get_values(ds1, tags)

    df = pydcm.read_dataframe(tmp, files, tags)
    v0_df = df.iloc[0].values.tolist()
    v1_df = df.iloc[1].values.tolist()

    remove_tmp_database(tmp)

    assert v0 == v0_df
    assert v1 == v1_df

    

if __name__ == "__main__":

    # test_rw()
    # test_to_set_type()
    # test_get_values()
    # test_set_values()
    test_read_dataframe()

    print('-------------------------')
    print('pydicom passed all tests!')
    print('-------------------------')

