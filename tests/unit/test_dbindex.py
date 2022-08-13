import os
import shutil
import numpy as np

from dbdicom.dbindex import DbIndex

datapath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fixtures')
twofiles = os.path.join(datapath, 'TWOFILES')
onefile = os.path.join(datapath, 'ONEFILE')
rider = os.path.join(datapath, 'RIDER')
zipped = os.path.join(datapath, 'ZIP')
multiframe = os.path.join(datapath, 'MULTIFRAME')

# Helper functions

def create_tmp_database(path):
    tmp = os.path.join(os.path.dirname(__file__), 'tmp')
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    shutil.copytree(path, tmp)
    return tmp

def remove_tmp_database(tmp):
    shutil.rmtree(tmp)


def test_init():

    dbi = DbIndex()
    assert dbi.dataframe is None

    tmp = create_tmp_database(rider)

    dbi = DbIndex(tmp)
    assert dbi.dataframe is None
    assert dbi.path == tmp

    remove_tmp_database(tmp)


def test_read_dataframe():

    tmp = create_tmp_database(rider)
    dbi = DbIndex(tmp)
    dbi.read_dataframe()
    assert dbi.dataframe.shape == (24, 2+len(dbi.columns))
    remove_tmp_database(tmp)

def test_rw_df():

    tmp = create_tmp_database(twofiles)
    dbi = DbIndex(tmp)
    dbi.read_dataframe()
    dbi._write_df()
    df1 = dbi.dataframe
    dbi._read_df()
    df2 = dbi.dataframe

    remove_tmp_database(tmp)

    assert np.array_equal(df1.to_numpy(), df2.to_numpy())

def test_multiframe_to_singleframe():

    tmp = create_tmp_database(multiframe)
    dbi = DbIndex(tmp)
    dbi.read_dataframe()
    assert dbi.dataframe.shape == (2, 14)
    dbi._multiframe_to_singleframe()
    assert dbi.dataframe.shape == (124, 14)
    remove_tmp_database(tmp)

def test_scan():

    tmp = create_tmp_database(rider)
    dbi = DbIndex(tmp)
    dbi.scan()
    assert dbi.dataframe.shape == (24, 14)
    remove_tmp_database(tmp)

def test_open_close():

    tmp = create_tmp_database(rider)
    dbi = DbIndex(tmp)
    dbi.open()
    assert dbi.dataframe.shape == (24, 14)
    dbi.close()
    assert dbi.dataframe is None
    try:
        dbi.open()
    except ValueError:
        assert True
    else:
        assert False
    dbi.close()
    remove_tmp_database(tmp)

    tmp = create_tmp_database(twofiles)
    dbi.open(tmp)
    assert dbi.dataframe.shape == (2, 14)
    dbi.close()
    assert dbi.dataframe is None
    remove_tmp_database(tmp)

def test_inmemory_vs_ondisk():

    tmp = create_tmp_database(rider)

    dbi = DbIndex()

    # open a database on disk
    # get the dataframe and close it again
    dbi.open(tmp)   
    df = dbi.dataframe
    dbi.close()
    assert df.shape == (24, 14)
    assert dbi.dataframe is None

    remove_tmp_database(tmp)

    # create a database in memory
    # Try to read a dataframe and check 
    # that this produces an exception
    dbi.dataframe = df
    assert dbi.dataframe.shape == (24, 14)
    try:
        dbi.read_dataframe()
    except ValueError:
        assert True
    else:
        assert False

def test_keys():

    tmp = create_tmp_database(rider)
    dbi = DbIndex()
    dbi.open(tmp)

    keys = dbi.keys()
    assert 24 == len(keys)
    keys = dbi.keys(patient='RIDER Neuro MRI-3369019796')
    assert 12 == len(keys)
    keys = dbi.keys(study='1.3.6.1.4.1.9328.50.16.139212415517744996696223190563412871264')
    assert 4 == len(keys)
    keys = dbi.keys(series='1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242')
    assert 2 == len(keys)
    keys = dbi.keys(dataset='1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022')
    assert 1 == len(keys)
    keys = dbi.keys(series='1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022')
    assert 0 == len(keys)

def test_value():

    tmp = create_tmp_database(rider)
    dbi = DbIndex()
    dbi.open(tmp)

    key = dbi.keys()[5]
    assert dbi.value(key, 'PatientName') == '281949'
    assert dbi.value(key, 'StudyDescription') == 'BRAIN^RESEARCH'
    assert dbi.value(key, 'SeriesDescription') == 'ax 10 flip'
    assert dbi.value(key, 'InstanceNumber') == 9
    try:
        dbi.value(key, 'dummy')
    except:
        assert True
    else:
        assert False

def test_label():

    tmp = create_tmp_database(rider)

    dbi = DbIndex()
    dbi.open(tmp)
    key = dbi.keys()[5]

    assert dbi.label(key, 'Patient') == 'Patient 281949 [RIDER Neuro MRI-3369019796]'
    assert dbi.label(key, 'Study') == 'Study BRAIN^RESEARCH [19040321]'
    assert dbi.label(key, 'Series') == 'Series 007 [ax 10 flip]'
    assert dbi.label(key, 'Instance') == 'MRImage 000009'

    remove_tmp_database(tmp)


def test_read_and_clear():

    tmp = create_tmp_database(rider)
    dbi = DbIndex()
    dbi.open(tmp)

    assert 0 == len(dbi._datasets)
    dbi.read()
    assert 24 == len(dbi._datasets)
    dbi.clear()
    assert 0 == len(dbi._datasets)
    dbi.read(patient='RIDER Neuro MRI-3369019796')
    assert 12 == len(dbi._datasets)
    dbi.clear()
    dbi.read(study='1.3.6.1.4.1.9328.50.16.139212415517744996696223190563412871264')
    assert 4 == len(dbi._datasets)
    dbi.clear()
    dbi.read(series='1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242')
    assert 2 == len(dbi._datasets)
    dbi.clear()
    dbi.read(dataset='1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022')
    assert 1 == len(dbi._datasets)
    dbi.clear()

    # Try to read something that does not exist
    dbi.read(series='1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022')
    assert 0 == len(dbi._datasets)

    # read a patient, then a study in that patient - reading study does nothing
    dbi.read(patient='RIDER Neuro MRI-3369019796')
    assert 12 == len(dbi._datasets)
    dbi.read(study='1.3.6.1.4.1.9328.50.16.139212415517744996696223190563412871264')
    assert 12 == len(dbi._datasets)

    # Clear only that study
    dbi.clear(study='1.3.6.1.4.1.9328.50.16.139212415517744996696223190563412871264')
    assert 8 == len(dbi._datasets)

    # Clear a dataset from the study that was just cleared (does nothing)
    dbi.clear(dataset='1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022')
    assert 8 == len(dbi._datasets)

    # Clear all datasets from the patient
    dbi.clear(patient='RIDER Neuro MRI-3369019796')
    assert 0 == len(dbi._datasets)

    remove_tmp_database(tmp)


def test_write():

    tmp = create_tmp_database(rider)
    dbi = DbIndex()
    dbi.open(tmp)

    # read a dataset and check its properties.
    uid = '1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022'

    datasets = dbi.read(dataset=uid)
    assert 1 == len(datasets)
    assert uid == datasets[0].SOPInstanceUID
    assert '281949' == datasets[0].PatientName

    # Change the name in memory, clear memory and read again to test nothing has changed.
    datasets[0].PatientName = 'Anonymous' # Note this is not the correct way of setting values - use dbi.set_value() instead.
    dbi.clear()
    datasets = dbi.read(dataset=uid)
    assert 1 == len(datasets)
    assert uid == datasets[0].SOPInstanceUID
    assert '281949' == datasets[0].PatientName

    # Do the same thing but now write before clearing memory, and test that the name has now changed
    datasets = dbi.read(dataset=uid)
    datasets[0].PatientName = 'Anonymous' # Note this is not the correct way of setting values - use dbi.set_value() instead.
    dbi.write()
    dbi.clear()
    datasets = dbi.read(dataset=uid)
    assert 1 == len(datasets)
    assert uid == datasets[0].SOPInstanceUID
    assert 'Anonymous' == datasets[0].PatientName

    # Read a series, change all of the datasets, write only one before clearing memory.
    # Then check that only one of the values has changed
    series_uid='1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    for ds in dbi.read(series=series_uid):
        ds.PatientName = 'Anonymous'
    dbi.write(dataset=uid)
    dbi.clear()
    for ds in dbi.read(series=series_uid):
        if ds.SOPInstanceUID == uid:
            assert ds.PatientName == 'Anonymous'
        else:
            assert ds.PatientName == '281949'

    # Read a series, change all of the datasets, write all of them before clearing memory.
    # Then check that all of the values has changed
    for ds in dbi.read(series=series_uid):
        ds.PatientName = 'Anonymous'
    dbi.write(series=series_uid)
    dbi.clear()
    for ds in dbi.read(series=series_uid):
        assert ds.PatientName == 'Anonymous'

    remove_tmp_database(tmp)


def test_datasets():

    tmp = create_tmp_database(rider)
    dbi = DbIndex()
    dbi.open(tmp)

    # Get a list of all datasets from disk
    ds = dbi.datasets()
    assert 24 == len(ds)

    # Get a list of all datasets from memory
    dbi.read()
    ds = dbi.datasets()
    dbi.clear()
    assert 24 == len(ds)

    # Read all datasets for one series from disk
    series_uid='1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    ds = dbi.datasets(series=series_uid)
    assert 2 == len(ds)

    # Read one of the datasets first, check that the result is the same
    uid = '1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022'
    dbi.read(dataset=uid)
    ds = dbi.datasets(series=series_uid)
    dbi.clear()
    assert 2 == len(ds)

    remove_tmp_database(tmp)


if __name__ == "__main__":

    test_init()
    test_read_dataframe()
    test_rw_df()
    test_multiframe_to_singleframe()
    test_scan()
    test_open_close()
    test_inmemory_vs_ondisk() # This may need some revision
    test_keys()
    test_value()
    test_label()
    test_read_and_clear()
    test_write()
    test_datasets()

    # Next steps:
    # Copy, delete, move, merge, group - correctly flagging removed/created, revisit close
    # save and restore from memory and from disk
    # set_values in datasets, correctly also changing entries in dataframe and flagging removed/create
    # read_values from memory or disk as appropriate.
    

    print('-------------------------')
    print('dbindex passed all tests!')
    print('-------------------------')

