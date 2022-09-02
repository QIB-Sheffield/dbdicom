import os
import shutil
import numpy as np

from dbdicom.dbindex import DbIndex
import dbdicom.utils.pydicom as pydcm

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

def test_type():

    tmp = create_tmp_database(rider)
    dbi = DbIndex()
    dbi.open(tmp)

    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    patient = 'RIDER Neuro MRI-5244517593'
    assert dbi.type(series) == 'SeriesInstanceUID'
    assert dbi.type(patient) == 'PatientID'
    assert dbi.type('Database') == 'Database'
    assert dbi.type() is None

def test_keys():

    tmp = create_tmp_database(rider)
    dbi = DbIndex()
    dbi.open(tmp)

    keys = dbi.keys()
    assert keys == []

    keys = dbi.keys('Database')
    assert 24 == len(keys)

    # Patient
    keys = dbi.keys(patient='RIDER Neuro MRI-3369019796')
    assert 12 == len(keys)
    keys = dbi.keys(
        patient = [
            'RIDER Neuro MRI-3369019796',
            'RIDER Neuro MRI-5244517593',
            ]
        )
    assert 24 == len(keys)

    # Study
    keys = dbi.keys(study='1.3.6.1.4.1.9328.50.16.139212415517744996696223190563412871264')
    assert 4 == len(keys)
    keys = dbi.keys(
        study = [
            '1.3.6.1.4.1.9328.50.16.139212415517744996696223190563412871264',
            '1.3.6.1.4.1.9328.50.16.10388995917728582202615060907395571964',
            ]
        )
    assert 6 == len(keys)

    # Series
    keys = dbi.keys(series='1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242')
    assert 2 == len(keys)
    keys = dbi.keys(
        series = [
            '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242',
            '1.3.6.1.4.1.9328.50.16.163870745718873861235299152775293374260',
            '1.3.6.1.4.1.9328.50.16.39537076883396884954303966295061604769'
            ]
        )
    assert 6 == len(keys)

    # Datasets
    keys = dbi.keys(dataset='1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022')
    assert 1 == len(keys)
    keys = dbi.keys(
        dataset = [
            '1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022',
            '1.3.6.1.4.1.9328.50.16.180826285525298270030493768487939618219',
            '1.3.6.1.4.1.9328.50.16.243004851579310565813723110219735642931',
            '1.3.6.1.4.1.9328.50.16.68617558011242394133101958688682703605',
            ]
        )
    assert 4 == len(keys)

    # Entering non-existent entries
    keys = dbi.keys(series='1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022')
    assert 0 == len(keys)
    keys = dbi.keys('abc')
    assert 0 == len(keys)

    # Mixed arguments
    keys = dbi.keys(
        patient = 'RIDER Neuro MRI-3369019796', 
        study = '1.3.6.1.4.1.9328.50.16.139212415517744996696223190563412871264')
    assert 12 == len(keys)

    keys = dbi.keys(
        patient = 'RIDER Neuro MRI-3369019796', 
        study = [
            '1.3.6.1.4.1.9328.50.16.139212415517744996696223190563412871264',
            '1.3.6.1.4.1.9328.50.16.168701627691879645008036315574545460110',
            ]
        )
    assert 12 == len(keys)

    keys = dbi.keys(
        patient = 'RIDER Neuro MRI-3369019796', 
        study = '1.3.6.1.4.1.9328.50.16.10388995917728582202615060907395571964',
        )
    assert 14 == len(keys)

    keys = dbi.keys(
        patient = 'RIDER Neuro MRI-5244517593',
        series = '1.3.6.1.4.1.9328.50.16.233636027937248405570338506686080257722')
    assert 12 == len(keys)

    keys = dbi.keys(
        patient = 'RIDER Neuro MRI-5244517593',
        series = [
            '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242',
            '1.3.6.1.4.1.9328.50.16.39537076883396884954303966295061604769'
            ]
        )
    assert 16 == len(keys)

    keys = dbi.keys(
        patient = 'RIDER Neuro MRI-5244517593',
        series = [
            '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242',
            '1.3.6.1.4.1.9328.50.16.39537076883396884954303966295061604769'
            ],
        dataset = [
            '1.3.6.1.4.1.9328.50.16.8428147229483429640255033081444174773',
            '1.3.6.1.4.1.9328.50.16.302536631490375078745691772311115895736',
            '1.3.6.1.4.1.9328.50.16.68617558011242394133101958688682703605'],
        )
    assert 18 == len(keys)

    # Any UID
    keys = dbi.keys(uid = 'RIDER Neuro MRI-5244517593')
    assert 12 == len(keys) 

    keys = dbi.keys(
        uid = [
            'RIDER Neuro MRI-5244517593',
            '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242',
            '1.3.6.1.4.1.9328.50.16.39537076883396884954303966295061604769',
            '1.3.6.1.4.1.9328.50.16.8428147229483429640255033081444174773',
            '1.3.6.1.4.1.9328.50.16.302536631490375078745691772311115895736',
            '1.3.6.1.4.1.9328.50.16.68617558011242394133101958688682703605'
            ],
        )
    assert 18 == len(keys)   

    keys = dbi.keys(
            [None, 
            'RIDER Neuro MRI-5244517593',
            '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242',
            '1.3.6.1.4.1.9328.50.16.39537076883396884954303966295061604769',
            '1.3.6.1.4.1.9328.50.16.8428147229483429640255033081444174773',
            '1.3.6.1.4.1.9328.50.16.302536631490375078745691772311115895736',
            '1.3.6.1.4.1.9328.50.16.68617558011242394133101958688682703605'
            ],
        )
    assert 18 == len(keys) 

    keys = dbi.keys(
            [ None, 
            'Database', 
            'RIDER Neuro MRI-5244517593',
            '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242',
            '1.3.6.1.4.1.9328.50.16.39537076883396884954303966295061604769',
            '1.3.6.1.4.1.9328.50.16.8428147229483429640255033081444174773',
            '1.3.6.1.4.1.9328.50.16.302536631490375078745691772311115895736',
            '1.3.6.1.4.1.9328.50.16.68617558011242394133101958688682703605'
            ],
        )
    assert 24 == len(keys)         

    keys = dbi.keys([None], patient = None)
    assert keys == []

def test_value():

    tmp = create_tmp_database(rider)
    dbi = DbIndex()
    dbi.open(tmp)

    key = dbi.keys('Database')[5]
    assert dbi.value(key, 'PatientName') == '281949'
    assert dbi.value(key, 'StudyDescription') == 'BRAIN^RESEARCH'
    assert dbi.value(key, 'SeriesDescription') == 'ax 10 flip'
    assert dbi.value(key, 'InstanceNumber') == 9
    assert None is dbi.value(key, 'dummy')

    key = dbi.keys('RIDER Neuro MRI-5244517593')
    arr = dbi.value(key, ['PatientName','StudyDescription'])
    assert arr.shape == (12, 2)
    assert arr[0,0] == '281949'
    assert 'BRAIN^ROUTINE BRAIN' in arr[:,1] 

def test_parent():

    tmp = create_tmp_database(rider)
    dbi = DbIndex()
    dbi.open(tmp)

    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    patient = 'RIDER Neuro MRI-5244517593'
    assert dbi.parent(series) == '1.3.6.1.4.1.9328.50.16.139212415517744996696223190563412871264'
    assert dbi.parent(patient) == 'Database'
    assert dbi.parent() is None
    assert dbi.parent('abc') is None

def test_children():

    tmp = create_tmp_database(rider)
    dbi = DbIndex()
    dbi.open(tmp)

    instance = '1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022'
    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    patient = 'RIDER Neuro MRI-5244517593'
    assert len(dbi.children(series)) == 2
    assert len(dbi.children(patient)) == 4
    assert dbi.type(dbi.children(patient)[0]) == 'StudyInstanceUID'
    assert dbi.children(dbi.children(series)[0]) == []
    assert len(dbi.children('Database')) == 2
    assert [] == dbi.children(instance)
    assert [] == dbi.children()
    assert 6 == len(dbi.children([series, patient]))

def test_instances():

    tmp = create_tmp_database(rider)
    dbi = DbIndex()
    dbi.open(tmp)

    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    patient = 'RIDER Neuro MRI-5244517593'
    assert len(dbi.instances(series)) == 2
    assert len(dbi.instances(patient)) == 12
    assert len(dbi.instances('Database')) == 24
    assert dbi.instances() == []

def test_series():

    tmp = create_tmp_database(rider)
    dbi = DbIndex()
    dbi.open(tmp)

    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    patient = 'RIDER Neuro MRI-5244517593'
    assert len(dbi.series(series)) == 1
    assert len(dbi.series(patient)) == 6
    assert dbi.series() == []

def test_studies():

    tmp = create_tmp_database(rider)
    dbi = DbIndex()
    dbi.open(tmp)

    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    patient = 'RIDER Neuro MRI-5244517593'
    assert len(dbi.studies(series)) == 1
    assert len(dbi.studies(patient)) == 4

def test_patients():

    tmp = create_tmp_database(rider)
    dbi = DbIndex()
    dbi.open(tmp)

    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    patient = 'RIDER Neuro MRI-5244517593'
    assert len(dbi.patients(series)) == 1
    assert len(dbi.patients(patient)) == 1
    assert len(dbi.patients('Database')) == 2
    assert dbi.patients() == []

def test_label():

    tmp = create_tmp_database(rider)
    dbi = DbIndex()
    dbi.open(tmp)

    patient = dbi.patients('Database')[0]
    assert dbi.label(patient) in ['Patient 281949 [RIDER Neuro MRI-3369019796]','Patient 281949 [RIDER Neuro MRI-5244517593]']
    study = dbi.studies(patient)[0]
    try:
        dbi.label(study)
    except: 
        assert False
    else:
        assert True
    series = dbi.series(study)[0]
    try:
        dbi.label(series)
    except: 
        assert False
    else:
        assert True
    instance = dbi.instances(series)[0]
    try:
        dbi.label(instance)
    except:
        assert False
    else:
        assert True

    remove_tmp_database(tmp)

def test_print():

    tmp = create_tmp_database(rider)

    dbi = DbIndex()
    dbi.open(tmp)
    try:
        dbi.print()
    except:
        assert False
    else:
        assert True

    remove_tmp_database(tmp)

def test_read_and_clear():

    tmp = create_tmp_database(rider)
    dbi = DbIndex()
    dbi.open(tmp)

    assert 0 == len(dbi._datasets)
    dbi.read('Database')
    assert 24 == len(dbi._datasets)
    dbi.clear('Database')
    assert 0 == len(dbi._datasets)
    dbi.read(patient='RIDER Neuro MRI-3369019796')
    assert 12 == len(dbi._datasets)
    dbi.clear('Database')
    dbi.read(study='1.3.6.1.4.1.9328.50.16.139212415517744996696223190563412871264')
    assert 4 == len(dbi._datasets)
    dbi.clear('Database')
    dbi.read(series='1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242')
    assert 2 == len(dbi._datasets)
    dbi.clear('Database')
    dbi.read(dataset='1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022')
    assert 1 == len(dbi._datasets)
    dbi.clear('Database')

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

    uid = '1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022'

    # Change a name in memory, write, clear memory and read again to test this has changed.
    dbi.read(dataset=uid)
    dbi.set_values(uid, 'PatientName', 'Anonymous')
    dbi.write('Database')
    dbi.clear('Database')
    dbi.read(dataset=uid)
    datasets = dbi.datasets(dbi.keys(dataset=uid))
    assert 1 == len(datasets)
    assert uid == datasets[0].SOPInstanceUID
    assert 'Anonymous' == datasets[0].PatientName

    # Read a series, change all of the datasets, write only one before clearing memory.
    # Then check that only one of the values has changed
    series_uid='1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    dbi.read(series=series_uid)
    dbi.set_values(series_uid, 'PatientName', 'Anonymous')
    dbi.write('Database')
    dbi.clear('Database')
    dbi.read(dataset=uid)
    for ds in dbi.datasets(dbi.keys(dataset=uid)):
        assert ds.PatientName == 'Anonymous'

    remove_tmp_database(tmp)

def test_open_close():

    tmp = create_tmp_database(rider)
    dbi = DbIndex(tmp)
    dbi.open()
    assert dbi.dataframe.shape == (24, 14)
    dbi.save()
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
    dbi.save()
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


def test_datasets():

    tmp = create_tmp_database(rider)
    dbi = DbIndex()
    dbi.open(tmp)

    # Get a list of all datasets from disk
    ds = dbi.datasets(dbi.keys('Database'))
    assert 24 == len(ds)

    # Get a list of all datasets from memory
    dbi.read('Database')
    ds = dbi.datasets(dbi.keys('Database'))
    dbi.clear('Database')
    assert 24 == len(ds)

    # Read all datasets for one series from disk
    series_uid='1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    ds = dbi.datasets(dbi.keys(series=series_uid))
    assert 2 == len(ds)

    # Read one of the datasets first, check that the result is the same
    uid = '1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022'
    dbi.read(dataset=uid)
    ds = dbi.datasets(dbi.keys(series=series_uid))
    dbi.clear('Database')
    assert 2 == len(ds)

    remove_tmp_database(tmp)

def test_delete():
    
    tmp = create_tmp_database(rider)
    dbi = DbIndex()
    dbi.open(tmp)

    # Delete all datasets for one series
    uid = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    dbi.delete(series=uid)
    assert len(dbi.keys('Database')) == 22

    # Delete it again
    dbi.delete(series=uid)
    assert len(dbi.keys('Database')) == 22
    assert dbi.keys(series=uid) == []

    # Delete the patient containing this series
    dbi.delete(patient='RIDER Neuro MRI-3369019796')
    assert len(dbi.keys('Database')) == 12
    assert len(dbi.keys(patient='RIDER Neuro MRI-5244517593')) == 12
    assert len(dbi.keys('RIDER Neuro MRI-5244517593')) == 12
    assert len(dbi.keys('RIDER Neuro MRI-3369019796')) == 0

    # Delete the other patient too
    dbi.delete(patient='RIDER Neuro MRI-5244517593')
    assert dbi.keys('Database') == []

    remove_tmp_database(tmp)

def test_copy_to_series():

    tmp = create_tmp_database(rider)
    dbi = DbIndex()
    dbi.open(tmp)

    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    orig_keys = dbi.keys(patient='RIDER Neuro MRI-5244517593')
    copy_keys = dbi.copy_to_series('RIDER Neuro MRI-5244517593', series)
    assert len(dbi.keys(series)) == 14
    assert len(dbi.keys('RIDER Neuro MRI-5244517593')) == 12
    assert len(dbi.keys('RIDER Neuro MRI-3369019796')) == 24
    assert len(dbi.keys('Database')) == 36
    assert dbi.value(orig_keys[0], 'PatientID') == 'RIDER Neuro MRI-5244517593'
    assert dbi.value(copy_keys[0], 'PatientID') == 'RIDER Neuro MRI-3369019796'
    # Check that all new instance numbers are unique
    nrs = dbi.value(dbi.keys(series), 'InstanceNumber')
    assert len(set(nrs)) == len(nrs) 
    remove_tmp_database(tmp)

def test_copy_to_study():

    tmp = create_tmp_database(rider)
    dbi = DbIndex()
    dbi.open(tmp)

    study = '1.3.6.1.4.1.9328.50.16.168701627691879645008036315574545460110'
    patient = 'RIDER Neuro MRI-5244517593'
    assert len(dbi.series(study)) == 1
    assert len(dbi.series(patient)) == 6
    orig_keys = dbi.keys(patient=patient)
    copy_keys = dbi.copy_to_study(patient, study)
    assert len(dbi.instances(study)) == 14
    assert len(dbi.series(study)) == 7
    assert len(dbi.instances('RIDER Neuro MRI-5244517593')) == 12
    assert len(dbi.instances('RIDER Neuro MRI-3369019796')) == 24
    assert len(dbi.instances('Database')) == 36
    assert dbi.value(orig_keys[0], 'PatientID') == 'RIDER Neuro MRI-5244517593'
    assert dbi.value(copy_keys[0], 'PatientID') == 'RIDER Neuro MRI-3369019796'
    nrs = dbi.get_values(study, 'SeriesNumber')
    assert len(nrs) == len(dbi.series(study))
    remove_tmp_database(tmp)

def test_copy_to_patient():

    tmp = create_tmp_database(rider)
    dbi = DbIndex()
    dbi.open(tmp)

    study = '1.3.6.1.4.1.9328.50.16.168701627691879645008036315574545460110'
    patient = 'RIDER Neuro MRI-5244517593'
    assert len(dbi.series(study)) == 1
    assert len(dbi.series(patient)) == 6
    assert len(dbi.instances(study)) == 2
    assert len(dbi.instances(patient)) == 12
    dbi.copy_to_patient(study, patient)
    assert len(dbi.series(study)) == 1
    assert len(dbi.instances(study)) == 2
    assert len(dbi.series(patient)) == 7
    assert len(dbi.instances(patient)) == 14
    remove_tmp_database(tmp)

def test_copy_to():

    tmp = create_tmp_database(rider)
    dbi = DbIndex()
    dbi.open(tmp)

    instance ='1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022'
    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    study = '1.3.6.1.4.1.9328.50.16.168701627691879645008036315574545460110'
    patient = 'RIDER Neuro MRI-5244517593'  
    assert len(dbi.instances(series)) == 2
    dbi.copy_to(instance, series)   # copy an instance to its own series.
    assert len(dbi.instances(series)) == 3
    assert len(dbi.series(study)) == 1
    dbi.copy_to(series, study) # copy a series to another study
    assert len(dbi.series(study)) == 2
    assert len(dbi.studies(patient)) == 4
    dbi.copy_to(study, patient) # copy a study to another patient
    assert len(dbi.studies(patient)) == 5
    remove_tmp_database(tmp)


def test_move_to_series():

    tmp = create_tmp_database(rider)
    dbi = DbIndex()
    dbi.open(tmp)

    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    orig_keys = dbi.keys(patient='RIDER Neuro MRI-5244517593')
    copy_keys = dbi.move_to_series('RIDER Neuro MRI-5244517593', series)
    assert len(dbi.keys(series)) == 14
    assert len(dbi.keys('RIDER Neuro MRI-5244517593')) == 0
    assert len(dbi.keys('RIDER Neuro MRI-3369019796')) == 24
    assert len(dbi.keys('Database')) == 24
    assert dbi.value(orig_keys[0], 'PatientID') == 'RIDER Neuro MRI-5244517593'
    assert dbi.value(copy_keys[0], 'PatientID') == 'RIDER Neuro MRI-3369019796'
    # Check that all new instance numbers are unique
    nrs = dbi.value(dbi.keys(series), 'InstanceNumber')
    assert len(set(nrs)) == len(nrs)
    remove_tmp_database(tmp)


def test_move_to_study():

    tmp = create_tmp_database(rider)
    dbi = DbIndex()
    dbi.open(tmp)

    study = '1.3.6.1.4.1.9328.50.16.168701627691879645008036315574545460110'
    patient = 'RIDER Neuro MRI-5244517593'
    assert len(dbi.series(study)) == 1
    assert len(dbi.series(patient)) == 6
    copy_keys = dbi.move_to_study(patient, study) # move to a study of another patient
    assert len(dbi.instances(study)) == 14
    assert len(dbi.series(study)) == 7
    assert len(dbi.instances(patient)) == 0
    assert len(dbi.instances('RIDER Neuro MRI-3369019796')) == 24
    assert len(dbi.instances('Database')) == 24
    assert dbi.get_values(patient, 'PatientID') is None
    assert dbi.value(copy_keys[0], 'PatientID') == 'RIDER Neuro MRI-3369019796'
    nrs = dbi.get_values(study, 'SeriesNumber')
    assert len(nrs) == len(dbi.series(study))
    remove_tmp_database(tmp)

def test_move_to_patient():

    tmp = create_tmp_database(rider)
    dbi = DbIndex()
    dbi.open(tmp)

    study = '1.3.6.1.4.1.9328.50.16.168701627691879645008036315574545460110'
    patient = 'RIDER Neuro MRI-5244517593'
    assert len(dbi.series(study)) == 1
    assert len(dbi.series(patient)) == 6
    assert len(dbi.instances(study)) == 2
    assert len(dbi.instances(patient)) == 12
    dbi.move_to_patient(study, patient)
    assert len(dbi.series(study)) == 1
    assert len(dbi.instances(study)) == 2
    assert len(dbi.series(patient)) == 7
    assert len(dbi.instances(patient)) == 14
    remove_tmp_database(tmp)

def test_move_to():

    tmp = create_tmp_database(rider)
    dbi = DbIndex()
    dbi.open(tmp)

    instance ='1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022'
    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    study = '1.3.6.1.4.1.9328.50.16.168701627691879645008036315574545460110'
    patient = 'RIDER Neuro MRI-5244517593'  
    assert len(dbi.instances(series)) == 2
    dbi.move_to(instance, series)   # move an instance to its own series.
    assert dbi.get_values(instance, 'SeriesInstanceUID') == series
    assert len(dbi.instances(series)) == 2
    assert len(dbi.series(study)) == 1
    dbi.move_to(series, study) # copy a series to another study
    assert dbi.get_values(series, 'StudyInstanceUID') == study
    assert len(dbi.series(study)) == 2
    assert len(dbi.studies(patient)) == 4
    dbi.move_to(study, patient) # move a study to another patient
    assert dbi.get_values(study, 'PatientID') == patient
    assert len(dbi.studies(patient)) == 5
    remove_tmp_database(tmp)


def test_set_values():

    tmp = create_tmp_database(rider)
    dbi = DbIndex()
    dbi.open(tmp)

    patient = 'RIDER Neuro MRI-5244517593'
    for name in dbi.value(dbi.keys(patient), 'PatientName'):
        assert name != 'Anonymous'
    for t in dbi.value(dbi.keys(patient), 'SeriesDescription'):
        assert t != 'Desc'
    for key in dbi.keys(patient):
        assert pydcm.get_values(dbi.dataset(key), 'AcquisitionTime') != '000000.00'

    dbi.set_values(patient, ['PatientName', 'SeriesDescription', 'AcquisitionTime'], ['Anonymous', 'Desc', '000000.00'])

    for name in dbi.value(dbi.keys(patient), 'PatientName'):
        assert name == 'Anonymous'
    for t in dbi.value(dbi.keys(patient), 'SeriesDescription'):
        assert t == 'Desc'
    for key in dbi.keys(patient):
        assert pydcm.get_values(dbi.dataset(key), 'AcquisitionTime') == '000000.00'

    remove_tmp_database(tmp)


def test_get_values():

    tmp = create_tmp_database(rider)
    dbi = DbIndex()
    dbi.open(tmp)

    assert dbi.get_values('', 'PatientName') is None

    patient = 'RIDER Neuro MRI-5244517593'
    attributes = ['PatientName','SeriesDescription','AcquisitionTime']

    values = dbi.get_values(patient, 'PatientName')
    assert values == '281949'

    values = dbi.get_values(patient, 'SeriesDescription')
    assert set(values) == set(['ax 10 flip', 'ax 5 flip', 'sag 3d gre +c'])

    values = dbi.get_values(patient, 'AcquisitionTime')
    assert set(values) == set(['074424.477508', '074443.472496', '074615.472480', '074634.479981', '075505.670007', '075611.105017'])

    values = dbi.get_values(patient, attributes)
    assert values[0] == '281949'
    assert set(values[1]) == set(['ax 10 flip', 'ax 5 flip', 'sag 3d gre +c'])
    assert set(values[2]) == set(['074424.477508', '074443.472496', '074615.472480', '074634.479981', '075505.670007', '075611.105017'])

    values = dbi.get_values(patient, 'PatientName')
    assert values == '281949'

    patient = [patient, 'RIDER Neuro MRI-3369019796']
    values = dbi.get_values(patient, attributes)
    assert values[0][0] == '281949'
    assert set(values[0][1]) == set(['ax 10 flip', 'ax 5 flip', 'sag 3d gre +c'])
    assert set(values[0][2]) == set(['074424.477508', '074443.472496', '074615.472480', '074634.479981', '075505.670007', '075611.105017'])
    assert values[1][0] == '281949'
    assert set(values[1][1]) == set(['ax 10 flip', 'ax 5 flip', 'sag 3d gre +c'])
    assert set(values[1][2]) == set(['073903.155010', '073921.462495', '074511.204995', '074529.432496', '075138.639989', '075649.057496'])

    remove_tmp_database(tmp)

def test_restore():

    tmp = create_tmp_database(rider)
    dbi = DbIndex()
    dbi.open(tmp)

    patient = 'RIDER Neuro MRI-5244517593'

    assert '' == dbi.get_values(patient, 'PatientSex')
    dbi.set_values(patient, 'PatientSex', 'M')
    assert 'M' == dbi.get_values(patient, 'PatientSex')
    dbi.restore('Database')
    assert '' == dbi.get_values(patient, 'PatientSex')

    dbi.read(patient)
    assert '' == dbi.get_values(patient, 'PatientSex')
    dbi.set_values(patient, 'PatientSex', 'M')
    assert 'M' == dbi.get_values(patient, 'PatientSex')
    dbi.restore('Database')
    assert '' == dbi.get_values(patient, 'PatientSex')

    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    study = '1.3.6.1.4.1.9328.50.16.168701627691879645008036315574545460110'

    assert len(dbi.series(study)) == 1
    dbi.move_to(series, study) 
    assert len(dbi.series(study)) == 2
    dbi.restore('Database')
    assert len(dbi.series(study)) == 1

    dbi.read(series)
    assert len(dbi.series(study)) == 1
    dbi.move_to(series, study) 
    assert len(dbi.series(study)) == 2
    dbi.restore('Database')
    assert len(dbi.series(study)) == 1

    remove_tmp_database(tmp)

def test_save():

    tmp = create_tmp_database(rider)
    dbi = DbIndex()
    dbi.open(tmp)

    # Three objects that are not nested: study not in patient and series not in study.
    patient = 'RIDER Neuro MRI-5244517593'
    study = '1.3.6.1.4.1.9328.50.16.168701627691879645008036315574545460110'
    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'

    assert '' == dbi.get_values(patient, 'PatientSex')
    dbi.set_values(patient, 'PatientSex', 'M')
    assert 'M' == dbi.get_values(patient, 'PatientSex')
    dbi.save('Database')
    assert 'M' == dbi.get_values(patient, 'PatientSex')

    dbi.read(patient)
    assert 'M' == dbi.get_values(patient, 'PatientSex')
    dbi.set_values(patient, 'PatientSex', '')
    assert '' == dbi.get_values(patient, 'PatientSex')
    dbi.save('Database')
    assert '' == dbi.get_values(patient, 'PatientSex')

    assert len(dbi.series(study)) == 1
    dbi.move_to(series, study) 
    assert len(dbi.series(study)) == 2
    dbi.save('Database')
    assert len(dbi.series(study)) == 2

    dbi.read(series)
    assert len(dbi.series(study)) == 2
    dbi.move_to(series, study) 
    assert len(dbi.series(study)) == 2
    dbi.save('Database')
    dbi.restore()
    assert len(dbi.series(study)) == 2

    remove_tmp_database(tmp)

def test_new_patient():

    tmp = create_tmp_database(rider)
    dbi = DbIndex()
    dbi.open(tmp)

    n = len(dbi.patients('Database'))
    patient = dbi.new_patient()
    assert len(dbi.patients('Database')) == n+1
    assert [] == dbi.children(patient) 
    assert [] == dbi.instances(patient)
    assert 1 == len(dbi.keys(patient))
    try:
        dbi.print()
    except:
        assert False
    else:
        assert True

    remove_tmp_database(tmp)

def test_new_study():

    tmp = create_tmp_database(rider)
    dbi = DbIndex()
    dbi.open(tmp)

    n = len(dbi.studies('Database'))
    study = dbi.new_study()
    assert len(dbi.studies('Database')) == n+1
    assert [] == dbi.children(study)
    assert [] == dbi.instances(study)
    assert 1 == len(dbi.keys(study))

    patient = 'RIDER Neuro MRI-5244517593'
    n = len(dbi.studies(patient))
    study = dbi.new_study(patient)
    assert len(dbi.studies(patient)) == n+1
    assert 0 == len(dbi.children(study))

    remove_tmp_database(tmp)

def test_new_series():

    tmp = create_tmp_database(rider)
    dbi = DbIndex()
    dbi.open(tmp)

    n = len(dbi.series('Database'))
    series = dbi.new_series()
    assert len(dbi.series('Database')) == n+1
    assert [] == dbi.children(series)
    assert [] == dbi.instances(series)
    assert 1 == len(dbi.keys(series))

    study = '1.3.6.1.4.1.9328.50.16.168701627691879645008036315574545460110'
    n = len(dbi.series(study))
    series = dbi.new_series(study)
    assert len(dbi.series(study)) == n+1
    assert [] == dbi.children(series)

    remove_tmp_database(tmp)


if __name__ == "__main__":

    test_init()
    test_read_dataframe()
    test_rw_df()
    test_multiframe_to_singleframe()
    test_scan()
    test_type()
    test_keys()
    test_value()
    test_parent()
    test_children()
    test_instances()
    test_series()
    test_studies()
    test_patients()
    test_label()
    test_print()
    test_read_and_clear()
    test_write()
    test_open_close()
    test_inmemory_vs_ondisk() # This may need some revision
    test_datasets()
    test_delete()
    test_copy_to_series()
    test_copy_to_study()
    test_copy_to_patient()
    test_copy_to()
    test_move_to_series()
    test_move_to_study()
    test_move_to_patient()
    test_move_to()
    test_set_values()
    test_get_values()
    test_restore()
    test_save()
    test_new_patient()
    test_new_study()
    test_new_series()

    # new_cousin, new_sibling, new_pibling  

    # Next steps:
    # Merge, group
    # import and export
    # include attributes in children, instances, series etc to filter on.
    

    print('-------------------------')
    print('dbindex passed all tests!')
    print('-------------------------')

