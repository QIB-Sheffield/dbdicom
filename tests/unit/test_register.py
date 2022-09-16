import os
import shutil
import numpy as np

from dbdicom.register import DbRegister
from dbdicom.dataset_classes.mr_image import MRImage


datapath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fixtures')
twofiles = os.path.join(datapath, 'TWOFILES')
onefile = os.path.join(datapath, 'ONEFILE')
rider = os.path.join(datapath, 'RIDER')
zipped = os.path.join(datapath, 'ZIP')
multiframe = os.path.join(datapath, 'MULTIFRAME')

# Helper functions

def create_tmp_database(path=None, name='tmp'):
    tmp = os.path.join(os.path.dirname(__file__), name)
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    if path is not None:
        shutil.copytree(path, tmp)
    else:
        os.makedirs(tmp)
    return tmp

def remove_tmp_database(tmp):
    shutil.rmtree(tmp)


def test_init():

    dbr = DbRegister()
    assert dbr.dataframe.empty

    tmp = create_tmp_database(rider)

    dbr = DbRegister(tmp)
    assert dbr.dataframe.empty
    assert dbr.path == tmp

    remove_tmp_database(tmp)


def test_read_dataframe():

    tmp = create_tmp_database(rider)
    dbr = DbRegister(tmp)
    dbr.read_dataframe()
    assert dbr.dataframe.shape == (24, 2+len(dbr.columns))
    remove_tmp_database(tmp)


def test_read_write_df():

    tmp = create_tmp_database(twofiles)
    dbr = DbRegister(tmp)
    dbr.read_dataframe()
    dbr._write_df()
    df1 = dbr.dataframe
    dbr._read_df()
    df2 = dbr.dataframe

    remove_tmp_database(tmp)

    assert np.array_equal(df1.to_numpy(), df2.to_numpy())

def test_multiframe_to_singleframe():

    tmp = create_tmp_database(multiframe)
    dbr = DbRegister(tmp)
    dbr.read_dataframe()
    assert dbr.dataframe.shape == (2, 14)
    dbr._multiframe_to_singleframe()
    assert dbr.dataframe.shape == (124, 14)
    remove_tmp_database(tmp)

def test_scan():

    tmp = create_tmp_database(rider)
    dbr = DbRegister(tmp)
    dbr.scan()
    assert dbr.dataframe.shape == (24, 14)
    remove_tmp_database(tmp)

def test_type():

    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    patient = 'RIDER Neuro MRI-5244517593'
    assert dbr.type(series) == 'Series'
    assert dbr.type(patient) == 'Patient'
    assert dbr.type('Database') == 'Database'
    assert dbr.type() is None

def test_keys():

    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    keys = dbr.keys()
    assert keys == []

    keys = dbr.keys('Database')
    assert 24 == len(keys)

    # Patient
    keys = dbr.keys(patient='RIDER Neuro MRI-3369019796')
    assert 12 == len(keys)
    keys = dbr.keys(
        patient = [
            'RIDER Neuro MRI-3369019796',
            'RIDER Neuro MRI-5244517593',
            ]
        )
    assert 24 == len(keys)

    # Study
    keys = dbr.keys(study='1.3.6.1.4.1.9328.50.16.139212415517744996696223190563412871264')
    assert 4 == len(keys)
    keys = dbr.keys(
        study = [
            '1.3.6.1.4.1.9328.50.16.139212415517744996696223190563412871264',
            '1.3.6.1.4.1.9328.50.16.10388995917728582202615060907395571964',
            ]
        )
    assert 6 == len(keys)

    # Series
    keys = dbr.keys(series='1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242')
    assert 2 == len(keys)
    keys = dbr.keys(
        series = [
            '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242',
            '1.3.6.1.4.1.9328.50.16.163870745718873861235299152775293374260',
            '1.3.6.1.4.1.9328.50.16.39537076883396884954303966295061604769'
            ]
        )
    assert 6 == len(keys)

    # Datasets
    keys = dbr.keys(instance='1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022')
    assert 1 == len(keys)
    keys = dbr.keys(
        instance = [
            '1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022',
            '1.3.6.1.4.1.9328.50.16.180826285525298270030493768487939618219',
            '1.3.6.1.4.1.9328.50.16.243004851579310565813723110219735642931',
            '1.3.6.1.4.1.9328.50.16.68617558011242394133101958688682703605',
            ]
        )
    assert 4 == len(keys)

    # Entering non-existent entries
    keys = dbr.keys(series='1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022')
    assert 0 == len(keys)
    keys = dbr.keys('abc')
    assert 0 == len(keys)

    # Mixed arguments
    keys = dbr.keys(
        patient = 'RIDER Neuro MRI-3369019796', 
        study = '1.3.6.1.4.1.9328.50.16.139212415517744996696223190563412871264')
    assert 12 == len(keys)

    keys = dbr.keys(
        patient = 'RIDER Neuro MRI-3369019796', 
        study = [
            '1.3.6.1.4.1.9328.50.16.139212415517744996696223190563412871264',
            '1.3.6.1.4.1.9328.50.16.168701627691879645008036315574545460110',
            ]
        )
    assert 12 == len(keys)

    keys = dbr.keys(
        patient = 'RIDER Neuro MRI-3369019796', 
        study = '1.3.6.1.4.1.9328.50.16.10388995917728582202615060907395571964',
        )
    assert 14 == len(keys)

    keys = dbr.keys(
        patient = 'RIDER Neuro MRI-5244517593',
        series = '1.3.6.1.4.1.9328.50.16.233636027937248405570338506686080257722')
    assert 12 == len(keys)

    keys = dbr.keys(
        patient = 'RIDER Neuro MRI-5244517593',
        series = [
            '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242',
            '1.3.6.1.4.1.9328.50.16.39537076883396884954303966295061604769'
            ]
        )
    assert 16 == len(keys)

    keys = dbr.keys(
        patient = 'RIDER Neuro MRI-5244517593',
        series = [
            '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242',
            '1.3.6.1.4.1.9328.50.16.39537076883396884954303966295061604769'
            ],
        instance = [
            '1.3.6.1.4.1.9328.50.16.8428147229483429640255033081444174773',
            '1.3.6.1.4.1.9328.50.16.302536631490375078745691772311115895736',
            '1.3.6.1.4.1.9328.50.16.68617558011242394133101958688682703605'],
        )
    assert 18 == len(keys)

    # Any UID
    keys = dbr.keys(uid = 'RIDER Neuro MRI-5244517593')
    assert 12 == len(keys) 

    keys = dbr.keys(
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

    keys = dbr.keys(
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

    keys = dbr.keys(
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

    keys = dbr.keys([None], patient = None)
    assert keys == []

def test_value():

    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    key = dbr.keys('Database')[5]
    assert dbr.value(key, 'PatientName') == '281949'
    assert dbr.value(key, 'StudyDescription') == 'BRAIN^RESEARCH'
    assert dbr.value(key, 'SeriesDescription') == 'ax 10 flip'
    assert dbr.value(key, 'InstanceNumber') == 9
    assert None is dbr.value(key, 'dummy')

    key = dbr.keys('RIDER Neuro MRI-5244517593')
    arr = dbr.value(key, ['PatientName','StudyDescription'])
    assert arr.shape == (12, 2)
    assert arr[0,0] == '281949'
    assert 'BRAIN^ROUTINE BRAIN' in arr[:,1] 

def test_parent():

    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    patient = 'RIDER Neuro MRI-5244517593'
    assert dbr.parent(series) == '1.3.6.1.4.1.9328.50.16.139212415517744996696223190563412871264'
    assert dbr.parent(patient) == 'Database'
    assert dbr.parent() is None
    assert dbr.parent('abc') is None

def test_children():

    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    instance = '1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022'
    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    patient = 'RIDER Neuro MRI-5244517593'
    assert len(dbr.children(series)) == 2
    assert len(dbr.children(patient)) == 4
    assert dbr.type(dbr.children(patient)[0]) == 'Study'
    assert dbr.children(dbr.children(series)[0]) == []
    assert len(dbr.children('Database')) == 2
    assert [] == dbr.children(instance)
    assert [] == dbr.children()
    assert 6 == len(dbr.children([series, patient]))

    remove_tmp_database(tmp)

def test_siblings():

    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    patient = 'RIDER Neuro MRI-5244517593'
    assert dbr.siblings('Database') is None
    assert len(dbr.children('Database')) == 2
    assert len(dbr.siblings(dbr.children('Database')[0])) == 1
    assert len(dbr.children(patient)) == 4
    assert len(dbr.siblings(dbr.children(patient)[0])) == 3
    assert len(dbr.children(series)) == 2
    assert len(dbr.siblings(dbr.children(series)[0])) == 1

    remove_tmp_database(tmp)

def test_instances():

    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    patient = 'RIDER Neuro MRI-5244517593'
    assert len(dbr.instances(series)) == 2
    assert len(dbr.instances(patient)) == 12
    assert len(dbr.instances('Database')) == 24
    assert dbr.instances() == []

    remove_tmp_database(tmp)

def test_series():

    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    patient = 'RIDER Neuro MRI-5244517593'
    assert len(dbr.series(series)) == 1
    assert len(dbr.series(patient)) == 6
    assert dbr.series() == []

    remove_tmp_database(tmp)

def test_studies():

    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    patient = 'RIDER Neuro MRI-5244517593'
    assert len(dbr.studies(series)) == 1
    assert len(dbr.studies(patient)) == 4

    remove_tmp_database(tmp)

def test_patients():

    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    patient = 'RIDER Neuro MRI-5244517593'
    assert len(dbr.patients(series)) == 1
    assert len(dbr.patients(patient)) == 1
    assert len(dbr.patients('Database')) == 2
    assert dbr.patients() == []

    remove_tmp_database(tmp)


def test_new_patient():

    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    n = len(dbr.patients('Database'))
    patient = dbr.new_patient()
    assert len(dbr.patients('Database')) == n+1
    assert [] == dbr.children(patient) 
    assert [] == dbr.instances(patient)
    assert 1 == len(dbr.keys(patient))
    try:
        dbr.print()
    except:
        assert False
    else:
        assert True

    remove_tmp_database(tmp)


def test_new_study():

    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    n = len(dbr.studies('Database'))
    study = dbr.new_study()
    assert len(dbr.studies('Database')) == n+1
    assert [] == dbr.children(study)
    assert [] == dbr.instances(study)
    assert 1 == len(dbr.keys(study))

    patient = 'RIDER Neuro MRI-5244517593'
    n = len(dbr.studies(patient))
    study = dbr.new_study(patient)
    assert len(dbr.studies(patient)) == n+1
    assert 0 == len(dbr.children(study))

    remove_tmp_database(tmp)

def test_new_series():

    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    n = len(dbr.series('Database'))
    series = dbr.new_series()
    assert len(dbr.series('Database')) == n+1
    assert [] == dbr.children(series)
    assert [] == dbr.instances(series)
    assert 1 == len(dbr.keys(series))

    study = '1.3.6.1.4.1.9328.50.16.168701627691879645008036315574545460110'
    n = len(dbr.series(study))
    series = dbr.new_series(study)
    assert len(dbr.series(study)) == n+1
    assert [] == dbr.children(series)

    remove_tmp_database(tmp)

def test_new_instance():

    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    n = len(dbr.instances('Database'))
    instance = dbr.new_instance()
    assert len(dbr.instances('Database')) == n+1
    assert None is dbr.get_dataset(instance)
    assert 1 == len(dbr.keys(instance))

    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    n = len(dbr.instances(series))
    instance = dbr.new_instance(series)
    assert len(dbr.instances(series)) == n+1
    assert None is dbr.get_dataset(instance)

    remove_tmp_database(tmp)


def test_series_header():

    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'

    instance = dbr.instances(series)[0]
    attr, vals = dbr.series_header(dbr.keys(instance)[0])
    series_vals = dbr.get_values(series, attr)
    for i in range(len(vals)):
        assert series_vals[i] == vals[i]

    instance = dbr.new_instance(series)
    attr, vals = dbr.series_header(dbr.keys(instance)[0])
    series_vals = dbr.get_values(series, attr)
    for i in range(len(vals)):
        assert series_vals[i] == vals[i]

def test_set_instance_dataset():

    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    series1 = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'

    instance1 = dbr.instances(series1)[0]
    ds1 = dbr.get_dataset(instance1)
    ds1.PatientName = 'Blabla'
    ds1.AcquisitionTime = '000000.00'
    dbr.set_instance_dataset(instance1, ds1)
    assert ds1.PatientName != 'Blabla'
    assert ds1.AcquisitionTime == dbr.get_values(instance1, 'AcquisitionTime')

    instance1 = dbr.instances(series1)[0]
    ds1 = dbr.get_dataset(instance1)
    ds1.PatientName = 'Blabla'
    ds1.AcquisitionTime = '000001.00'
    instance = dbr.new_instance(series1)
    dbr.set_instance_dataset(instance, ds1)
    assert ds1.PatientName != 'Blabla'
    assert ds1.AcquisitionTime == dbr.get_values(instance, 'AcquisitionTime')

    instance1 = dbr.instances(series1)[0]
    ds1 = MRImage()
    ds1.PatientName = 'Blabla'
    ds1.AcquisitionTime = '000002.00'
    dbr.set_instance_dataset(instance1, ds1)
    assert ds1.PatientName != 'Blabla'
    assert ds1.AcquisitionTime == dbr.get_values(instance1, 'AcquisitionTime')


def test_set_dataset():

    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    series1 = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    series2 = '1.3.6.1.4.1.9328.50.16.39537076883396884954303966295061604769'

    # Save datasets in new instances
    dataset = dbr.get_dataset(series1)
    dataset[0].AcquisitionTime = '000000'
    n = len(dbr.instances(series2))
    dbr.set_dataset(series2, dataset[:2])
    assert len(dbr.instances(series2)) == n+2
    assert '000000' in dbr.get_values(series2, 'AcquisitionTime')
    dbr.restore('Database')
    assert len(dbr.instances(series2)) == n
    assert '000000' not in dbr.get_values(series2, 'AcquisitionTime')

    # Save datasets in existing instances
    dataset = dbr.get_dataset(series2)
    dataset[0].AcquisitionTime = '000000'
    n = len(dbr.instances(series2))
    assert n == len(dataset)
    dbr.set_dataset(series2, dataset[:2])
    assert len(dbr.instances(series2)) == n
    assert '000000' in dbr.get_values(series2, 'AcquisitionTime')
    
    remove_tmp_database(tmp)

def test_new_child():

    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    # Three objects that are not nested: study not in patient and series not in study.
    patient = 'RIDER Neuro MRI-5244517593'
    study = '1.3.6.1.4.1.9328.50.16.168701627691879645008036315574545460110'
    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    instance = '1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022'

    new_patient = dbr.new_child('Database')
    assert dbr.parent(new_patient) == 'Database'
    new_study = dbr.new_child(patient)
    assert dbr.parent(new_study) == patient
    new_series = dbr.new_child(study)
    assert dbr.parent(new_series) == study
    new_instance = dbr.new_child(series)
    assert dbr.parent(new_instance) == series
    new_instance = dbr.new_child(new_series)
    assert dbr.parent(new_instance) == new_series
    assert None is dbr.new_child(instance)

    remove_tmp_database(tmp)

def test_new_sibling():

    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    # Three objects that are not nested: study not in patient and series not in study.
    patient = 'RIDER Neuro MRI-5244517593'
    study = '1.3.6.1.4.1.9328.50.16.168701627691879645008036315574545460110'
    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    instance = '1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022'

    assert dbr.new_sibling('Database') is None
    new_patient = dbr.new_sibling(patient)
    assert patient in dbr.children(dbr.parent(new_patient))
    new_study = dbr.new_sibling(study)
    assert study in dbr.children(dbr.parent(new_study))
    new_series = dbr.new_sibling(series)
    assert series in dbr.children(dbr.parent(new_series))
    new_instance = dbr.new_sibling(instance)
    assert instance in dbr.children(dbr.parent(new_instance))

    remove_tmp_database(tmp)

def test_new_pibling():

    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    # Three objects that are not nested: study not in patient and series not in study.
    patient = 'RIDER Neuro MRI-5244517593'
    study = '1.3.6.1.4.1.9328.50.16.168701627691879645008036315574545460110'
    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    instance = '1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022'

    assert dbr.new_pibling('Database') is None
    assert dbr.new_pibling(patient) is None
    new_patient = dbr.new_pibling(study)
    assert new_patient in dbr.siblings(dbr.parent(study))
    new_study = dbr.new_pibling(series)
    assert new_study in dbr.siblings(dbr.parent(series))
    new_series = dbr.new_pibling(instance)
    assert new_series in dbr.siblings(dbr.parent(instance))

    remove_tmp_database(tmp)

def test_label():

    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    patient = dbr.patients('Database')[0]
    assert dbr.label(patient) in ['Patient 281949 [RIDER Neuro MRI-3369019796]','Patient 281949 [RIDER Neuro MRI-5244517593]']
    study = dbr.studies(patient)[0]
    try:
        dbr.label(study)
    except: 
        assert False
    else:
        assert True
    series = dbr.series(study)[0]
    try:
        dbr.label(series)
    except: 
        assert False
    else:
        assert True
    instance = dbr.instances(series)[0]
    try:
        dbr.label(instance)
    except:
        assert False
    else:
        assert True

    remove_tmp_database(tmp)

def test_print():

    tmp = create_tmp_database(rider)

    dbr = DbRegister()
    dbr.open(tmp)
    try:
        dbr.print()
    except:
        assert False
    else:
        assert True

    remove_tmp_database(tmp)

def test_read_and_clear():

    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    assert 0 == len(dbr.dataset)
    dbr.read('Database')
    assert 24 == len(dbr.dataset)
    dbr.clear('Database')
    assert 0 == len(dbr.dataset)
    dbr.read(patient='RIDER Neuro MRI-3369019796')
    assert 12 == len(dbr.dataset)
    dbr.clear('Database')
    dbr.read(study='1.3.6.1.4.1.9328.50.16.139212415517744996696223190563412871264')
    assert 4 == len(dbr.dataset)
    dbr.clear('Database')
    dbr.read(series='1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242')
    assert 2 == len(dbr.dataset)
    dbr.clear('Database')
    dbr.read(instance='1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022')
    assert 1 == len(dbr.dataset)
    dbr.clear('Database')

    # Try to read something that does not exist
    dbr.read(series='1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022')
    assert 0 == len(dbr.dataset)

    # read a patient, then a study in that patient - reading study does nothing
    dbr.read(patient='RIDER Neuro MRI-3369019796')
    assert 12 == len(dbr.dataset)
    dbr.read(study='1.3.6.1.4.1.9328.50.16.139212415517744996696223190563412871264')
    assert 12 == len(dbr.dataset)

    # Clear only that study
    dbr.clear(study='1.3.6.1.4.1.9328.50.16.139212415517744996696223190563412871264')
    assert 8 == len(dbr.dataset)

    # Clear a dataset from the study that was just cleared (does nothing)
    dbr.clear(instance='1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022')
    assert 8 == len(dbr.dataset)

    # Clear all datasets from the patient
    dbr.clear(patient='RIDER Neuro MRI-3369019796')
    assert 0 == len(dbr.dataset)

    remove_tmp_database(tmp)

def test_write():

    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    uid = '1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022'

    # Change a name in memory, write, clear memory and read again to test this has changed.
    dbr.read(instance=uid)
    dbr.set_values(uid, 'PatientName', 'Anonymous')
    dbr.write('Database')
    dbr.clear('Database')
    dbr.read(instance=uid)
    dataset = dbr.get_dataset(uid)
    assert uid == dataset.SOPInstanceUID
    assert 'Anonymous' == dataset.PatientName

    # Read a series, change all of the datasets, write only one before clearing memory.
    # Then check that only one of the values has changed
    series_uid='1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    dbr.read(series=series_uid)
    dbr.set_values(series_uid, 'PatientName', 'Anonymous')
    dbr.write('Database')
    dbr.clear('Database')
    dbr.read(instance=uid)
    ds = dbr.get_dataset(uid)
    assert ds.PatientName == 'Anonymous'

    remove_tmp_database(tmp)

def test_open_close():

    tmp = create_tmp_database(rider)
    dbr = DbRegister(tmp)
    dbr.open()
    assert dbr.dataframe.shape == (24, 14)
    dbr.save()
    dbr.close()
    assert dbr.dataframe is None
    try:
        dbr.open()
    except ValueError:
        assert True
    else:
        assert False
    dbr.close()
    remove_tmp_database(tmp)

    tmp = create_tmp_database(twofiles)
    dbr.open(tmp)
    assert dbr.dataframe.shape == (2, 14)
    dbr.save()
    dbr.close()
    assert dbr.dataframe is None
    remove_tmp_database(tmp)

def test_inmemory_vs_ondisk():

    tmp = create_tmp_database(rider)

    dbr = DbRegister()

    # open a database on disk
    # get the dataframe and close it again
    dbr.open(tmp)   
    df = dbr.dataframe
    dbr.close()
    assert df.shape == (24, 14)
    assert dbr.dataframe is None

    remove_tmp_database(tmp)

    # create a database in memory
    # Try to read a dataframe and check 
    # that this produces an exception
    dbr.dataframe = df
    assert dbr.dataframe.shape == (24, 14)
    try:
        dbr.read_dataframe()
    except ValueError:
        assert True
    else:
        assert False


def test_get_dataset():

    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    # Get a list of all datasets from disk
    ds = dbr.get_dataset('Database')
    assert 24 == len(ds)

    # Get a list of all datasets from memory
    dbr.read('Database')
    ds = dbr.get_dataset('Database')
    dbr.clear('Database')
    assert 24 == len(ds)

    # Read all datasets for one series from disk
    series_uid='1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    ds = dbr.get_dataset(series_uid)
    assert 2 == len(ds)

    # Read one of the datasets first, check that the result is the same
    uid = '1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022'
    dbr.read(instance=uid)
    ds = dbr.get_dataset(series_uid)
    dbr.clear('Database')
    assert 2 == len(ds)

    remove_tmp_database(tmp)

def test_delete():
    
    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    # Delete all datasets for one series
    uid = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    dbr.delete(series=uid)
    assert len(dbr.keys('Database')) == 22

    # Delete it again
    dbr.delete(series=uid)
    assert len(dbr.keys('Database')) == 22
    assert dbr.keys(series=uid) == []

    # Delete the patient containing this series
    dbr.delete(patient='RIDER Neuro MRI-3369019796')
    assert len(dbr.keys('Database')) == 12
    assert len(dbr.keys(patient='RIDER Neuro MRI-5244517593')) == 12
    assert len(dbr.keys('RIDER Neuro MRI-5244517593')) == 12
    assert len(dbr.keys('RIDER Neuro MRI-3369019796')) == 0

    # Delete the other patient too
    dbr.delete(patient='RIDER Neuro MRI-5244517593')
    assert dbr.keys('Database') == []

    remove_tmp_database(tmp)

def test_copy_to_series():

    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    orig_keys = dbr.keys(patient='RIDER Neuro MRI-5244517593')
    copy_instances = dbr.copy_to_series('RIDER Neuro MRI-5244517593', series)
    assert len(dbr.keys(series)) == 14
    assert len(dbr.keys('RIDER Neuro MRI-5244517593')) == 12
    assert len(dbr.keys('RIDER Neuro MRI-3369019796')) == 24
    assert len(dbr.keys('Database')) == 36
    assert dbr.value(orig_keys[0], 'PatientID') == 'RIDER Neuro MRI-5244517593'
    assert dbr.value(dbr.keys(copy_instances)[0], 'PatientID') == 'RIDER Neuro MRI-3369019796'
    # Check that all new instance numbers are unique
    nrs = dbr.value(dbr.keys(series), 'InstanceNumber')
    assert len(set(nrs)) == len(nrs) 
    remove_tmp_database(tmp)

def test_copy_to_study():

    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    study = '1.3.6.1.4.1.9328.50.16.168701627691879645008036315574545460110'
    patient = 'RIDER Neuro MRI-5244517593'
    assert len(dbr.series(study)) == 1
    assert len(dbr.series(patient)) == 6
    orig_keys = dbr.keys(patient=patient)
    copy_series = dbr.copy_to_study(patient, study)
    assert len(dbr.instances(study)) == 14
    assert len(dbr.series(study)) == 7
    assert len(dbr.instances('RIDER Neuro MRI-5244517593')) == 12
    assert len(dbr.instances('RIDER Neuro MRI-3369019796')) == 24
    assert len(dbr.instances('Database')) == 36
    assert dbr.value(orig_keys[0], 'PatientID') == 'RIDER Neuro MRI-5244517593'
    assert dbr.value(dbr.keys(copy_series)[0], 'PatientID') == 'RIDER Neuro MRI-3369019796'
    nrs = dbr.get_values(study, 'SeriesNumber')
    assert len(nrs) == len(dbr.series(study))
    remove_tmp_database(tmp)

def test_copy_to_patient():

    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    study = '1.3.6.1.4.1.9328.50.16.168701627691879645008036315574545460110'
    patient = 'RIDER Neuro MRI-5244517593'
    assert len(dbr.series(study)) == 1
    assert len(dbr.series(patient)) == 6
    assert len(dbr.instances(study)) == 2
    assert len(dbr.instances(patient)) == 12
    dbr.copy_to_patient(study, patient)
    assert len(dbr.series(study)) == 1
    assert len(dbr.instances(study)) == 2
    assert len(dbr.series(patient)) == 7
    assert len(dbr.instances(patient)) == 14
    remove_tmp_database(tmp)

def test_copy_to():

    # Need to include some scenarios involving copying to from empty objects

    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    instance ='1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022'
    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    study = '1.3.6.1.4.1.9328.50.16.168701627691879645008036315574545460110'
    patient = 'RIDER Neuro MRI-5244517593'  
    assert len(dbr.instances(series)) == 2
    dbr.copy_to(instance, series)   # copy an instance to its own series.
    assert len(dbr.instances(series)) == 3
    assert len(dbr.series(study)) == 1
    dbr.copy_to(series, study) # copy a series to another study
    assert len(dbr.series(study)) == 2
    assert len(dbr.studies(patient)) == 4
    dbr.copy_to(study, patient) # copy a study to another patient
    assert len(dbr.studies(patient)) == 5
    remove_tmp_database(tmp)


def test_move_to_series():

    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    orig_keys = dbr.keys(patient='RIDER Neuro MRI-5244517593')
    instances = dbr.move_to_series('RIDER Neuro MRI-5244517593', series)
    assert len(dbr.keys(series)) == 14
    assert len(dbr.keys('RIDER Neuro MRI-5244517593')) == 0
    assert len(dbr.keys('RIDER Neuro MRI-3369019796')) == 24
    assert len(dbr.keys('Database')) == 24
    assert dbr.value(orig_keys[0], 'PatientID') == 'RIDER Neuro MRI-5244517593'
    assert dbr.value(dbr.keys(instances)[0], 'PatientID') == 'RIDER Neuro MRI-3369019796'
    # Check that all new instance numbers are unique
    nrs = dbr.value(dbr.keys(series), 'InstanceNumber')
    assert len(set(nrs)) == len(nrs)
    remove_tmp_database(tmp)


def test_move_to_study():

    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    study = '1.3.6.1.4.1.9328.50.16.168701627691879645008036315574545460110'
    patient = 'RIDER Neuro MRI-5244517593'
    assert len(dbr.series(study)) == 1
    assert len(dbr.series(patient)) == 6
    series = dbr.move_to_study(patient, study) # move to a study of another patient
    assert len(dbr.instances(study)) == 14
    assert len(dbr.series(study)) == 7
    assert len(dbr.instances(patient)) == 0
    assert len(dbr.instances('RIDER Neuro MRI-3369019796')) == 24
    assert len(dbr.instances('Database')) == 24
    assert dbr.get_values(patient, 'PatientID') is None
    assert dbr.value(dbr.keys(series)[0], 'PatientID') == 'RIDER Neuro MRI-3369019796'
    nrs = dbr.get_values(study, 'SeriesNumber')
    assert len(nrs) == len(dbr.series(study))
    remove_tmp_database(tmp)

def test_move_to_patient():

    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    study = '1.3.6.1.4.1.9328.50.16.168701627691879645008036315574545460110'
    patient = 'RIDER Neuro MRI-5244517593'
    assert len(dbr.series(study)) == 1
    assert len(dbr.series(patient)) == 6
    assert len(dbr.instances(study)) == 2
    assert len(dbr.instances(patient)) == 12
    dbr.move_to_patient(study, patient)
    assert len(dbr.series(study)) == 1
    assert len(dbr.instances(study)) == 2
    assert len(dbr.series(patient)) == 7
    assert len(dbr.instances(patient)) == 14
    remove_tmp_database(tmp)

def test_move_to():

    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    instance ='1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022'
    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    study = '1.3.6.1.4.1.9328.50.16.168701627691879645008036315574545460110'
    patient = 'RIDER Neuro MRI-5244517593'  
    assert len(dbr.instances(series)) == 2
    dbr.move_to(instance, series)   # move an instance to its own series.
    assert dbr.get_values(instance, 'SeriesInstanceUID') == series
    assert len(dbr.instances(series)) == 2
    assert len(dbr.series(study)) == 1
    dbr.move_to(series, study) # copy a series to another study
    assert dbr.get_values(series, 'StudyInstanceUID') == study
    assert len(dbr.series(study)) == 2
    assert len(dbr.studies(patient)) == 4
    dbr.move_to(study, patient) # move a study to another patient
    assert dbr.get_values(study, 'PatientID') == patient
    assert len(dbr.studies(patient)) == 5
    remove_tmp_database(tmp)


def test_set_values():

    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    patient = 'RIDER Neuro MRI-5244517593'
    for name in dbr.value(dbr.keys(patient), 'PatientName'):
        assert name != 'Anonymous'
    for t in dbr.value(dbr.keys(patient), 'SeriesDescription'):
        assert t != 'Desc'
    for instance in dbr.instances(patient):
        assert dbr.get_dataset(instance).get_values('AcquisitionTime') != '000000.00'

    dbr.set_values(patient, ['PatientName', 'SeriesDescription', 'AcquisitionTime'], ['Anonymous', 'Desc', '000000.00'])

    for name in dbr.value(dbr.keys(patient), 'PatientName'):
        assert name == 'Anonymous'
    for t in dbr.value(dbr.keys(patient), 'SeriesDescription'):
        assert t == 'Desc'
    for instance in dbr.instances(patient):
        assert dbr.get_dataset(instance).get_values('AcquisitionTime') == '000000.00'

    remove_tmp_database(tmp)


def test_get_values():

    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    assert dbr.get_values('', 'PatientName') is None

    patient = 'RIDER Neuro MRI-5244517593'
    attributes = ['PatientName','SeriesDescription','AcquisitionTime']

    values = dbr.get_values(patient, 'PatientName')
    assert values == '281949'

    values = dbr.get_values(patient, 'SeriesDescription')
    assert set(values) == set(['ax 10 flip', 'ax 5 flip', 'sag 3d gre +c'])

    values = dbr.get_values(patient, 'AcquisitionTime')
    assert set(values) == set(['074424.477508', '074443.472496', '074615.472480', '074634.479981', '075505.670007', '075611.105017'])

    values = dbr.get_values(patient, attributes)
    assert values[0] == '281949'
    assert set(values[1]) == set(['ax 10 flip', 'ax 5 flip', 'sag 3d gre +c'])
    assert set(values[2]) == set(['074424.477508', '074443.472496', '074615.472480', '074634.479981', '075505.670007', '075611.105017'])

    values = dbr.get_values(patient, 'PatientName')
    assert values == '281949'

    patient = [patient, 'RIDER Neuro MRI-3369019796']
    values = dbr.get_values(patient, attributes)
    assert values[0][0] == '281949'
    assert set(values[0][1]) == set(['ax 10 flip', 'ax 5 flip', 'sag 3d gre +c'])
    assert set(values[0][2]) == set(['074424.477508', '074443.472496', '074615.472480', '074634.479981', '075505.670007', '075611.105017'])
    assert values[1][0] == '281949'
    assert set(values[1][1]) == set(['ax 10 flip', 'ax 5 flip', 'sag 3d gre +c'])
    assert set(values[1][2]) == set(['073903.155010', '073921.462495', '074511.204995', '074529.432496', '075138.639989', '075649.057496'])

    remove_tmp_database(tmp)

def test_restore():

    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    patient = 'RIDER Neuro MRI-5244517593'

    assert '' == dbr.get_values(patient, 'PatientSex')
    dbr.set_values(patient, 'PatientSex', 'M')
    assert 'M' == dbr.get_values(patient, 'PatientSex')
    dbr.restore('Database')
    assert '' == dbr.get_values(patient, 'PatientSex')

    dbr.read(patient)
    assert '' == dbr.get_values(patient, 'PatientSex')
    dbr.set_values(patient, 'PatientSex', 'M')
    assert 'M' == dbr.get_values(patient, 'PatientSex')
    dbr.restore('Database')
    assert '' == dbr.get_values(patient, 'PatientSex')

    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    study = '1.3.6.1.4.1.9328.50.16.168701627691879645008036315574545460110'

    assert len(dbr.series(study)) == 1
    dbr.move_to(series, study) 
    assert len(dbr.series(study)) == 2
    dbr.restore('Database')
    assert len(dbr.series(study)) == 1

    dbr.read(series)
    assert len(dbr.series(study)) == 1
    dbr.move_to(series, study) 
    assert len(dbr.series(study)) == 2
    dbr.restore('Database')
    assert len(dbr.series(study)) == 1

    remove_tmp_database(tmp)

def test_save():

    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    # Three objects that are not nested: study not in patient and series not in study.
    patient = 'RIDER Neuro MRI-5244517593'
    study = '1.3.6.1.4.1.9328.50.16.168701627691879645008036315574545460110'
    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'

    assert '' == dbr.get_values(patient, 'PatientSex')
    dbr.set_values(patient, 'PatientSex', 'M')
    assert 'M' == dbr.get_values(patient, 'PatientSex')
    dbr.save('Database')
    assert 'M' == dbr.get_values(patient, 'PatientSex')

    dbr.read(patient)
    assert 'M' == dbr.get_values(patient, 'PatientSex')
    dbr.set_values(patient, 'PatientSex', '')
    assert '' == dbr.get_values(patient, 'PatientSex')
    dbr.save('Database')
    assert '' == dbr.get_values(patient, 'PatientSex')

    assert len(dbr.series(study)) == 1
    dbr.move_to(series, study) 
    assert len(dbr.series(study)) == 2
    dbr.save('Database')
    assert len(dbr.series(study)) == 2

    dbr.read(series)
    assert len(dbr.series(study)) == 2
    dbr.move_to(series, study) 
    assert len(dbr.series(study)) == 2
    dbr.save('Database')
    dbr.restore()
    assert len(dbr.series(study)) == 2

    remove_tmp_database(tmp)



def test_template():

    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    remove_tmp_database(tmp)

def test_filter():

    tmp = create_tmp_database(rider)
    dbr = DbRegister()
    dbr.open(tmp)

    patients = dbr.patients('Database')
    patients = dbr.filter(patients, PatientName='281949')
    assert len(patients) == 2
    patients = dbr.filter(patients, PatientID='RIDER Neuro MRI-5244517593')
    assert len(patients) == 1

    series = dbr.series('Database')
    series = dbr.filter(series, SeriesDescription='ax 5 flip')
    assert len(series) == 4

    series = dbr.series('Database')
    series = dbr.filter(series, SeriesDescription='ax 5 flip', PatientID='RIDER Neuro MRI-5244517593')
    assert len(series) == 2

    remove_tmp_database(tmp)


def test_import_datasets():

    source = create_tmp_database(rider, name='source')
    source_dbr = DbRegister()
    source_dbr.open(source)
    source_files = source_dbr.filepaths('Database')

    # Create empty database and import all source files
    target = create_tmp_database(name='target')
    target_dbr = DbRegister()
    target_dbr.open(target)
    target_dbr.import_datasets(source_files)

    # Check that the number of datasets equals nr of source files
    assert len(target_dbr.instances('Database')) == len(source_files)

    # Import all source files again and check that nothing has changed
    target_dbr.import_datasets(source_files)
    assert len(target_dbr.instances('Database')) == len(source_files)

    # Delete one patient, import all source files again and check that nothing has changed
    patient = target_dbr.patients('Database', PatientID='RIDER Neuro MRI-5244517593')
    target_dbr.delete(patient)
    target_dbr.import_datasets(source_files)
    assert len(target_dbr.instances('Database')) == len(source_files)

    # Save new database and check that nothing has changed
    target_dbr.save('Database')
    assert len(target_dbr.instances('Database')) == len(source_files)

    # Delete one patient and import files from that patient again
    # Check that nothing has changed
    patient = target_dbr.patients('Database', PatientID='RIDER Neuro MRI-5244517593')
    target_dbr.delete(patient)
    patient = source_dbr.patients('Database', PatientID='RIDER Neuro MRI-5244517593')
    patient_files = source_dbr.filepaths(patient)
    assert len(target_dbr.instances('Database')) == len(source_files)-len(patient_files)
    target_dbr.import_datasets(patient_files)
    assert len(target_dbr.instances('Database')) == len(source_files)

    # Delete one patient and import files from another patient
    # Check that files are not imported again
    patient1 = target_dbr.patients('Database', PatientID='RIDER Neuro MRI-5244517593')
    patient1_files = target_dbr.filepaths(patient1)
    patient2 = source_dbr.patients('Database', PatientID='RIDER Neuro MRI-3369019796')
    patient2_files = source_dbr.filepaths(patient2)
    target_dbr.delete(patient1)
    assert len(target_dbr.instances('Database')) == len(source_files)-len(patient1_files)
    target_dbr.import_datasets(patient2_files)
    assert len(target_dbr.instances('Database')) == len(source_files)-len(patient1_files)

    # Start over, this time with a full target databese
    # Import all source files and check nothing has changed
    target = create_tmp_database(rider, name='target')
    target_dbr = DbRegister()
    target_dbr.open(target)
    assert len(target_dbr.instances('Database')) == len(source_files)
    target_dbr.import_datasets(source_files)
    assert len(target_dbr.instances('Database')) == len(source_files)

    remove_tmp_database(source)
    remove_tmp_database(target)

def test_export_datasets():

    source = create_tmp_database(rider, name='source')
    source_dbr = DbRegister()
    source_dbr.open(source)
    patient = source_dbr.patients('Database', PatientID='RIDER Neuro MRI-5244517593')

    target = create_tmp_database(name='target')
    target_dbr = DbRegister()
    target_dbr.open(target)
    source_dbr.export_datasets(patient, target_dbr)

    assert len(target_dbr.instances('Database')) == len(source_dbr.instances(patient))
    assert len(target_dbr.series('Database')) == len(source_dbr.series(patient))
    assert len(target_dbr.studies('Database')) == len(source_dbr.studies(patient))

    remove_tmp_database(source)
    remove_tmp_database(target)

def test_new_database():

    dbr = DbRegister()

    p1 = dbr.new_patient('Database')
    p2 = dbr.new_patient('Database')
    p3 = dbr.new_patient('Database')

    p1v1 = dbr.new_study(p1)
    p1v2 = dbr.new_study(p1)

    p1v1s1 = dbr.new_series(p1v1)
    p1v1s2 = dbr.new_series(p1v1)
    p1v1s3 = dbr.new_series(p1v1)

    dbr.new_instance(p1v1s1, dataset=MRImage())
    dbr.new_instance(p1v1s1, dataset=MRImage())
    dbr.new_instance(p1v1s1, dataset=MRImage())
    dbr.new_instance(p1v1s1, dataset=MRImage())
    dbr.new_instance(p1v1s1, dataset=MRImage())
    dbr.new_instance(p1v1s1, dataset=MRImage())

    assert 6 == len(dbr.instances(p1v1))

if __name__ == "__main__":

    test_init()
    test_read_dataframe()
    test_read_write_df()
    test_multiframe_to_singleframe()
    test_scan()
    test_type()
    test_keys()
    test_value()
    test_parent()
    test_children()
    test_siblings()
    test_instances()
    test_series()
    test_studies()
    test_patients()
    test_new_patient()
    test_new_study()
    test_new_series()
    test_new_instance()
    test_series_header()
    test_set_instance_dataset()
    test_set_dataset()
    test_new_child()
    test_new_sibling()
    test_new_pibling()
    test_label()
    test_print()
    test_read_and_clear()
    test_write()
    test_open_close()
    test_inmemory_vs_ondisk() 
    test_get_dataset()
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
    test_filter()
    test_import_datasets()
    test_export_datasets()
    test_new_database()
    

    print('--------------------------')
    print('register passed all tests!')
    print('--------------------------')

