import os
import shutil
import timeit
import numpy as np

from dbdicom.new_manager import Manager
from dbdicom.ds import MRImage

datapath = os.path.join(os.path.dirname(__file__), 'data')
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


# CONSTANTS
N_COLUMNS = 20


def test_init():
    mgr = Manager()
    assert mgr.register is None
    tmp = create_tmp_database(rider)
    mgr = Manager(tmp)
    assert mgr.register is None
    assert mgr.path == tmp
    remove_tmp_database(tmp)


def test_read_register():
    tmp = create_tmp_database(rider)
    mgr = Manager(tmp)
    mgr.read_register()
    assert mgr.register.shape == (24, N_COLUMNS)
    remove_tmp_database(tmp)


def test_read_write_register():
    tmp = create_tmp_database(twofiles)
    mgr = Manager(tmp)
    mgr.read_register()
    cols1 = mgr.register.columns()
    mgr.save_register()
    mgr.open_register()
    cols2 = mgr.register.columns()
    remove_tmp_database(tmp)
    assert cols1 == cols2


def test_multiframe_to_singleframe():
    tmp = create_tmp_database(multiframe)
    mgr = Manager(tmp)
    mgr.read_register()
    mgr.multiframe_to_singleframe()
    assert mgr.register.shape == (124, N_COLUMNS)
    remove_tmp_database(tmp)


def test_scan():
    tmp = create_tmp_database(rider)
    mgr = Manager(tmp)
    mgr.scan()
    assert mgr.register.shape == (24, N_COLUMNS-1)
    remove_tmp_database(tmp)


def test_open():
    tmp = create_tmp_database(rider)
    mgr = Manager(tmp)
    mgr.open(tmp) 
    assert mgr.register.shape == (24, N_COLUMNS-1)
    mgr.save_register()
    mgr.open(tmp)
    assert mgr.register.shape == (24, N_COLUMNS-1)
    remove_tmp_database(tmp)


def test_type():
    tmp = create_tmp_database(rider)
    mgr = Manager()
    mgr.open(tmp)
    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    patient = 'RIDER Neuro MRI-5244517593'
    assert mgr.type('007') is None
    assert mgr.type(series) == 'Series'
    assert mgr.type(patient) == 'Patient'
    assert mgr.type('Database') == 'Database'
    assert mgr.type() is None
    assert mgr.type(patient, mgr.register.index()[-1]) == 'Patient'
    assert mgr.type(patient, mgr.register.index()[0]) is None
    remove_tmp_database(tmp)


def test_keys():
    tmp = create_tmp_database(rider)
    mgr = Manager()
    mgr.open(tmp)

    keys = mgr.keys()
    assert keys == []

    keys = mgr.keys('Database')
    assert 24 == len(keys)

    # Patient
    keys = mgr.keys(patient='RIDER Neuro MRI-3369019796')
    assert 12 == len(keys)
    keys = mgr.keys(
        patient = [
            'RIDER Neuro MRI-3369019796',
            'RIDER Neuro MRI-5244517593',
            ]
        )
    assert 24 == len(keys)

    # Study
    keys = mgr.keys(study='1.3.6.1.4.1.9328.50.16.139212415517744996696223190563412871264')
    assert 4 == len(keys)
    keys = mgr.keys(
        study = [
            '1.3.6.1.4.1.9328.50.16.139212415517744996696223190563412871264',
            '1.3.6.1.4.1.9328.50.16.10388995917728582202615060907395571964',
            ]
        )
    assert 6 == len(keys)

    # Series
    keys = mgr.keys(series='1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242')
    assert 2 == len(keys)
    keys = mgr.keys(
        series = [
            '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242',
            '1.3.6.1.4.1.9328.50.16.163870745718873861235299152775293374260',
            '1.3.6.1.4.1.9328.50.16.39537076883396884954303966295061604769'
            ]
        )
    assert 6 == len(keys)

    # Datasets
    keys = mgr.keys(instance='1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022')
    assert 1 == len(keys)
    keys = mgr.keys(
        instance = [
            '1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022',
            '1.3.6.1.4.1.9328.50.16.180826285525298270030493768487939618219',
            '1.3.6.1.4.1.9328.50.16.243004851579310565813723110219735642931',
            '1.3.6.1.4.1.9328.50.16.68617558011242394133101958688682703605',
            ]
        )
    assert 4 == len(keys)

    # Entering non-existent entries
    keys = mgr.keys(series='1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022')
    assert 0 == len(keys)
    keys = mgr.keys('abc')
    assert 0 == len(keys)

    # Any UID
    keys = mgr.keys(uid = 'RIDER Neuro MRI-5244517593')
    assert 12 == len(keys) 

    keys = mgr.keys(
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

    keys = mgr.keys(
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

    keys = mgr.keys(
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

    keys = mgr.keys([None], patient = None)
    assert keys == []


def test_filepath():
    tmp = create_tmp_database(rider)
    mgr = Manager()
    mgr.open(tmp)
    try:
        mgr.filepath(mgr.keys('Database')[0])
    except:
        assert False
    else:
        assert True
    remove_tmp_database(tmp)


def test_filepaths():
    tmp = create_tmp_database(rider)
    mgr = Manager()
    mgr.open(tmp)
    try:
        mgr.filepaths(mgr.keys('Database'))
    except:
        assert False
    else:
        assert True    
    remove_tmp_database(tmp)


def test_value():
    tmp = create_tmp_database(rider)
    mgr = Manager()
    mgr.open(tmp)
    key = mgr.keys('Database')[5]
    assert mgr.value(key, 'PatientName') == '281949'
    assert mgr.value(key, 'StudyDescription') == 'BRAIN^RESEARCH'
    assert mgr.value(key, 'SeriesDescription') == 'ax 10 flip'
    assert mgr.value(key, 'InstanceNumber') == 9
    assert None is mgr.value(key, 'dummy')
    key = mgr.keys('RIDER Neuro MRI-5244517593')
    arr = mgr.value(key, ['PatientName','StudyDescription'])
    assert arr.shape == (12, 2)
    assert arr[0,0] == '281949'
    assert 'BRAIN^ROUTINE BRAIN' in arr[:,1] 


def test_get_instance_dataset():
    tmp = create_tmp_database(rider)
    mgr = Manager()
    mgr.open(tmp)
    uid = '1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022'
    ds = mgr.get_instance_dataset(mgr.keys(uid)[0])
    assert ds.SeriesDescription == 'ax 10 flip'
    assert ds.InstanceNumber == 9
    remove_tmp_database(tmp)


def test_get_dataset():
    tmp = create_tmp_database(rider)
    mgr = Manager()
    mgr.open(tmp)
    # Get a list of all datasets from disk
    ds = mgr.get_dataset('Database')
    assert 24 == len(ds)
    # Read all datasets for one series from disk
    series_uid = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    ds = mgr.get_dataset(series_uid)
    assert 2 == len(ds)
    # Read one of the datasets first, check that the result is the same
    uid = '1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022'
    ds = mgr.get_dataset(uid)
    assert ds.SeriesDescription == 'ax 10 flip'
    assert ds.InstanceNumber == 9
    remove_tmp_database(tmp)


def test_get_values():
    tmp = create_tmp_database(rider)
    mgr = Manager()
    mgr.open(tmp)
    assert mgr.get_values('PatientName', uid='') is None
    patient = 'RIDER Neuro MRI-5244517593'
    attributes = ['PatientName','SeriesDescription','AcquisitionTime']
    values = mgr.get_values('PatientName', uid=patient)
    assert values == '281949'
    values = mgr.get_values('SeriesDescription', uid=patient)
    assert set(values) == set(['ax 10 flip', 'ax 5 flip', 'sag 3d gre +c'])
    values = mgr.get_values('AcquisitionTime', uid=patient)
    assert set(values) == set(['074424.477508', '074443.472496', '074615.472480', '074634.479981', '075505.670007', '075611.105017'])
    values = mgr.get_values(attributes, uid=patient)
    assert values[0] == '281949'
    assert set(values[1]) == set(['ax 10 flip', 'ax 5 flip', 'sag 3d gre +c'])
    assert set(values[2]) == set(['074424.477508', '074443.472496', '074615.472480', '074634.479981', '075505.670007', '075611.105017'])
    values = mgr.get_values('PatientName', uid=patient)
    assert values == '281949'
    remove_tmp_database(tmp)



def test_parent():
    tmp = create_tmp_database(rider)
    mgr = Manager()
    mgr.open(tmp)
    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    patient = 'RIDER Neuro MRI-5244517593'
    assert mgr.parent(series) == '1.3.6.1.4.1.9328.50.16.139212415517744996696223190563412871264'
    assert mgr.parent(patient) == 'Database'
    assert mgr.parent() is None
    assert mgr.parent('abc') is None
    remove_tmp_database(tmp)


def test_instances():
    tmp = create_tmp_database(rider)
    mgr = Manager()
    mgr.open(tmp)
    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    patient = 'RIDER Neuro MRI-5244517593'
    assert mgr.instances(series, sort=False).shape[0] == 2
    assert mgr.instances(patient).shape[0] == 12
    assert mgr.instances('Database', sortby='PatientID').shape[0] == 24
    assert mgr.instances().empty
    remove_tmp_database(tmp)


def test_filter():
    tmp = create_tmp_database(rider)
    mgr = Manager()
    mgr.open(tmp)
    patients = set(mgr.register.values(column='PatientID'))
    patients = mgr.filter(patients, PatientName='281949')
    assert len(patients) == 2
    patients = mgr.filter(patients, PatientID='RIDER Neuro MRI-5244517593')
    assert len(patients) == 1
    series = set(mgr.register.values(column='SeriesInstanceUID'))
    series = mgr.filter(series, SeriesDescription='ax 5 flip')
    assert len(series) == 4
    series = set(mgr.register.values(column='SeriesInstanceUID'))
    series = mgr.filter(series, SeriesDescription='ax 5 flip', PatientID='RIDER Neuro MRI-5244517593')
    assert len(series) == 2
    remove_tmp_database(tmp)


def test_series():
    tmp = create_tmp_database(rider)
    mgr = Manager()
    mgr.open(tmp)
    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    patient = 'RIDER Neuro MRI-5244517593'
    assert len(mgr.series(series)) == 1
    assert len(mgr.series(patient)) == 6
    assert mgr.series() == []
    remove_tmp_database(tmp)


def test_studies():
    tmp = create_tmp_database(rider)
    mgr = Manager()
    mgr.open(tmp)
    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    patient = 'RIDER Neuro MRI-5244517593'
    assert len(mgr.studies(series)) == 1
    assert len(mgr.studies(patient)) == 4
    remove_tmp_database(tmp)


def test_patients():
    tmp = create_tmp_database(rider)
    mgr = Manager()
    mgr.open(tmp)
    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    patient = 'RIDER Neuro MRI-5244517593'
    assert len(mgr.patients(series)) == 1
    assert len(mgr.patients(patient)) == 1
    assert len(mgr.patients('Database')) == 2
    assert mgr.patients() == []
    remove_tmp_database(tmp)


def test_new_patient():
    tmp = create_tmp_database(rider)
    mgr = Manager()
    mgr.open(tmp)
    n = len(mgr.patients('Database'))
    patient, _ = mgr.new_patient()
    assert len(mgr.patients('Database')) == n+1
    assert [] == mgr.studies(patient) 
    assert mgr.instances(patient).empty
    assert 1 == len(mgr.keys(patient))
    assert False in mgr.value(column='removed')
    assert True in mgr.value(column='created')
    remove_tmp_database(tmp)


def test_new_study():
    tmp = create_tmp_database(rider)
    mgr = Manager()
    mgr.open(tmp)
    n = len(mgr.studies('Database'))
    study, _ = mgr.new_study()
    assert len(mgr.studies('Database')) == n+1
    assert [] == mgr.series(study)
    assert mgr.instances(study).empty
    assert 1 == len(mgr.keys(study))
    patient = 'RIDER Neuro MRI-5244517593'
    n = len(mgr.studies(patient))
    study, _ = mgr.new_study(patient)
    assert len(mgr.studies(patient)) == n+1
    assert 0 == len(mgr.series(study))
    remove_tmp_database(tmp)


def test_new_series():
    tmp = create_tmp_database(rider)
    mgr = Manager()
    mgr.open(tmp)
    n = len(mgr.series('Database'))
    series, _ = mgr.new_series()
    assert len(mgr.series('Database')) == n+1
    assert mgr.instances(series).empty
    assert 1 == len(mgr.keys(series))
    study = '1.3.6.1.4.1.9328.50.16.168701627691879645008036315574545460110'
    n = len(mgr.series(study))
    series, _ = mgr.new_series(study)
    assert len(mgr.series(study)) == n+1
    assert mgr.instances(series).empty
    remove_tmp_database(tmp)


def test_new_instance():
    tmp = create_tmp_database(rider)
    mgr = Manager()
    mgr.open(tmp)
    n = mgr.instances('Database').shape[0]
    instance, _ = mgr.new_instance()
    assert mgr.instances('Database').shape[0] == n+1
    assert None is mgr.get_dataset(instance)
    assert 1 == len(mgr.keys(instance))
    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    n = mgr.instances(series).shape[0]
    instance, _ = mgr.new_instance(series)
    assert mgr.instances(series).shape[0] == n+1
    assert None is mgr.get_dataset(instance)
    remove_tmp_database(tmp)


def test_series_header():

    tmp = create_tmp_database(rider)
    mgr = Manager()
    mgr.open(tmp)

    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'

    instance = mgr.instances(series).values.tolist()[0]
    attr, vals = mgr.series_header(mgr.keys(instance)[0])
    series_vals = mgr.get_values(attr, uid=series)
    for i in range(len(vals)):
        assert series_vals[i] == vals[i]

    instance, _ = mgr.new_instance(series)
    attr, vals = mgr.series_header(mgr.keys(instance)[0])
    series_vals = mgr.get_values(attr, uid=series)
    for i in range(len(vals)):
        assert series_vals[i] == vals[i]

def test_set_instance_dataset():

    tmp = create_tmp_database(rider)
    mgr = Manager()
    mgr.open(tmp)

    series1 = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'

    instance1 = mgr.instances(series1).values.tolist()[0]
    ds1 = mgr.get_dataset(instance1)
    ds1.PatientName = 'Blabla'
    ds1.AcquisitionTime = '000000.00'
    mgr.set_instance_dataset(instance1, ds1)
    assert ds1.PatientName != 'Blabla'
    assert ds1.AcquisitionTime == mgr.get_values('AcquisitionTime', uid=instance1)

    instance1 = mgr.instances(series1).values.tolist()[0]
    ds1 = mgr.get_dataset(instance1)
    ds1.PatientName = 'Blabla'
    ds1.AcquisitionTime = '000001.00'
    instance, _ = mgr.new_instance(series1)
    mgr.set_instance_dataset(instance, ds1)
    assert ds1.PatientName != 'Blabla'
    assert ds1.AcquisitionTime == mgr.get_values('AcquisitionTime', uid=instance)

    instance1 = mgr.instances(series1).values.tolist()[0]
    ds1 = MRImage()
    ds1.PatientName = 'Blabla'
    ds1.AcquisitionTime = '000002.00'
    mgr.set_instance_dataset(instance1, ds1)
    assert ds1.PatientName != 'Blabla'
    assert ds1.AcquisitionTime == mgr.get_values('AcquisitionTime', uid=instance1)


def test_set_dataset():

    tmp = create_tmp_database(rider)
    mgr = Manager()
    mgr.open(tmp)

    series1 = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    series2 = '1.3.6.1.4.1.9328.50.16.39537076883396884954303966295061604769'

    # Save datasets in new instances
    dataset = mgr.get_dataset(series1)
    dataset[0].ContentTime = '000000'
    n = len(mgr.instances(series2))
    mgr.set_dataset(series2, dataset[:2])
    assert len(mgr.instances(series2)) == n+2
    assert '000000' in mgr.get_values('ContentTime', uid=series2)
    mgr.restore()
    assert len(mgr.instances(series2)) == n
    assert '000000' not in mgr.get_values('ContentTime', uid=series2)

    # Save datasets in existing instances
    dataset = mgr.get_dataset(series2)
    dataset[0].ContentTime = '000000'
    n = len(mgr.instances(series2))
    assert n == len(dataset)
    mgr.set_dataset(series2, dataset[:2])
    assert len(mgr.instances(series2)) == n
    assert '000000' in mgr.get_values('ContentTime', uid=series2)
    
    remove_tmp_database(tmp)



def test_label():

    tmp = create_tmp_database(rider)
    mgr = Manager()
    mgr.open(tmp)

    patient = mgr.patients('Database')[0]
    assert mgr.label(patient) == 'Patient 281949'
    study = mgr.studies(patient)[0]
    try:
        mgr.label(study)
    except: 
        assert False
    else:
        assert True
    series = mgr.series(study)[0]
    try:
        mgr.label(series)
    except: 
        assert False
    else:
        assert True
    instance = mgr.instances(series).values.tolist()[0]
    try:
        mgr.label(instance)
    except:
        assert False
    else:
        assert True

    remove_tmp_database(tmp)

def test_print():

    tmp = create_tmp_database(rider)

    mgr = Manager()
    mgr.open(tmp)
    try:
        mgr.print()
    except:
        assert False
    else:
        assert True

    remove_tmp_database(tmp)

def test_read_and_clear():

    tmp = create_tmp_database(rider)
    mgr = Manager()
    mgr.open(tmp)

    assert 0 == len(mgr.dataset)
    mgr.read('Database')
    assert 24 == len(mgr.dataset)
    mgr.clear('Database')
    assert 0 == len(mgr.dataset)
    mgr.read(patient='RIDER Neuro MRI-3369019796')
    assert 12 == len(mgr.dataset)
    mgr.clear('Database')
    mgr.read(study='1.3.6.1.4.1.9328.50.16.139212415517744996696223190563412871264')
    assert 4 == len(mgr.dataset)
    mgr.clear('Database')
    mgr.read(series='1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242')
    assert 2 == len(mgr.dataset)
    mgr.clear('Database')
    mgr.read(instance='1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022')
    assert 1 == len(mgr.dataset)
    mgr.clear('Database')

    # Try to read something that does not exist
    mgr.read(series='1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022')
    assert 0 == len(mgr.dataset)

    # read a patient, then a study in that patient - reading study does nothing
    mgr.read(patient='RIDER Neuro MRI-3369019796')
    assert 12 == len(mgr.dataset)
    mgr.read(study='1.3.6.1.4.1.9328.50.16.139212415517744996696223190563412871264')
    assert 12 == len(mgr.dataset)

    # Clear only that study
    mgr.clear(study='1.3.6.1.4.1.9328.50.16.139212415517744996696223190563412871264')
    assert 8 == len(mgr.dataset)

    # Clear a dataset from the study that was just cleared (does nothing)
    mgr.clear(instance='1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022')
    assert 8 == len(mgr.dataset)

    # Clear all datasets from the patient
    mgr.clear(patient='RIDER Neuro MRI-3369019796')
    assert 0 == len(mgr.dataset)

    remove_tmp_database(tmp)

def test_write():

    tmp = create_tmp_database(rider)
    mgr = Manager()
    mgr.open(tmp)

    uid = '1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022'

    # Change a name in memory, write, clear memory and read again to test this has changed.
    mgr.read(instance=uid)
    mgr.set_values('PatientName', 'Anonymous', uid=uid)
    mgr.write('Database')
    mgr.clear('Database')
    mgr.read(instance=uid)
    dataset = mgr.get_dataset(uid)
    assert uid == dataset.SOPInstanceUID
    assert 'Anonymous' == dataset.PatientName

    # Read a series, change all of the datasets, write only one before clearing memory.
    # Then check that only one of the values has changed
    series_uid='1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    mgr.read(series=series_uid)
    mgr.set_values('PatientName', 'Anonymous', uid=series_uid)
    mgr.write('Database')
    mgr.clear('Database')
    mgr.read(instance=uid)
    ds = mgr.get_dataset(uid)
    assert ds.PatientName == 'Anonymous'

    remove_tmp_database(tmp)

def test_open_close():

    tmp = create_tmp_database(rider)
    mgr = Manager(tmp)
    mgr.open()
    assert mgr.register.shape == (24*2, N_COLUMNS)
    mgr.save()
    #mgr.save('Database')
    mgr.close()
    assert mgr.register is None
    try:
        mgr.open()
    except ValueError:
        assert True
    else:
        assert False
    mgr.close()
    remove_tmp_database(tmp)

    tmp = create_tmp_database(twofiles)
    mgr.open(tmp)
    assert mgr.register.shape == (2*2, N_COLUMNS)
    mgr.save()
    #mgr.save('Database')
    mgr.close()
    assert mgr.register is None
    remove_tmp_database(tmp)

def test_inmemory_vs_ondisk():

    tmp = create_tmp_database(rider)

    mgr = Manager()

    # open a database on disk
    # get the dataframe and close it again
    mgr.open(tmp)   
    df = mgr.register
    mgr.close()
    assert df.shape == (24*2, N_COLUMNS)
    assert mgr.register is None

    remove_tmp_database(tmp)

    # create a database in memory
    # Try to read a dataframe and check 
    # that this this is empty
    mgr.register = df
    assert mgr.register.shape == (24*2, N_COLUMNS)
    mgr.scan()
    assert mgr.register.empty




def test_delete():
    
    tmp = create_tmp_database(rider)
    mgr = Manager()
    mgr.open(tmp)

    # Delete all datasets for one series
    uid = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    mgr.delete(series=uid)
    assert len(mgr.keys('Database')) == 22

    # Delete it again
    mgr.delete(series=uid)
    assert len(mgr.keys('Database')) == 22
    assert mgr.keys(series=uid) == []

    # Delete the patient containing this series
    mgr.delete(patient='RIDER Neuro MRI-3369019796')
    assert len(mgr.keys('Database')) == 12
    assert len(mgr.keys(patient='RIDER Neuro MRI-5244517593')) == 12
    assert len(mgr.keys('RIDER Neuro MRI-5244517593')) == 12
    assert len(mgr.keys('RIDER Neuro MRI-3369019796')) == 0

    # Delete the other patient too
    mgr.delete(patient='RIDER Neuro MRI-5244517593')
    assert mgr.keys('Database') == []

    remove_tmp_database(tmp)

def test_copy_to_series():

    tmp = create_tmp_database(rider)
    mgr = Manager()
    mgr.open(tmp)

    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    orig_keys = mgr.keys(patient='RIDER Neuro MRI-5244517593')
    copy_instances = mgr.copy_to_series('RIDER Neuro MRI-5244517593', series)
    assert len(mgr.keys(series)) == 14
    assert len(mgr.keys('RIDER Neuro MRI-5244517593')) == 12
    assert len(mgr.keys('RIDER Neuro MRI-3369019796')) == 24
    assert len(mgr.keys('Database')) == 36
    assert mgr.value(orig_keys[0], 'PatientID') == 'RIDER Neuro MRI-5244517593'
    assert mgr.value(mgr.keys(copy_instances)[0], 'PatientID') == 'RIDER Neuro MRI-3369019796'
    # Check that all new instance numbers are unique
    nrs = mgr.value(mgr.keys(series), 'InstanceNumber')
    assert len(set(nrs)) == len(nrs) 
    remove_tmp_database(tmp)

def test_copy_to_study():

    tmp = create_tmp_database(rider)
    mgr = Manager()
    mgr.open(tmp)

    study = '1.3.6.1.4.1.9328.50.16.168701627691879645008036315574545460110'
    patient = 'RIDER Neuro MRI-5244517593'
    assert len(mgr.series(study)) == 1
    assert len(mgr.series(patient)) == 6
    orig_keys = mgr.keys(patient=patient)
    copy_series = mgr.copy_to_study(patient, study)
    assert len(mgr.instances(study)) == 14
    assert len(mgr.series(study)) == 7
    assert len(mgr.instances('RIDER Neuro MRI-5244517593')) == 12
    assert len(mgr.instances('RIDER Neuro MRI-3369019796')) == 24
    assert len(mgr.instances('Database')) == 36
    assert mgr.value(orig_keys[0], 'PatientID') == 'RIDER Neuro MRI-5244517593'
    assert mgr.value(mgr.keys(copy_series)[0], 'PatientID') == 'RIDER Neuro MRI-3369019796'
    nrs = mgr.get_values('SeriesNumber', uid=study)
    assert len(nrs) == len(mgr.series(study))
    remove_tmp_database(tmp)

def test_copy_to_patient():

    tmp = create_tmp_database(rider)
    mgr = Manager()
    mgr.open(tmp)

    study = '1.3.6.1.4.1.9328.50.16.168701627691879645008036315574545460110'
    patient = 'RIDER Neuro MRI-5244517593'
    assert len(mgr.series(study)) == 1
    assert len(mgr.series(patient)) == 6
    assert len(mgr.instances(study)) == 2
    assert len(mgr.instances(patient)) == 12
    mgr.copy_to_patient(study, mgr.keys(patient=patient)[0])
    assert len(mgr.series(study)) == 1
    assert len(mgr.instances(study)) == 2
    assert len(mgr.series(patient)) == 7
    assert len(mgr.instances(patient)) == 14
    remove_tmp_database(tmp)

# def test_copy_to():

#     # Need to include some scenarios involving copying to from empty objects

#     tmp = create_tmp_database(rider)
#     mgr = Manager()
#     mgr.open(tmp)

#     instance ='1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022'
#     series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
#     study = '1.3.6.1.4.1.9328.50.16.168701627691879645008036315574545460110'
#     patient = 'RIDER Neuro MRI-5244517593'  
#     assert len(mgr.instances(series)) == 2
#     mgr.copy_to(instance, series, 'Series')   # copy an instance to its own series.
#     assert len(mgr.instances(series)) == 3
#     assert len(mgr.series(study)) == 1
#     mgr.copy_to(series, study, 'Study') # copy a series to another study
#     assert len(mgr.series(study)) == 2
#     assert len(mgr.studies(patient)) == 4
#     mgr.copy_to(study, patient, 'Patient') # copy a study to another patient
#     assert len(mgr.studies(patient)) == 5
#     remove_tmp_database(tmp)


def test_move_to_series():

    tmp = create_tmp_database(rider)
    mgr = Manager()
    mgr.open(tmp)

    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    orig_keys = mgr.keys(patient='RIDER Neuro MRI-5244517593')
    nr_instances_in_patient_before_move = len(mgr.instances('RIDER Neuro MRI-5244517593'))
    nr_instances_in_other_patient_before_move = len(mgr.instances('RIDER Neuro MRI-3369019796'))
    nr_instances_in_series_before_move =  len(mgr.instances(series))
    nr_instances_in_database_before_move =  len(mgr.instances('Database'))
    instances = mgr.move_to_series('RIDER Neuro MRI-5244517593', series)
    nr_instances_in_patient_after_move = len(mgr.instances('RIDER Neuro MRI-5244517593'))
    nr_instances_in_other_patient_after_move = len(mgr.instances('RIDER Neuro MRI-3369019796'))
    nr_instances_in_series_after_move =  len(mgr.instances(series))
    nr_instances_in_database_after_move =  len(mgr.instances('Database'))
    assert nr_instances_in_patient_after_move == 0
    assert nr_instances_in_series_after_move == nr_instances_in_series_before_move + nr_instances_in_patient_before_move    
    assert nr_instances_in_other_patient_after_move == nr_instances_in_other_patient_before_move + nr_instances_in_patient_before_move
    assert nr_instances_in_database_after_move == nr_instances_in_database_before_move
    # Check that the patient name is updated in the move
    assert mgr.value(orig_keys[0], 'PatientID') == 'RIDER Neuro MRI-5244517593'
    assert mgr.value(mgr.keys(instances)[0], 'PatientID') == 'RIDER Neuro MRI-3369019796'
    # Check that all new instance numbers are unique
    nrs = mgr.value(mgr.keys(series), 'InstanceNumber')
    assert len(set(nrs)) == len(nrs)
    remove_tmp_database(tmp)


def test_move_to_study():

    tmp = create_tmp_database(rider)
    mgr = Manager()
    mgr.open(tmp)

    study = '1.3.6.1.4.1.9328.50.16.168701627691879645008036315574545460110'
    patient = 'RIDER Neuro MRI-5244517593'
    assert len(mgr.series(study)) == 1
    assert len(mgr.series(patient)) == 6
    series = mgr.move_to_study(patient, study) # move to a study of another patient
    assert len(mgr.instances(study)) == 14
    assert len(mgr.series(study)) == 7
    assert len(mgr.instances(patient)) == 0
    assert len(mgr.instances('RIDER Neuro MRI-3369019796')) == 24
    assert len(mgr.instances('Database')) == 24
    # Check that the empty patient still exists
    assert mgr.get_values('PatientID', uid=patient) == 'RIDER Neuro MRI-5244517593'
    # Check that patient ID is correctly updated
    assert mgr.value(mgr.keys(series)[0], 'PatientID') == 'RIDER Neuro MRI-3369019796'
    # Check that all series numbers are unique
    nrs = mgr.get_values('SeriesNumber', uid=study)
    assert len(nrs) == len(mgr.series(study))
    remove_tmp_database(tmp)

def test_move_to_patient():

    tmp = create_tmp_database(rider)
    mgr = Manager()
    mgr.open(tmp)

    study = '1.3.6.1.4.1.9328.50.16.168701627691879645008036315574545460110'
    patient = 'RIDER Neuro MRI-5244517593'
    assert len(mgr.series(study)) == 1
    assert len(mgr.series(patient)) == 6
    assert len(mgr.instances(study)) == 2
    assert len(mgr.instances(patient)) == 12
    mgr.move_to_patient(study, patient)
    assert len(mgr.series(study)) == 1
    assert len(mgr.instances(study)) == 2
    assert len(mgr.series(patient)) == 7
    assert len(mgr.instances(patient)) == 14
    remove_tmp_database(tmp)

def test_move_to():

    tmp = create_tmp_database(rider)
    mgr = Manager()
    mgr.open(tmp)

    instance ='1.3.6.1.4.1.9328.50.16.251746227724893696781798455517674264022'
    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    study = '1.3.6.1.4.1.9328.50.16.168701627691879645008036315574545460110'
    patient = 'RIDER Neuro MRI-5244517593'  
    assert len(mgr.instances(series)) == 2
    mgr.move_to(instance, series)   # move an instance to its own series.
    assert mgr.get_values('SeriesInstanceUID', uid=instance) == series
    assert len(mgr.instances(series)) == 2
    assert len(mgr.series(study)) == 1
    mgr.move_to(series, study) # copy a series to another study
    assert mgr.get_values('StudyInstanceUID', uid=series) == study
    assert len(mgr.series(study)) == 2
    assert len(mgr.studies(patient)) == 4
    mgr.move_to(study, patient) # move a study to another patient
    assert mgr.get_values('PatientID', uid=study) == patient
    assert len(mgr.studies(patient)) == 5
    remove_tmp_database(tmp)


def test_set_values():

    tmp = create_tmp_database(rider)
    mgr = Manager()
    mgr.open(tmp)

    patient = 'RIDER Neuro MRI-5244517593'
    for name in mgr.value(mgr.keys(patient), 'PatientName'):
        assert name != 'Anonymous'
    for t in mgr.value(mgr.keys(patient), 'SeriesDescription'):
        assert t != 'Desc'
    for instance in mgr.instances(patient).values.tolist():
        assert mgr.get_dataset(instance).get_values('AcquisitionTime') != '000000.00'

    mgr.set_values(['PatientName', 'SeriesDescription', 'AcquisitionTime'], ['Anonymous', 'Desc', '000000.00'], uid=patient)

    for name in mgr.value(mgr.keys(patient), 'PatientName'):
        assert name == 'Anonymous'
    for t in mgr.value(mgr.keys(patient), 'SeriesDescription'):
        assert t == 'Desc'
    for instance in mgr.instances(patient).values.tolist():
        assert mgr.get_dataset(instance).get_values('AcquisitionTime') == '000000.00'

    remove_tmp_database(tmp)




def test_restore():

    tmp = create_tmp_database(rider)
    mgr = Manager()
    mgr.open(tmp)

    patient = 'RIDER Neuro MRI-5244517593'

    assert '' == mgr.get_values('PatientSex', uid=patient)
    mgr.set_values('PatientSex', 'M', uid=patient)
    assert 'M' == mgr.get_values('PatientSex', uid=patient)
    mgr.restore()
    assert '' == mgr.get_values('PatientSex', uid=patient)

    mgr.read(patient)
    assert '' == mgr.get_values('PatientSex', uid=patient)
    mgr.set_values('PatientSex', 'M', uid=patient)
    assert 'M' == mgr.get_values('PatientSex', uid=patient)
    mgr.restore()
    assert '' == mgr.get_values('PatientSex', uid=patient)

    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'
    study = '1.3.6.1.4.1.9328.50.16.168701627691879645008036315574545460110'

    assert len(mgr.series(study)) == 1
    mgr.move_to(series, study) 
    assert len(mgr.series(study)) == 2
    mgr.restore()
    assert len(mgr.series(study)) == 1

    mgr.read(series)
    assert len(mgr.series(study)) == 1
    mgr.move_to(series, study) 
    assert len(mgr.series(study)) == 2
    mgr.restore()
    assert len(mgr.series(study)) == 1

    remove_tmp_database(tmp)

def test_save():

    tmp = create_tmp_database(rider)
    mgr = Manager()
    mgr.open(tmp)

    # Three objects that are not nested: study not in patient and series not in study.
    patient = 'RIDER Neuro MRI-5244517593'
    study = '1.3.6.1.4.1.9328.50.16.168701627691879645008036315574545460110'
    series = '1.3.6.1.4.1.9328.50.16.63380113333602578570923656300898710242'

    assert '' == mgr.get_values('PatientSex', uid=patient)
    mgr.set_values('PatientSex', 'M', uid=patient)
    assert 'M' == mgr.get_values('PatientSex', uid=patient)
    mgr.save()
    #mgr.save('Database')
    assert 'M' == mgr.get_values('PatientSex', uid=patient)

    mgr.read(patient)
    assert 'M' == mgr.get_values('PatientSex', uid=patient)
    mgr.set_values('PatientSex', '', uid=patient)
    assert '' == mgr.get_values('PatientSex', uid=patient)
    mgr.save()
    #mgr.save('Database')
    assert '' == mgr.get_values('PatientSex', uid=patient)

    assert len(mgr.series(study)) == 1
    mgr.move_to(series, study) 
    assert len(mgr.series(study)) == 2
    mgr.save()
    #mgr.save('Database')
    assert len(mgr.series(study)) == 2

    mgr.read(series)
    assert len(mgr.series(study)) == 2
    mgr.move_to(series, study) 
    assert len(mgr.series(study)) == 2
    mgr.save()
    #mgr.save('Database')
    mgr.restore()
    assert len(mgr.series(study)) == 2

    remove_tmp_database(tmp)



def test_template():

    tmp = create_tmp_database(rider)
    mgr = Manager()
    mgr.open(tmp)

    remove_tmp_database(tmp)




def test_import_datasets():

    source = create_tmp_database(rider, name='source')
    source_mgr = Manager()
    source_mgr.open(source)
    source_files = source_mgr.filepaths('Database')

    # Create empty database and import all source files
    target = create_tmp_database(name='target')
    target_mgr = Manager()
    target_mgr.open(target)
    target_mgr.import_datasets(source_files)

    # Check that the number of datasets equals nr of source files
    assert len(target_mgr.instances('Database')) == len(source_files)

    # Import all source files again and check that nothing has changed
    target_mgr.import_datasets(source_files)
    assert len(target_mgr.instances('Database')) == len(source_files)

    # Delete one patient, import all source files again and check that nothing has changed
    patient = target_mgr.patients('Database', PatientID='RIDER Neuro MRI-5244517593')
    target_mgr.delete(patient)
    target_mgr.import_datasets(source_files)
    assert len(target_mgr.instances('Database')) == len(source_files)

    # Save new database and check that nothing has changed
    target_mgr.save()
    #target_mgr.save('Database')
    assert len(target_mgr.instances('Database')) == len(source_files)

    # Delete one patient and import files from that patient again
    # Check that nothing has changed
    patient = target_mgr.patients('Database', PatientID='RIDER Neuro MRI-5244517593')
    target_mgr.delete(patient)
    patient = source_mgr.patients('Database', PatientID='RIDER Neuro MRI-5244517593')
    patient_files = source_mgr.filepaths(patient)
    assert len(target_mgr.instances('Database')) == len(source_files)-len(patient_files)
    target_mgr.import_datasets(patient_files)
    assert len(target_mgr.instances('Database')) == len(source_files)

    # Delete one patient and import files from another patient
    # Check that files are not imported again
    patient1 = target_mgr.patients('Database', PatientID='RIDER Neuro MRI-5244517593')
    patient1_files = target_mgr.filepaths(patient1)
    patient2 = source_mgr.patients('Database', PatientID='RIDER Neuro MRI-3369019796')
    patient2_files = source_mgr.filepaths(patient2)
    target_mgr.delete(patient1)
    assert len(target_mgr.instances('Database')) == len(source_files)-len(patient1_files)
    target_mgr.import_datasets(patient2_files)
    assert len(target_mgr.instances('Database')) == len(source_files)-len(patient1_files)

    # Start over, this time with a full target databese
    # Import all source files and check nothing has changed
    target = create_tmp_database(rider, name='target')
    target_mgr = Manager()
    target_mgr.open(target)
    assert len(target_mgr.instances('Database')) == len(source_files)
    target_mgr.import_datasets(source_files)
    assert len(target_mgr.instances('Database')) == len(source_files)

    remove_tmp_database(source)
    remove_tmp_database(target)

def test_export_datasets():

    source = create_tmp_database(rider, name='source')
    source_mgr = Manager()
    source_mgr.open(source)
    patient = source_mgr.patients('Database', PatientID='RIDER Neuro MRI-5244517593')

    target = create_tmp_database(name='target')
    target_mgr = Manager()
    target_mgr.open(target)
    source_mgr.export_datasets(patient, target_mgr)

    assert len(target_mgr.instances('Database')) == len(source_mgr.instances(patient))
    assert len(target_mgr.series('Database')) == len(source_mgr.series(patient))
    assert len(target_mgr.studies('Database')) == len(source_mgr.studies(patient))

    remove_tmp_database(source)
    remove_tmp_database(target)

def test_new_database():

    mgr = Manager()

    p1, _ = mgr.new_patient('Database')
    p2, _ = mgr.new_patient('Database')
    p3, _ = mgr.new_patient('Database')

    p1v1, _ = mgr.new_study(p1)
    p1v2, _ = mgr.new_study(p1)

    p1v1s1, _ = mgr.new_series(p1v1)
    p1v1s2, _ = mgr.new_series(p1v1)
    p1v1s3, _ = mgr.new_series(p1v1)

    mgr.new_instance(p1v1s1, dataset=MRImage())
    mgr.new_instance(p1v1s1, dataset=MRImage())
    mgr.new_instance(p1v1s1, dataset=MRImage())
    mgr.new_instance(p1v1s1, dataset=MRImage())
    mgr.new_instance(p1v1s1, dataset=MRImage())
    mgr.new_instance(p1v1s1, dataset=MRImage())

    assert 6 == len(mgr.instances(p1v1))

def test_tree():

    tmp = create_tmp_database(rider)
    mgr = Manager()
    mgr.open(tmp)

    start = timeit.default_timer()
    tree = mgr.tree(depth=1)
    tree = mgr.tree(depth=2)
    tree = mgr.tree(depth=3)
    stop = timeit.default_timer()

    print(tree['uid'])
    for patient in tree['patients']:
        for study in patient['studies']:
            for series in study['series']:
                print('Patient: ', patient['uid'])
                print('Study: ', study['uid'])
                print('Series: ', series['uid'])
    print(tree['patients'][0]['studies'][0]['series'][0])

    assert len(tree) == 2
    assert stop-start <= 0.5

    print('time to build tree:', stop-start)

    remove_tmp_database(tmp)

if __name__ == "__main__":

    exit()

    test_init()
    test_read_register()
    test_read_write_register()
    test_multiframe_to_singleframe()
    test_scan()
    test_open()
    test_type()
    test_keys()
    test_filepath()
    test_filepaths()
    test_value()
    test_get_instance_dataset()
    test_get_dataset()
    test_get_values()
    test_parent()
    test_instances()
    test_filter()
    test_series()
    test_studies()
    test_patients()
    test_new_patient()
    test_new_study()
    test_new_series()
    test_new_instance()
    # test_series_header()
    # test_set_instance_dataset()
    # test_set_dataset()
    # test_label()
    # test_print()
    # test_read_and_clear()
    # test_write()
    # test_open_close()
    # test_inmemory_vs_ondisk() 
    # test_delete()
    # test_copy_to_series()
    # test_copy_to_study()
    # test_copy_to_patient()
    # test_move_to_series()
    # test_move_to_study()
    # test_move_to_patient()
    # test_move_to()
    # test_set_values()
    # test_restore()
    # test_save()
    # test_import_datasets()
    # test_export_datasets()
    # test_new_database()
    # test_tree()



    print('---------------------------------')
    print('The new manager passed all tests!')
    print('---------------------------------')

