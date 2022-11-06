import os
import shutil
import timeit
import numpy as np
import matplotlib.pyplot as plt

import dbdicom as db
from dbdicom.ds import MRImage



datapath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
ct = os.path.join(datapath, '2_skull_ct')
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


# Test functions

def test_read():

    tmp = create_tmp_database(ct)
    database = db.database(tmp)
    database.print()
    remove_tmp_database(tmp)

def test_database():

    tmp = create_tmp_database(rider)
    database = db.database(tmp)

    try:
        database.print()
    except:
        assert False
    assert 24 == len(database.instances())
    assert 24 == len(database.files())

    database.close()
    remove_tmp_database(tmp)


def test_children():

    tmp = create_tmp_database(rider)
    database = db.database(tmp)

    patients = database.children(PatientID='RIDER Neuro MRI-3369019796')
    assert patients[0].label() == 'Patient 281949'
    studies = patients[0].children()
    assert len(studies) == 4

    database.close()
    remove_tmp_database(tmp)


def test_read_dicom_data_elements():

    tmp = create_tmp_database(rider)
    database = db.database(tmp)

    patient_id = database.PatientID
    patients = database.children()
    for patient in patients:
        assert patient.PatientID in patient_id

    acq_time = database.AcquisitionTime
    instances = database.instances()
    for instance in instances:
        assert instance.AcquisitionTime in acq_time

    desc = patients[0].StudyDescription
    study = patients[0].children()[0]
    assert study.StudyDescription in desc

    desc = study.SeriesDescription
    series = study.children()[0]
    assert series.SeriesDescription in desc

    instance = series.children()[0]
    assert instance.AcquisitionTime in acq_time

    database.close()
    remove_tmp_database(tmp)


def test_read_dicom_data_elements_from_memory(): 

    tmp = create_tmp_database(rider)
    database = db.database(tmp)

    database.read()

    patient_id = database.PatientID
    patients = database.children()
    for patient in patients:
        assert patient.PatientID in patient_id

    acq_time = database.AcquisitionTime
    instances = database.instances()
    for instance in instances:
        assert instance.AcquisitionTime in acq_time

    desc = patients[0].StudyDescription
    study = patients[0].children()[0]
    assert study.StudyDescription in desc

    desc = study.SeriesDescription
    series = study.children()[0]
    assert series.SeriesDescription in desc

    instance = series.children()[0]
    assert instance.AcquisitionTime in acq_time

    database.close()
    remove_tmp_database(tmp)


def test_hierarchy():

    tmp = create_tmp_database(rider)
    database = db.database(tmp)

    patients = database.patients() 
    assert len(patients) == 2
    studies = database.studies()
    assert len(studies) == 8
    series = database.series()
    assert len(series) == 12
    instances = database.instances()
    assert len(instances) == 24

    nr_series = 0
    nr_instances = 0
    for patient in patients:
        nr_series += len(patient.series())
        nr_instances += len(patient.instances())
    assert nr_instances == 24

    nr_series = 0
    nr_instances = 0
    for study in studies:
        nr_series += len(study.series())
        nr_instances += len(study.instances())
    assert nr_instances == 24

    assert patients[0].instances()[0].patients()[0].PatientID in database.PatientID
    assert patients[0].instances()[-1].patients()[0].PatientID in database.PatientID
    assert studies[1].instances()[0].studies()[0].StudyDescription in database.StudyDescription
    assert studies[1].instances()[-1].studies()[0].StudyDescription in database.StudyDescription

    remove_tmp_database(tmp)
    
def test_hierarchy_in_memory_v1():

    tmp = create_tmp_database(rider)
    database = db.database(tmp)

    database.read()

    patients = database.patients() 
    assert len(patients) == 2
    studies = database.studies()
    assert len(studies) == 8
    series = database.series()
    assert len(series) == 12
    instances = database.instances()
    assert len(instances) == 24

    nr_series = 0
    nr_instances = 0
    for patient in patients:
        nr_series += len(patient.series())
        nr_instances += len(patient.instances())
    assert nr_instances == 24

    nr_series = 0
    nr_instances = 0
    for study in studies:
        nr_series += len(study.series())
        nr_instances += len(study.instances())
    assert nr_instances == 24

    assert patients[0].instances()[0].patients()[0].PatientID in database.PatientID
    assert patients[0].instances()[-1].patients()[0].PatientID in database.PatientID
    assert studies[1].instances()[0].studies()[0].StudyDescription in database.StudyDescription
    assert studies[1].instances()[-1].studies()[0].StudyDescription in database.StudyDescription

    remove_tmp_database(tmp)

def test_hierarchy_in_memory_v2():

    tmp = create_tmp_database(rider)
    database = db.database(tmp)

    patients = database.patients() 
    assert len(patients) == 2
    studies = database.studies()
    assert len(studies) == 8
    series = database.series()
    assert len(series) == 12
    instances = database.instances()
    assert len(instances) == 24

    nr_series = 0
    nr_instances = 0
    for patient in patients:
        patient.read()
        nr_series += len(patient.series())
        nr_instances += len(patient.instances())
    assert nr_instances == 24

    nr_series = 0
    nr_instances = 0
    for study in studies:
        study.read()
        nr_series += len(study.series())
        nr_instances += len(study.instances())
    assert nr_instances == 24

    assert patients[0].instances()[0].patients()[0].PatientID in database.PatientID
    assert patients[0].instances()[-1].patients()[0].PatientID in database.PatientID
    assert studies[1].instances()[0].studies()[0].StudyDescription in database.StudyDescription
    assert studies[1].instances()[-1].studies()[0].StudyDescription in database.StudyDescription

    remove_tmp_database(tmp)

def test_find_by_value():

    tmp = create_tmp_database(rider)
    database = db.database(tmp)

    series = database.series(
        SeriesDescription = 'ax 20 flip', 
        PatientID = 'RIDER Neuro MRI-5244517593')
    assert len(series) == 0
    series = database.series(
        SeriesDescription = 'ax 10 flip', 
        PatientID = 'RIDER Neuro MRI-5244517593')
    assert len(series) == 2
    series = database.series(
        StudyDescription = 'BRAIN^ROUTINE BRAIN', 
        SeriesDescription = 'ax 10 flip', 
        PatientID = 'RIDER Neuro MRI-5244517593')
    assert len(series) == 1

    remove_tmp_database(tmp)

def test_find_by_value_in_memory():

    tmp = create_tmp_database(rider)
    database = db.database(tmp)
    database.read()

    series = database.series(
        SeriesDescription = 'ax 20 flip', 
        PatientID = 'RIDER Neuro MRI-5244517593')
    assert len(series) == 0
    series = database.series(
        SeriesDescription = 'ax 10 flip', 
        PatientID = 'RIDER Neuro MRI-5244517593')
    assert len(series) == 2
    series = database.series(
        StudyDescription = 'BRAIN^ROUTINE BRAIN', 
        SeriesDescription = 'ax 10 flip', 
        PatientID = 'RIDER Neuro MRI-5244517593')
    assert len(series) == 1

    remove_tmp_database(tmp)

def test_read_item_instance():

    tmp = create_tmp_database(rider)
    database = db.database(tmp)
    tags = [
        'SeriesDescription', 
        (0x0010, 0x0020), 
        (0x0010, 0x0020), 
        'PatientID', 
        'Rows',
        (0x0011, 0x0020)]
    instance = '1.3.6.1.4.1.9328.50.16.175333593952805976694548436931998383940'
    instance = database.instances(SOPInstanceUID=instance)[0]
    assert instance[tags] == [
        'sag 3d gre +c', 
        'RIDER Neuro MRI-3369019796', 
        'RIDER Neuro MRI-3369019796', 
        'RIDER Neuro MRI-3369019796', 
        256,
        None]

    remove_tmp_database(tmp)

def test_read_item():

    tmp = create_tmp_database(rider)
    database = db.database(tmp)
    tags = [
        'SeriesDescription', 
        (0x0010, 0x0020), 
        (0x0010, 0x0020), 
        'PatientID', 
        'Rows',
        (0x0011, 0x0020)]
    uid = '1.3.6.1.4.1.9328.50.16.300508905663376267701233831747863284128'
    series = database.series(SeriesInstanceUID=uid)[0]
    assert series[tags] == [
        'ax 10 flip', 
        'RIDER Neuro MRI-5244517593', 
        'RIDER Neuro MRI-5244517593', 
        'RIDER Neuro MRI-5244517593', 
        256,
        None]
    patient = database.patients(PatientID='RIDER Neuro MRI-5244517593')[0]
    assert set(patient[tags][0]) == set(['ax 10 flip', 'ax 5 flip', 'sag 3d gre +c'])

    remove_tmp_database(tmp)


def test_set_attr_instance():

    tmp = create_tmp_database(rider)
    database = db.database(tmp)

    instance = database.instances()[0]

    orig_slice_loc = instance.SliceLocation
    instance.SliceLocation = orig_slice_loc + 100
    new_slice_loc = instance.SliceLocation
    assert 100 == np.round(new_slice_loc-orig_slice_loc)

    orig_acq_time = instance.AcquisitionTime
    instance.AcquisitionTime = str(float(orig_acq_time) + 3.0) 
    new_acq_time = instance.AcquisitionTime
    assert 3 == np.round(float(new_acq_time)-float(orig_acq_time))

    uid = instance.uid
    try:
        instance.SOPInstanceUID = '007'
    except:
        assert True
    else:
        assert False
    assert instance.SOPInstanceUID == uid

    remove_tmp_database(tmp)


def test_set_attr_instance_in_memory_v1():

    tmp = create_tmp_database(rider)
    database = db.database(tmp)

    instance = database.instances()[0]
    instance.read()

    orig_slice_loc = instance.SliceLocation
    instance.SliceLocation = orig_slice_loc + 100
    new_slice_loc = instance.SliceLocation
    assert 100 == np.round(new_slice_loc-orig_slice_loc)

    orig_acq_time = instance.AcquisitionTime
    instance.AcquisitionTime = str(float(orig_acq_time) + 3.0) 
    new_acq_time = instance.AcquisitionTime
    assert 3 == np.round(float(new_acq_time)-float(orig_acq_time))

    uid = instance.uid
    try:
        instance.SOPInstanceUID = '007'
    except:
        assert True
    else:
        assert False
    assert instance.SOPInstanceUID == uid

    remove_tmp_database(tmp)


def test_set_attr_instance_in_memory_v2():

    tmp = create_tmp_database(rider)
    database = db.database(tmp)

    database.read()
    instance = database.instances()[0]

    orig_slice_loc = instance.SliceLocation
    instance.SliceLocation = orig_slice_loc + 100
    new_slice_loc = instance.SliceLocation
    assert 100 == np.round(new_slice_loc-orig_slice_loc)

    orig_acq_time = instance.AcquisitionTime
    instance.AcquisitionTime = str(float(orig_acq_time) + 3.0) 
    new_acq_time = instance.AcquisitionTime
    assert 3 == np.round(float(new_acq_time)-float(orig_acq_time))

    uid = instance.uid
    try:
        instance.SOPInstanceUID = '007'
    except:
        assert True
    else:
        assert False
    assert instance.SOPInstanceUID == uid

    remove_tmp_database(tmp)


def test_set_item_instance():

    tmp = create_tmp_database(rider)
    database = db.database(tmp)

    instance = database.instances()[0]

    tags = ['SliceLocation', 'AcquisitionTime', (0x0012, 0x0010)]
    new_values = [0.0, '000000.00', 'University of Sheffield']

    assert instance[tags] != new_values
    instance[tags] = new_values
    assert instance[tags] == new_values

    remove_tmp_database(tmp)


def test_set_item_instance_in_memory():

    tmp = create_tmp_database(rider)
    database = db.database(tmp)

    instance = database.instances()[0]
    instance.read()

    tags = ['SliceLocation', 'AcquisitionTime', (0x0012, 0x0010)]
    new_values = [0.0, '000000.00', 'University of Sheffield']

    assert instance[tags] != new_values
    instance[tags] = new_values
    assert instance[tags] == new_values

    remove_tmp_database(tmp)


def test_set_item():

    tmp = create_tmp_database(rider)
    database = db.database(tmp)

    series = '1.3.6.1.4.1.9328.50.16.121915437221985680060436367746350988049'
    instance = '1.3.6.1.4.1.9328.50.16.243004851579310565813723110219735642931'
    series = database.series(SeriesInstanceUID=series)[0]
    instance = series.instances(SOPInstanceUID=instance)[0]

    tags = ['SliceLocation', 'AcquisitionTime', (0x0012, 0x0010)]
    values = series[tags]
    assert set(values[0]) == set([59.872882843018, 64.872882843018])
    assert values[1:] == ['074424.477508', None]

    values = instance[tags]
    assert values == [59.872882843018, '074424.477508', None]

    new_values = [0.0, '000000.00', 'University of Sheffield']
    series[tags] = new_values
    assert series[tags] == new_values
    assert instance[tags] == new_values

    remove_tmp_database(tmp)


def test_set_item_in_memory():

    tmp = create_tmp_database(rider)
    database = db.database(tmp)

    series = '1.3.6.1.4.1.9328.50.16.121915437221985680060436367746350988049'
    instance = '1.3.6.1.4.1.9328.50.16.243004851579310565813723110219735642931'
    series = database.series(SeriesInstanceUID=series)[0]
    instance = series.instances(SOPInstanceUID=instance)[0]

    series.read()

    tags = ['SliceLocation', 'AcquisitionTime', (0x0012, 0x0010)]
    values = series[tags]
    assert set(values[0]) == set([59.872882843018, 64.872882843018])
    assert values[1:] == ['074424.477508', None]

    values = instance[tags]
    assert values == [59.872882843018, '074424.477508', None]

    new_values = [0.0, '000000.00', 'University of Sheffield']
    series[tags] = new_values
    assert series[tags] == new_values
    assert instance[tags] == new_values

    remove_tmp_database(tmp)
    

def test_create_records():

    tmp = create_tmp_database(rider)
    database = db.database(tmp)

    patient = database.new_child()
    study = patient.new_child()
    series = study.new_child()
    instance = series.new_child()

    assert instance.parent() == series
    assert series.parent() == study
    assert study.parent() == patient
    assert patient.parent() == database

    database.read()

    patient = database.new_child()
    study = patient.new_child()
    series = study.new_child()
    instance = series.new_child()

    assert instance.parent() == series
    assert series.parent() == study
    assert study.parent() == patient
    assert patient.parent() == database

    remove_tmp_database(tmp)


def test_copy_remove_instance():

    tmp = create_tmp_database(rider)
    database = db.database(tmp)
    
    #
    # move and copy an instance from one series to another
    #

    parent = database.patients()[0].studies()[0].series()[0] 
    instance = parent.instances()[0]
    new_parent = database.patients()[1].studies()[0].series()[0] 

    # copy the instance to the new parent and remove the original
    n = len(new_parent.instances())
    desc = new_parent.SeriesDescription
    copy = instance.copy_to(new_parent)

    assert len(new_parent.instances()) == 1 + n
    assert desc == new_parent.SeriesDescription
    assert instance.SeriesDescription == parent.SeriesDescription
    assert instance.SOPInstanceUID in parent.SOPInstanceUID
    assert copy.SOPInstanceUID in new_parent.SOPInstanceUID
    assert copy.SeriesDescription == new_parent.SeriesDescription

    # remove the original instance
    n0 = len(parent.instances())
    uid = instance.SOPInstanceUID
    instance.remove()

    assert len(parent.instances()) == n0-1
    assert uid not in parent.SOPInstanceUID

    # move the copy back to the original parent
    # this should restore the situation
    copy = copy.move_to(parent)

    assert len(parent.instances()) == n0
    assert len(new_parent.instances()) == n
    assert copy.SOPInstanceUID not in new_parent.SOPInstanceUID
    assert copy.SOPInstanceUID in parent.SOPInstanceUID

    #
    # move and copy an instance from one study to another
    #

    patients = database.patients()
    parent = patients[0].studies()[0]
    new_parent = patients[1].studies()[0] 
    instance = parent.instances()[0]
    
    # copy the instance to the new parent and remove the original
    n0 = len(parent.instances())
    n = len(new_parent.instances())
    desc = new_parent.StudyDescription
    series = new_parent.new_child()

    copy = instance.copy_to(series)

    assert len(parent.instances()) == n0
    assert len(new_parent.instances()) == 1 + n
    assert desc == new_parent.StudyDescription
    assert instance.StudyDescription == parent.StudyDescription
    assert instance.SOPInstanceUID in parent.SOPInstanceUID
    assert copy.SOPInstanceUID in new_parent.SOPInstanceUID
    assert copy.StudyDescription == new_parent.StudyDescription

    # remove the original instance
    uid = instance.SOPInstanceUID
    series = instance.parent()

    instance.remove() 

    assert len(parent.instances()) == n0-1
    assert uid not in parent.SOPInstanceUID

    # move the copy back to the original parent
    # this should restore the situation
    
    copy = copy.move_to(series)

    assert len(parent.instances()) == n0
    assert len(new_parent.instances()) == n
    assert copy.SOPInstanceUID not in new_parent.SOPInstanceUID
    assert copy.SOPInstanceUID in parent.SOPInstanceUID

    remove_tmp_database(tmp)


def test_copy_remove():

    tmp = create_tmp_database(rider)
    database = db.database(tmp)

    # Copy a series of patient 1 to a new study of patient 0
    patients = database.patients()
    new_patient0_study = patients[0].new_child()
    patient1_study = patients[1].studies()[0] 
    patient1_series = patient1_study.series()[0]

    nr_patient1_study = len(patient1_study.series())
    nr_new_patient0_study = len(new_patient0_study.series())

    copy_patient1_series = patient1_series.copy_to(new_patient0_study)

    assert len(patient1_study.series()) == nr_patient1_study
    assert len(new_patient0_study.series()) == 1 + nr_new_patient0_study
    assert patient1_series.SeriesDescription in patient1_study.SeriesDescription
    assert copy_patient1_series.SeriesDescription in new_patient0_study.SeriesDescription

    # Check that series header of copy is preserved
    attr, values_copy = copy_patient1_series.series_data()
    _, values_source = patient1_series.series_data()
    for i, v_copy in enumerate(values_copy):
        if attr[i] not in ['SeriesInstanceUID', 'SeriesNumber', 'SeriesDescription']:
            assert v_copy == values_source[i]

    # Check that study header has changed
    attr, values_copy = copy_patient1_series.study_data()
    _, values_source = patient1_series.study_data()
    for i, v_copy in enumerate(values_copy):
        if attr[i] not in ['StudyInstanceUID', 'StudyDate', 'StudyDescription']:
            assert v_copy == values_source[i]
    
    remove_tmp_database(tmp)

def test_inherit_attributes():
    
    tmp = create_tmp_database(rider)
    database = db.database(tmp)
    
    #
    # Copy RIDER series to new patient and study
    #

    patients = database.patients()
    rider_series1 = patients[0].studies()[0].series()[0]
    rider_series2 = patients[1].studies()[0].series()[0]

    james_bond = database.new_patient(
        PatientName = 'James Bond', 
        PatientSex = 'M', 
        PatientBirthDate = '19111972',
    ) 
    james_bond_mri = james_bond.new_study(
        PatientName = 'Joanne Bond', # Ignore - alread set in patient
        StudyDescription = 'MRI hip replacement',
        Occupation = 'Secret agent',
    )

    copy_series = rider_series1.copy_to(james_bond_mri)

    assert copy_series.PatientName == 'James Bond'
    assert copy_series.PatientSex == 'M'
    assert copy_series.PatientBirthDate == '19111972'
    assert copy_series.StudyDescription == 'MRI hip replacement'
    assert copy_series.Occupation == 'Secret agent'

    rider_series2.move_to(james_bond_mri)

    assert rider_series2.PatientName == 'James Bond'
    assert rider_series2.PatientSex == 'M'
    assert rider_series2.PatientBirthDate == '19111972'
    assert rider_series2.StudyDescription == 'MRI hip replacement'
    assert rider_series2.Occupation == 'Secret agent'


def test_merge():

    tmp = create_tmp_database(rider)
    database = db.database(tmp)

    # first create two new patients
    # and copy series into them
    patient1 = database.new_child()
    patient2 = database.new_child()
    series = database.series(SeriesDescription = 'sag 3d gre +c')
    for s in series:
        new_study = patient1.new_child()
        s.copy_to(new_study)
    for s in database.series(SeriesDescription = 'ax 5 flip'):
        new_study = patient2.new_child()
        s.copy_to(new_study) 

    n_instances = len(patient1.instances()) + len(patient2.instances())
    n_studies = len(patient1.studies()) + len(patient2.studies())
    n_series = len(patient1.series()) + len(patient2.series())

    # merge the two patients into a third
    patient3 = db.merge([patient1, patient2])
    assert len(patient3.studies()) == n_studies
    assert len(patient3.series()) == n_series
    assert len(patient3.instances()) == n_instances
    assert patient1.SeriesDescription == 'sag 3d gre +c'
    assert patient2.SeriesDescription == 'ax 5 flip'
    assert set(patient3.SeriesDescription) == set(['sag 3d gre +c', 'ax 5 flip'])

    # now merge all studies of the new patient 
    # into a new study of the same patient.
    new_study = db.merge(patient3.studies())
    
    assert len(patient3.studies()) == n_studies + 1
    assert len(patient3.series()) == 2 * n_series
    assert len(new_study.series()) == n_series
    assert len(patient3.instances()) == 2 * n_instances

    # now merge all series of the new patient into
    # a new series in a new study of the same patient
    study = patient3.new_study()
    db.merge(patient3.series(), into=study.new_series())

    assert len(patient3.studies()) == n_studies + 2
    assert len(patient3.series()) == 2 * n_series + 1
    assert len(patient3.instances()) == 4 * n_instances
    assert len(study.series()) == 1

    remove_tmp_database(tmp)


def test_merge_empty():

    database = db.database()

    james_bond = database.new_patient(PatientName='James Bond')
    james_bond_mri = james_bond.new_study(StudyDescription='MRI')
    james_bond_mri_localizer = james_bond_mri.new_series(SeriesDescription='Localizer')
    james_bond_mri_T2w = james_bond_mri.new_series(SeriesDescription='T2w')
    james_bond_xray = james_bond.new_study(StudyDescription='Xray')
    james_bond_xray_chest = james_bond_xray.new_series(SeriesDescription='Chest')
    james_bond_xray_head = james_bond_xray.new_series(SeriesDescription='Head')

    scarface = database.new_patient(PatientName='Scarface')
    scarface_mri = scarface.new_study(StudyDescription='MRI')
    scarface_mri_localizer = scarface_mri.new_series(SeriesDescription='Localizer')
    scarface_mri_T2w = scarface_mri.new_series(SeriesDescription='T2w')
    scarface_xray = scarface.new_study(StudyDescription='Xray')
    scarface_xray_chest = scarface_xray.new_series(SeriesDescription='Chest')
    scarface_xray_head = scarface_xray.new_series(SeriesDescription='Head')

    assert len(database.studies(StudyDescription='Xray')) == 2
    assert len(database.studies(StudyDescription='Xray', PatientName='Scarface')) == 1
    assert len(database.series(SeriesDescription='T2w')) == 2
    assert len(database.series(SeriesDescription='T2w', PatientName='Scarface')) == 1

    batman = db.merge([scarface, james_bond])

    assert len(batman.studies()) == 4
    assert len(batman.studies(StudyDescription='MRI')) == 2
    assert len(batman.studies(PatientName='James Bond')) == 0
    assert len(batman.studies(StudyDescription='MRI', PatientName='James Bond')) == 0

    new_study = db.merge(batman.studies())

    assert len(batman.studies()) == 5
    assert len(new_study.series()) == 8

    scarface_xray.copy_to(james_bond)

    assert len(james_bond.studies(StudyDescription='Xray')) == 2
    assert len(james_bond.studies(StudyDescription='Xray', PatientName='James Bond')) == 2

    scarface_mri.move_to(james_bond)

    assert len(scarface.studies(StudyDescription='MRI')) == 0
    assert len(james_bond.studies(PatientName='James Bond')) == 4
    assert len(james_bond.studies(StudyDescription='MRI')) == 2
    assert len(james_bond.studies(StudyDescription='MRI', PatientName='James Bond')) == 2

    localizer_slice1 = MRImage()
    localizer_slice2 = MRImage()
    T2w = MRImage()

    scarface_mri_localizer.set_dataset(localizer_slice1)
    scarface_mri_localizer.set_dataset(localizer_slice2)
    scarface_mri_T2w.set_dataset(T2w)

    ds = scarface_mri_localizer.get_dataset()
    
    assert ds[0].SeriesDescription == 'Localizer'
    assert ds[0].PatientName == 'James Bond'
    assert ds[1].SeriesDescription == 'Localizer'
    assert ds[1].PatientName == 'James Bond'

    assert set([ds[0].InstanceNumber, ds[1].InstanceNumber]) == set([1,2])

    tmp = create_tmp_database()
    database.save(tmp)

    database = db.database(tmp)
    assert set(database.PatientName) == set(['James Bond', 'Scarface', 'New Patient'])
    assert set(database.StudyDescription) == set(['MRI', 'Xray', 'New Study'])

    remove_tmp_database(tmp)


def test_save_restore():

    tmp = create_tmp_database(rider)
    database = db.database(tmp)

    nr_patients = len(database.patients())
    nr_studies = len(database.studies())

    # copy the first series into a new patient
    # restore and check that the state is restored
    patient = database.new_child()
    database.studies()[0].copy_to(patient)
    assert len(database.patients()) == nr_patients + 1
    assert len(database.studies()) == nr_studies + 1
    assert len(patient.studies()) == 1
    database.restore()
    assert len(database.patients()) == nr_patients
    assert len(database.studies()) == nr_studies
    
    # Do the same thing again but this time save
    patient = database.new_child()
    database.studies()[0].copy_to(patient)
    assert len(database.patients()) == nr_patients + 1
    assert len(database.studies()) == nr_studies + 1
    assert len(patient.studies()) == 1
    database.save()
    assert len(database.patients()) == nr_patients + 1
    assert len(database.studies()) == nr_studies + 1
    assert len(patient.studies()) == 1

    # Close and open again, and check the state is the same
    database.close()
    database = db.database(tmp)
    assert len(database.patients()) == nr_patients + 1
    assert len(database.studies()) == nr_studies + 1

    remove_tmp_database(tmp)


def test_read_write_dataset():

    tmp = create_tmp_database(rider)
    database = db.database(tmp)
    instances = database.instances()

    instance = instances[0]
    rows = instance.Rows
    instance.read() 
    assert rows == instance.Rows
    instance.save()
    assert rows == instance.Rows
    instance.clear()
    assert rows == instance.Rows
    instance2 = instances[1]
    rows2 = instance2.Rows
    instance2.read()
    instance2.Rows = 0
    assert instance2.Rows == 0
    instance2.restore()
    assert instance2.Rows == rows2
    instance.Rows = 1
    instance2.Rows = 0
    instance.save()
    instance2.restore()
    assert instance.Rows == 1
    assert instance2.Rows == rows2
    instance3 = instance.parent().new_instance()
    assert instance3.Rows is None
    database.save()
    instance3.Rows = 256 
    assert instance3.Rows == 256
    database.restore()
    assert instance3.Rows is None

    remove_tmp_database(tmp)


def test_read_write_image():

    tmp = create_tmp_database(onefile)
    database = db.database(tmp)
    image = database.instances()[0]
    
    array = image.get_pixel_array()
    image.set_pixel_array(-array)
    array_invert = image.get_pixel_array()

    norm_diff = np.abs(np.sum(np.square(array)) - np.sum(np.square(array_invert)))
    norm_mean = np.abs(np.sum(np.square(array)) + np.sum(np.square(array_invert)))
    max_diff = np.amax(np.abs(array + array_invert))

    assert norm_diff/norm_mean < 0.0001
    assert max_diff < 0.05

    plot = False

    if plot==True:

        col = 3
        fig = plt.figure(figsize=(16,16))
        i=0
        fig.add_subplot(1,col,i+1)
        plt.imshow(array)
        plt.colorbar()
        i=1
        fig.add_subplot(1,col,i+1)
        plt.imshow(array_invert)
        plt.colorbar()
        i=2
        fig.add_subplot(1,col,i+1)
        plt.imshow(array+array_invert)
        plt.colorbar()
        plt.show()

    remove_tmp_database(tmp)

def test_read_write_series():

    tmp = create_tmp_database(rider)
    database = db.database(tmp)
    all_series = database.series()
    series = all_series[0]

    # Read array, modify and save in same series
    array, headers = series.get_pixel_array('SliceLocation')
    series.set_pixel_array(-array, headers)
    array_invert, _ = series.get_pixel_array('SliceLocation')

    norm_diff = np.abs(np.sum(np.square(array)) - np.sum(np.square(array_invert)))
    norm_mean = np.abs(np.sum(np.square(array)) + np.sum(np.square(array_invert)))
    max_diff = np.amax(np.abs(array + array_invert))

    assert norm_diff/norm_mean < 0.0001
    assert max_diff < 0.05

    # Set array in new series

    inverse = series.new_sibling()
    inverse.set_pixel_array(-array, headers)
    array_invert, _ = inverse.get_pixel_array('SliceLocation')

    norm_diff = np.abs(np.sum(np.square(array)) - np.sum(np.square(array_invert)))
    norm_mean = np.abs(np.sum(np.square(array)) + np.sum(np.square(array_invert)))
    max_diff = np.amax(np.abs(array + array_invert))

    assert norm_diff/norm_mean < 0.0001
    assert max_diff < 0.05

    plot = False

    if plot==True:

        col = 3
        fig = plt.figure(figsize=(16,16))
        i=0
        fig.add_subplot(1,col,i+1)
        plt.imshow(array[1,0,:,:])
        plt.colorbar()
        i=1
        fig.add_subplot(1,col,i+1)
        plt.imshow(array_invert[1,0,:,:])
        plt.colorbar()
        i=2
        fig.add_subplot(1,col,i+1)
        plt.imshow(array[1,0,:,:] + array_invert[1,0,:,:])
        plt.colorbar()
        plt.show()

    remove_tmp_database(tmp)

def test_instance_map_to():

    return # needs some work

    tmp = create_tmp_database(rider)
    database = db.database(tmp)
    images = database.instances()
    try:
        map = images[0].map_to(images[1])
    except:
        assert False

    remove_tmp_database(tmp)

def test_instance_map_mask_to():

    tmp = create_tmp_database(rider)
    database = db.database(tmp)
    images = database.instances()
    try:
        map = images[0].map_mask_to(images[1])
    except:
        assert False
    remove_tmp_database(tmp)


def test_series_map_mask_to():

    tmp = create_tmp_database(rider)
    database = db.database(tmp)
    series = database.series()
    try:
        map = series[0].map_mask_to(series[1])
    except:
        assert False
    remove_tmp_database(tmp)

def test_set_colormap():

    tmp = create_tmp_database(rider)
    database = db.database(tmp)
    images = database.instances()
    images[0].colormap = 'viridis'
    assert images[0].colormap == 'viridis'
    remove_tmp_database(tmp)

def test_instance_export_as_csv():

    tmp = create_tmp_database(rider)
    database = db.database(tmp)
    export = create_tmp_database(path=None, name='export')
    images = database.instances()
    try:
        images[0].export_as_csv(export)
    except:
        assert False
    remove_tmp_database(tmp)
    remove_tmp_database(export)

def test_instance_export_as_png():

    tmp = create_tmp_database(rider)
    database = db.database(tmp)
    export = create_tmp_database(path=None, name='export')
    images = database.instances()
    try:
        images[0].export_as_png(export)
    except:
        assert False
    remove_tmp_database(tmp)
    remove_tmp_database(export)

def test_instance_export_as_nifti():

    tmp = create_tmp_database(rider)
    database = db.database(tmp)
    export = create_tmp_database(path=None, name='export')
    images = database.instances()
    try:
        images[0].export_as_nifti(export)
    except:
        assert False
    remove_tmp_database(tmp)
    remove_tmp_database(export)

def test_series_export_as_csv():

    tmp = create_tmp_database(rider)
    database = db.database(tmp)
    export = create_tmp_database(path=None, name='export')
    series = database.series()
    try:
        series[0].export_as_csv(export)
    except:
        assert False
    remove_tmp_database(tmp)
    remove_tmp_database(export)

def test_series_export_as_png():

    tmp = create_tmp_database(rider)
    database = db.database(tmp)
    export = create_tmp_database(path=None, name='export')
    series = database.series()
    try:
        series[0].export_as_png(export)
    except:
        assert False
    remove_tmp_database(tmp)
    remove_tmp_database(export)

def test_series_export_as_nifti():

    tmp = create_tmp_database(rider)
    database = db.database(tmp)
    export = create_tmp_database(path=None, name='export')
    series = database.series()
    try:
        series[0].export_as_nifti(export)
    except:
        assert False
    remove_tmp_database(tmp)
    remove_tmp_database(export)

def test_series_export_as_npy():

    tmp = create_tmp_database(rider)
    database = db.database(tmp)
    export = create_tmp_database(path=None, name='export')
    series = database.series()
    try:
        series[0].export_as_npy(export)
    except:
        assert False
    remove_tmp_database(tmp)
    remove_tmp_database(export)


def test_subseries():

    tmp = create_tmp_database(rider)
    database = db.database(tmp)
    series = database.series()
    for i in series[0].instances():
        assert i.image_type == 'MAGNITUDE'
    magn = series[0].subseries(image_type='MAGNITUDE')
    for i in magn.instances():
        assert i.image_type == 'MAGNITUDE'

    remove_tmp_database(tmp)

def test_export_as_dicom():

    tmp = create_tmp_database(rider)
    database = db.database(tmp)
    export = create_tmp_database(path=None, name='export')
    series = database.series()
    try:
        series[0].export_as_dicom(export)
        database.export_as_dicom(export)
    except:
        assert False
    remove_tmp_database(tmp)
    remove_tmp_database(export)

def test_import_dicom():

    tmp = create_tmp_database(rider)
    database = db.database(tmp)
    export = create_tmp_database(path=None, name='export')
    series = database.series()
    series[0].export_as_dicom(export)
    dbexport = db.database(export)
    try:
        series[0].import_dicom(dbexport.files())
        database.import_dicom(dbexport.files())
    except:
        assert False
    remove_tmp_database(tmp)
    remove_tmp_database(export)

def test_series():

    path = create_tmp_database(path=None, name='export')
    # array = np.random.normal(size=(10, 128, 192))
    array = np.zeros((10, 128, 192))
    series = db.series(array)
    series.PatientName = 'Random noise'
    series.StudyDate = '19112022'
    series.AcquisitionTime = '120000'
    series.save(path)

def test_custom_attributes():

    tmp = create_tmp_database(rider)
    database = db.database(tmp)
    series = database.series()
    for i in series[0].instances():
        assert i.image_type == 'MAGNITUDE'
        assert i.colormap is None
        assert i.lut is None
    for i in series[0].instances():
        i.image_type = 'PHASE'
    for i in series[0].instances():
        assert i.image_type == 'PHASE'

    remove_tmp_database(tmp)


if __name__ == "__main__":

    test_read()
    test_database()
    test_children()
    test_read_dicom_data_elements()
    test_read_dicom_data_elements_from_memory()
    test_hierarchy()
    test_hierarchy_in_memory_v1()
    test_hierarchy_in_memory_v2()
    test_find_by_value()
    test_find_by_value_in_memory()
    test_read_item_instance()
    test_read_item()
    test_set_attr_instance()
    test_set_attr_instance_in_memory_v1()
    test_set_attr_instance_in_memory_v2()
    test_set_item_instance()
    test_set_item_instance_in_memory()
    test_set_item()
    test_set_item_in_memory()
    test_create_records()
    test_copy_remove_instance()
    test_copy_remove()
    test_inherit_attributes()
    test_merge()
    test_merge_empty()
    test_save_restore()
    test_read_write_dataset()
    test_read_write_image()
    test_read_write_series()
    test_instance_map_to()
    test_instance_map_mask_to()
    test_series_map_mask_to()
    test_set_colormap()
    test_instance_export_as_csv()
    test_instance_export_as_png()
    test_instance_export_as_nifti()
    test_series_export_as_csv()
    test_series_export_as_png()
    test_series_export_as_nifti()
    test_series_export_as_npy()
    test_subseries()
    test_export_as_dicom()
    test_import_dicom()
    test_series()
    test_custom_attributes()


    print('------------------------')
    print('record passed all tests!')
    print('------------------------')