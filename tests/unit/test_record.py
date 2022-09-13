import os
import shutil
import numpy as np

import dbdicom.record as record


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


# Test functions

def test_database():

    tmp = create_tmp_database(rider)
    database = record.open(tmp)

    try:
        database.print()
    except:
        assert False
    assert 24 == len(database.instances())

    database.close()
    remove_tmp_database(tmp)


def test_children():

    tmp = create_tmp_database(rider)
    database = record.open(tmp)

    patients = database.children(PatientID='RIDER Neuro MRI-3369019796')
    assert patients[0].label() == 'Patient 281949 [RIDER Neuro MRI-3369019796]'
    studies = patients[0].children()
    assert (len(studies) == 4)

    database.close()
    remove_tmp_database(tmp)


def test_read_dicom_data_elements():

    tmp = create_tmp_database(rider)
    database = record.open(tmp)

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
    database = record.open(tmp)

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
    database = record.open(tmp)

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
    database = record.open(tmp)

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
    database = record.open(tmp)

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
    database = record.open(tmp)

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
    database = record.open(tmp)
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
    database = record.open(tmp)
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
    database = record.open(tmp)
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
    database = record.open(tmp)

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
    database = record.open(tmp)

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
    database = record.open(tmp)

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
    database = record.open(tmp)

    instance = database.instances()[0]

    tags = ['SliceLocation', 'AcquisitionTime', (0x0012, 0x0010)]
    new_values = [0.0, '000000.00', 'University of Sheffield']

    assert instance[tags] != new_values
    instance[tags] = new_values
    assert instance[tags] == new_values

    remove_tmp_database(tmp)


def test_set_item_instance_in_memory():

    tmp = create_tmp_database(rider)
    database = record.open(tmp)

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
    database = record.open(tmp)

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
    database = record.open(tmp)

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
    database = record.open(tmp)

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
    database = record.open(tmp)
    
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
    instance.delete()

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

    instance.delete() 

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

    tmp_path = create_tmp_database()
    database = db.open(tmp_path)

    # Create a new study and copy the first series to it.
    new_study = database.patients(0).new_child()
    study = database.patients(1).studies(0) # BRAIN^ROUTINE BRAIN (2)
    series = study.series(0)
    copy = series.copy_to(new_study)

    assert len(study.series()) == 2
    assert len(new_study.series()) == 1
    assert set(study.SeriesDescription) == set(['ax 5 flip', 'ax 10 flip'])
    assert set(new_study.SeriesDescription) == set(['ax 5 flip'])
    assert set(series.SeriesInstanceUID) <= set(study.SeriesInstanceUID)
    assert set(copy.SeriesInstanceUID) <= set(new_study.SeriesInstanceUID)
    
    remove_tmp_database(tmp_path)

def test_merge():

    tmp_path = create_tmp_database()
    database = db.open(tmp_path)

    # first create two new patients
    # and copy series into them

    patient1 = database.new_child()
    patient2 = database.new_child()
    for s in database.series(SeriesDescription = ['sag 3d gre +c']):
        s.copy_to(patient1)
    for s in database.series(SeriesDescription = ['ax 5 flip']):
        s.copy_to(patient2) 

    assert len(patient1.studies()) == 4
    assert len(patient2.studies()) == 4
    assert len(patient1.series()) == 4
    assert len(patient2.series()) == 4
    assert len(database.patients()) == 4

    # merge the two patients into a third

    patients_to_merge = [patient1, patient2]
    patient3 = database.new_child()
    for patient in patients_to_merge:
        for study in patient.studies():
            study.copy_to(patient3)

    assert len(patient3.studies()) == 8
    assert len(patient3.series()) == 8
    assert set(patient1.SeriesDescription) == set(['sag 3d gre +c'])
    assert set(patient2.SeriesDescription) == set(['ax 5 flip'])
    assert set(patient3.SeriesDescription) == set(['sag 3d gre +c', 'ax 5 flip'])

    # now merge all studies of the new patient 
    # into a new study of the same patient.
    studies_to_merge = patient3.studies()
    new_study = patient3.new_child()
    for study in studies_to_merge:
        for series in study.series():
            series.copy_to(new_study)
    
    assert len(patient3.studies()) == 9
    assert len(patient3.series()) == 16
    assert len(new_study.series()) == 8
    assert len(patient3.instances()) == 32

    # now merge all series of the new patient into
    # a new series in a new study of the same patient
    series_to_merge = patient3.series()
    new_study = patient3.new_child()
    new_series = new_study.new_child()
    for series in series_to_merge:
        for instance in series.instances():
            instance.copy_to(new_series)

    assert len(patient3.studies()) == 10
    assert len(patient3.series()) == 17
    assert len(patient3.instances()) == 64
    assert len(new_study.series()) == 1

    remove_tmp_database(tmp_path)

def test_save_restore():

    tmp_path = create_tmp_database()
    database = db.open(tmp_path)

    nr_patients = len(database.patients())
    nr_series = len(database.series())

    # copy the first series into a new patient
    # restore and check that the state is restored
    patient = database.new_child()
    database.series(0).copy_to(patient)
    assert len(database.patients()) == nr_patients + 1
    assert len(database.series()) == nr_series + 1
    assert len(patient.series()) == 1
    database.restore()
    assert len(database.patients()) == nr_patients
    assert len(database.series()) == nr_series
    
    # Do the same thing again but this time save
    patient = database.new_child()
    database.series(0).copy_to(patient)
    assert len(database.patients()) == nr_patients + 1
    assert len(database.series()) == nr_series + 1
    assert len(patient.series()) == 1
    database.save()
    assert len(database.patients()) == nr_patients + 1
    assert len(database.series()) == nr_series + 1
    assert len(patient.series()) == 1

    # Close and open again, and check the state is the same
    database.close()
    database = db.open(tmp_path)
    assert len(database.patients()) == nr_patients + 1
    assert len(database.series()) == nr_series + 1

    remove_tmp_database(tmp_path)

def test_read_write_dataset():

    tmp_path = create_tmp_database()
    database = db.open(tmp_path)

    instance = database.instances(0)
    dataset = instance.read() 
    
    assert dataset.Rows == instance.Rows
    assert dataset.Columns == instance.Columns

    dataset.Rows = dataset.Rows * 2
    dataset.Columns = dataset.Columns * 2
    assert dataset.Rows == instance.Rows * 2
    assert dataset.Columns == instance.Columns * 2

    matrix = ['Rows','Columns']
    d = dataset[matrix]
    dataset[matrix] = [int(d[0]*2), int(d[1]*2)]
    assert dataset.Rows == instance.Rows * 4
    assert dataset.Columns == instance.Columns * 4

    instance.write(dataset)

    assert instance.Rows == dataset.Rows
    assert instance.Columns == dataset.Columns
    
    remove_tmp_database(tmp_path)

def test_read_write_image():

    tmp_path = create_tmp_database()
    database = db.open(tmp_path)
    instance = database.series(0).instances(0)
    array = instance.array()
    instance.set_array(-array)
    array_invert = instance.array()

    print('These should be equal: ', np.sum(np.square(array)), np.sum(np.square(array_invert)))
    print('This should be zero: ', np.sum(np.square(array + array_invert)))

    remove_tmp_database(tmp_path)


if __name__ == "__main__":

    # test_database()
    # test_children()
    # test_read_dicom_data_elements()
    # test_read_dicom_data_elements_from_memory()
    # test_hierarchy()
    # test_hierarchy_in_memory_v1()
    # test_hierarchy_in_memory_v2()
    # test_find_by_value()
    # test_find_by_value_in_memory()
    # test_read_item_instance()
    # test_read_item()
    # test_set_attr_instance()
    # test_set_attr_instance_in_memory_v1()
    # test_set_attr_instance_in_memory_v2()
    # test_set_item_instance()
    # test_set_item_instance_in_memory()
    # test_set_item()
    # test_set_item_in_memory()
    # test_create_records()
    test_copy_remove_instance()
    # test_copy_remove()
    # test_merge()
    # test_save_restore()
    # test_read_write_dataset()

    # test_read_write_image()


    print('------------------------')
    print('record passed all tests!')
    print('------------------------')