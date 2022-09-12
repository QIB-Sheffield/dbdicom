import os
import shutil

import dbdicom.record as db


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



def test_database():

    tmp = create_tmp_database(rider)

    dbr = db.open(tmp)
    assert 24 == len(dbr.register.instances('Database'))

    remove_tmp_database(tmp)

# Test functions

def test_opendatabase_and_list_contents(path=datapath):

    database = db.open(path)
    patient = database.children(1)
    assert patient.label() == '281949 [RIDER Neuro MRI-5244517593]'
    study = patient.children(2)
    assert study.label() == 'BRAIN^RESEARCH [19041007]'
    series = study.children(1)
    assert series.label() == '[009] ax 10 flip'
    assert len(series.children()) == 2
    database.close()

def test_read_dicom_data_elements():

    database = db.open(datapath)
    patient = database.children(1)
    assert patient.PatientID == ['RIDER Neuro MRI-5244517593']
    study = patient.children(2)
    assert study.StudyDescription == ['BRAIN^RESEARCH']
    series = study.children(1)
    assert series.SeriesDescription == ['ax 10 flip']
    instance = series.children(1)
    assert instance.AcquisitionTime == '074634.479981'
    database.close()

def test_read_dicom_data_elements_from_memory_v1(): 

    database = db.open(datapath)
    patient = database.children(1).read()
    assert patient.PatientID == ['RIDER Neuro MRI-5244517593']
    study = patient.children(2).read()
    assert study.StudyDescription == ['BRAIN^RESEARCH']
    series = study.children(1).read()
    assert series.SeriesDescription == ['ax 10 flip']
    instance = series.children(1).read()
    assert instance.AcquisitionTime == '074634.479981'
    database.close()

def test_read_dicom_data_elements_from_memory_v2(): 

    database = db.open(datapath).read()
    patient = database.children(1)
    assert patient.PatientID == ['RIDER Neuro MRI-5244517593']
    study = patient.children(2)
    assert study.StudyDescription == ['BRAIN^RESEARCH']
    series = study.children(1)
    assert series.SeriesDescription == ['ax 10 flip']
    instance = series.children(1)
    assert instance.AcquisitionTime == '074634.479981'

def test_hierarchy():

    database = db.open(datapath)

    patients = database.patients()
#    assert patients[0].database().path == database.path 
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

    assert patients[0].instances(0).patients(0).PatientID == patients[0].PatientID
    assert patients[0].instances(-1).patients(0).PatientID == patients[0].PatientID
    assert studies[1].instances(0).studies(0).StudyDescription == studies[1].StudyDescription
    assert studies[1].instances(-1).studies(0).StudyDescription == studies[1].StudyDescription
    
def test_hierarchy_in_memory_v1():

    database = db.open(datapath).read()

    patients = database.patients()
#    assert patients[0].database().path == database.path 
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

    assert patients[0].instances(0).patients(0).PatientID == patients[0].PatientID
    assert patients[0].instances(-1).patients(0).PatientID == patients[0].PatientID
    assert studies[1].instances(0).studies(0).StudyDescription == studies[1].StudyDescription
    assert studies[1].instances(-1).studies(0).StudyDescription == studies[1].StudyDescription

def test_hierarchy_in_memory_v2():

    database = db.open(datapath)

    patients = database.patients()
    nr_series = 0
    nr_instances = 0
    for patient in patients:
        p = patient.read()
        nr_series += len(p.series())
        nr_instances += len(p.instances())
    assert nr_instances == 24

    studies = database.studies()
    nr_series = 0
    nr_instances = 0
    for study in studies:
        s = study.read()
        nr_series += len(s.series())
        nr_instances += len(s.instances())
    assert nr_instances == 24

    p0 = patients[0].read()
    assert p0.instances(0).patients(0).PatientID == p0.PatientID
    assert p0.instances(-1).patients(0).PatientID == p0.PatientID
    s1 = studies[1].read()
    assert s1.instances(0).studies(0).StudyDescription == s1.StudyDescription
    assert s1.instances(-1).studies(0).StudyDescription == s1.StudyDescription

def test_find_by_value():

    database = db.open(datapath)
    series = database.series(
        SeriesDescription = ['ax 20 flip'], 
        PatientID = ['RIDER Neuro MRI-5244517593'])
    assert len(series) == 0
    series = database.series(
        SeriesDescription = ['ax 10 flip'], 
        PatientID = ['RIDER Neuro MRI-5244517593'])
    assert len(series) == 2
    series = database.series(
        StudyDescription = ['BRAIN^ROUTINE BRAIN'], 
        SeriesDescription = ['ax 10 flip'], 
        PatientID = ['RIDER Neuro MRI-5244517593'])
    assert len(series) == 1

def test_find_by_value_in_memory():

    database = db.open(datapath).read()
    series = database.series(
        SeriesDescription = ['ax 20 flip'], 
        PatientID = ['RIDER Neuro MRI-5244517593'])
    assert len(series) == 0
    series = database.series(
        SeriesDescription = ['ax 10 flip'], 
        PatientID = ['RIDER Neuro MRI-5244517593'])
    assert len(series) == 2
    series = database.series(
        StudyDescription = ['BRAIN^ROUTINE BRAIN'], 
        SeriesDescription = ['ax 10 flip'], 
        PatientID = ['RIDER Neuro MRI-5244517593'])
    assert len(series) == 1

# Reading DICOM attributes


def test_read_item_instance():

    database = db.open(datapath)
    tags = [
        'SeriesDescription', 
        (0x0010, 0x0020), 
        (0x0010, 0x0020), 
        'PatientID', 
        (0x0011, 0x0020)]
    instance = database.instances(0)
    assert instance[tags] == [
        'sag 3d gre +c', 
        'RIDER Neuro MRI-3369019796', 
        'RIDER Neuro MRI-3369019796', 
        'RIDER Neuro MRI-3369019796', 
        None]

def test_read_item():

    database = db.open(datapath)
    tags = [
        'SeriesDescription', 
        (0x0010, 0x0020), 
        (0x0010, 0x0020), 
        'PatientID', 
        (0x0011, 0x0020)]
    series = database.series(0)
    assert series[tags] == [
        ['sag 3d gre +c'], 
        ['RIDER Neuro MRI-3369019796'], 
        ['RIDER Neuro MRI-3369019796'], 
        ['RIDER Neuro MRI-3369019796'], 
        [None]]
    patient = database.patients(0)
    assert set(patient[tags][0]) == set(['sag 3d gre +c', 'ax 5 flip', 'ax 10 flip'])

def test_export():

    tmp_path = create_tmp_database()
    test_opendatabase_and_list_contents(tmp_path)
    remove_tmp_database(tmp_path)

def test_set_attr_instance():

    # What if the attribute is one of the UIDs?
    # What if it is one of the attributes?

    tmp_path = create_tmp_database()

    database = db.open(tmp_path)
    instance = database.instances(0)

    orig_slice_loc = instance.SliceLocation
    instance.SliceLocation = orig_slice_loc + 100
    new_slice_loc = instance.SliceLocation
    assert 100 == np.round(new_slice_loc-orig_slice_loc)

    orig_acq_time = instance.AcquisitionTime
    instance.AcquisitionTime = str(float(orig_acq_time) + 3.0) 
    new_acq_time = instance.AcquisitionTime
    assert 3 == np.round(float(new_acq_time)-float(orig_acq_time))

    remove_tmp_database(tmp_path)


def test_set_attr_instance_in_memory_v1():

    tmp_path = create_tmp_database()

    database = db.open(tmp_path)
    instance = database.instances(0).read()

    orig_slice_loc = instance.SliceLocation
    instance.SliceLocation = orig_slice_loc + 100
    new_slice_loc = instance.SliceLocation
    assert 100 == np.round(new_slice_loc-orig_slice_loc)

    orig_acq_time = instance.AcquisitionTime
    instance.AcquisitionTime = str(float(orig_acq_time) + 3.0) 
    new_acq_time = instance.AcquisitionTime
    assert 3 == np.round(float(new_acq_time)-float(orig_acq_time))

    remove_tmp_database(tmp_path)


def test_set_attr_instance_in_memory_v2():

    tmp_path = create_tmp_database()

    database = db.open(tmp_path).read()
    instance = database.instances(0)

    orig_slice_loc = instance.SliceLocation
    instance.SliceLocation = orig_slice_loc + 100
    new_slice_loc = instance.SliceLocation
    assert 100 == np.round(new_slice_loc-orig_slice_loc)

    orig_acq_time = instance.AcquisitionTime
    instance.AcquisitionTime = str(float(orig_acq_time) + 3.0) 
    new_acq_time = instance.AcquisitionTime
    assert 3 == np.round(float(new_acq_time)-float(orig_acq_time))

    remove_tmp_database(tmp_path)


def test_set_item_instance():

    tmp_path = create_tmp_database()
    database = db.open(tmp_path)
    instance = database.instances(0)

    tags = ['SliceLocation', 'AcquisitionTime', (0x0012, 0x0010)]
    new_values = [0.0, '000000.00', 'University of Sheffield']

    instance[tags] = new_values
    assert instance[tags] == new_values

    remove_tmp_database(tmp_path)


def test_set_item_instance_in_memory():

    tmp_path = create_tmp_database()
    database = db.open(tmp_path)
    
    tags = ['SliceLocation', 'AcquisitionTime', (0x0012, 0x0010)]
    current_values = [75.561665058136, '075649.057496', None]
    new_values = [0.0, '000000.00', 'University of Sheffield']

    record = database.instances(0)
    data = record.read()
    
    data[tags] = new_values
    assert data[tags] == new_values

    # check that data on disk have not changed
    assert record[tags] == current_values

    # write out data in memory and check that data on disk now have changed
    record.write(data)
    assert record[tags] == new_values

    remove_tmp_database(tmp_path)


def test_set_item():

    tmp_path = create_tmp_database()
    database = db.open(tmp_path)
    series = database.series(0)
    instance = series.instances(1)

    tags = ['SliceLocation', 'AcquisitionTime', (0x0012, 0x0010)]
    values = series[tags]
    assert set(values[0]) == set([74.561665058136, 75.561665058136])
    assert values[1:] == [['075649.057496'], [None]]

    values = instance[tags]
    assert values == [74.561665058136, '075649.057496', None]

    new_values = [0.0, '00:00:00', 'University of Sheffield']
    series[tags] = new_values
    series_values = series[tags]
    assert series_values[0][0] == new_values[0]
    assert series_values[1][0] == new_values[1]
    assert series_values[2][0] == new_values[2]
    
    instance_values = instance[tags]
    assert instance_values[0] == series_values[0][0]
    assert instance_values[1] == series_values[1][0]
    assert instance_values[2] == series_values[2][0]

    remove_tmp_database(tmp_path)

def test_set_item_in_memory():

    tmp_path = create_tmp_database()
    database = db.open(tmp_path)
    series_record = database.series(0)
    instance_record = series_record.instances(1)
    series = series_record.read()
    instance = series.instances(1)
    tags = ['SliceLocation', 'AcquisitionTime', (0x0012, 0x0010)]

    values = series[tags]
    assert set(values[0]) == set([74.561665058136, 75.561665058136])
    assert values[1:] == [['075649.057496'], [None]]

    values = instance[tags]
    assert values == [74.561665058136, '075649.057496', None]

    new_values = [0.0, '00:00:00', 'University of Sheffield']
    series[tags] = new_values
    series_values = series[tags]
    assert series_values[0][0] == new_values[0]
    assert series_values[1][0] == new_values[1]
    assert series_values[2][0] == new_values[2]
    
    instance_values = instance[tags]
    assert instance_values[0] == new_values[0]
    assert instance_values[1] == new_values[1]
    assert instance_values[2] == new_values[2]

    # check that data on record have not been changed

    values = series_record[tags]
    assert set(values[0]) == set([74.561665058136, 75.561665058136])
    assert values[1:] == [['075649.057496'], [None]]

    values = instance_record[tags]
    assert values == [74.561665058136, '075649.057496', None]

    remove_tmp_database(tmp_path)

def test_create_records():

    tmp_path = create_tmp_database()
    database = db.open(tmp_path)  
    patient = database.new_child()
    study = patient.new_child()
    series = study.new_child()
    instance = series.new_child()

    # considering specifying a _ds attribute in DbRecord as well
    # so that you can travers the hierarchy even if no data are saved

    # assert instance.patients(0).UID == patient.UID 
    # assert instance.studies(0).UID == study.UID 
    # assert series.children(0).UID == instance.UID 

    assert instance.UID[0] == patient.UID[0] 
    assert instance.UID[:2] == study.UID
    assert instance.UID[:3] == series.UID

    database = database.read()
    patient = database.new_child()
    study = patient.new_child()
    series = study.new_child()
    instance = series.new_child()

    assert instance.UID[0] == patient.UID[0] 
    assert instance.UID[:2] == study.UID
    assert instance.UID[:3] == series.UID

    assert instance.patients(0).UID == patient.UID 
    assert instance.studies(0).UID == study.UID 
    assert series.children(0).UID == instance.UID 

    remove_tmp_database(tmp_path)

def test_copy_remove_instance():

    tmp_path = create_tmp_database()
    database = db.open(tmp_path)
    
    #
    # move and copy an instance from one series to another
    #

    parent = database.patients(1).studies(1).series(0) # [016] sag 3d gre +c (2)
    instance = parent.instances(0)
    new_parent = database.patients(0).studies(1).series(0) # [006] ax 5 flip (2)

    # copy the instance to the new parent and remove the original
    copy = instance.copy_to(new_parent)

    assert len(parent.instances()) == 2
    assert len(new_parent.instances()) == 3
    assert set(parent.SeriesDescription) == set(['sag 3d gre +c'])
    assert set(new_parent.SeriesDescription) == set(['ax 5 flip', 'sag 3d gre +c'])
    assert instance.SOPInstanceUID in set(parent.SOPInstanceUID)
    assert copy.SOPInstanceUID in set(new_parent.SOPInstanceUID)

    # remove the original instance
    uid = instance.SOPInstanceUID
    instance.remove()

    assert len(parent.instances()) == 1
    assert set(parent.SeriesDescription) == set(['sag 3d gre +c'])
    assert uid not in set(parent.SOPInstanceUID)

    # move the copy back to the original parent
    # this should restore the situation
    copy = copy.move_to(parent)

    assert len(parent.instances()) == 2
    assert len(new_parent.instances()) == 2
    assert set(parent.SeriesDescription) == set(['sag 3d gre +c'])
    assert set(new_parent.SeriesDescription) == set(['ax 5 flip'])
    assert copy.SOPInstanceUID not in set(new_parent.SOPInstanceUID)
    assert copy.SOPInstanceUID in set(parent.SOPInstanceUID)

    #
    # move and copy an instance from one study to another
    #

    parent = database.patients(1).studies(1) # BRAIN^ROUTINE BRAIN (2)
    instance = parent.instances(0)
    new_parent = database.patients(0).studies(1) # BRAIN^RESEARCH (4)

    # copy the instance to the new parent and remove the original
    copy = instance.copy_to(new_parent)

    assert len(parent.instances()) == 2
    assert len(new_parent.instances()) == 5
    assert set(parent.StudyDescription) == set(['BRAIN^ROUTINE BRAIN'])
    assert set(new_parent.StudyDescription) == set(['BRAIN^ROUTINE BRAIN', 'BRAIN^RESEARCH'])
    assert instance.SOPInstanceUID in set(parent.SOPInstanceUID)
    assert copy.SOPInstanceUID in set(new_parent.SOPInstanceUID)

    # remove the original instance
    uid = instance.SOPInstanceUID
    instance.remove()

    assert len(parent.instances()) == 1
    assert set(parent.StudyDescription) == set(['BRAIN^ROUTINE BRAIN'])
    assert uid not in set(parent.SOPInstanceUID)

    # move the copy back to the original parent
    # this should restore the situation
    copy = copy.move_to(parent)

    assert len(parent.instances()) == 2
    assert len(new_parent.instances()) == 4
    assert set(parent.StudyDescription) == set(['BRAIN^ROUTINE BRAIN'])
    assert set(new_parent.StudyDescription) == set(['BRAIN^RESEARCH'])
    assert copy.SOPInstanceUID not in set(new_parent.SOPInstanceUID)
    assert copy.SOPInstanceUID in set(parent.SOPInstanceUID)

    remove_tmp_database(tmp_path)

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

    test_opendatabase_and_list_contents()
    test_read_dicom_data_elements()
    test_read_dicom_data_elements_from_memory_v1()
    test_read_dicom_data_elements_from_memory_v2()
    test_hierarchy()
    test_hierarchy_in_memory_v1()
    test_hierarchy_in_memory_v2()
    test_find_by_value()
    test_find_by_value_in_memory()
    test_read_item_instance()
    test_read_item()
    test_export()
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
    test_merge()
    test_save_restore()
    test_read_write_dataset()

    # test_read_write_image()


    print('------------------------')
    print('record passed all tests!')
    print('------------------------')