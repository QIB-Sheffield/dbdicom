import os
import shutil
import numpy as np

import dbdicom as db

datapath = os.path.join(os.path.dirname(__file__), 'fixtures')


# Helper functions

def create_tmp_database():

    tmp_path = os.path.join(os.path.dirname(__file__), 'tmp')
    
    if os.path.isdir(tmp_path):
        shutil.rmtree(tmp_path)
    os.mkdir(tmp_path)
    database = db.open(datapath)
    database.export(tmp_path)
    database.close()

    return tmp_path

def remove_tmp_database(path):

    shutil.rmtree(path)

# Test functions

def test_opendatabase_and_list_contents(path=datapath):

    database = db.open(path)
    for i, patient in enumerate(database.children()):
        if i==0:
            assert patient.label() == '281949 [RIDER Neuro MRI-3369019796]'
        for j, study in enumerate(patient.children()):
            if j==0:
                assert study.label() == 'BRAIN^RESEARCH [19040323]'
            for k, series in enumerate(study.children()):
                if k==0:
                    assert series.label() == '[006] ax 5 flip'
                    assert len(series.children()) == 16
    database.close()

def test_read_dicom_data_elements():

    database = db.open(datapath)
    for series in database.series():
        assert series.SeriesDescription == ['ax 5 flip']
    rows = 0
    for instance in database.series(0).instances():
        rows += instance.Rows
    assert rows == 4096
    for patient in database.patients():
        assert patient.PatientName == ['281949']
    assert database.patients(0).PatientID == ['RIDER Neuro MRI-3369019796']
    assert database.patients(0).series(0).SeriesDescription == ['ax 5 flip']
    assert database.patients(0).studies(0).series(0).SeriesDescription == ['ax 5 flip']

    # need more structure in the test database - more series, studies etc

def test_read_dicom_data_elements_from_memory(): # update __setitem__ for DataSet

    database = db.open(datapath)
    for series in database.series():
        assert series.read().SeriesDescription == 'ax 5 flip'
    rows = 0
    for instance in database.series(0).instances():
        rows += instance.read().Rows
    assert rows == 4096
    for patient in database.patients():
        assert patient.read().PatientName == '281949'

def test_find_by_value():

    database = db.open(datapath)
    series = database.patients(0).series(SeriesDescription="ax 15 flip", PatientID='RIDER Neuro MRI-5244517593')
    assert series == [] 
    series = database.patients(0).series(SeriesDescription="ax 5 flip", PatientID='RIDER Neuro MRI-3369019796')
    for s in series: 
        assert s.SeriesDescription == "ax 5 flip"

def test_hierarchy():

    database = db.open(datapath)
    series = database.patients(0).series(SeriesDescription="ax 5 flip", PatientID='RIDER Neuro MRI-3369019796')
    series2 = series[0].parent.children()
    assert series[0].SeriesDescription == series2[0].SeriesDescription


# Reading DICOM attributes


def test_read_item_instance():

    database = db.open(datapath)
    tags = ['SeriesDescription', (0x0010, 0x0020), (0x0010, 0x0020), 'PatientID', (0x0011, 0x0020)]
    instance = database.instances(0)
    assert instance[tags] == ['ax 5 flip', 'RIDER Neuro MRI-3369019796', 'RIDER Neuro MRI-3369019796', 'RIDER Neuro MRI-3369019796', None]

def test_read_item():

    database = db.open(datapath)
    tags = ['SeriesDescription', (0x0010, 0x0020), (0x0010, 0x0020), 'PatientID', (0x0011, 0x0020)]
    series = database.series(0)
    assert series[tags] == ['ax 5 flip', 'RIDER Neuro MRI-3369019796', 'RIDER Neuro MRI-3369019796', 'RIDER Neuro MRI-3369019796', None]
    patient = database.patients(0)
    assert patient[tags] == ['ax 5 flip', 'RIDER Neuro MRI-3369019796', 'RIDER Neuro MRI-3369019796', 'RIDER Neuro MRI-3369019796', None]

def test_export():

    tmp_path = create_tmp_database()
    test_opendatabase_and_list_contents(tmp_path)
    remove_tmp_database(tmp_path)


def test_set_attr_instance():

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


def test_set_attr_instance_in_memory():

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


def test_set_item_instance():

    tmp_path = create_tmp_database()
    database = db.open(tmp_path)
    instance = database.instances(0)

    tags = ['SliceLocation', 'AcquisitionTime', (0x0012, 0x0010)]
    new_values = [0.0, '000000.00', 'University of Sheffield']

    # assign new values and check that they are correctly assigned
    instance[tags] = new_values
    assert instance[tags] == new_values

    remove_tmp_database(tmp_path)


def test_set_item_instance_in_memory():

    tmp_path = create_tmp_database()
    database = db.open(tmp_path)
    
    tags = ['SliceLocation', 'AcquisitionTime', (0x0012, 0x0010)]
    current_values = [36.991525650024, '073903.155010', None]
    new_values = [0.0, '000000.00', 'University of Sheffield']

    record = database.instances(0)
    data = record.read()
    
    # assign new values in memory and check that they are correctly assigned
    data[tags] = new_values
    assert data[tags] == new_values

    # check that data on disk have not changed
    assert record[tags] == current_values

    # write out data in memory and check that data on disk now have changed
    record.write(data)
    assert record[tags] == new_values

    remove_tmp_database(tmp_path)



if __name__ == "__main__":

    test_opendatabase_and_list_contents()
    test_read_dicom_data_elements()
    # test_find_by_value()
    # test_hierarchy()
    # test_read_item_instance()
    # test_read_item()
    # test_read_dicom_data_elements_from_memory()
    # test_export()
    # test_set_attr_instance()
    # test_set_attr_instance_in_memory()
    # test_set_item_instance()
    # test_set_item_instance_in_memory()


    print('-------------------------')
    print('dbdicom passed all tests!')
    print('-------------------------')