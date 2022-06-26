import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


from .folder import Folder

test_folder = 'C:\\Users\\steve\\Dropbox\\Data\\WeaselDevSmallBrain'
test_folder2 = 'C:\\Users\\steve\\Dropbox\\Data\\WeaselDevSmallKid'
test_export_folder = 'C:\\Users\\steve\\Dropbox\\Data\\Tmp'


def test_df():

    columns = ['Bool', 'Name', 'Age', 'Stream', 'Percentage']
    record = {
        'Bool' : [True, True, False, True, False, True] , 
        'Name': ['Ankit', 'Amit', 'Aishwarya', 'Priyanka', 'Priya', 'Shaurya' ],
        'Age': [21, 19, 20, 18, 17, 21],
        'Stream': ['Math', 'Commerce', 'Science', 'Math', 'Math', 'Science'],
        'Percentage': [88, 92, 95, 70, 65, 78] } 
    df = pd.DataFrame(record, 
        columns = columns, 
        index = ['A','A','C',None,'E','F'])
    print(df)

def test_set_array_mip():
    pass

def test_set_slice_array_invert():

    folder = Folder(test_folder2).open() # enhancement: open automatically on init
    instance = folder.series(12).instances(0)
    array = instance.array()
    instance.set_array(-array)
    array_invert = instance.array()

    print('These should be equal: ', np.sum(np.square(array)), np.sum(np.square(array_invert)))
    print('This should be zero: ', np.sum(np.square(array + array_invert)))

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
    folder.restore()

def test_set_array_invert():

    folder = Folder(test_folder2).open() # enhancement: open automatically on init
    series = folder.series(12) # T1-MOLLI multi-TI
    array, _ = series.array()
    invert = series.copy().set_array(-array)
    array_invert, _ = invert.array()
    print('Round-off error (%): ', 100*np.sum(np.square(array + array_invert))/np.sum(np.square(array)))
    col = 3
    fig = plt.figure(figsize=(16,16))
    i=0
    fig.add_subplot(1,col,i+1)
    plt.imshow(array[0,:,:])
    plt.colorbar()
    i=1
    fig.add_subplot(1,col,i+1)
    plt.imshow(array_invert[0,:,:])
    plt.colorbar()
    i=2
    fig.add_subplot(1,col,i+1)
    plt.imshow(array[0,:,:]+array_invert[0,:,:])
    plt.colorbar()
    plt.show()

    folder.restore()

def test_sort_series():

    folder = Folder(test_folder2).open()
    series = folder.series(12) # T1-MOLLI multi-TI
#    data = series.sort(['SliceLocation','InversionTime', 'PatientName'])
    data = series.dataset(['SliceLocation','InversionTime', 'PatientName'])
    print(data[0,0,0,0].__class__.__name__)
    print(data.shape)
    print(data[0,0,0,0].SliceLocation)
    print(data[0,0,0,0].InversionTime)
    print(data[0,0,0,0].PatientName)
    loc = [data[z,0,0,0].SliceLocation for z in range(data.shape[0])]
    print(loc)

def test_read_series_array():

    folder = Folder(test_folder2).open()
    series = folder.series(12) # T1-MOLLI multi-TI
    array, data = series.array(['SliceLocation','InversionTime', 'PatientName'], pixels_first=True)
    print(array.shape)
    print(data.shape)
    print([data[0,0,0,0].SliceLocation, data[0,0,0,0].InversionTime, data[0,0,0,0].PatientName])

def test_read_and_open():

    folder = Folder(path = test_folder)
    folder.scan()
    folder.print()
    folder.open()
    folder.print()

def test_retrieve():

    folder = Folder(path = test_folder) 
    folder.scan()

    print('Printing SeriesDescription of all series in the folder')
    for series in folder.series():
        print(series.SeriesDescription)
    print('Printing Rows for first series in the folder')
    for instance in folder.series(0).instances():
        print(instance.Rows)
    print('Printing PatientName of all patients in the folder')
    for patient in folder.patients():
        print(patient.PatientName)
    print('ID of the first patient in the folder')
    print(folder.patients(0).PatientID)
    print('Description of the second series for the first patient')
    print(folder.patients(0).series(1).SeriesDescription)
    print('Description of the first series of the first study of the first patient')
    print(folder.patients(0).studies(0).series(0).SeriesDescription)

def test_find():

    folder = Folder(path = test_folder) 
    folder.scan()
    print('Find series with series description ax 15 flip and given PatientID')
    series = folder.patients(0).series(SeriesDescription="ax 15 flip", PatientID='RIDER Neuro MRI-5244517593')
    for s in series: print(s.SeriesDescription) 

def test_read_item_instance():

    folder = Folder(path = test_folder) 
    folder.scan()
    tags = ['SeriesDescription', (0x0010, 0x0020), (0x0010, 0x0020), 'PatientID', (0x0011, 0x0020)]
    instance = folder.instances(0)
    print(instance.PatientID)
    print(instance[tags])

def test_read_item():

    folder = Folder(path = test_folder)  
    folder.scan()
    tags = ['SeriesDescription', (0x0010, 0x0020), (0x0010, 0x0020), 'PatientID', (0x0011, 0x0020)]
    series = folder.series(0)
    print(series.PatientID)
    print(series[tags])
    patient = folder.patients(0)
    print(patient[tags])

def test_set_attr_instance():

    folder = Folder(path = test_folder) 
    folder.scan()
    instance = folder.instances(0)

    slice_loc = instance.SliceLocation
    print('Original Slice location: ' + str(slice_loc))
    instance.SliceLocation = slice_loc + 100
    slice_loc = instance.SliceLocation
    print('New Slice location: ' + str(slice_loc))

    acq_time = instance.AcquisitionTime
    print('Original Acquisition Time: ' + str(acq_time))
    instance.AcquisitionTime = '00.00.00'
    acq_time = instance.AcquisitionTime
    print('New Acquisition location: ' + str(acq_time))

    folder.restore()

def test_set_item_instance():

    folder = Folder(path = test_folder) 
    folder.scan()
    instance = folder.instances(0)

    tags = ['SliceLocation', 'AcquisitionTime', (0x0012, 0x0010)]
    values = instance[tags]
    print('Original Slice location: ' + str(values[0]))
    print('Original Acquisition Time: ' + str(values[1]))
    print('Original Clinical Trial Sponsor: ' + str(values[2]))

    instance[tags] = [0.0, '00:00:00', 'University of Sheffield']
    values = instance[tags]
    print('New Slice location: ' + str(values[0]))
    print('New Acquisition Time: ' + str(values[1]))
    print('New Clinical Trial Sponsor: ' + str(values[2]))

    folder.restore()

def test_set_item():

    folder = Folder(path = test_folder) 
    folder.scan()
    series = folder.series(0)
    instance = series.instances(1)

    tags = ['SliceLocation', 'AcquisitionTime', (0x0012, 0x0010)]
    values = series[tags]
    print('SERIES')
    print('Original Slice location: ' + str(values[0]))
    print('Original Acquisition Time: ' + str(values[1]))
    print('Original Clinical Trial Sponsor: ' + str(values[2]))
    values = instance[tags]
    print('2nd IMAGE')
    print('Original Slice location: ' + str(values[0]))
    print('Original Acquisition Time: ' + str(values[1]))
    print('Original Clinical Trial Sponsor: ' + str(values[2]))

    series[tags] = [0.0, '00:00:00', 'University of Sheffield']
    values = series[tags]
    print('SERIES')
    print('New Slice location: ' + str(values[0]))
    print('New Acquisition Time: ' + str(values[1]))
    print('New Clinical Trial Sponsor: ' + str(values[2]))
    values = instance[tags]
    print('2nd IMAGE')
    print('New Slice location: ' + str(values[0]))
    print('New Acquisition Time: ' + str(values[1]))
    print('New Clinical Trial Sponsor: ' + str(values[2]))    

    folder.restore()

def test_copy_remove_instance():

    folder = Folder(path = test_folder) 
    folder.scan()
    folder.print()
    
    print('****************************************')
    print('MOVE AND COPY FROM ONE SERIES TO ANOTHER')
    print('****************************************')
    
    parent = folder.patients(0).studies(0).series(7)
    instance = parent.instances(0)
    new_parent = folder.patients(0).studies(0).series(6)

    # copy the instance to the new parent and remove the original
    copy = instance.copy_to(new_parent)
    instance.remove()
    folder.print()

    # move the copy back to the original parent
    # this should restore the situation
    copy.move_to(parent)
    folder.print()

    print('')
    print('***************************************')
    print('MOVE AND COPY FROM ONE STUDY TO ANOTHER')
    print('***************************************')

    # take an instance from the first study
    # and create a new study for the copies
    parent = folder.patients(0).studies(0)
    instance = parent.series(7).instances(0)
    new_parent = folder.patients(0).new_child()

    # copy the instance to the new parent and remove the original
    # this is the same as instance.move_to(new_parent) but slower
    copy = instance.copy_to(new_parent)
    instance.remove()
    folder.print()

    # move the copy back to the original parent
    # this restores the original situation
    copy.move_to(instance.parent)
    folder.print()

    folder.restore()

def test_create():

    folder = Folder(path = test_folder) 
    folder.scan()  
    patient = folder.new_child()
    study = patient.new_child()
    series = study.new_child()
    instance = series.new_child()
    print(instance.UID)

def test_copy_remove():

    folder = Folder(path = test_folder) 
    folder.scan()
    folder.print() 
    # create a new study and copy the first series to it.
    study = folder.patients(0).new_child()
    folder.series(0).copy_to(study)
    folder.print()
    folder.restore()

def test_merge():



    # first create two new patients
    folder = Folder(path = test_folder) 
    folder.scan()

    print('')
    print('***************************************')
    print('            CREATE 2 NEW PATIENTS      ')
    print('***************************************')

    patient1 = folder.new_child()
    patient2 = folder.new_child()
    folder.patients(0).series(0).copy_to(patient1)
    folder.patients(0).series(1).copy_to(patient2)
    folder.print()

    print('')
    print('***************************************')
    print('            MERGE 2 PATIENTS           ')
    print('***************************************')

    # then merge the two new patients into a third
    patients_to_merge = [patient1, patient2]
    patient3 = folder.new_child()
    for patient in patients_to_merge:
        for study in patient.studies():
            study.copy_to(patient3)
    folder.print()


    print('')
    print('***************************************')
    print('            MERGE ALL STUDIES          ')
    print('***************************************')

    # now merge all studies of the new patient 
    # into a new study of the same patient.
    studies_to_merge = patient3.studies()
    new_study = patient3.new_child()
    for study in studies_to_merge:
        for series in study.series():
            series.copy_to(new_study)
    folder.print()

    print('')
    print('***************************************')
    print('            MERGE ALL SERIES          ')
    print('***************************************')

    # now merge all series of the new patient into
    # a new series in a new study of the same patient
    series_to_merge = patient3.series()
    new_study = patient3.new_child()
    new_series = new_study.new_child()
    for series in series_to_merge:
        for instance in series.instances():
            instance.copy_to(new_series)
    folder.print()

    folder.restore()


def test_save_restore():

    print('***************************************')
    print('            ORIGINAL FOLDER            ')
    print('***************************************')

    folder = Folder(path = test_folder) 
    folder.scan()
    folder.print()

    # copy the first series into a new patient

    print('***************************************')
    print('        NEW PATIENT CREATED            ')
    print('***************************************')

    patient = folder.new_child()
    folder.series(0).copy_to(patient)
    folder.print()

    # restore and print again

    print('***************************************')
    print('       ORIGINAL FOLDER RESTORED        ')
    print('***************************************')

    folder.restore()
    folder.print()

    # copy the first series into a new patient

    print('***************************************')
    print('       NEW PATIENT CREATED AGAIN       ')
    print('***************************************')

    patient = folder.new_child()
    folder.series(0).copy_to(patient)
    folder.print()

    # save and print again

    print('***************************************')
    print('          NEW PATIENT SAVED            ')
    print('***************************************')

    folder.save()
    folder.print()

def test_read_write_dataset():

    folder = Folder(path = test_folder) 
    folder.scan()

    instance = folder.instances(0)
    ds = instance.read() # work from memory
    print('Original rows and columns')
    print(ds.Rows)
    print(ds.Columns)

    ds.Rows = ds.Rows * 2
    ds.Columns = ds.Columns * 2
    print('Modified rows and columns')
    print(ds.Rows)
    print(ds.Columns)

    matrix = ['Rows','Columns']
    d = instance[matrix]
    instance[matrix] = [int(d[0]*2), int(d[1]*2)]
    print('Modified rows and columns again')
    print(ds.Rows)
    print(ds.Columns)
    
    ds = instance.read() # read original values from disk again
    print('Original rows and columns')
    print(ds.Rows)
    print(ds.Columns)
    ds.Rows = ds.Rows * 2
    ds.Columns = ds.Columns * 2
    instance.write()
    instance.clear()

    instance.read() # read original values from disk again
    print('Modified rows and columns')
    print(instance.Rows)
    print(instance.Columns)
    instance.clear()

    print('Modified rows and columns') #  read from disk
    print(instance.Rows)
    print(instance.Columns)

    folder.restore()    

def test_export():

    folder = Folder(path = test_folder) 
    folder.scan()
    instance = folder.instances(0)
    instance.export(test_export_folder)
    series = folder.series(0)
    series.export(test_export_folder)

def test_SOPClass():

    folder = Folder(path = test_folder) 
    folder.scan()
    for instance in folder.instances():
        print(instance.__class__.__name__)


def test_checking():

    folder = Folder(path = test_folder) 
    folder.open()
    folder.check()
    folder.uncheck()
    series = folder.series(1)
    print('The second series in the folder')
    print(series.label())
    series.check()
    folder.series(0).check()
    folder.save()
    print('The second checked series in the folder')
    series = folder.series(checked=True)
    print(series[1].label())
    print('The second of the checked series')
    series = folder.series(1, checked=True)
    print(series.label())
    instance = series.instances(0)
    instance.uncheck()
    series = folder.series(checked=True)
    print('Nr of checked series after unchecking an instance')
    print(len(series))


def test_all():

    test_read_and_open()
    test_retrieve()
    test_find()
    test_read_item_instance()
    test_read_item()
    test_set_attr_instance()
    test_set_item_instance()
    test_set_item()
    test_create()
    test_copy_remove_instance()
    test_copy_remove()
    test_merge()
    test_save_restore()
    test_read_write_dataset()
    test_export()
    test_SOPClass() 
    test_checking()
    


# For development purposes
# test_tmp()
# test_df()

# test_all()

# test_read_and_open()
# test_retrieve()
# test_find()
# test_read_item_instance()
# test_read_item()
# test_set_attr_instance()
# test_set_item_instance()
# test_set_item()
# test_copy_remove_instance()
# test_create()
# test_copy_remove()
# test_merge()
# test_save_restore()
# test_read_write_dataset()
# test_export()
# test_SOPClass()
# test_checking()
# test_read_series_array()
# test_sort_series()
# test_set_slice_array_invert()
# test_set_array_invert()
# test_set_array_mip()