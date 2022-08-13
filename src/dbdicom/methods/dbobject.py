import os
import pandas as pd
import numpy as np
import dbdicom.utils.pydicom as pydcm
import dbdicom.utils.arrays as arrays

def type(dbobject):

    if dbobject.generation == 0:
        return 'Database'
    if dbobject.generation == 1:
        return 'Patient'
    if dbobject.generation == 2:
        return 'Study'
    if dbobject.generation == 3:
        return 'Series'
    if dbobject.generation == 4:
        return pydcm.SOPClass(dbobject._SOPClassUID)

def _filter(objects, **kwargs):
    # Note also works on pydicom ds's
    """
    Filters a list of DICOM classes by DICOM tags and values.
    
    Example
    -------
    instances = _filter(instances, PatientName="Harry")
    """

    filtered = []
    for obj in objects:
        select = True
        for tag, value in kwargs.items():
            if getattr(obj, tag) != value:
                select = False
                break
        if select: 
            filtered.append(obj)
    return filtered

 
def copy_instances(dbobject, instances, concatenate=True, message=None):  # Needs status updates
    # Copies instances to dbobject

    if dbobject.generation == 4:
        series = dbobject.parent
    elif dbobject.generation == 3:
        series = dbobject
    else:
        series = dbobject.new_series()
    
    # Attributes that are inherited from the series
    attr = {}
    attr['PatientID'] = series.UID[0]
    attr['StudyInstanceUID'] = series.UID[1]
    attr['SeriesInstanceUID'] = series.UID[2]
    if dbobject.attributes is not None:
        attr.update(dbobject.attributes)

    copy_data = []
    copy_files = []
    copy_instances = []

    for i, instance in enumerate(instances):

        if message is not None:
            dbobject.status.progress(i, len(instances), message)

        # get a new UID and file for the copy
        attr['SOPInstanceUID'] = pydcm.new_uid()
        copy_file = dbobject.dbindex.new_file()
        filepath = os.path.join(dbobject.dbindex.path, copy_file)
        copy_instance = instance.__class__(dbobject, UID=attr[instance.key])

        # read dataset, set new attributes, and write result to new file
        ds = instance.read()._ds
        pydcm.set_values(ds, list(attr.keys()), list(attr.values()))
        pydcm.write(ds, filepath, dbobject.dialog)
        if copy_instance.in_memory():
            copy_instance._ds = ds

        # Get new data for the dataframe
        row = pydcm.get_values(ds, dbobject.dbindex.columns)
        copy_data.append(row)
        copy_files.append(copy_file)
        copy_instances.append(copy_instance)
        
    # Update the dataframe in the index
    df = pd.DataFrame(copy_data, index=copy_files, columns=dbobject.dbindex.columns)
    df['removed'] = False
    df['created'] = True
    if concatenate:
        dbobject.dbindex.dataframe = pd.concat([dbobject.dbindex.dataframe, df])

    dbobject.status.hide()
    return copy_instances, df

def _copy_list_to(dbobjects, parent, message=None):
    """Copy a list of dbobjects to a parent dbobject"""

    while parent.generation < dbobjects[0].generation-1:
        parent = parent.new_child()

    if dbobjects[0].generation == 4:
        instance_copies, _ = copy_instances(parent, dbobjects, message=message)
        return instance_copies

    if dbobjects[0].generation == 3:
        new_df = [parent.dbindex.dataframe]
        series_copies = []
        for series in dbobjects:
            if message is not None:
                parent.status.progress(i, len(dbobjects), message)
            series_copy = parent.new_child()
            series_copies.append(series_copy)
            _, df = copy_instances(series_copy, series.instances(), concatenate=False)
            new_df.append(df)
        parent.dbindex.dataframe = pd.concat(new_df)
        parent.status.hide()
        return series_copies

    if dbobjects[0].generation == 2:
        new_df = [parent.dbindex.dataframe]
        study_copies = []
        for study in dbobjects:
            if message is not None:
                parent.status.progress(i, len(dbobjects), message)
            study_copy = parent.new_child()
            study_copies.append(study_copy)
            for series in study.children():
                series_copy = study_copy.new_child()
                _, df = copy_instances(series_copy, series.instances(), concatenate=False) 
                new_df.append(df)
        parent.dbindex.dataframe = pd.concat(new_df)
        parent.status.hide()
        return study_copies

    if dbobjects[0].generation == 1:
        new_df = [parent.dbindex.dataframe]
        patient_copies = []
        for patient in dbobjects:
            if message is not None:
                parent.status.progress(i, len(dbobjects), message)
            patient_copy = parent.new_child()
            patient_copies.append(patient_copy)
            for study in patient.children():
                study_copy = patient_copy.new_child()
                for series in study.children():
                    series_copy = study_copy.new_child()
                    _, df = series_copy.copy_instances(series.instances(), concatenate=False) 
                    new_df.append(df)
        parent.dbindex.dataframe = pd.concat(new_df)
        parent.status.hide()
        return patient_copies

    if dbobjects[0].generation == 0:
        raise ValueError('Cannot copy database - use export instead')


def copy_to(dbobject, parent, message=None):

    if isinstance(dbobject, list):
        return _copy_list_to(dbobject, parent, message=message)
    else:
        return _copy_list_to([dbobject], parent, message=message)[0]


def merge(dbobjects, message=None):

    for obj in dbobjects[1:]:
        if obj.generation != dbobjects[0].generation:
            raise ValueError('Cannot merge objects of a different class')
    if dbobjects[0] == 4:
        raise ValueError('Instances cannot be merged.')
    if dbobjects[0] == 0:
        raise ValueError('Databases cannot be merged. Use import instead.')

    if dbobjects[0].generation == 3:
        merged = dbobjects[0].new_sibling(SeriesDescription = 'Merged Series')
    elif dbobjects[0].generation == 2:
        merged = dbobjects[0].new_sibling(StudyDescription = 'Merged Study')
    elif dbobjects[0].generation == 1:
        merged = dbobjects[0].new_sibling(PatientName = 'Merged Patient')

    all_children = []
    for obj in dbobjects:
        all_children.extend(obj.children())
    copy_to(all_children, merged, message=message)

    return merged


def sort_instances(dbobject, sortby=None, status=True): 
    """Sort instances by a list of attributes.
    
    Args:
        sortby: 
            List of DICOM keywords by which the series is sorted
    Returns:
        An ndarray holding the instances sorted by sortby.
    """
    if sortby is None:
        df = dbobject.data()
        return dbobject._dataset_from_df(df)
    else:
        if set(sortby) <= set(dbobject.dbindex.dataframe):
            df = dbobject.dbindex.dataframe.loc[dbobject.data().index, sortby]
        else:
            df = dbobject.get_dataframe(sortby)
        df.sort_values(sortby, inplace=True) 
        return _sorted_dataset_from_df(dbobject, df, sortby, status=status)

def _sorted_dataset_from_df(dbobject, df, sortby, status=True): 

    data = []
    vals = df[sortby[0]].unique()
    for i, c in enumerate(vals):
        if status: 
            dbobject.status.progress(i, len(vals), message='Sorting..')
        dfc = df[df[sortby[0]] == c]
        if len(sortby) == 1:
            datac = _dataset_from_df(dbobject, dfc)
        else:
            datac = _sorted_dataset_from_df(dbobject, dfc, sortby[1:], status=False)
        data.append(datac)
    return arrays._stack(data, align_left=True)

def _dataset_from_df(dbobject, df): 
    """Return datasets as numpy array of object type"""

    data = np.empty(df.shape[0], dtype=object)
    cnt = 0
    for file, _ in df.iterrows(): # just enumerate over df.index
        #dbobject.status.progress(cnt, df.shape[0])
        data[cnt] = dbobject.new_instance(file)
        cnt += 1
    #dbobject.status.hide()
    return data


def array(dbobject, sortby=None, pixels_first=False): 

    if dbobject.generation == 4:
        return pydcm._image_array(dbobject.read()._ds)
    source = dbobject.sort_instances(sortby)
    array = []
    ds = source.ravel()
    for i, im in enumerate(ds):
        dbobject.status.progress(i, len(ds), 'Reading pixel data..')
        if im is None:
            array.append(np.zeros((1,1)))
        else:
            array.append(im.array())
    dbobject.status.hide()
    #array = [im.array() for im in dataset.ravel() if im is not None]
    array = arrays._stack(array)
    array = array.reshape(source.shape + array.shape[1:])
    if pixels_first:
        array = np.moveaxis(array, -1, 0)
        array = np.moveaxis(array, -1, 0)
    return array, source # REPLACE BY DBARRAY?


def set_array(dbobject, array, source=None, pixels_first=False): 

    if pixels_first:    # Move to the end (default)
        array = np.moveaxis(array, 0, -1)
        array = np.moveaxis(array, 0, -1)

    if source is None:
        source = dbobject.sort_instances()

    # Return with error message if dataset and array do not match.
    nr_of_slices = np.prod(array.shape[:-2])
    if nr_of_slices != np.prod(source.shape):
        message = 'Error in set_array(): array and source do not match'
        message += '\n Array has ' + str(nr_of_slices) + ' elements'
        message += '\n Source has ' + str(np.prod(source.shape)) + ' elements'
        dbobject.dialog.error(message)
        raise ValueError(message)

    # Identify the parent object and set 
    # attributes that are inherited from the parent
    if dbobject.generation in [3,4]:
        parent = dbobject
    else:
        parent = dbobject.new_series()
    attr = {}
    attr['PatientID'] = parent.UID[0]
    attr['StudyInstanceUID'] = parent.UID[1]
    attr['SeriesInstanceUID'] = parent.UID[2]
    if parent.attributes is not None:
        attr.update(parent.attributes)

    # Flatten array and source for iterating
    array = array.reshape((nr_of_slices, array.shape[-2], array.shape[-1])) # shape (i,x,y)
    source = source.reshape(nr_of_slices) # shape (i,)

    # Load each dataset in the source files
    # Replace the array and parent attributes
    # save the result in a new file
    # and update the index dataframe
    copy_data = []
    copy_files = []
    for i, instance in enumerate(source):

        dbobject.status.progress(i, len(source), 'Writing array to file..')

        # get a new UID and file for the copy
        attr['SOPInstanceUID'] = pydcm.new_uid()
        copy_file = dbobject.dbindex.new_file()
        filepath = os.path.join(dbobject.dbindex.path, copy_file)

        # read dataset, set new attributes & array, and write result to new file
        ds = instance.read()._ds 
        pydcm.set_values(ds, list(attr.keys()), list(attr.values()))
        pydcm._set_image_array(ds, array[i,...])
        if not dbobject.in_memory():
            pydcm.write(ds, filepath, dbobject.dialog)

        # Get new data for the dataframe
        row = pydcm.get_values(ds, dbobject.dbindex.columns)
        copy_data.append(row)
        copy_files.append(copy_file)
        
    # Update the dataframe in the index
    df = pd.DataFrame(copy_data, index=copy_files, columns=dbobject.dbindex.columns)
    df['removed'] = False
    df['created'] = True
    dbobject.dbindex.dataframe = pd.concat([dbobject.dbindex.dataframe, df])

    return parent