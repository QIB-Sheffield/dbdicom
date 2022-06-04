__all__ = ['set_value', 'merge', 'write', 'read', 'find_series', 'move', 'copy']

import os
from copy import deepcopy
import pandas as pd
import pydicom
from . import utilities

def write(ds, file, dialog=None):

    try:
        ds.save_as(file) 
    except:
        message = "Failed to write to " + file
        message += "\n The file may be open in another application, or is being synchronised by a cloud service."
        message += "\n Please close the file or pause the synchronisation and try again."
        if dialog is not None:
            dialog.information(message) 
        else:
            print(message)  

def read(file, dialog=None):

    try:
        return pydicom.dcmread(file)
    except:
        message = "Failed to read " + file
        message += "\n The file may be open in another application, or is being synchronised by a cloud service."
        message += "\n Please close the file or pause the synchronisation and try again."
        if dialog is not None:
            dialog.information(message) 
        else:
            print(message)           
    

# This needs to be generalized to objects of any type
def set_value(instances, status=None, **kwargs):
    """Set value of a DICOM tag"""

    if not isinstance(instances, list):
        instances = [instances]
    dialog = instances[0].dialog
    folder = instances[0].folder
    
    if status is not None: 
        status.message('Finding instances in register..')

    # Find the instances in the dataframe. 
    # Separate those that are changing for the first time 
    # from those that have been changed before
    uids = [i.UID[-1] for i in instances]
    df = folder.dataframe
    df = df.loc[df.SOPInstanceUID.isin(uids) & (df.removed == False)]
    df_first_change = df.loc[df.created == False]
    df_prevs_change = df.loc[df.created == True]

    if status is not None: 
        status.message('Extending register with new files..')

    # Create a copy of those that are changing for the first time.
    # Allocate new filenames and append to the dataframe.
    df_created = df_first_change.copy(deep=True)
    df_created.removed = False
    df_created.created = True
    df_created['file'] = [folder.new_file() for _ in range(df_created.shape[0])]
    df_created.set_index('file', inplace=True)
    folder.__dict__['dataframe'] = pd.concat([folder.dataframe, df_created])

    # Mark the original files for removal.
    folder.dataframe.loc[df_first_change.index, 'removed'] = True

    if status is not None: 
        status.message('Writing values to files..')

    # Read all the datasets, update the attributes, write the results to disk.
    # Either in the same file (if the dataset has been changed before)
    # Or in a new file (if this is the first change)
    n = df_prevs_change.shape[0]+df_first_change.shape[0]
    cnt = 1
    for i, filename in enumerate(df_prevs_change.index.values):
        if status is not None: status.progress(cnt, n)
        file = os.path.join(folder.path, filename)
        ds = read(file, dialog)
        for tag, value in kwargs.items():
            ds = utilities._set_tags(ds, tag, value)
            if tag in folder._columns:
                folder.dataframe.loc[filename, tag] = value
        write(ds, file, dialog)
        cnt+=1
    for i, filename in enumerate(df_first_change.index.values):
        if status is not None: status.progress(cnt, n)
        file = os.path.join(folder.path, filename)
        newfilename = df_created.index.values[i]
        newfile = os.path.join(folder.path, newfilename)
        ds = read(file, dialog)
        for tag, value in kwargs.items():
            ds = utilities._set_tags(ds, tag, value)
            if tag in folder._columns:
                folder.dataframe.loc[newfilename, tag] = value
        write(ds, newfile, dialog)
        cnt+=1

    if status is not None: 
        status.message('Finished changing values..')

def copy(instances, series=None, status=None):

    if not isinstance(instances, list):
        instances = [instances]
    #instances = [deepcopy(i) for i in instances if i is not None]
    instances = [i.__class__(i.folder, UID=i.UID) for i in instances if i is not None]
    if instances == []:
        return []
    dialog = instances[0].dialog
    folder = instances[0].folder
    if status is not None: 
        status.message('Moving datasets..')
    if series is None:
        series = instances[0].new_pibling(SeriesDescription='Copy')

    attributes = {}
    attributes['PatientID'] = series.UID[0]
    attributes['StudyInstanceUID'] = series.UID[1]
    attributes['SeriesInstanceUID'] = series.UID[2]
    if series.attributes is not None:
        attributes.update(series.attributes)
        #for key, value in series.attributes.items():
        #    attributes[key] = value

    # Find the instances in the dataframe. 
    uids = [i.UID[-1] for i in instances]
    df = folder.dataframe
    df = df.loc[df.SOPInstanceUID.isin(uids) & (df.removed == False)]
    instance_ind = []
    for filename in df.index.values:
        uid = df.at[filename, 'SOPInstanceUID']
        instance_ind.append(uids.index(uid))

    if status is not None: 
        status.message('Extending register with new files..')

    # Create a copy of the df
    # Allocate new filenames and append to the dataframe.
    df_created = df.copy(deep=True)
    df_created.removed = False
    df_created.created = True
    df_created['file'] = [folder.new_file() for _ in range(df_created.shape[0])]
    df_created.set_index('file', inplace=True)
    folder.__dict__['dataframe'] = pd.concat([folder.dataframe, df_created])

    if status is not None: 
        status.message('Writing values to files..')

    # Read all the datasets, update the attributes, 
    # write the results in the new file
    cnt, n = 1, df.shape[0]
    for i, filename in enumerate(df.index.values):
        if status is not None: 
            status.progress(cnt, n)
        file = os.path.join(folder.path, filename)
        newfilename = df_created.index.values[i]
        newfile = os.path.join(folder.path, newfilename)
        ds = read(file, dialog)
        for tag, value in attributes.items():
            ds = utilities._set_tags(ds, tag, value)
            if tag in folder._columns:
                folder.dataframe.loc[newfilename, tag] = value
        uid = folder.new_uid()
        ds = utilities._set_tags(ds, 'SOPInstanceUID', uid)
        folder.dataframe.loc[newfilename, 'SOPInstanceUID'] = uid
        instances[instance_ind[i]].UID[-1] = uid
        instances[instance_ind[i]].UID[:-1] = series.UID
        write(ds, newfile, dialog)
        cnt+=1

    if status is not None: 
        status.message('Finished changing values..')
    
    return instances

def move(instances, series=None, status=None):

    if status is not None: 
        status.message('Moving datasets..')
    if series is None:
        series = instances[0].new_pibling(SeriesDescription='Copy')
    set_value(instances, 
        status = status,
        PatientID = series.UID[0],
        StudyInstanceUID = series.UID[1],
        SeriesInstanceUID = series.UID[2],
    )
    return instances

def merge(series_list, merged=None, status=None):

    if status is not None: 
        status.message('Merging..')
    if merged is None:
        merged = series_list[0].new_cousin(
            SeriesDescription='Merged series',
        )
    instances = find_instances(series_list)
#    instances = []
#    for i, series in enumerate(series_list):
#        status.progress(i, len(series_list), 'Reading instances..')
#        instances += series.instances() # slow - go via df
    set_value(instances, 
        status = status,
        PatientID = merged.UID[0],
        StudyInstanceUID = merged.UID[1],
        SeriesInstanceUID = merged.UID[2],
    )
    return merged

# Gneralize this to other parents & **kwargs
def find_instances(series, status=None):

    if status is not None:
        status.message('Finding instances..')
    if not isinstance(series, list):
        series = [series]
    UID = [s.UID[-1] for s in series]
    df = series[0].folder.dataframe
    df = df.loc[(df.removed == False) & (df.SeriesInstanceUID.isin(UID))]
    key = series[0].folder._columns[:4]
    all_instances = []
    cnt=0
    for _, row in df.iterrows():
        if status is not None:
            status.progress(cnt, df.shape[0], 'Finding instances..')
        instance = series[0].dicm.instance(
            series[0].folder, 
            row[key].values.tolist(),
            SOPClassUID = row.SOPClassUID)
        all_instances.append(instance)
        cnt+=1
    return all_instances

# Gneralize this to **kwargs but searching df where feasible
def find_series(studies, SeriesDescription=[], status=None):

    if status is not None:
        status.message('Finding series..')
    if not isinstance(studies, list):
        studies = [studies]
    if not isinstance(SeriesDescription, list):
        SeriesDescription = [SeriesDescription]
    UID = [s.UID[-1] for s in studies]
    df = studies[0].folder.dataframe
    select = (df.removed == False) & (df.StudyInstanceUID.isin(UID))
    if SeriesDescription != []:
        desc = SeriesDescription[0]
        series_select = (df.SeriesDescription == desc)
        for desc in SeriesDescription[1:]:
            series_select = series_select | (df.SeriesDescription == desc)
        select = select & series_select
    df = df.loc[select]
    key = studies[0].folder._columns[:3]
    unique_series = df.SeriesInstanceUID.unique()
    all_series = []
    cnt=0
    for uid in unique_series:
        if status is not None:
            status.progress(cnt, len(unique_series), 'Finding series..')
        row = df.loc[df.SeriesInstanceUID == uid].iloc[0]
        UID = row[key].values.tolist()
        series = studies[0].dicm.series(studies[0].folder, UID)
        all_series.append(series)
        cnt+=1
    return all_series
    