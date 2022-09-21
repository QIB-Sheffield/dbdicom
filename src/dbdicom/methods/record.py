import numpy as np
import dbdicom.utils.arrays as arrays
import dbdicom.dataset as dbdataset

def get_pixel_array(record, sortby=None, pixels_first=False): 
    """Pixel values of the object as an ndarray
    
    Args:
        sortby: 
            Optional list of DICOM keywords by which the volume is sorted
        pixels_first: 
            If True, the (x,y) dimensions are the first dimensions of the array.
            If False, (x,y) are the last dimensions - this is the default.

    Returns:
        An ndarray holding the pixel data.

        An ndarry holding the datasets (instances) of each slice.

    Examples:
        ``` ruby
        # return a 3D array (z,x,y)
        # with the pixel data for each slice
        # in no particular order (z)
        array, _ = series.array()    

        # return a 3D array (x,y,z)   
        # with pixel data in the leading indices                               
        array, _ = series.array(pixels_first = True)    

        # Return a 4D array (x,y,t,k) sorted by acquisition time   
        # The last dimension (k) enumerates all slices with the same acquisition time. 
        # If there is only one image for each acquision time, 
        # the last dimension is a dimension of 1                               
        array, data = series.array('AcquisitionTime', pixels_first=True)                         
        v = array[:,:,10,0]                 # First image at the 10th location
        t = data[10,0].AcquisitionTIme      # acquisition time of the same image

        # Return a 4D array (loc, TI, x, y) 
        sortby = ['SliceLocation','InversionTime']
        array, data = series.array(sortby) 
        v = array[10,6,0,:,:]            # First slice at 11th slice location and 7th inversion time    
        Loc = data[10,6,0][sortby[0]]    # Slice location of the same slice
        TI = data[10,6,0][sortby[1]]     # Inversion time of the same slice
        ```  
    """
    if sortby is not None:
        if not isinstance(sortby, list):
            sortby = [sortby]
    source = instance_array(record, sortby)
    array = []
    instances = source.ravel()
    for i, im in enumerate(instances):
        record.status.progress(i, len(instances), 'Reading pixel data..')
        if im is None:
            array.append(np.zeros((1,1)))
        else:
            array.append(im.get_pixel_array())
    array = arrays._stack(array)
    array = array.reshape(source.shape + array.shape[1:])
    if pixels_first:
        array = np.moveaxis(array, -1, 0)
        array = np.moveaxis(array, -1, 0)
    return array, source 


def instance_array(record, sortby=None, status=True): 
    """Sort instances by a list of attributes.
    
    Args:
        sortby: 
            List of DICOM keywords by which the series is sorted
    Returns:
        An ndarray holding the instances sorted by sortby.
    """
    if sortby is None:
        instances = record.instances()
        array = np.empty(len(instances), dtype=object)
        for i, instance in enumerate(instances): 
            array[i] = instance
        return array
    else:
        if set(sortby) <= set(record.manager.register):
            df = record.manager.register.loc[dataframe(record).index, sortby]
        else:
            ds = record.get_dataset()
            df = dbdataset.get_dataframe(ds, sortby)
        df.sort_values(sortby, inplace=True) 
        return df_to_sorted_instance_array(record, df, sortby, status=status)


def df_to_sorted_instance_array(record, df, sortby, status=True): 

    data = []
    vals = df[sortby[0]].unique()
    for i, c in enumerate(vals):
        if status: 
            record.status.progress(i, len(vals), message='Sorting..')
        dfc = df[df[sortby[0]] == c]
        if len(sortby) == 1:
            datac = df_to_instance_array(record, dfc)
        else:
            datac = df_to_sorted_instance_array(record, dfc, sortby[1:], status=False)
        data.append(datac)
    return arrays._stack(data, align_left=True)


def df_to_instance_array(record, df): 
    """Return datasets as numpy array of object type"""

    data = np.empty(df.shape[0], dtype=object)
    for i, uid in enumerate(df.index.values): 
        data[i] = record.instance(uid)
    return data


def dataframe(record):

    keys = record.manager.keys(record.uid)
    return record.manager.register.loc[keys, :]





#
# Functions on a list of records of the same database
#

def get_values(records, attributes):

    uids = [rec.uid for rec in records]
    dbr = records[0].manager
    return dbr.get_values(uids, attributes)

def set_values(records, attributes, values):

    uids = [rec.uid for rec in records]
    dbr = records[0].manager
    dbr.set_values(uids, attributes, values)

def children(records, **kwargs):

    dbr = records[0].manager
    uids = [rec.uid for rec in records]
    uids = dbr.children(uids, **kwargs)
    return [records[0].create(dbr, uid) for uid in uids]

def instances(records, **kwargs):

    dbr = records[0].manager
    uids = [rec.uid for rec in records]
    uids = dbr.instances(uids, **kwargs)
    return [records[0].create(dbr, uid, 'Instance') for uid in uids]

def series(records, **kwargs):

    dbr = records[0].manager
    uids = [rec.uid for rec in records]
    uids = dbr.series(uids, **kwargs)
    return [records[0].create(dbr, uid, 'Series') for uid in uids]

def studies(records, **kwargs):

    dbr = records[0].manager
    uids = [rec.uid for rec in records]
    uids = dbr.studies(uids, **kwargs)
    return [records[0].create(dbr, uid, 'Study') for uid in uids]

def patients(records, **kwargs):

    dbr = records[0].manager
    uids = [rec.uid for rec in records]
    uids = dbr.patients(uids, **kwargs)
    return [records[0].create(dbr, uid, 'Patient') for uid in uids]

def copy_to(records, target):

    dbr = records[0].manager
    uids = [rec.uid for rec in records]
    uids = dbr.copy_to(uids, target.uid, **target.attributes)
    if isinstance(uids, list):
        return [records[0].create(dbr, uid) for uid in uids]
    else:
        return [records[0].create(dbr, uids)]

def move_to(records, target):

    dbr = records[0].manager
    uids = [rec.uid for rec in records]
    dbr.move_to(uids, target.uid, **target.attributes)
    return records

def group(records, into=None):

    if into is None:
        into = records[0].new_pibling()
    copy_to(records, into)
    return into

def merge(records, into=None):

    return group(children(records), into=into)