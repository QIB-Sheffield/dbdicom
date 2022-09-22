import os
import math
import numpy as np
from dbdicom.record import DbRecord, copy_to
import dbdicom.ds.dataset as dbdataset
from dbdicom.ds import MRImage


class Series(DbRecord):

    def get_pixel_array(*args, **kwargs):
        return get_pixel_array(*args, **kwargs)

    def set_pixel_array(*args, **kwargs):
        set_pixel_array(*args, **kwargs)

    def map_mask_to(*args, **kwargs):
        map_mask_to(*args, **kwargs)

    def export_as_npy(*args, **kwargs):
        export_as_npy(*args, **kwargs)

    def subseries(*args, **kwargs):
        return subseries(*args, **kwargs)

    def import_dicom(*args, **kwargs):
        import_dicom(*args, **kwargs)


def import_dicom(series, files):
    uids = series.manager.import_datasets(files)
    series.manager.move_to(uids, series.uid)

def subseries(record, **kwargs):
    """Extract subseries"""

    series = record.new_sibling()
    for instance in record.instances(**kwargs):
        instance.copy_to(series)
    return series

def read_npy(record):
    # Not in use - loading of temporary numpy files
    file = record.manager.npy()
    if not os.path.exists(file):
        return
    with open(file, 'rb') as f:
        array = np.load(f)
    return array


def export_as_npy(record, directory=None, filename=None, sortby=None, pixels_first=False):
    """Export array in numpy format"""

    if directory is None: 
        directory = record.dialog.directory(message='Please select a folder for the png data')
    if filename is None:
        filename = record.SeriesDescription
    array, _ = record.get_pixel_array(sortby=sortby, pixels_first=pixels_first)
    file = os.path.join(directory, filename + '.npy')
    with open(file, 'wb') as f:
        np.save(f, array)


def map_mask_to(series, target):
    """Map non-zero pixels onto another series"""

    source_images = series.instances()
    target_images = target.instances() 
    mapped_series = series.new_sibling(
        SeriesDescription = series.SeriesDescription + ' mapped to ' + target.SeriesDescription
    )
    for i, target_image in enumerate(target_images):
        series.status.progress(i, len(target_images))
        pixel_array = np.zeros((target_image.Columns, target_image.Rows), dtype=np.bool) 
        for j, source_image in enumerate(source_images):
            series.status.message(
                'Mapping image ' + str(j) + 
                ' of ' + series.SeriesDescription + 
                ' to image ' + str(i) + 
                ' of ' + target.SeriesDescription 
            )
            im = source_image.map_mask_to(target_image)
            array = im.get_pixel_array().astype(np.bool)
            np.logical_or(pixel_array, array, out=pixel_array)
            im.remove()
        if pixel_array.any():
            mapped_image = target_image.copy_to(mapped_series)
            mapped_image.set_pixel_array(pixel_array.astype(np.float32))
    return mapped_series


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
    array = _stack(array)
    array = array.reshape(source.shape + array.shape[1:])
    if pixels_first:
        array = np.moveaxis(array, -1, 0)
        array = np.moveaxis(array, -1, 0)
    return array, source 


def set_pixel_array(series, array, source=None, pixels_first=False): 
    """
    Set pixel values of a series from a numpy ndarray.

    Since the pixel data do not hold any information about the 
    image such as geometry, or other metainformation,
    a dataset must be provided as well with the same 
    shape as the array except for the slice dimensions. 

    If a dataset is not provided, header info is 
    derived from existing instances in order.

    Args:
        array: 
            numpy ndarray with pixel data.

        dataset: 
            numpy ndarray

            Instances holding the header information. 
            This *must* have the same shape as array, minus the slice dimensions.

        pixels_first: 
            bool

            Specifies whether the pixel dimensions are the first or last dimensions of the series.
            If not provided it is assumed the slice dimensions are the last dimensions
            of the array.

        inplace: 
            bool

            If True (default) the current pixel values in the series 
            are overwritten. If set to False, the new array is added to the series.
    
    Examples:
        ```ruby
        # Invert all images in a series:
        array, _ = series.array()
        series.set_array(-array)

        # Create a maximum intensity projection of the series.
        # Header information for the result is taken from the first image.
        # Results are saved in a new sibling series.
        array, data = series.array()
        array = np.amax(array, axis=0)
        data = np.squeeze(data[0,...])
        series.new_sibling().set_array(array, data)

        # Create a 2D maximum intensity projection along the SliceLocation direction.
        # Header information for the result is taken from the first slice location.
        # Current data of the series are overwritten.
        array, data = series.array('SliceLocation')
        array = np.amax(array, axis=0)
        data = np.squeeze(data[0,...])
        series.set_array(array, data)

        # In a series with multiple slice locations and inversion times,
        # replace all images for each slice location with that of the shortest inversion time.
        array, data = series.array(['SliceLocation','InversionTime']) 
        for loc in range(array.shape[0]):               # loop over slice locations
            slice0 = np.squeeze(array[loc,0,0,:,:])     # get the slice with shortest TI 
            TI0 = data[loc,0,0].InversionTime           # get the TI of that slice
            for TI in range(array.shape[1]):            # loop over TIs
                array[loc,TI,0,:,:] = slice0            # replace each slice with shortest TI
                data[loc,TI,0].InversionTime = TI0      # replace each TI with shortest TI
        series.set_array(array, data)
        ```
    """
    if pixels_first:    # Move to the end (default)
        array = np.moveaxis(array, 0, -1)
        array = np.moveaxis(array, 0, -1)

    # if no header data is provided, use template headers.
    if source is None:
        n = np.prod(array.shape[:-2])
        source = np.empty(n, dtype=object)
        for i in range(n): 
            source[i] = series.new_instance(MRImage())  
        source = source.reshape(array.shape[:-2])
        series.set_pixel_array(array, source)
        #source = instance_array(series)

    # Return with error message if dataset and array do not match.
    nr_of_slices = np.prod(array.shape[:-2])
    if nr_of_slices != np.prod(source.shape):
        message = 'Error in set_array(): array and source do not match'
        message += '\n Array has ' + str(nr_of_slices) + ' elements'
        message += '\n Source has ' + str(np.prod(source.shape)) + ' elements'
        series.dialog.error(message)
        raise ValueError(message)

    # Flatten array and source for iterating
    array = array.reshape((nr_of_slices, array.shape[-2], array.shape[-1])) # shape (i,x,y)
    source = source.reshape(nr_of_slices) # shape (i,)

    # set_array replaces current array
    for i in series.instances():
        if i not in source.tolist():
            i.remove()
    if series.instances() == []:
        copy = copy_to(source.tolist(), series)
        #for i, s in enumerate(source.tolist()):
        #    print(copy[i].SliceLocation, s.SliceLocation) # Not matching up
    else:
        copy = source.tolist()

    series.manager.pause_extensions()
    for i, instance in enumerate(copy):
        series.status.progress(i, len(copy), 'Writing array to file..')
        instance.set_pixel_array(array[i,...])
    series.manager.resume_extensions()


def amax(record, axis=None):
    """Calculate the maximum of the image array along a given dimension.
    
    This function is included as a placeholder reminder 
    to build up functionality at series level that emulates 
    numpy behaviour.

    Args:
        axis: DICOM KeyWord string to specify the dimension
        along which the maximum is taken.

    Returns:
        a new sibling series holding the result.

    Example:
    ```ruby
    # Create a maximum intensity projection along the slice dimension:
    mip = series.amax(axis='SliceLocation')
    ```
    """

    array, header = record.get_pixel_array(axis)
    array = np.amax(array, axis=0)
    header = np.squeeze(header[0,...])
    series = record.new_sibling()
    series.set_pixel_array(array, header)
    return series



##
## Helper functions
##


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

def dataframe(record):

    keys = record.manager.keys(record.uid)
    return record.manager.register.loc[keys, :]


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
    return _stack(data, align_left=True)


def df_to_instance_array(record, df): 
    """Return datasets as numpy array of object type"""

    data = np.empty(df.shape[0], dtype=object)
    for i, uid in enumerate(df.index.values): 
        data[i] = record.instance(uid)
    return data

def _stack(arrays, align_left=False):
    """Stack a list of arrays of different shapes but same number of dimensions.
    
    This generalises numpy.stack to arrays of different sizes.
    The stack has the size of the largest array.
    If an array is smaller it is zero-padded and centred on the middle.
    """

    # Get the dimensions of the stack
    # For each dimension, look for the largest values across all arrays
    ndim = len(arrays[0].shape)
    dim = [0] * ndim
    for array in arrays:
        for i, d in enumerate(dim):
            dim[i] = max((d, array.shape[i])) # changing the variable we are iterating over!!
    #    for i in range(ndim):
    #        dim[i] = max((dim[i], array.shape[i]))

    # Create the stack
    # Add one dimension corresponding to the size of the stack
    n = len(arrays)
    #stack = np.full([n] + dim, 0, dtype=arrays[0].dtype)
    stack = np.full([n] + dim, None, dtype=arrays[0].dtype)

    for k, array in enumerate(arrays):
        index = [k]
        for i, d in enumerate(dim):
            if align_left:
                i0 = 0
            else: # align center and zero-pad missing values
                i0 = math.floor((d-array.shape[i])/2)
            i1 = i0 + array.shape[i]
            index.append(slice(i0,i1))
        stack[tuple(index)] = array

    return stack

