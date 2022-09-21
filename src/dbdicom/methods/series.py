import os
import numpy as np
from dbdicom.record import DbRecord
import dbdicom.methods.record as record_methods


class Series(DbRecord):
    
    def get_pixel_array(*args, **kwargs):
        return record_methods.get_pixel_array(*args, **kwargs)

    def set_pixel_array(*args, **kwargs):
        set_pixel_array(*args, **kwargs)

    def map_mask_to(*args, **kwargs):
        map_mask_to(*args, **kwargs)

    def export_as_csv(*args, **kwargs):
        export_as_csv(*args, **kwargs)

    def export_as_png(*args, **kwargs):
        export_as_png(*args, **kwargs)

    def export_as_nifti(*args, **kwargs):
        export_as_nifti(*args, **kwargs)

    def export_as_npy(*args, **kwargs):
        export_as_npy(*args, **kwargs)

    def subseries(*args, **kwargs):
        return subseries(*args, **kwargs)


def subseries(record, **kwargs):
    """Extract subseries"""

    series = record.new_sibling()
    for instance in record.instances(**kwargs):
        instance.copy_to(series)
    return series

def read_npy(record):
    # Not in use - loading of temporary numpy files
    file = record.register.npy()
    if not os.path.exists(file):
        return
    with open(file, 'rb') as f:
        array = np.load(f)
    return array

def export_as_csv(record, directory=None, filename=None, columnHeaders=None):
    """Export all images as csv files"""

    if directory is None: 
        directory = record.dialog.directory(message='Please select a folder for the csv data')
    if filename is None:
        filename = record.SeriesDescription
    for i, instance in enumerate(record.instances()):
        instance.export_as_csv( 
            directory = directory, 
            filename = filename + ' [' + str(i) + ']', 
            columnHeaders = columnHeaders)

def export_as_png(record, directory=None, filename=None):
    """Export all images as png files"""

    if directory is None: 
        directory = record.dialog.directory(message='Please select a folder for the png data')
    if filename is None:
        filename = record.SeriesDescription
    for i, instance in enumerate(record.instances()):
        instance.export_as_png( 
            directory = directory, 
            filename = filename + ' [' + str(i) + ']')

def export_as_nifti(record, directory=None, filename=None):
    """Export all images as nifti files"""

    if directory is None: 
        directory = record.dialog.directory(message='Please select a folder for the png data')
    if filename is None:
        filename = record.SeriesDescription
    for i, instance in enumerate(record.instances()):
        instance.export_as_nifti( 
            directory = directory, 
            filename = filename + ' [' + str(i) + ']')

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

    if source is None:
        source = record_methods.instance_array(series)

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
        copy = record_methods.copy_to(source.tolist(), series)
        #for i, s in enumerate(source.tolist()):
        #    print(copy[i].SliceLocation, s.SliceLocation) # Not matching up
    else:
        copy = source.tolist()

    series.register.pause_extensions()
    for i, instance in enumerate(copy):
        series.status.progress(i, len(copy), 'Writing array to file..')
        instance.set_pixel_array(array[i,...])
    series.register.resume_extensions()


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

