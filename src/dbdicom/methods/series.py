import numpy as np
from dbdicom.record import DbRecord
import dbdicom.methods.record as record_methods


class Series(DbRecord):
    
    def get_pixel_array(*args, **kwargs):
        return record_methods.get_pixel_array(*args, **kwargs)

    def set_pixel_array(*args, **kwargs):
        set_pixel_array(*args, **kwargs)

    def map_mask_onto(*args, **kwargs):
        map_mask_onto(*args, **kwargs)


def map_mask_onto(record, target):
    """Map non-zero pixels onto another series"""

    source_images = record.children()
    mapped_series = record.new_sibling()
    target_images = target.children() # create record.images() to return children of type image

    for i, target_image in enumerate(target_images):
        record.status.progress(i, len(target_images))
        pixel_array = np.zeros((target_image.Rows, target_image.Columns), dtype=np.bool) 
        for j, source_image in enumerate(source_images):
            message = (
                'Mapping image ' + str(j) + 
                ' of ' + record.SeriesDescription + 
                ' to image ' + str(i) + 
                ' of ' + target.SeriesDescription )
            record.status.message(message)
            im = source_image.map_mask_onto(target_image)
            array = im.get_pixel_array().astype(np.bool)
            np.logical_or(pixel_array, array, out=pixel_array)
        if pixel_array.any():
            mapped_image = target_image.copy_to(mapped_series)
            mapped_image.set_pixel_array(pixel_array.astype(np.float32))
            mapped_image.SeriesDescription = record.SeriesDescription
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

