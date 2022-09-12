import os
import numpy as np
import nibabel as nib
import pandas as pd

import matplotlib.pyplot as plt

import dbdicom.utils.arrays as arrays
import dbdicom.utils.image as image
import dbdicom.dataset as dbdataset


def array(dbobject, sortby=None, pixels_first=False): 
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
    if dbobject.generation == 4:
        return dbdataset._image_array(dbobject.read()._ds)
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
        attr['SOPInstanceUID'] = dbdataset.new_uid()
        copy_file = dbobject.dbindex.new_file()
        filepath = os.path.join(dbobject.dbindex.path, copy_file)

        # read dataset, set new attributes & array, and write result to new file
        ds = instance.read()._ds 
        dbdataset.set_values(ds, list(attr.keys()), list(attr.values()))
        dbdataset._set_image_array(ds, array[i,...])
        if not dbobject.in_memory():
            dbdataset.write(ds, filepath, dbobject.dialog)

        # Get new data for the dataframe
        row = dbdataset.get_values(ds, dbobject.dbindex.columns)
        copy_data.append(row)
        copy_files.append(copy_file)
        
    # Update the dataframe in the index
    df = pd.DataFrame(copy_data, index=copy_files, columns=dbobject.dbindex.columns)
    df['removed'] = False
    df['created'] = True
    dbobject.dbindex.dataframe = pd.concat([dbobject.dbindex.dataframe, df])

    return parent


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


def map_onto(record, target):
    """Map non-zero pixels onto another series"""

    if record.generation == 4:
        return _image_map_onto(record, target)

    if record.type() != 'Series':
        return

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
            array = _image_map_onto(source_image, target_image).array().astype(np.bool)
            np.logical_or(pixel_array, array, out=pixel_array)
        if pixel_array.any():
            mapped_image = target_image.copy_to(mapped_series)
            mapped_image.set_array(pixel_array.astype(np.float32))
            mapped_image.SeriesDescription = record.SeriesDescription
    record.status.hide()
    return mapped_series

def _image_map_onto(record, target):
    """Map non-zero image pixels onto a target image.
    
    Overwrite pixel values in the target"""

    # Create a coordinate array of non-zero pixels
    coords = np.transpose(np.where(record.array() != 0)) 
    coords = [[coord[0], coord[1], 0] for coord in coords] 
    coords = np.array(coords)

    # Determine coordinate transformation matrix
    affineSource = record._affine_matrix()
    affineTarget = target._affine_matrix()
    sourceToTarget = np.linalg.inv(affineTarget).dot(affineSource)

    # Apply coordinate transformation
    coords = nib.affines.apply_affine(sourceToTarget, coords)
    coords = np.round(coords, 3).astype(int)
    x = tuple([coord[0] for coord in coords if coord[2] == 0])
    y = tuple([coord[1] for coord in coords if coord[2] == 0])

    # Set values in the target image
    # Note - replace by actual values rather than 1 & 0.
    result = target.zeros()
    pixelArray = result.array()
    pixelArray[(x, y)] = 1.0
    result.set_array(pixelArray)

    return result


def export_as_nifti(record, directory=None, filename=None):
    """Export series as a single Nifty file"""

    if record.generation == 4:
        ds = record.read()
    else:
        ds = record.instances(0).read()

    if directory is None: 
        directory = record.directory(message='Please select a folder for the nifty data')
    if filename is None:
        filename = record.SeriesDescription
    dicomHeader = nib.nifti1.Nifti1DicomExtension(2, ds)
    pixelArray = np.flipud(np.rot90(np.transpose(record).array()))
    niftiObj = nib.Nifti1Instance(pixelArray, ds.affine)
    niftiObj.header.extensions.append(dicomHeader)
    nib.save(niftiObj, directory + '/' + filename + '.nii.gz')


def export_as_csv(record, directory=None, filename=None, columnHeaders=None):
    """Export all images as csv files"""

    if record.generation == 4:
        _export_instance_as_csv(record, directory=directory, filename=filename, columnHeaders=columnHeaders)
        return

    if directory is None: 
        directory = record.directory(message='Please select a folder for the csv data')
    if filename is None:
        filename = record.SeriesDescription
    for i, instance in enumerate(record.instances()):
        _export_instance_as_csv(instance, 
            directory = directory, 
            filename = filename + '(' + str(i) + ')', 
            columnHeaders = columnHeaders)

def _export_instance_as_csv(record, directory=None, filename=None, columnHeaders=None):
    """Export 2D pixel Array in csv format"""

    if directory is None: 
        directory = record.directory(message='Please select a folder for the csv data')
    if filename is None:
        filename = record.SeriesDescription
    filename = os.path.join(directory, filename + '.csv')
    table = record.array()
    if columnHeaders is None:
        columnHeaders = []
        counter = 0
        for _ in table:
            counter += 1
            columnHeaders.append("Column" + str(counter))
    df = pd.DataFrame(np.transpose(table), columns=columnHeaders)
    df.to_csv(filename, index=False)


def magnitude(record):
    "Creates a sibling series with the magnitude images"

    if record.type() != 'Series':
        return

    return _extractImageType(record, 'MAGNITUDE')

def phase(record):
    "Creates a sibling series with the phase images"

    if record.type() != 'Series':
        return

    return _extractImageType(record, 'PHASE')

def real(record):
    "Creates a sibling series with the real images"

    if record.type() != 'Series':
        return

    return _extractImageType(record, 'REAL')

def imaginary(record):
    "Creates a sibling series with the imaginary images"

    if record.type() != 'Series':
        return

    return _extractImageType(record, 'IMAGINARY')

def _extractImageType(record, image_type):
    """Extract subseries with images of given imageType"""

    if record.type() != 'Series':
        return

    series = record.new_sibling()
    for instance in record.instances():
        if instance.image_type() == image_type:
            instance.copy_to(series)
    return series

def _amax(record, axis=None):
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

    if record.type() != 'Series':
        return

    array, data = record.array(axis)
    array = np.amax(array, axis=0)
    data = np.squeeze(data[0,...])
    series = record.new_sibling()
    series.set_array(array, data)
    return series



def zeros(record): # only for images

    if record.generation < 4:
        return
    array = np.zeros((record.Rows, record.Columns))
    new = record.copy()
    new.set_array(array)
    return new


def _affine_matrix(record):
    """Affine transformation matrix for a DICOM image"""

    ds = record.read()
    return image.affine_matrix(
        ds.ImageOrientationPatient, 
        ds.ImagePositionPatient, 
        ds.PixelSpacing, 
        ds.SliceThickness)

def _enhanced_mri_affine_matrix_list(record):
    """Affine transformation matrix for all images in a multiframe image"""

    ds = record.read()
    affineList = list()
    for frame in ds.PerFrameFunctionalGroupsSequence:
        affine = image.affine_matrix(
            frame.PlaneOrientationSequence[0].ImageOrientationPatient, 
            frame.PlanePositionSequence[0].ImagePositionPatient, 
            frame.PixelMeasuresSequence[0].PixelSpacing, 
            frame.PixelMeasuresSequence[0].SpacingBetweenSlices)
        affineList.append(affine)
    return np.squeeze(np.array(affineList))


def get_colormap(record):
    """Returns the colormap if there is any."""

    if record.generation < 4:
        return

    ds = record.read()
    return dbdataset.colormap(ds)

def get_lut(record):

    if record.generation < 4:
        return
    ds = record.read()
    return dbdataset.lut(ds)


def set_colormap(record, *args, **kwargs):
    """Set the colour table of the image."""

    if record.generation < 4:
        return
    dataset = record.read()
    dbdataset.set_colormap(dataset.to_pydicom(), *args, **kwargs)
    record.write(dataset)  


def export_as_png(record, fileName):
    """Export image in png format."""

    if record.generation < 4:
        return

    colourTable, _ = record.get_colormap()
    pixelArray = np.transpose(record.array())
    centre, width = record.window()
    minValue = centre - width/2
    maxValue = centre + width/2
    cmap = plt.get_cmap(colourTable)
    plt.imshow(pixelArray, cmap=cmap)
    plt.clim(int(minValue), int(maxValue))
    cBar = plt.colorbar()
    cBar.minorticks_on()
    plt.savefig(fname=fileName + '_' + record.label() + '.png')
    plt.close() 


def window(record):
    """Centre and width of the pixel data after applying rescale slope and intercept"""

    ds = record.read()
    if 'WindowCenter' in ds: centre = ds.WindowCenter
    if 'WindowWidth' in ds: width = ds.WindowWidth
    if centre is None or width is None:
        array = record.array()
    if centre is None: 
        centre = np.median(array)
    if width is None: 
        p = np.percentile(array, [25, 75])
        width = p[1] - p[0]
    
    return centre, width

def _enhanced_mri_window(record):
    """Centre and width of the pixel data after applying rescale slope and intercept.
    
    In this case retrieve the centre and width values of the first frame
    NOT In USE
    """
    ds = record.read()
    centre = ds.PerFrameFunctionalGroupsSequence[0].FrameVOILUTSequence[0].WindowCenter 
    width = ds.PerFrameFunctionalGroupsSequence[0].FrameVOILUTSequence[0].WindowWidth
    if centre is None or width is None:
        array = record.array()
    if centre is None: 
        centre = np.median(array)
    if width is None: 
        p = np.percentile(array, [25, 75])
        width = p[1] - p[0]
    
    return centre, width


def QImage(record):

    array = record.array()
    return image.QImage(array, width=record.WindowWidth, center=record.WindowCenter)


def image_type(record):
    """Determine if an image is Magnitude, Phase, Real or Imaginary image or None"""

    ds = record.read().to_pydicom()

    if record.type == 'MRImage':
        return dbdataset.mr_image_type(ds)
    if record.type == 'EnhancedMRImage':
        return dbdataset.enhanced_mr_image_type(ds)


def signal_type(record):
    """Determine if an image is Water, Fat, In-Phase, Out-phase image or None"""

    ds = record.read().to_pydicom()

    if record.type == 'MRImage':
        return dbdataset.mr_image_signal_type(ds)
    if record.type == 'EnhancedMRImage':
        return dbdataset.enhanced_mr_image_signal_type(ds)


