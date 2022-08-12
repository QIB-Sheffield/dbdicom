import os
import numpy as np
import nibabel as nib
import pandas as pd

import matplotlib.pyplot as plt

import dbdicom.utils.image as image
import dbdicom.utils.pydicom as pydcm


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
    return pydcm.colormap(ds)

def get_lut(record):

    if record.generation < 4:
        return
    ds = record.read()
    return pydcm.lut(ds)


def set_colormap(record, *args, **kwargs):
    """Set the colour table of the image."""

    if record.generation < 4:
        return
    dataset = record.read()
    pydcm.set_colormap(dataset.to_pydicom(), *args, **kwargs)
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
        return pydcm.mr_image_type(ds)
    if record.type == 'EnhancedMRImage':
        return pydcm.enhanced_mr_image_type(ds)


def signal_type(record):
    """Determine if an image is Water, Fat, In-Phase, Out-phase image or None"""

    ds = record.read().to_pydicom()

    if record.type == 'MRImage':
        return pydcm.mr_image_signal_type(ds)
    if record.type == 'EnhancedMRImage':
        return pydcm.enhanced_mr_image_signal_type(ds)


