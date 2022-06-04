"""Defines the `Series` class for reading and writing series of DICOM instances.

`Series` provides an interface for reading and sorting images from a series
of instances, extracting subseries of certain type such as magnitude images
in MRI, or mapping one series volume into the coordinate space of another, etc 

    # Example: Get a 3D numpy array from the first series in a folder
    # sorted by slice location and acquisition time

    from dbdicom import Folder

    series = Folder('C:\\Users\\MyName\\MyData\\DICOMtestData').open().series(0)
    array, _ = series.array(['SliceLocation', 'AcquisitionTime'])
"""

__all__ = ['Series']

import numpy as np
import nibabel as nib

from .record import Record


class Series(Record):
    """Programming interface for reading and writing DICOM data series"""

    def __init__(self, folder, UID=[], **attributes):
        """Initialise the series with an instance of Folder.

        Args:
            folder: instance of Folder()
            UID: optional 3-element list with UID's of the patient and study the series is part of.
                The final element is the UID of the series to be created.
                
                If UID is not provided or has less than 3 elements, the missing elements
                are automatically created.
        """
        super().__init__(folder, UID, generation=3, **attributes)

    def label(self, row=None):
        """
        A human-readable label to describe the series.
        """
        if row is None:
            data = self.data()
            if data.empty: return "New Series"
            file = data.index[0]
            descr = data.at[file, 'SeriesDescription']
            nr = data.at[file, 'SeriesNumber']
        else:
            descr = row.SeriesDescription
            nr = row.SeriesNumber
            
        label = '[' + str(nr).zfill(3) + '] ' 
        label += str(descr)
        return label
    
    def map_onto(self, target):
        """Map non-zero pixels onto another series"""

        in_memory = self.in_memory()
        source_images = self.children()
        mapped_series = self.parent.new_child()
        if in_memory: mapped_series.read()
        target_images = target.children() # create record.images() to return children of type image
        for i, target_image in enumerate(target_images):
            target_image.read()
            self.status.progress(i, len(target_images))
            pixel_array = np.zeros((target_image.Rows, target_image.Columns), dtype=np.bool) 
            for j, source_image in enumerate(source_images):
                message = (
                    'Mapping image ' + str(j) + 
                    ' of ' + self.SeriesDescription + 
                    ' to image ' + str(i) + 
                    ' of ' + target.SeriesDescription )
                self.status.message(message)
                if not in_memory: source_image.read()
                array = source_image.map_onto(target_image).array().astype(np.bool)
                np.logical_or(pixel_array, array, out=pixel_array)
            if pixel_array.any():
                mapped_image = target_image.copy_to(mapped_series)
                mapped_image.set_array(pixel_array.astype(np.float32))
                mapped_image.SeriesDescription = self.SeriesDescription
                if not in_memory: mapped_image.write()
        self.status.hide()
        return mapped_series

    def export_as_nifti(self, directory=None, filename=None):
        """Export series as a single Nifty file"""

        if directory is None: 
            directory = self.directory(message='Please select a folder for the nifty data')
        if filename is None:
            filename = self.SeriesDescription
        dicomHeader = nib.nifti1.Nifti1DicomExtension(2, self.instances(0).read())
        pixelArray = np.flipud(np.rot90(np.transpose(self.array())))
        niftiObj = nib.Nifti1Instance(pixelArray, self.instances(0).affine)
        niftiObj.header.extensions.append(dicomHeader)
        nib.save(niftiObj, directory + '/' + filename + '.nii.gz')

    def export_as_csv(self, directory=None, filename=None, columnHeaders=None):
        """Export all images as csv files"""

        if directory is None: 
            directory = self.directory(message='Please select a folder for the csv data')
        if filename is None:
            filename = self.SeriesDescription
        for i, instance in enumerate(self.instances()):
            instance.export_as_csv(self, 
                directory = directory, 
                filename = filename + '(' + str(i) + ')', 
                columnHeaders = columnHeaders)

    def magnitude(self):
        "Creates a sibling series with the magnitude images"

        return self._extractImageType('MAGNITUDE')

    def phase(self):
        "Creates a sibling series with the phase images"

        return self._extractImageType('PHASE')

    def real(self):
        "Creates a sibling series with the real images"

        return self._extractImageType('REAL')

    def imaginary(self):
        "Creates a sibling series with the imaginary images"

        return self._extractImageType('IMAGINARY')

    def _extractImageType(self, type):
        """Extract subseries with images of given imageType"""

        series = self.parent.new_child()
        for instance in self.instances():
            if instance.image_type() == type:
                instance.copy_to(series)
        return series

    def _amax(self, axis=None):
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
        array, data = self.array(axis)
        array = np.amax(array, axis=0)
        data = np.squeeze(data[0,...])
        series = self.new_sibling()
        series.set_array(array, data)
        return series
