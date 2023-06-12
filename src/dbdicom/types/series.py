# Importing annotations to handle or sign in import type hints
from __future__ import annotations

import os
import math

import numpy as np

from dbdicom.record import Record, read_dataframe_from_instance_array
from dbdicom.ds import MRImage
import dbdicom.utils.image as image_utils
from dbdicom.manager import Manager
# import dbdicom.wrappers.scipy as scipy_utils
from dbdicom.utils.files import export_path


class Series(Record):

    name = 'SeriesInstanceUID'

    def _set_key(self):
        self._key = self.keys()[0]

    def remove(self):
        self.manager.delete_series([self.uid])

    def parent(self):
        #uid = self.manager.register.at[self.key(), 'StudyInstanceUID']
        uid = self.manager._at(self.key(), 'StudyInstanceUID')
        return self.record('Study', uid, key=self.key())

    def children(self, **kwargs):
        return self.instances(**kwargs)

    def new_child(self, dataset=None, **kwargs): 
        attr = {**kwargs, **self.attributes}
        return self.new_instance(dataset=dataset, **attr)

    def new_sibling(self, suffix=None, **kwargs):
        if suffix is not None:
            desc = self.manager._at(self.key(), 'SeriesDescription') 
            kwargs['SeriesDescription'] = desc + ' [' + suffix + ']'
        return self.parent().new_child(**kwargs)

    def new_instance(self, dataset=None, **kwargs):
        attr = {**kwargs, **self.attributes}
        uid, key = self.manager.new_instance(parent=self.uid, dataset=dataset, key=self.key(), **attr)
        return self.record('Instance', uid, key, **attr)

    # replace by clone(). Adopt implies move rather than copy
    def adopt(self, instances): 
        uids = [i.uid for i in instances]
        uids = self.manager.copy_to_series(uids, self.uid, **self.attributes)
        if isinstance(uids, list):
            return [self.record('Instance', uid) for uid in uids]
        else:
            return self.record('Instance', uids)        

    def _copy_from(self, record, **kwargs):
        attr = {**kwargs, **self.attributes}
        uids = self.manager.copy_to_series(record.uid, self.uid, **attr)
        if isinstance(uids, list):
            return [self.record('Instance', uid, **attr) for uid in uids]
        else:
            return self.record('Instance', uids, **attr)




    def export_as_npy(self, directory=None, filename=None, sortby=None, pixels_first=False):
        """Export array in numpy format"""

        if directory is None: 
            directory = self.dialog.directory(message='Please select a folder for the png data')
        if filename is None:
            filename = self.SeriesDescription
        array, _ = self.get_pixel_array(sortby=sortby, pixels_first=pixels_first)
        file = os.path.join(directory, filename + '.npy')
        with open(file, 'wb') as f:
            np.save(f, array)


    def export_as_dicom(self, path): 
        # instance = self.instance()
        # patient = "".join([c if c.isalnum() else "_" for c in instance.PatientID])
        # study = "".join([c if c.isalnum() else "_" for c in instance.StudyDescription])
        # series = "".join([c if c.isalnum() else "_" for c in instance.SeriesDescription])
        # path = os.path.join(os.path.join(os.path.join(path, patient), study), series)
        # path = export_path(path)

        folder = self.label()
        path = export_path(path, folder)

        copy = self.copy()
        mgr = Manager(path, status=self.status)
        mgr.open(path)
        mgr.import_datasets(copy.files())
        copy.remove()


    def export_as_png(self, path):
        """Export all images as png files"""
        folder = self.label()
        path = export_path(path, folder)
        images = self.images()
        for i, img in enumerate(images):
            img.status.progress(i+1, len(images), 'Exporting png..')
            img.export_as_png(path)


    def export_as_csv(self, path):
        """Export all images as csv files"""
        folder = self.label()
        path = export_path(path, folder)
        images = self.images()
        for i, img in enumerate(images):
            img.status.progress(i+1, len(images), 'Exporting csv..')
            img.export_as_csv(path)


    def export_as_nifti(self, path: str):
        """Export images in nifti format.

        Args:
            path (str): path where results are to be saved.
        """
        folder = self.label()
        path = export_path(path, folder)
        affine = self.affine_matrix()
        if not isinstance(affine, list):
            affine = [affine]
        for a in affine:
            matrix = a[0]
            images = a[1]
            for i, img in enumerate(images):
                img.status.progress(i+1, len(images), 'Exporting nifti..')
                img.export_as_nifti(path, matrix)


    def subseries(*args, move=False, **kwargs):
        return subseries(*args, move=move, **kwargs)

    
    def split_by(self, keyword: str | tuple) -> list:
        """Split the series into multiple subseries based on keyword value.

        Args:
            keyword (str | tuple): A valid DICOM keyword or hexadecimal (group, element) tag.

        Raises:
            ValueError: if an invalid or missing keyword is provided.
            ValueError: if all images have the same value for the keyword, so no subseries can be derived. An exception is raised rather than a copy of the series to avoid unnecessary copies being made. If that is the intention, use series.copy() instead.

        Returns:
            list: A list of subseries, where each element has the same value of the given keyword.

        Example: 

            Create a single-slice series with multiple flip angles and repetition times:

            >>> coords = {
            ...    'FlipAngle': [2, 15, 30],
            ...    'RepetitionTime': [2.5, 5.0, 7.5],
            ... }
            >>> zeros = db.zeros((3,2,128,128), coords)
            >>> print(zeros)
            ---------- SERIES --------------
            Series 001 [New Series]
                Nr of instances: 6
                    MRImage 000001
                    MRImage 000002
                    MRImage 000003
                    MRImage 000004
                    MRImage 000005
                    MRImage 000006
            --------------------------------

            Splitting this series by FlipAngle now creates 3 new series in the same study, with 2 images each. By default the fixed value of the splitting attribute is written in the series description:

            >>> zeros_FA = zeros.split_by('FlipAngle')
            >>> zeros.study().print()
            ---------- STUDY ---------------
            Study New Study [None]
                Series 001 [New Series]
                    Nr of instances: 6
                Series 002 [New Series[FlipAngle = 2.0]]
                    Nr of instances: 2
                Series 003 [New Series[FlipAngle = 15.0]]
                    Nr of instances: 2
                Series 004 [New Series[FlipAngle = 30.0]]
                    Nr of instances: 2
            --------------------------------
        """
        
        self.status.message('Reading values..')
        try:
            values = self[keyword]
        except:
            msg = str(keyword) + ' is not a valid DICOM keyword'
            raise ValueError(msg)
        if len(values) == 1:
            msg = 'Cannot split by ' + str(keyword) + '\n' 
            msg += 'All images have the same value'
            raise ValueError(msg)
        
        self.status.message('Splitting series..')
        split_series = []
        desc = self.instance().SeriesDescription + '[' + keyword + ' = '
        for v in values:
            kwargs = {keyword: v}
            new = self.subseries(**kwargs)
            new.SeriesDescription = desc + str(v) + ']'
            split_series.append(new)
        return split_series


    def import_dicom(self, files):
        uids = self.manager.import_datasets(files)
        self.manager.move_to(uids, self.uid)

    def slice_groups(*args, **kwargs):
        return slice_groups(*args, **kwargs)


    def affine_matrix(self):
        return affine_matrix(self)
    

    def ndarray(self, dims=('InstanceNumber',)) -> np.ndarray:
        """Return a numpy.ndarray with pixel data.

        Args:
            dims (tuple, optional): Dimensions of the result, as a tuple of valid DICOM tags of any length. Defaults to ('InstanceNumber',).

        Returns:
            np.ndarray: pixel data. The number of dimensions will be 2 plus the number of elements in dim. The first two indices will enumerate (x,y) coordinates in the slice, the other dimensions are as specified by the dims argument.

        See also:
            :func:`~set_ndarray`

        Example:
            Create a zero-filled array, describing 8 MRI slices each measured at 3 flip angles and 2 repetition times:

            >>> coords = {
            ...    'SliceLocation': np.arange(8),
            ...    'FlipAngle': [2, 15, 30],
            ...    'RepetitionTime': [2.5, 5.0],
            ... }
            >>> zeros = db.zeros((128,128,8,3,2), coords)

            To retrieve the array, the dimensions need to be provided:

            >>> dims = ('SliceLocation', 'FlipAngle', 'RepetitionTime')
            >>> array = zeros.ndarray(dims)
            >>> print(array.shape)
            (128, 128, 8, 3, 2)

            The dimensions are the keys of the coordinate dictionary, so this could also have been called as:

            >>> array = zeros.ndarray(dims=tuple(coords)) 
            >>> print(array.shape)
            (128, 128, 8, 3, 2)
        """
        array, _ = get_pixel_array(self, sortby=list(dims), first_volume=True, pixels_first=True)
        return array


    def set_ndarray(self, array:np.ndarray, dims=('InstanceNumber',), coords:dict=None):
        """Assign new pixel data with a new numpy.ndarray. 

        Args:
            array (np.ndarray): array with new pixel data.
            dims (tuple, optional): Dimensions of the result, as a tuple of valid DICOM tags of any length. Defaults to ('InstanceNumber',). Must be provided if coords are not given.
            coords (dict, optional): Provide coordinates for the array explicitly, using a dictionary with dimensions as keys and as values either 1D or meshgrid arrays of coordinates. If coords are not provided, then dimensions a default range array will be used. If coordinates are provided, then the dimensions argument is ignored.

        Raises:
            ValueError: if dimensions and coordinates are both provided with incompatible dimensions.

        See also:
            :func:`~ndarray`

        Warning:
            Currently this function assumes that the new array has the same shape as the current array. This will be generalised in an upcoming update - for now please look at the pipelines examples for saving different dimensions using the current interface. 

        Example:
            Create a zero-filled array, describing 8 MRI slices each measured at 3 flip angles and 2 repetition times:

            >>> coords = {
            ...    'SliceLocation': np.arange(8),
            ...    'FlipAngle': [2, 15, 30],
            ...    'RepetitionTime': [2.5, 5.0],
            ... }
            >>> series = db.zeros((128,128,8,3,2), coords)

            Retrieve the array and check that it is populated with zeros:

            >>> array = series.ndarray(dims=tuple(coords)) 
            >>> print(np.mean(array))
            0.0

            Now overwrite the values with a new array of ones. Coordinates are not changed so only dimensions need to be specified:

            >>> ones = np.ones((128,128,8,3,2))
            >>> series.set_ndarray(ones, dims=tuple(coords))

            Retrieve the array and check that it is now populated with ones:

            >>> array = series.ndarray(dims=tuple(coords)) 
            >>> print(np.mean(array))
            1.0
        """
        # TODO: Include a reshaping option!!!!
        
        # TODO: set_pixel_array has **kwargs to allow setting other properties on the fly to save extra reading and writing. This makes sense but should be handled by a more general function, such as:
        # #  
        # series.set_properties(ndarray:np.ndarray, coords:{}, affine:np.ndarray, **kwargs)
        # #

        # Lazy solution - first get the header information (slower than propagating explicitly but conceptually more convenient - can be rationalised later - pixel values can be set on the fly as the header is retrieved)

        # If coordinates are provided, the dimensions are taken from that. Dimensions are not needed in this case but if they are set they need to be the same as those specified in the coordinates. Else an error is raised.
        if coords is not None:
            if dims != tuple(coords):
                msg = 'Coordinates do not have the correct dimensions \n'
                msg += 'Note: if coordinates are defined than the dimensions argument is ignored. Hence you can remove the dimensions argument in this call, or else make sure it matches up with the dimensions in coordinates.'
                raise ValueError(msg)
            else:
                dims = tuple(coords)
        _, headers = get_pixel_array(self, sortby=list(dims), first_volume=True, pixels_first=True)
        set_pixel_array(self, array, source=headers, pixels_first=True, coords=coords)


    #
    # Following APIs are obsolete and will be removed in future versions
    #


    def array(*args, **kwargs):
        return get_pixel_array(*args, **kwargs)

    def set_array(*args, **kwargs):
        set_pixel_array(*args, **kwargs)

    def get_pixel_array(*args, **kwargs): 
        return get_pixel_array(*args, **kwargs)

    def set_pixel_array(*args, **kwargs):
        set_pixel_array(*args, **kwargs)






def slice_groups(series): # not yet in use
    slice_groups = []
    for orientation in series.ImageOrientationPatient:
        sg = series.instances(ImageOrientationPatient=orientation)
        slice_groups.append(sg)
    return slice_groups

def subseries(record, move=False, **kwargs):
    """Extract subseries"""
    series = record.new_sibling()
    instances = record.instances(**kwargs)
    for i, instance in enumerate(instances):
        record.status.progress(i+1, len(instances), 'Extracting subseries..')
        if move:
            instance.move_to(series)
        else:
            instance.copy_to(series)
    # This should be faster:
    # instances = record.instances(**kwargs)
    # series.adopt(instances)
    return series

def read_npy(record):
    # Not in use - loading of temporary numpy files
    file = record.manager.npy()
    if not os.path.exists(file):
        return
    with open(file, 'rb') as f:
        array = np.load(f)
    return array


def affine_matrix(series):
    """Returns the affine matrix of a series.
    
    If the series consists of multiple slice groups with different 
    image orientations, then a list of affine matrices is returned,
    one for each slice orientation.
    """
    image_orientation = series.ImageOrientationPatient
    if image_orientation is None:
        msg = 'ImageOrientationPatient not defined in the DICOM header \n'
        msg = 'This is a required DICOM field \n'
        msg += 'The data may be corrupted - please check'
        raise ValueError(msg)
    # Multiple slice groups in series - return list of affine matrices
    if isinstance(image_orientation[0], list):
        affine_matrices = []
        for dir in image_orientation:
            slice_group = series.instances(ImageOrientationPatient=dir)
            affine = _slice_group_affine_matrix(slice_group, dir)
            affine_matrices.append((affine, slice_group))
        return affine_matrices
    # Single slice group in series - return a single affine matrix
    else:
        slice_group = series.instances()
        affine = _slice_group_affine_matrix(slice_group, image_orientation)
        return affine, slice_group


def _slice_group_affine_matrix(slice_group, image_orientation):
    """Return the affine matrix of a slice group"""

    # single slice
    if len(slice_group) == 1:
        return slice_group[0].affine_matrix
    # multi slice
    else:
        pos = [s.ImagePositionPatient for s in slice_group]
        # Find unique elements
        pos = [x for i, x in enumerate(pos) if i==pos.index(x)]

        # One slice location
        if len(pos) == 1: 
            return slice_group[0].affine_matrix
        
        # Slices with different locations
        else:
            return image_utils.affine_matrix_multislice(
                image_orientation, pos,
                slice_group[0].PixelSpacing)    # assume all the same pixel spacing


def array(record, **kwargs):
    if isinstance(record, list): # array of instances
        arr = np.empty(len(record), dtype=object)
        for i, rec in enumerate(record):
            arr[i] = rec
        return _get_pixel_array_from_instance_array(arr, **kwargs)
    elif isinstance(record, np.ndarray): # array of instances
        return _get_pixel_array_from_instance_array(record, **kwargs)
    else:
        return get_pixel_array(record, **kwargs)
    

def get_pixel_array(record, sortby=None, first_volume=False, **kwargs): 
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

    source = instance_array(record, sortby)
    array, headers = _get_pixel_array_from_sorted_instance_array(source, **kwargs)
    if first_volume:
        return array[...,0], headers[...,0]
    else:
        return array, headers


def _get_pixel_array_from_instance_array(instance_array, sortby=None, **kwargs):
    source = sort_instance_array(instance_array, sortby)
    return _get_pixel_array_from_sorted_instance_array(source, **kwargs)   


def _get_pixel_array_from_sorted_instance_array(source, pixels_first=False):

    array = []
    instances = source.ravel()
    im = None
    for i, im in enumerate(instances):
        if im is None:
            array.append(np.zeros((1,1)))
        else:
            im.progress(i+1, len(instances), 'Reading pixel data..')
            array.append(im.get_pixel_array())
    if im is not None:
        im.status.hide()
    array = _stack(array)
    if array is None:
        msg = 'Pixel array is empty. \n'
        msg += 'Either because one or more of the keywords used for sorting does not exist; \n'
        msg += 'or the series does not have any image data..'
        raise ValueError(msg)
    array = array.reshape(source.shape + array.shape[1:])
    if pixels_first:
        array = np.moveaxis(array, -1, 0)
        array = np.moveaxis(array, -1, 0)
    return array, source 


def set_pixel_array(series, array, source=None, pixels_first=False, coords=None, **kwargs): 
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

    # Move pixels to the end (default)
    if pixels_first:    
        array = np.moveaxis(array, 0, -1)
        array = np.moveaxis(array, 0, -1)

    # If source data are provided, then coordinates are optional. 
    # If no source data are given, then coordinates MUST be defined to ensure array data can be retrieved in the proper order..
    if source is None:
        if coords is None:
            if array.ndim > 4:
                msg = 'For arrays with more than 4 dimensions, \n'
                msg += 'either coordinate labels or headers must be provided'
                raise ValueError(msg)
            elif array.ndim == 4:
                coords = {
                    'SliceLocation':np.arange(array.shape[0]),
                    'AcquisitionTime':np.arange(array.shape[1]),
                }
            elif array.ndim == 3:
                coords = {
                    'SliceLocation':np.arange(array.shape[0]),
                }

    # If coordinates are given as 1D arrays, turn them into grids and flatten for iteration.
    if coords is not None:
        v0 = list(coords.values())[0]
        if np.array(v0).ndim==1: # regular grid
            pos = tuple([coords[c] for c in coords])
            pos = np.meshgrid(*pos)
            for i, c in enumerate(coords):
                coords[c] = pos[i].ravel()

    # if no header data are provided, use template headers.
    nr_of_slices = int(np.prod(array.shape[:-2]))
    if source is None:
        source = [series.new_instance(MRImage()) for _ in range(nr_of_slices)]

    # If the header data are not the same size, use only the first one.
    else:
        if isinstance(source, list):
            pass
        elif isinstance(source, np.ndarray):
            source = source.ravel().tolist()
        else: # assume scalar
            source = [source] * nr_of_slices
        if nr_of_slices != len(source):
            source = [source[0]] * nr_of_slices

    # Copy all sources to the series, if they are not part of it
    copy_source = []
    instances = series.instances()
    for i, s in enumerate(source):
        if s in instances:
            copy_source.append(s)
        else:
            series.progress(i+1, len(source), 'Copying series..')
            copy_source.append(s.copy_to(series))

    # Faster but does not work if all sources are the same
    # series.status.message('Saving array (1/2): Copying series..')
    # instances = series.instances()
    # to_copy = [i for i in range(len(source)) if source[i] not in instances]
    # copied = series.adopt([source[i] for i in to_copy])
    # for i, c in enumerate(copied):
    #     source[to_copy[i]] = c

    # Flatten array for iterating
    array = array.reshape((nr_of_slices, array.shape[-2], array.shape[-1])) # shape (i,x,y)
    for i, image in enumerate(copy_source):
        series.progress(i+1, len(copy_source), 'Saving array..')
        image.read()

        for attr, vals in kwargs.items(): 
            if isinstance(vals, list):
                setattr(image, attr, vals[i])
            else:
                setattr(image, attr, vals)

        # If coordinates are provided, these will override the values from the sources.
        if coords is not None: # ADDED 31/05/2023
            for c in coords:
                image[c] = coords[c][i]
        image.set_pixel_array(array[i,...])
        image.clear()



    # More compact but does not work with pause extensions
    # for i, s in enumerate(source):
    #     series.status.progress(i+1, len(source), 'Writing array..')
    #     if s not in instances:
    #         s.copy_to(series).set_pixel_array(array[i,...])
    #     else:
    #         s.set_pixel_array(array[i,...])






##
## Helper functions
##

def sort_instance_array(instance_array, sortby=None, status=True):
    if sortby is None:
        return instance_array
    else:
        if not isinstance(sortby, list):
            sortby = [sortby]
        df = read_dataframe_from_instance_array(instance_array, sortby + ['SOPInstanceUID'])
        df.sort_values(sortby, inplace=True) 
        return df_to_sorted_instance_array(instance_array[0], df, sortby, status=status)
        

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
        if not isinstance(sortby, list):
            sortby = [sortby]
        df = record.read_dataframe(sortby + ['SOPInstanceUID'])
        df.sort_values(sortby, inplace=True) 
        return df_to_sorted_instance_array(record, df, sortby, status=status)


def df_to_sorted_instance_array(record, df, sortby, status=True): 
    # note record here only passed for access to the function instance() and progress()
    # This really should be db.instance()

    data = []
    vals = df[sortby[0]].unique()
    for i, c in enumerate(vals):
        if status: 
            record.progress(i, len(vals), message='Sorting pixel data..')
        # if a type is not supported by np.isnan()
        # assume it is not a nan
        try: 
            nan = np.isnan(c)
        except: 
            nan = False
        if nan:
            dfc = df[df[sortby[0]].isnull()]
        else:
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
    for i, item in enumerate(df.SOPInstanceUID.items()):
        data[i] = record.instance(key=item[0])
    return data


def _stack(arrays, align_left=False):
    """Stack a list of arrays of different shapes but same number of dimensions.
    
    This generalises numpy.stack to arrays of different sizes.
    The stack has the size of the largest array.
    If an array is smaller it is zero-padded and centred on the middle.
    None items are removed first before stacking
    """

    # Get the dimensions of the stack
    # For each dimension, look for the largest values across all arrays
    arrays = [a for a in arrays if a is not None]
    if arrays == []:
        return
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

