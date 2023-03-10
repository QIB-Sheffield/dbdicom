import os
import math

import numpy as np

from dbdicom.record import DbRecord, read_dataframe_from_instance_array
from dbdicom.ds import MRImage
import dbdicom.utils.image as image_utils
from dbdicom.manager import Manager
# import dbdicom.wrappers.scipy as scipy_utils
from dbdicom.utils.files import export_path


class Series(DbRecord):

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
            return [self.record('Instance', uid) for uid in uids]
        else:
            return self.record('Instance', uids)

    def affine_matrix(self):
        return affine_matrix(self)

    def array(*args, **kwargs):
        return get_pixel_array(*args, **kwargs)

    def set_array(*args, **kwargs):
        set_pixel_array(*args, **kwargs)


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


    def export_as_nifti(self, path):
        """Export all images as nii files"""
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
    
    def split_by(self, keyword):
        
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

    #
    # Following APIs are obsolete and will be removed in future versions
    #

    # Obsolete - use array()
    def get_pixel_array(*args, **kwargs): 
        return get_pixel_array(*args, **kwargs)

    # Obsolete - use set_array()
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
    

def get_pixel_array(record, sortby=None, **kwargs): 
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
    return _get_pixel_array_from_sorted_instance_array(source, **kwargs)


def _get_pixel_array_from_instance_array(instance_array, sortby=None, **kwargs):
    source = sort_instance_array(instance_array, sortby)
    return _get_pixel_array_from_sorted_instance_array(source, **kwargs)   


def _get_pixel_array_from_sorted_instance_array(source, pixels_first=False):

    array = []
    instances = source.ravel()
    for i, im in enumerate(instances):
        im.progress(i, len(instances), 'Reading pixel data..')
        if im is None:
            array.append(np.zeros((1,1)))
        else:
            array.append(im.get_pixel_array())
    im.status.message('Reshaping pixel array..')
    array = _stack(array)
    if array is None:
        return None, None
    array = array.reshape(source.shape + array.shape[1:])
    if pixels_first:
        array = np.moveaxis(array, -1, 0)
        array = np.moveaxis(array, -1, 0)
    return array, source 


def set_pixel_array(series, array, source=None, pixels_first=False, **kwargs): 
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
        series.status.progress(i+1, len(source), 'Saving array (1/2): Copying series..')
        if s in instances:
            copy_source.append(s)
        else:
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
    series.manager.pause_extensions()
    for i, image in enumerate(copy_source):
        series.status.progress(i+1, len(copy_source), 'Saving array (2/2): Writing array..')
        image.read()
        for attr, vals in kwargs.items():
            if isinstance(vals, list):
                setattr(image, attr, vals[i])
            else:
                setattr(image, attr, vals)
        image.set_pixel_array(array[i,...])
        image.clear()
    series.manager.resume_extensions()


    # More compact but does not work with pause extensions
    # series.manager.pause_extensions()
    # for i, s in enumerate(source):
    #     series.status.progress(i+1, len(source), 'Writing array..')
    #     if s not in instances:
    #         s.copy_to(series).set_pixel_array(array[i,...])
    #     else:
    #         s.set_pixel_array(array[i,...])
    # series.manager.resume_extensions()





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

