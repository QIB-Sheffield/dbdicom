# Importing annotations to handle or sign in import type hints
from __future__ import annotations

import os
import math
from numbers import Number

import numpy as np
import pandas as pd
import nibabel as nib

from dbdicom.record import Record, read_dataframe_from_instance_array
from dbdicom.ds import MRImage
import dbdicom.utils.image as image_utils
from dbdicom.manager import Manager
# import dbdicom.extensions.scipy as scipy_utils
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

    def export_as_dicom(self, path): 
        folder = self.label()
        path = export_path(path, folder)
        # Create a copy so that exported datasets have different UIDs.
        copy = self.copy()
        mgr = Manager(path, status=self.status)
        mgr.open(path)
        for i in copy.instances():
            ds = i.get_dataset()
            mgr.import_dataset(ds)
        copy.remove()

    def export_as_png(self, path, **kwargs):
        #Export all images as png files
        folder = self.label()
        path = export_path(path, folder)
        images = self.images()
        for i, img in enumerate(images):
            img.progress(i+1, len(images), 'Exporting png..')
            img.export_as_png(path, **kwargs)

    def export_as_csv(self, path):
        #Export all images as csv files
        folder = self.label()
        path = export_path(path, folder)
        images = self.images()
        for i, img in enumerate(images):
            img.progress(i+1, len(images), 'Exporting csv..')
            img.export_as_csv(path)

    def export_as_npy(self, path, dims=None):
        if dims is None:
            folder = self.label()
            path = export_path(path, folder)
            images = self.images()
            for i, img in enumerate(images):
                img.progress(i+1, len(images), 'Exporting npy..')
                img.export_as_npy(path)
        else:
            array = self.pixel_values(dims)
            filepath = self.label()
            filepath = os.path.join(path, filepath + '.npy')
            with open(filepath, 'wb') as f:
                np.save(f, array)

    def export_as_nifti(self, path, dims=None):
        if dims is None:
            folder = self.label()
            path = export_path(path, folder)
            affine = self.affine_matrix()
            if not isinstance(affine, list):
                affine = [affine]
            for a in affine:
                matrix = a[0]
                images = a[1]
                for i, img in enumerate(images):
                    img.progress(i+1, len(images), 'Exporting nifti..')
                    img.export_as_nifti(path, matrix)
        else:
            ds = self.instance().get_dataset()
            sgroups = self.slice_groups(dims=dims)
            for i, sg in enumerate(sgroups):
                self.progress(i+1, len(sgroups), 'Exporting nifti..')
                dicom_header = nib.nifti1.Nifti1DicomExtension(2, ds)
                nifti1_image = nib.Nifti1Image(sg['ndarray'], image_utils.affine_to_RAH(sg['affine']))
                nifti1_image.header.extensions.append(dicom_header)
                filepath = self.label()
                filepath = os.path.join(path, filepath + '[' + str(i) + '].nii')
                nib.save(nifti1_image, filepath)

    def import_dicom(self, files):
        uids = self.manager.import_datasets(files)
        self.manager.move_to(uids, self.uid)




    def _frames(self, dims:tuple=None, coords:dict=None, return_coords=False):
        """Return the frames of given coordinates in the correct order"""
        
        if coords is None:
            if dims is None:
                dims = ('InstanceNumber', )

            # Read and sort coordinates
            frames_sel = self.instances()
            fvalues = [f[list(dims)] for f in frames_sel]
            values = _meshvals(np.array(fvalues).T)
            shape = values.shape[1:]
            values = values.reshape((values.shape[0],-1))
             
            # Look up coordinates of each frame.
            values_ind = []
            for fv in fvalues:
                for iv in range(values.shape[1]):
                    if np.array_equal(fv, values[:,iv]):
                        values_ind.append(iv)
                        break 

            # If requested, build coordinates
            if return_coords:
                coords = {}
                for i, c in enumerate(dims):
                    coords[c] = values[i,:].reshape(shape)         

        else:
            # Check if the coordinates are well-defined and get the values.
            coords = _check_if_coords(coords)
            shape = _coords_shape(coords)
            values = _coords_vals(coords)
            dims = tuple(coords)

            # Look up coordinates of each frame.
            frames = self.instances()
            values_ind = []
            frames_sel = []
            for f in frames:
                fv = f[list(dims)]
                for iv in range(values.shape[1]):
                    if np.array_equal(fv, values[:,iv]):
                        frames_sel.append(f)
                        values_ind.append(iv)
                        break

        # Create object array of frames, sort and reshape
        frames = np.empty(len(frames_sel), dtype=object)
        for i, fi in enumerate(frames_sel):
            frames[i] = fi
        frames = frames[np.argsort(values_ind)]
        frames = frames.reshape(shape)

        if return_coords:
            return frames, coords
        else:
            return frames


    def coords(self, dims:tuple=None)->dict:
        """return a dictionary of coordinates.

        Args:
            dims (tuple, optional): Dimensions along which the shape is to be determined. If dims is not provided, they default to InstanceNumber. 

        Raises:
            ValueError: If the dimensions do not produce suitable coordinates.

        Returns:
            dict: dictionary of coordinates, one entry for each dimension. The values for each coordinate are returned as an darray with one dimension.

        See also:
            `set_coords`

        Example:

            Create an empty series with 3 slice dimensions:

            >>> coords = {
            ...     'SliceLocation': np.array([0,1,2,0,1,2]),
            ...     'FlipAngle': np.array([2,2,2,10,10,10]),
            ...     'RepetitionTime': np.array([1,5,15,1,5,15]),
            ... }
            >>> series = db.empty_series(coords)
            
            Retrieve the coordinates:

            >>> coords = series.coords(tuple(coords))
            >>> coords['FlipAngle']
            [2,10,2,10,2,10]
            >>> coords['RepetitionTime']
            [1,1,5,5,15,15]

            Check the result in default dimensions:

            >>> coords = series.coords()
            >>> coords['InstanceNumber']
            [1,2,3,4,5,6]

            In this case the slice location and flip angle along are sufficient to identify the frames, so these are valid coordinates:

            >>> coords = series.coords(('SliceLocation', 'FlipAngle'))
            >>> coords['SliceLocation']
            [0,0,1,1,2,2]

            # However slice location and acquisition time are not sufficient as coordinates because each combination appears twice. So this throws an error:

            >>> series.coords(('SliceLocation','RepetitionTime'))
            ValueError: These are not proper coordinates. Coordinate values must be unique.
        """
        if dims is None:
            dims = ('InstanceNumber', )

        # Read coordinate values
        values = [inst[list(dims)] for inst in self.instances()]

        # Sort and create array
        values.sort()
        values = np.array(values).T

        # Convert array to dictionary
        coords = {}
        for d, dim in enumerate(dims):
            coords[dim] = values[d,:]

        # return if valid
        return _check_if_coords(coords)


    def set_coords(self, new_coords:dict, dims:tuple=None, coords:dict=None):
        """Set a dictionary of coordinates.

        Args:
            new_coords (dict): Dictionary of coordinates.
            dims (tuple, optional): Dimensions of at which the new coordinates are to be best. If *dims* is not set, the dimensions are assumed to be the same as those of *coords* or *grid*. Defaults to None.
            coords (tuple, optional): Locations at which the new coordinates are to be set. If not provided, all frames recieve a new value for the coordinates

        Raises:
            ValueError: if the coordinates provided are not properly formatted or have the wrong shape.

        See also:
            `coords`
            `set_gridcoords`

        Example:

            Create an empty series:

            >>> coords = {
            ...     'SliceLocation': np.array([0,1,2,0,1,2]),
            ...     'FlipAngle': np.array([2,2,2,10,10,10]),
            ...     'RepetitionTime': np.array([1,5,15,1,5,15]),
            ... }
            >>> series = db.empty_series(coords)
            
            Change the flip angle of 15 to 12:

            >>> coords = series.coords(tuple(coords))
            >>> fa = coords['FlipAngle']
            >>> fa[np.where(fa==2)] = 5
            >>> series.set_coords(coords)

            Check the new coordinates:

            >>> new_coords = series.coords(dims)
            >>> new_coords['FlipAngle']
            [5,10,5,10,5,10]

            Create a new set of coordinates along slice location and acquisition time:

            >>> new_coords = {
            ...     'SliceLocation': np.array([0,0,1,1,2,2]),
            ...     'AcquisitionTime': np.array([0,60,0,60,0,60]),
            ... }
            >>> series.set_coords(new_coords, ('SliceLocation', 'FlipAngle'))

            # Inspect the new coordinates - each slice now has two acquisition times corresponding to the flip angles:

            >>> coords['SliceLocation']
            [0,0,1,1,2,2]
            >>> coords['AcquisitionTime']
            [0,60,0,60,0,60]
            >>> coords['FlipAngle']
            [5,10,5,10,5,10]

            # Check that an error is raised if coordinate values have different sizes:
            >>> new_coords = {
            ...     'SliceLocation': np.zeros(24),
            ...     'AcquisitionTime': np.ones(25),
            ... }
            >>> series.set_coords(new_coords, dims)
            ValueError: Coordinate values must all have the same size

            # An error is also raised if they have all the same size but the values are not unique:

            >>> new_coords = {
            ...     'SliceLocation': np.zeros(24),
            ...     'AcquisitionTime': np.ones(24),
            ... }
            >>> series.set_coords(new_coords, dims)
            ValueError: Coordinate values must all have the same size

            # .. or when the number does not match up with the size of the series:

            >>> new_coords = {
            ...     'SliceLocation': np.arange(25),
            ...     'AcquisitionTime': np.arange(25),
            ... }
            >>> series.set_coords(new_coords, dims)
            ValueError: Shape of coordinates does not match up with the size of the series.

        """
        if coords is None:
            if dims is None:
                dims = tuple(new_coords)
        new_coords = _check_if_coords(new_coords)
        frames, coords = self._frames(dims=dims, coords=coords, return_coords=True)
        if _coords_size(new_coords) != _coords_size(coords):
            msg = 'Cannot set coordinates - new coordinates have a different size.'
            raise ValueError(msg)
        frames = frames.flatten()
        values = _coords_vals(new_coords)
        for f, frame in enumerate(frames):
            frame[list(new_coords)] = list(values[:,f])


    def meshcoords(self, dims:tuple=None)->dict:
        """return a dictionary of coordinates.

        Args:
            dims (tuple, optional): Dimensions along which the shape is to be determined. If dims is not provided, they default to InstanceNumber. 

        Raises:
            ValueError: If the dimensions do not produce suitable coordinates.

        Returns:
            dict: dictionary of coordinates, one entry for each dimension. The values for each coordinate are returned as an darray with one dimensions (mesh=False) or the same number of dimensions as *dims* (mesh=True).

        See also:
            `set_coords`

        Example:

            Create a zero-filled array with 3 slice dimensions:

            >>> coords = {
            ...     'SliceLocation': np.arange(4),
            ...     'FlipAngle': [2, 15, 30],
            ...     'RepetitionTime': [2.5, 5.0] }
            >>> series = db.zeros((128,128,4,3,2), coords)
            
            Retrieve the meshed coordinates:

            >>> coords = series.coords(tuple(coords), mesh=True)
            >>> coords['FlipAngle'].shape
            (4, 3, 2)
            >>> coords['SliceLocation'][1,1,1]
            1
            >>> coords['FlipAngle'][1,1,1]
            15
            >>> coords['RepetitionTime'][1,1,1]
            5

            Note that an error is thrown if invalid dimensions are provided:

            >>> coords = series.coords(('SliceLocation', 'FlipAngle'))
            ValueError: series shape is ambiguous in dimensions (SliceLocation, FlipAngle, )
            --> Multiple frames exist at some or all locations.
            --> Hint: use Series.unique() to list the values at all locations.

            The coordinates of the flat series are the instance numbers:
            >>> series.coords()
            {'InstanceNumber': array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])}
        """
        coords = self.coords(dims)
        return _as_meshcoords(coords)


    def set_meshcoords(self, new_coords:dict, dims:tuple=None, coords:dict=None):
        """Set a dictionary of coordinates.

        Args:
            new_coords (dict): Dictionary of coordinates.
            dims (tuple, optional): Dimensions of at which the new coordinates are to be best. If *dims* is not set, the dimensions are assumed to be the same as those of *coords* or *grid*. Defaults to None.
            coords (tuple, optional): Locations at which the new coordinates are to be set. If not provided, all frames recieve a new value for the coordinates

        Raises:
            ValueError: if the coordinates provided are not properly formatted or have the wrong shape.

        See also:
            `coords`
            `set_gridcoords`

        Example:

            Create a zero-filled array:

            >>> grid = {
            ...     'SliceLocation': np.arange(4),
            ...     'FlipAngle': np.array([2, 15, 30]),
            ...     'RepetitionTime': np.array([2.5, 5.0]) }
            >>> series = db.zeros((128,128,4,3,2), grid=grid)
            
            Change the flip angle of 15 to 12:

            >>> coords = series.coords(tuple(coords))
            >>> coords['FlipAngle'][:,1,:] = 12
            >>> series.set_coords(coords)

            Check the new coordinates:

            >>> new_coords = series.coords(tuple(coords))
            >>> fa = new_coords['FlipAngle'][:,1,:]
            >>> np.unique(fa)
            [12.]

            Create a new set of coordinates along slice location and acquisition time:

            >>> loc = np.arange(4)
            >>> tacq = 60*np.arange(6)
            >>> loc, tacq = np.meshgrid((loc, tacq), indexing='ij)
            >>> grid = {
            ...     'SliceLocation': ,
            ...     'AcquisitionTime': ,
            ... }
            
            Set the new coordinates and check the result:

            >>> series.set_coords(coords, dims)
            >>> c = series.coords(tuple(coords))
            >>> np.unique(c['AcquisitionTime'])
            [  0.  60. 120. 180. 240. 300.]

            # Check that an error is raised if coordinate values have different sizes:
            >>> new_coords = {
            ...     'SliceLocation': np.zeros(24),
            ...     'AcquisitionTime': np.ones(25),
            ... }
            >>> series.set_coords(new_coords, dims)
            ValueError: Coordinate values must all have the same size

            # An error is also raised if they have all the same size but the values are not unique:

            >>> new_coords = {
            ...     'SliceLocation': np.zeros(24),
            ...     'AcquisitionTime': np.ones(24),
            ... }
            >>> series.set_coords(new_coords, dims)
            ValueError: Coordinate values must all have the same size

            # .. or when the number does not match up with the size of the series:

            >>> new_coords = {
            ...     'SliceLocation': np.arange(25),
            ...     'AcquisitionTime': np.arange(25),
            ... }
            >>> series.set_coords(new_coords, dims)
            ValueError: Shape of coordinates does not match up with the size of the series.

        """
        new_coords = _mesh_to_coords(new_coords)
        self.set_coords(new_coords, dims=dims, coords=coords)



    def gridcoords(self, dims)->dict:
        """return a dictionary of grid coordinates.

        Args:
            dims (tuple): Attributes to be used as coordinates.

        Returns:
            dict: dictionary of coordinates, one entry for each dimension.

        See also:
            `coords`
            `set_gridcoords`

        Examples:

            Create a zero-filled array with 3 slice dimensions:

            >>> coords = {
            ...     'SliceLocation': np.arange(4),
            ...     'FlipAngle': [2, 15, 30],
            ...     'RepetitionTime': [2.5, 5.0] }
            >>> series = db.zeros((128,128,8,3,2), coords)

            Recover the grid coordinates:

            >>> coords_rec = series.gridcoords(tuple(coords))
            >>> coords_rec['SliceLocation']
            [0. 1. 2. 3. 4. 5. 6. 7.]
            >>> coords_rec['FlipAngle']
            [ 2. 15. 30.]
            >>> coords_rec['RepetitionTime']
            [2.5 5. ]
        """
        coords = {}
        for attribute in dims:
            coords[attribute] = self.unique(attribute)
        return coords


    def set_gridcoords(self, gridcoords:dict, dims:tuple=None, coords:dict=None):
        """ Set a dictionary of grid coordinates.

        Args:
            coords (dict): dictionary of grid coordinates
            dims (tuple, optional): provide the current dimensions of the series. If not provided, they are assumed to be the same as the dimensions of the new coordinates. Defaults to None.

        See also:
            `gridcoords`
            `set_coords`

        Examples:

            Create a zero-filled array with 3 slice dimensions:

            >>> coords = {
            ...     'SliceLocation': np.arange(4),
            ...     'FlipAngle': [2, 15, 30],
            ...     'RepetitionTime': [2.5, 5.0] }
            >>> series = db.zeros((128,128,8,3,2), coords)
            >>> dims = tuple(gridcoords)

            Change the flip angle of 15 to 12:

            >>> coords = series.gridcoords(dims)
            >>> coords['FlipAngle'][1] = 12
            >>> series.set_gridcoords(coords)

            Check coordinates:

            >>> new_coords = series.coords(dims)
            >>> fa = new_coords['FlipAngle'][:,1,:]
            >>> np.unique(fa)
            [12]

            Or, alternatively, check in the grid coordinates:

            >>> new_coords = series.gridcoords(dims)
            >>> new_coords['FlipAngle'][1]
            12

            Create a new set of coordinates along slice location and acquisition time:

            >>> new_coords = {
            ...     'SliceLocation': np.arange(8),
            ...     'AcquisitionTime': 60*np.arange(6),
            ... }
            
            Set the new coordinates and check the result:

            >>> series.set_gridcoords(new_coords, dims)
            >>> c = series.gridcoords(tuple(new_coords))
            >>> c['AcquisitionTime']
            [  0.  60. 120. 180. 240. 300.]
        """

        c = _grid_to_coords(gridcoords)
        self.set_coords(c, dims=dims, coords=coords)



    def expand(self, coords:dict=None, grid:dict=None):

        if coords is not None:
            new_coords = coords
        elif grid is not None:
            new_coords = _grid_to_coords(grid)

        new_size = _coords_size(new_coords)

        # If the series is not empty, first check that the new coordinates are valid.
        if not self.empty():
            current_coords = self.coords(tuple(new_coords))
            total_coords = _concatenate_coords((current_coords, new_coords))
            if _coords_size(total_coords) != _coords_size(current_coords) + new_size:
                msg = 'Cannot expand - the new coordinates overlap with existing coordinates.'
                raise ValueError(msg)
        
        # Expand the series to the new coordinates
        for i in range(new_size):
            ds = self.init_dataset()
            for c in new_coords:
                ds.set_values(c, new_coords[c][i])
            self.new_instance(ds)


    def shape(self, dims:tuple=None)->tuple:
        """Return the shape of the series along given dimensions.

        Args:
            dims (tuple, optional): Dimensions along which the shape is to be determined. If dims is not provided, the shape of the flattened series is returned. Defaults to None.
        
        Returns:
            tuple: one value for each element of dims.
        
        Raises:
            ValueError: if the shape in the specified dimensions is ambiguous (because the number of slices is not unique at each location) 
            ValueError: if the shape in the specified dimensions is not well defined (because there is no slice at one or more locations).

        See also:
            `coords`
            `gridcoords`
            `spacing`

        Example:

            Create a zero-filled series with 3 dimensions.

            >>> coords = {
            >>>     'SliceLocation': np.arange(4),
            >>>     'FlipAngle': [2, 15, 30],
            >>>     'RepetitionTime': [2.5, 5.0] }
            >>> series = db.zeros((128,128,4,3,2), coords)

            Check the shape of a flattened series:
            >>> series.shape()
            (24,)

            Check the shape along all 3 dimensions:

            >>> dims = tuple(coords)
            >>> series.shape(dims)
            (4, 3, 2)

            Swap the first two dimensions:

            >>> series.shape((dims[1], dims[0], dims[2]))
            (3, 4, 2)

            Determine the shape along another DICOM attribute:

            >>> series.shape(('FlipAngle', 'InstanceNumber'))
            (3, 8)

            The shape of an empty series is zero along any dimension:

            >>> series.new_sibling().shape(dims)
            (0, 0, 0)

            If one or more of the dimensions is not defined in the header, this raises an error:

            >>> series.shape(('FlipAngle', 'Gobbledigook'))
            ValueError: series shape is not well defined in dimensions (FlipAngle, Gobbledigook, )
            --> Some of the dimensions are not defined in the header.
            --> Hint: use Series.value() to find the undefined values.

            An error is also raised if the values are defined, but are not unique. In this case, all acquisition times are the same so this raises an error:

            >>> series.shape(('FlipAngle', 'AcquisitionTime'))
            ValueError: series shape is ambiguous in dimensions (FlipAngle, AcquisitionTime, )
            --> Multiple slices exist at some or all locations.
            --> Hint: use Series.unique() to list the values at all locations.

        """
        array = _instances(self, dims)
        return array.shape



    

    def values(self, tag:str|tuple, dims:tuple=None, inds:dict=None, select={}, **filters)->np.ndarray:
        """Return the values of an attribute along a given dimension.

        Args:
            tag (str or tuple): either a keyword string or a (group, element) tag of a DICOM data element.
            dims (tuple, optional): Dimensions of the resulting array. If *dims* is not provided, values are ordered by InstanceNumber. Defaults to None.
            inds (dict, optional): Dictionary with indices to retrieve a slice of the entire array. Defaults to None.
            select (dict, optional): A dictionary of values for DICOM attributes to filter the result. By default the data are not filtered.
            filters (dict, optional): keyword arguments to filter the data by value of DICOM attributes.

        Returns:
            An `numpy.ndarray` of values with dimensions as specified by *dims*. If the value is not defined in *one or more* of the slices, an empty array is returned.

        See also:
            `unique`
            `coords`
            `gridcoords`

        Note:
            In order to list the values in the case one or more are absent in the headers, use `Series.unique()` instead.

        Example:

            Create a zero-filled series with 3 slice dimensions:

            >>> coords = {
            ...     'SliceLocation': 10*np.arange(4),
            ...     'FlipAngle': np.array([2, 15, 30]),
            ...     'RepetitionTime': np.array([2.5, 5.0]), }
            >>> zeros = db.zeros((128,128,4,3,2), coords)

            # If values() is called without dimensions, a flat array is returned with one value per frame, ordered by instance number:

            >>> zeros.values('InstanceNumber')
            [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,191,20,21,22,23,24]
            >>> zros.values('FlipAngle')
            [2,2,15,15,30,30,2,2,15,15,30,30,2,2,15,15,30,30,2,2,15,15,30,30]

            if dimensions are provided, an array of the appropriate shape is returned:

            >>> dims = tuple(coords)
            >>> tacq = series.values('AcquisitionTime', dims)
            >>> tacq.shape
            (4,3,2)
            >>> tacq[0,0,0]
            28609.057496

            In this case all values are the same:

            >>> np.unique(tacq)
            [28609.057496]

            If a value is not defined in the header, None is returned:
            >>> series.values('Gobbledigook')[:2]
            [None None]

            Specify keywords to select a subset of values:

            >>> tacq = zeros.values('AcquisitionTime', dims, FlipAngle=15)
            >>> tacq.shape
            (4, 1, 2)

            If none exist, and emptry array is returned:

            >>> tacq = zeros.values('AcquisitionTime', dims, FlipAngle=0)
            >>> tacq.size
            0

            Multiple possible values can be selected with arrays:

            >>> tacq = zeros.values('AcquisitionTime', dims, FlipAngle=np.array([15,30]))
            >>> tacq.shape
            (4, 2, 2)

            Any number of keywords can be added as filters:

            >>> tacq = zeros.values('AcquisitionTime', dims, FlipAngle=np.array([15,30]), SliceLocation=np.array([10,20]))
            >>> tacq.shape
            (2, 2, 2)

            Filters can alos be set using the *select* argument: 

            >>> tacq = zeros.values('AcquisitionTime', dims, select={'FlipAngle': 15})
            >>> tacq.shape
            (4, 1, 2)

            This also allows (group, element) tags:

            >>> tacq = zeros.values('AcquisitionTime', dims, select={(0x0018, 0x1314): 15})
            >>> tacq.shape
            (4, 1, 2)

            Selections can also be made using indices rather than values:

            >>> tacq = zeros.values('FlipAngle', dims, inds={'FlipAngle': 1})
            >>> tacq.shape
            (4, 1, 2) 

            >>> tacq = zeros.values('AcquisitionTime', dims, inds={'FlipAngle':np.arange(2)})
            >>> tacq.shape
            (4, 2, 2)
        """
        if dims is None:
            dims = ('InstanceNumber', )
        array = _instances(self, dims=dims, inds=inds, select=select, **filters)
        shape = array.shape
        array = array.ravel()
        value = [frame[tag] for frame in array]
        return np.array(value).reshape(shape)

        # # If no dimensions are provided, return the single unique value.
        # # Raise an error if multiple are detected.
        # if dims is None:
        #     v = self.unique(attribute)
        #     v = np.array([x for x in v if x is not None])
        #     if len(v)==0:
        #         return None
        #     elif len(v)==1:
        #         return v[0]
        #     else: 
        #         msg = 'Multiple unique values detected for ' + str(attribute) + '.'
        #         msg += '\n--> Use Series.unique() to list the unique values.'
        #         raise ValueError(msg)

        # # If dims is provided, create an array of values
        # source = instance_array(self, sortby=list(dims))
        # if source.size == 0:
        #     return None
        # shape = source.shape
        # source = source.ravel()
        # value = np.array([s[attribute] for s in source])
        # value = np.reshape(value, shape)

        # # If there is only one instance at each location, return the array of values.
        # d = shape[-1]
        # if d == 1:
        #     return value[...,0]
        
        # # If there are multiple instances at some locations, find the unique value at each location. Raise an error if multiple unique values are found.
        # n = np.prod(shape[:-1])
        # value = value.reshape((n, d))
        # v = np.empty((n,))
        # for i in range(n):
        #     vi = np.unique(value[i,:])
        #     vi = np.array([x for x in vi if x is not None])
        #     if len(vi)==0:
        #         v[i] = None
        #     elif len(vi)==1:
        #         v[i] = vi[0]
        #     else:
        #         r = np.unravel_index(i, shape[:-1])
        #         loc = '('
        #         for x in range(len(r)):
        #             loc += str(dims[x]) + ' = ' + str(r[x])
        #             if x != len(r)-1:
        #                 loc += ', '
        #         loc += ')'
        #         msg = 'Multiple unique values detected for ' + str(attribute) + ' at ' + loc + '.'
        #         msg += '\n--> Use Series.unique() to list the unique values at each location.'
        #         raise ValueError(msg) 
        # return np.reshape(v, shape[:-1])


    def set_values(self, tag:str|tuple, value, dims:tuple=None, inds:dict=None, select={}, **filters):
        """Set the values of an attribute.

        Args:
            tag: either a keyword string or a (group, element) tag of a DICOM data element.
            value: a single value or a numpy array of values for the attribute. 
            dims (tuple, optional): Dimensions of *value*. If *value* is a single value, *dims* is ignored. Otherwise, if *dim* is not provided, values are ordered by instance number. Defaults to None.

        Raises: 
            ValueError: if the size of *value* does not match the size of the series.

        See also:
            `value`

        Example:

            Create a zero-filled series with 3 slice dimensions.

            >>> loc = np.arange(4)
            >>> fa = [2, 15, 30]
            >>> tr = [2.5, 5.0]
            >>> coords = {
            ...     'SliceLocation': np.arange(4),
            ...     'FlipAngle': [2, 15, 30],
            ...     'RepetitionTime': [2.5, 5.0] }
            >>> series = db.zeros((128,128,8,3,2), coords)

            Change the acquisition time of the series to midnight (0 sec):

            >>> series.value('AcquisitionTime')
            28609.057496
            >>> series.set_value('AcquisitionTime', 0)
            >>> series.value('AcquisitionTime')
            0

            Set the acquisition time to a different value for each flip angle:

            >>> tacq = np.repeat(60*np.arange(3), 8)
            >>> series.set_value('AcquisitionTime', tacq, dims=('FlipAngle','InstanceNumber'))

            Set the acquisition time to a different value for each flip angle and acquisition time:

            >>> tacq = np.repeat(60*np.arange(6), 4)
            >>> series.set_value('AcquisitionTime', tacq, dims=('FlipAngle','RepetitionTime','SliceLocation'))

            Note: the size of the value and of the series need to match up. If not, an error is raised:

            >>> series.set_value('AcquisitionTime', np.arange(25), dims=tuple(coords))
            ValueError: The size of the value array is different from the size of the series.
            The value array has shape (25,), but the series has shape (4, 3).

        """  
        # if the value is not an array, set as a constant
        if not isinstance(value, np.ndarray):
            for s in self.instances():
                s[tag] = value
            return
        
        if dims is None:
            dims = ('InstanceNumber', )

        array = _instances(self, dims=dims, inds=inds, select=select, **filters)
        if value.size != array.size:
            msg = 'The size of the value array is different from the size of the series.'
            msg += '\nThe value array has shape ' + str(value.shape) + ', '
            msg += 'but the series has shape ' + str(array.shape) + '.'
            raise ValueError(msg)
        array = array.ravel()
        value = value.ravel()
        for f, frame in enumerate(array):
            frame[tag] = value[f]


    def unique(self, tag:str|tuple, sortby:tuple=None) -> np.ndarray:
        """Return the unique values of an attribute, sorted by any number of variables.

        Args:
            tag: either a keyword string or a (group, element) tag of a DICOM data element.
            sortby (tuple, optional): Dimensions of the resulting array. If *sortby* is not provided, then an array of unique values is returned.

        Returns:
            np.ndarray: a sorted array of unique values of the attribute, with dimensions as specified by *dims*. If *dims* is provided, the result has the dimensions of *dims* and each element of the array is an array unique values.

        See also:
            `value`
            `unique_affines`
            `coords`
            `gridcoords`

        Example:
            Create a zero-filled series with 3 slice dimensions:

            >>> loc = np.arange(4)
            >>> fa = [2, 15, 30]
            >>> tr = [2.5, 5.0]
            >>> coords = {
            ...     'SliceLocation': np.arange(4),
            ...     'FlipAngle': [2, 15, 30],
            ...     'RepetitionTime': [2.5, 5.0] }
            >>> series = db.zeros((128,128,8,3,2), coords)

            Recover the unique values of any coordinate, such as the flip angle:

            >>> series.value('FlipAngle')
            [ 2. 15. 30.]

            List the flip angles for each slice location separately:

            >>> fa = series.unique('FlipAngle', sortby=('SliceLocation', ))
            >>> fa[0]
            [ 2. 15. 30.]
            >>> fa[3]
            [ 2. 15. 30.]

            List the flip angles for each slice location and repetition time:

            >>> fa = series.unique('FlipAngle', sortby=('SliceLocation', 'RepetitionTime'))
            >>> fa.shape
            (4, 2)
            >>> fa[1,1]
            [ 2. 15. 30.]

            Getting the values for a non-existing attribute produces an empty array:

            >>> gbbl = series.unique('Gobbledigook')
            >>> gbbl.size
            0
            >>> gbbl.shape
            (0,)

            Getting a non-existing attribute for each slice location produces an array of the expected shape, where each element is an empty array:

            >>> gbbl = series.unique('Gobbledigook', sortby=('SliceLocation',))
            >>> gbbl.shape
            (4,)
            >>> gbbl.size
            4
            >>> gbbl[-1].size
            0
        """
        # If no sorting is required, return an array of unique values
        if sortby is None:
            source = self.instances()
            value = [s[tag] for s in source]
            value = [x for x in value if x is not None]
            if value == []:
                return np.array([])
            else:
                return np.unique(value)
        
        # If sorting is required, return an array with arrays of unique values.
        source = instance_array(self, sortby=list(sortby))
        if source.size == 0:
            shape = tuple([0]*len(sortby))
            return np.array([]).reshape(shape)
        shape = source.shape
        source = source.ravel()
        value = np.array([s[tag] for s in source])
        
        # Return an array of unique values at each location.
        n = np.prod(shape[:-1])
        d = shape[-1]
        value = value.reshape((n, d))
        v = np.empty(n, dtype=np.ndarray)
        for i in range(n):
            vi = [x for x in value[i,:] if x is not None]
            if vi == []:
                v[i] = np.array([])
            else:
                v[i] = np.unique(vi)
        return np.reshape(v, shape[:-1])
    

    def pixel_values(self, dims:tuple=None, inds:dict=None, select={}, **filters) -> np.ndarray:
        """Return a numpy.ndarray with pixel data.

        Args:
            dims (tuple, optional): Dimensions of the result, as a tuple of valid DICOM tags of any length. If *dims* is not provided, pixel values are ordered by instance number. Defaults to None.
            inds (dict, optional): Dictionary with indices to retrieve a slice of the entire array. Defaults to None.
            select (dict, optional): A dictionary of values for DICOM attributes to filter the result. By default the data are not filtered.
            filters (dict, optional): keyword arguments to filter the data by value of DICOM attributes.

        Returns:
            np.ndarray: pixel data. The number of dimensions will be 2 plus the number of elements in *dim*. The first two indices will enumerate (column, row) indices in the slice, the other dimensions are as specified by the *dims* argument. 
            
            The function returns an empty array when no data are found at the specified locations.

        Raises:
            ValueError: Indices must be in the dimensions provided. If *ind* is set but keys are not part of *dims*.
            ValueError: if the images are different shapes.

        See also:
            `set_pixel_values`

        Example:
            Create a zero-filled array with 3 slice dimensions:

            >>> coords = {
            ...    'SliceLocation': 10*np.arange(4),
            ...    'FlipAngle': np.array([2, 15, 30]),
            ...    'RepetitionTime': np.array([2.5, 5.0]),
            ... }
            >>> zeros = db.zeros((128,64,4,3,2), coords)

            Retrieve the pixel array of the series:

            >>> dims = tuple(coords)
            >>> array = zeros.pixel_values(dims)
            >>> array.shape
            (128, 64, 4, 3, 2)

            To retrieve an array containing only the data with flip angle 15:

            >>> array = zeros.pixel_values(dims, FlipAngle=15)
            >>> array.shape
            (128, 64, 4, 1, 2)

            If no data fit the requirement, and empty array is returned:

            >>> array = zeros.pixel_values(dims, FlipAngle=15)
            >>> array.size
            0

            Multiple possible values can be specified as an array:

            >>> array = zeros.pixel_values(dims, FlipAngle=np.array([15,30]))
            >>> array.shape
            (128, 64, 4, 2, 2)

            And multiple filters can be specified by adding keyword arguments. The following returns an array of pixel values with flip angle of 15 or 30, and slice location of 10 or 20:

            >>> array = zeros.pixel_values(dims, FlipAngle=np.array([15,30]), SliceLocation=np.array([10,20]))
            >>> array.shape
            (128, 64, 2, 2, 2)

            The filters can be any DICOM attribute:

            >>> array = zeros.pixel_values(dims, AcquisitionTime=0)
            >>> array.size
            0

            The filters can also be specified as a dictionary of values:

            >>> array = zeros.pixel_values(dims, select={'FlipAngle': 15})
            >>> array.shape
            (128, 64, 4, 1, 2)

            Since keywords need to be strings in python, this is the only way to specify filters with (group, element) tags:

            >>> array = zeros.pixel_values(dims, select={(0x0018, 0x1314): 15})
            >>> array.shape
            (128, 64, 4, 1, 2)

            Using the *inds* argument, the pixel array can be indexed to avoid reading a large array if only a subarray is required:

            >>> array = zeros.pixel_values(dims, inds={'FlipAngle': 1})
            >>> array.shape
            (128, 64, 4, 1, 2)

            Note unlike filters defind by *value*, the indices must be provided in the dimensions of the array. If not, a `ValueError` is raised:

            >>> zeros.pixel_values(dims, inds={'AcquisitionTime':0})
            ValueError: Indices must be in the dimensions provided.
        """
        frames = _instances(self, dims=dims, inds=inds, select=select, **filters)
        if frames.size == 0:
            shape = (0,0) + frames.shape
            return np.array([]).reshape(shape)
        fshape = frames.shape
        frames = frames.ravel()
        array = []
        for i, im in enumerate(frames):
            self.progress(i+1, len(frames), 'Reading pixel values..')
            array.append(im.get_pixel_array())
        ashape = np.array([a.shape for a in array])
        ashape = ashape.reshape((2, len(array)))
        ashape = np.unique(ashape, axis=0)
        if len(ashape) > 1:
            msg = 'Cannot extract an array of pixel values - not all frames have the same size.'
            raise ValueError(msg)
        array = np.stack(array, axis=-1)
        return array.reshape(array.shape[:2] + fshape)
    

    def set_pixel_values(self, value:np.ndarray, dims:tuple=None, inds:dict=None, select={}, **filters):
        """Set a numpy.ndarray with pixel data.

        Args:
            dims (tuple, optional): Dimensions of the pixel values, as a tuple of valid DICOM tags of any length. If *dims* is not provided, pixel values are ordered by instance number. Defaults to None.
            inds (dict, optional): Dictionary with indices to set a slice of the entire array. Defaults to None.
            select (dict, optional): A dictionary of values for DICOM attributes to set specific frames. 
            filters (dict, optional): keyword arguments to set specific frames.

        Raises:
            ValueError: if the values are the incorrect shape for the dimensions.

        See also:
            `pixel_values`

        Example:
            Create a zero-filled array with 3 slice dimensions:

            >>> coords = {
            ...    'SliceLocation': 10*np.arange(4),
            ...    'FlipAngle': np.array([2, 15, 30]),
            ...    'RepetitionTime': np.array([2.5, 5.0]),
            ... }
            >>> zeros = db.zeros((128,64,4,3,2), coords)
        """
        if dims is None:
            dims = ('InstanceNumber', )
        frames = _instances(self, dims=dims, inds=inds, select=select, **filters)
        if np.prod(value.shape[2:]) != frames.size:
            msg = 'The size of the pixel value array is different from the size of the series.'
            msg += '\nThe pixel array has shape ' + str(value.shape[2:]) + ', '
            msg += 'but the series has shape ' + str(frames.shape) + '.'
            raise ValueError(msg)
        frames = frames.ravel()
        value = value.reshape(value.shape[:2] + (-1))
        for f, frame in enumerate(frames):
            self.progress(i+1, frames.size, 'Writing pixel values..')
            frame.set_pixel_array(value[:,:,f])
    

    def affine(self)->np.ndarray:
        """Return the affine of the Series.

        Raises:
            ValueError: if the DICOM file is corrupted
            ValueError: if the affine is not unique.

        Returns:
            np.ndarray: affine matrix as a 4x4 numpy array.

        See also:
            `set_affine`
            `unique_affines`

        Example:
            Check that the default affine is the identity:

            >>> zeros = db.zeros((128,128,10))
            >>> zeros.affine()
            [[1., 0., 0., 0.],
             [0., 1., 0., 0.],
             [0., 0., 1., 0.],
             [0., 0., 0., 1.]]
        """
        image_orientation = self.ImageOrientationPatient
        if image_orientation is None:
            msg = 'ImageOrientationPatient not defined in the DICOM header \n'
            msg += 'This is a required DICOM field \n'
            msg += 'The data may be corrupted - please check'
            raise ValueError(msg)
        
        # Multiple slice groups in series - raise Exception
        if isinstance(image_orientation[0], list):
            msg = 'The series has multiple affines. '
            msg += '\nUse Series.unique_affines() to return an array of unique affines.'
            raise ValueError(msg)
        
        # Single slice group in series - return a single affine matrix
        slice_group = self.instances()
        return _slice_group_affine_matrix(slice_group, image_orientation)


    def set_affine(self, affine:np.eye()):
        """Set the affine matrix of a series.

        The affine is defined as a 4x4 numpy array with bottom row [0,0,0,1]. The final column represents the position of the top right hand corner of the first slice. The first three columns represent rotation and scaling with respect to the axes of the reference frame.

        Args:
            affine (numpy.ndarray): 4x4 numpy array 

        Raises:
            ValueError: if the series is empty. The information of the affine matrix is stored in the header and can not be stored in an empty series.

        See also:
            `affine`
            `unique_affines`

        Example:
            Create a series with unit affine array:

            >>> zeros = db.zeros((128,128,10))
            >>> zeros.affine()
            [[1., 0., 0., 0.],
             [0., 1., 0., 0.],
             [0., 0., 1., 0.],
             [0., 0., 0., 1.]]

            Rotate the volume over 90 degrees in the xy-plane:

            >>> affine = np.array([
            ...     [1., 0., 0., 0.],
            ...     [0., 1., 0., 0.],
            ...     [0., 0., 1., 0.],
            ...     [0., 0., 0., 1.],
            ... ]) 
            >>> zeros.set_affine(affine)

            Apart from the rotation, also change the resolution to (3mm, 3mm, 1.5mm):

            >>> affine = np.array([
            ...     [0., -3., 0., 0.],
            ...     [3., 0., 0., 0.],
            ...     [0., 0., 1.5, 0.],
            ...     [0., 0., 0., 1.],
            ... ])  
            >>> zeros.set_affine(affine)

            Now rotate, change resolution, and shift the top right hand corner of the lowest slice to position (-30mm, 20mm, 120mm):

            >>> affine = np.array([
            ...     [0., -3., 0., -30.],
            ...     [3., 0., 0., 20.],
            ...     [0., 0., 1.5, 120.],
            ...     [0., 0., 0., 1.],
            ... ])  
            >>> zeros.set_affine(affine)

            Note: changing the affine will affect multiple DICOM tags, such as slice location and image positions:

            >>> zeros.SliceLocation
            [120.0, 121.5, 123.0, 124.5, 126.0, 127.5, 129.0, 130.5, 132.0, 133.5]

            In this case, since the slices are stacked in parallel to the z-axis, the slice location starts at the lower z-coordinate of 120mm and then increments slice-by-slice with the slice thickness of 1.5mm.
        
        """
        images = instance_array(self, sortby='SliceLocation')
        if images.size == 0:
            msg = 'Cannot set affine matrix in an empty series \n'
            msg += 'Set some data with series.pixel_values() and then try again.'
            raise ValueError(msg)
    
        # For each slice location, the slice position needs to be updated too
        # Need the coordinates of the vector parallel to the z-axis of the volume.
        a = image_utils.dismantle_affine_matrix(affine)
        ez = a['SpacingBetweenSlices']*np.array(a['slice_cosine'])

        # Set the affine slice-by-slice
        nz = images.shape[0]
        for d in range(images.shape[1]):
            affine_z = affine.copy()
            for z in range(nz):
                self.progress(z+1, nz, 'Writing affine..')
                affine_z[:3, 3] = affine[:3, 3] + z*ez
                images[z,d].read()
                images[z,d].affine_matrix = affine_z
                images[z,d].clear()

    
    def isnull(self, tag, dims:tuple) -> np.ndarray:
        array = _instances(self, dims)
        shape = array.shape
        array = array.ravel()
        null = [a[tag] is None for a in array]
        return np.array(null, dtype=bool).reshape(shape) 
      
    
    def split_by(self, keyword: str | tuple) -> list:
        """Split the series into multiple subseries based on keyword value.

        Args:
            keyword (str | tuple): A valid DICOM keyword or hexadecimal (group, element) tag.

        Raises:
            ValueError: if an invalid or missing keyword is provided.
            ValueError: if all images have the same value for the keyword, so no subseries can be derived. An exception is raised rather than a copy of the series to avoid unnecessary copies being made. If that is the intention, use series.copy() instead.

        Returns:
            list: A list of ``Series`` instances, where each element has the same value of the given keyword.

        See Also:
            `slice`
            `islice`

        Example: 

            Create a single-slice series with multiple flip angles and repetition times:

            >>> coords = {
            ...    'FlipAngle': [2, 15, 30],
            ...    'RepetitionTime': [2.5, 7.5],
            ... }
            >>> zeros = db.zeros((128, 128, 3, 2), coords)
            >>> zeros.print()
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

            >>> FA = zeros.split_by('FlipAngle')
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

            Check the flip angle of the split series:
            >>> for series in FA: 
            ...     print(series.FlipAngle)
            2.0
            15.0
            30.0
        """
        
        self.message('Reading values..')
        try:
            values = self[keyword]
        except:
            msg = str(keyword) + ' is not a valid DICOM keyword'
            raise ValueError(msg)
        if not isinstance(values, list):
            msg = 'Cannot split by ' + str(keyword) + '\n' 
            msg += 'All images have the same value'
            raise ValueError(msg)   
        if len(values) == 1:
            msg = 'Cannot split by ' + str(keyword) + '\n' 
            msg += 'All images have the same value'
            raise ValueError(msg)
        
        self.message('Splitting series..')
        split_series = []
        desc = self.instance().SeriesDescription + '[' + keyword + ' = '
        for v in values:
            kwargs = {keyword: v}
            new = self.subseries(**kwargs)
            new.SeriesDescription = desc + str(v) + ']'
            split_series.append(new)
        return split_series


    def spacing(self)->tuple:
        """3D pixel spacing in mm

        Returns:
            tuple: (x-spacing, y-spacing, z-spacing)

        See also:
            `shape`

        Examples:
            Check the spacing of a digital reference object:

            >>> series = db.dro.T1_mapping_vFATR()
            >>> series.spacing()
            (15, 15, 20)
        """
        affine = self.affine()
        column_spacing = np.linalg.norm(affine[:3, 0])
        row_spacing = np.linalg.norm(affine[:3, 1])
        slice_spacing = np.linalg.norm(affine[:3, 2])
        return column_spacing, row_spacing, slice_spacing
        

    def unique_affines(self)->np.ndarray:
        """Return the array of unique affine matrices.

        Raises:
            ValueError: if the DICOM file is corrupted.

        Returns:
            np.ndarray: array of 4x4 ndarrays with the unique affine matrices of the series.

        See also:
            `set_affine`
            `affine`

        Example:
            Check that the default affine is the identity:

            >>> zeros = db.zeros((128,128,10))
            >>> zeros.affine()
            [array([
                [1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.]], dtype=float32)]
        """
        image_orientation = self.ImageOrientationPatient
        if image_orientation is None:
            msg = 'ImageOrientationPatient not defined in the DICOM header \n'
            msg += 'This is a required DICOM field \n'
            msg += 'The data may be corrupted - please check'
            raise ValueError(msg)
        # Multiple slice groups in series - return list of affine matrices
        if isinstance(image_orientation[0], list):
            affine_matrices = []
            for dir in image_orientation:
                slice_group = self.instances(ImageOrientationPatient=dir)
                affine = _slice_group_affine_matrix(slice_group, dir)
                affine_matrices.append(affine)
            return np.unique(affine_matrices)
        # Single slice group in series - return a list with a single affine matrix
        else:
            slice_group = self.instances()
            affine = _slice_group_affine_matrix(slice_group, image_orientation)
            return np.array([affine])
        



    def slice(self, coordinates={}, **coords) -> Series:
        """Get a slice of the series by dimension values

        Args:
            coordinates (dict, optional): dictionary of tag:value pairs where the value is either a single value or an array of values.
            coords (dict): Provide coordinates for the slice, either as dimension=value pairs, or as a dictionary where the keys list the dimensions, and the values are provided as scalars, 1D or meshgrid arrays of coordinates. 

        See also:
            `islice`
            `split_by`

        Example:
            Create a zero-filled array, describing 8 MRI images each measured at 3 flip angles and 2 repetition times:

            >>> coords = {
            ...    'SliceLocation': np.arange(8),
            ...    'FlipAngle': [2, 15, 30],
            ...    'RepetitionTime': [2.5, 5.0],
            ... }
            >>> series = db.zeros((128,128,8,3,2), coords)

            Slice the series at flip angle 15:

            >>> fa15 = series.slice(FlipAngle=15)

            Retrieve the array and check the dimensions:

            >>> array = fa15.pixel_values(dims=tuple(coords))
            >>> print(array.shape)
            (128, 128, 8, 1, 2)

            Multiple possible values can be specified as a list or np.ndarray:

            >>> fa15 = series.slice(SliceLocation=[0,5], FlipAngle=15)
            >>> array = fa15.pixel_values(dims=tuple(coords))
            >>> print(array.shape)
            (128, 128, 2, 1, 2)

            Values can also be provided as a dictionary, which is useful for instance for private tags that do not have a keyword string. So the following are equivalent:

            >>> fa15 = series.slice(SliceLocation=[0,5], FlipAngle=15)
            >>> fa15 = series.slice({SliceLocation:[0,5], FlipAngle:15})
            >>> fa15 = series.slice({(0x0020, 0x1041):[0,5], (0x0018, 0x1314):15})
        """

        coords = {**coordinates, **coords}

        # Check whether the arguments are valid, and initialize dims.
        if coords == {}:
            return self.new_sibling()
        dims = list(coords.keys())
        source = instance_array(self, sortby=dims)
        
        # Retrieve the instances corresponding to the coordinates.
        if source.size != 0:
            for d, dim in enumerate(coords):
                ind = []
                for i in range(source.shape[d]):
                    si = source.take(i,axis=d).ravel()
                    try:
                        if si[0][dim] in coords[dim]:
                            ind.append(i)
                    except:
                        try:
                            if si[0][dim] == coords[dim]:
                                ind.append(i)
                        except:
                            pass    
                source = source.take(ind, axis=d)
                # Insert dimensions of 1 back in
                if len(ind)==1:
                    source = np.expand_dims(source, axis=d)

        result = self.new_sibling()
        source = source.ravel()
        for i in range(source.size):
            source[i].copy_to(result)
        return result
    

    def islice(self, indices={}, **inds) -> Series:
        """Get a slice of the series by dimension indics

        Args:
            indices (dict, optional): Dictionary with tag:value pairs, where the values are either a single index or an array of indices.
            inds (dict, optional): Provide indices for the slice, either as keyword=index pairs or as a dictionary. The indices must be provided either as a scalar, a list or a numpy array.

        Raises:
            IndexError: when the indices in inds are out of range of the existing coordinates. 

        See also:
            `slice`
            `split_by`

        Example:
            Create a zero-filled array, describing 8 MRI images each measured at 3 flip angles and 2 repetition times:

            >>> coords = {
            ...    'SliceLocation': np.arange(8),
            ...    'FlipAngle': [2, 15, 30],
            ...    'RepetitionTime': [2.5, 5.0],
            ... }
            >>> series = db.zeros((128,128,8,3,2), coords)

            Slice the series at flip angle 15 (i.e. index 1):

            >>> fa15 = series.islice(FlipAngle=1)

            Retrieve the array and check the dimensions:

            >>> array = fa15.pixel_values(dims=tuple(coords))
            >>> print(array.shape)
            (128, 128, 8, 1, 2)

            Multiple possible indices can be specified as a list or np.ndarray:

            >>> fa15 = series.slice(SliceLocation=[0,5], FlipAngle=1)
            >>> array = fa15.pixel_values(dims=tuple(coords))
            >>> print(array.shape)
            (128, 128, 2, 1, 2)

            Values can also be provided as a dictionary, which is useful for instance for private tags that do not have a keyword string. So the following are equivalent:

            >>> fa15 = series.slice(SliceLocation=[0,5], FlipAngle=1)
            >>> fa15 = series.slice({SliceLocation:[0,5], FlipAngle:1})
            >>> fa15 = series.slice({(0x0020, 0x1041):[0,5], (0x0018, 0x1314):1})

        """
        inds = {**indices, **inds}

        # Check whether the arguments are valid, and initialize dims.
        if inds == {}:
            return self.new_sibling()
        dims = list(inds.keys())
        source = instance_array(self, sortby=dims)
        
        # Retrieve the instances of the slice.
        for d, dim in enumerate(inds):
            ind = inds[dim]
            try:
                source = source.take(ind, axis=d)
                # Insert dimensions of 1 back in
                if isinstance(ind, Number):
                    source = np.expand_dims(source, axis=d)
            except IndexError as e:
                msg = str(e) + '\n'
                msg += 'The indices for ' + str(dim) + ' in the inds argument are out of bounds'
                raise IndexError(msg)
            
        result = self.new_sibling()
        source = source.ravel()
        for i in range(source.size):
            source[i].copy_to(result)
        return result


    #
    # Following APIs are obsolete and will be removed in future versions
    #


    def _old_set_pixel_values(self, array:np.ndarray, coords:dict=None, inds:dict=None):
        """Assign new pixel data with a new numpy.ndarray. 

        Args:
            array (np.ndarray): array with new pixel data.
            coords (dict, optional): Provide coordinates for the array, using a dictionary where the keys list the dimensions, and the values are provided as 1D or meshgrid arrays of coordinates. If data already exist at the specified coordinates, these will be overwritten. If not, the new data will be added to the series.
            inds (dict, optional): Provide a slice of existing data that will be overwritten with the new array. The format is the same as the dictionary of coordinates, except that the slice is identified by indices rather than values. 

        Raises:
            ValueError: if neither coords or inds or provided, if both are provided, or if the dimensions in coords or inds does not match up with the dimensions of the array.
            IndexError: when attempting to set a slice in an empty array, or when the indices in inds are out of range of the existing coordinates. 

        See also:
            `pixel_values`

        Example:
            Create a zero-filled array, describing 8 MRI images each measured at 3 flip angles and 2 repetition times:

            >>> coords = {
            ...    'SliceLocation': np.arange(8),
            ...    'FlipAngle': [2, 15, 30],
            ...    'RepetitionTime': [2.5, 5.0],
            ... }
            >>> series = db.zeros((128,128,8,3,2), coords)

            Retrieve the array and check that it is populated with zeros:

            >>> array = series.pixel_values(dims=tuple(coords)) 
            >>> print(np.mean(array))
            0.0

            Now overwrite the values with a new array of ones in a new shape:

            >>> new_shape = (128,128,8)
            >>> new_coords = {
            ...     'SliceLocation': np.arange(8),
            ... }
            >>> ones = np.ones(new_shape)
            >>> series.set_pixel_values(ones, coords=new_coords)

            Retrieve the new array and check shape:
            
            >>> array = series.pixel_values(dims=tuple(new_coords))
            >>> print(array.shape)
            (128,128,8)

            Check that the value is overwritten:

            >>> print(np.mean(array))
            1.0
        """

        # Check whether the arguments are valid, and initialize dims.
        cnt = 0
        if coords is not None:
            cnt+=1
            dims = tuple(coords)
            if len(dims) != array.ndim-2:
                msg = 'One coordinate must be specified for each dimensions in the array.'
                raise ValueError(msg)
            for d, dim in enumerate(coords):
                if len(coords[dim]) != array.shape[d+2]:
                    msg = str(dim) + ' in the coords must have the same number of elements as the corresponding dimension in the array'
                    raise ValueError(msg)
        if inds is not None:
            cnt+=1
            dims = tuple(inds)
            if len(dims) != array.ndim-2:
                msg = 'One coordinate must be specified for each dimensions in the array.'
                raise ValueError(msg)
        if cnt == 0:
            msg = 'At least one of the optional arguments coords or inds must be provided'
            raise ValueError(msg)
        if cnt == 2:
            msg = 'Only one of the optional arguments coords or inds must be provided'
            raise ValueError(msg)

        source = instance_array(self, sortby=list(dims))
        
        if coords is not None:
            # Retrieve the instances corresponding to the coordinates.
            if source.size != 0:
                for d, dim in enumerate(coords):
                    ind = []
                    for i in range(source.shape[d]):
                        si = source.take(i,axis=d).ravel()
                        if si[0][dim] in coords[dim]:
                            ind.append(i)
                    source = source.take(ind, axis=d)
                    # Insert dimensions of 1 back in
                    if len(ind)==1:
                        source = np.expand_dims(source, axis=d)
        elif inds is not None:
            # Retrieve the instances of the slice, as well as their coordinates.
            coords = {}
            for d, dim in enumerate(inds):
                ind = inds[dim]
                if isinstance(ind, np.ndarray):
                    ind = list(ind)
                try:
                    source = source.take(ind, axis=d)
                except IndexError as e:
                    msg = str(e) + '\n'
                    msg += 'The indices for ' + str(dim) + ' in the inds argument are out of bounds'
                    raise IndexError(msg)
                coords[dim] = []
                for i in range(source.shape[d]):
                    si = source.take(i,axis=d).ravel()
                    coords[dim].append(si[0][dim])  

        nr_of_slices = int(np.prod(array.shape[2:]))
        if source.size == 0:
            # If there are not yet any instances at the correct coordinates, they will be created from scratch
            source = [self.new_instance(MRImage()) for _ in range(nr_of_slices)]
            set_pixel_values(self, array, source=source, coords=coords)
        elif array.shape[2:] == source.shape:
            # If the new array has the same shape, use the exact headers.
            set_pixel_values(self, array, source=source.ravel().tolist(), coords=coords)
        else:
            # If the new array has a different shape, use the first header for all and delete all the others
            # This happens when some of the new coordinates are present, but not all.
            # TODO: This is overkill - only fill in the gaps with copies.
            source = source.ravel().tolist()
            for series in source[1:]:
                series.remove()
            source = [source[0]] + [source[0].copy_to(self) for _ in range(nr_of_slices-1)]
            set_pixel_values(self, array, source=source, coords=coords)
    
    def subseries(self, **kwargs)->Series:
        """Extract a subseries based on values of header elements.

        Args:
            kwargs: Any number of valid DICOM (tag, value) keyword arguments.

        Returns:
            Series: a new series as a sibling under the same parent.

        See Also:
            :func:`~split_by`

        Example:

            Create a multi-slice series with multiple flip angles and repetition times:

            >>> coords = {
            ...    'SliceLocation': np.arange(16),
            ...    'FlipAngle': [2, 15, 30],
            ...    'RepetitionTime': [2.5, 5.0, 7.5],
            ... }
            >>> zeros = db.zeros((128, 128, 16, 3, 2), coords)

            Create a new series containing only the data with flip angle 2 and repetition time 7.5:

            >>> volume = zeros.subseries(FlipAngle=2.0, RepetitionTime=7.5)

            Check that the volume series now has two dimensions of size 1:

            >>> array = volume.pixel_values(dims=tuple(coords))
            >>> print(array.shape)
            (128, 128, 16, 1, 1)

            and only one flip angle and repetition time:

            >>> print(volume.FlipAngle, volume.RepetitionTime)
            2.0 7.5

            and that the parent study now has two series:

            >>> volume.study().print()
            ---------- STUDY ---------------
            Study New Study [None]
            Series 001 [New Series]
                Nr of instances: 96
            Series 002 [New Series]
                Nr of instances: 16
            --------------------------------
        """
        return subseries(self, move=False, **kwargs)
    
    def slice_groups(self, dims=('InstanceNumber',)) -> list:
        """Return a list of slice groups in the series.

        In dbdicom, a *slice group* is defined as a series of slices that have the same orientation. It is common for a single series to have images with multiple orientations, such as in localizer series in MRI. For such a series, returning all data in a single array may not be meaningful. 

        Formally, a *slice group* is a dictionary with two entries: 'ndarray' is the numpy.ndarray with the data along the dimensions provided by the dims argument, and 'affine' is the 4x4 affine matrix of the slice group. The function returns a list of such dictionaries, one for each slice group in the series.

        Args:
            dims (tuple, optional): Dimensions for the returned arrays. Defaults to ('InstanceNumber',).

        Returns:
            list: A list of slice groups (dictionaries), one for each slice group in the series.

        Examples:

            >>> series = db.ones((128,128,5,10))
            >>> sgroups = series.slice_groups(dims=('SliceLocation', 'AcquisitionTime'))

            Since there is only one slice group in the series, ``sgroups`` is a list with one element:

            >>> print(len(sgroups))
            1

            The array of the slice group is the entire volume of the series:

            >>> print(sgroups[0]['ndarray'].shape)
            (128, 128, 5, 10)

            And the affine of the series has not changed from the default (identity):

            >>> print(sgroups[0]['affine'])
            [[1. 0. 0. 0.]
             [0. 1. 0. 0.]
             [0. 0. 1. 0.]
             [0. 0. 0. 1.]]

        """
        
        slice_groups = []
        image_orientation = self.ImageOrientationPatient

        # Multiple slice groups in series - return list of cuboids
        if isinstance(image_orientation[0], list):
            for dir in image_orientation:
                slice_group = instance_array(self, ImageOrientationPatient=dir)
                affine = _slice_group_affine_matrix(list(slice_group), dir)
                array, _ = _get_pixel_array_from_instance_array(slice_group, sortby=list(dims), pixels_first=True)
                slice_groups.append({'ndarray': array[...,0], 'affine': affine})
        
        # Single slice group in series - return a list with a single affine matrix
        else:
            slice_group = instance_array(self)
            affine = _slice_group_affine_matrix(list(slice_group), image_orientation)
            array, _ = _get_pixel_array_from_instance_array(slice_group, sortby=list(dims), pixels_first=True)
            slice_groups.append({'ndarray': array[...,0], 'affine': affine})

        return slice_groups
    
    def affine_matrix(self):
        return affine_matrix(self)
    
    def array(*args, **kwargs):
        return get_pixel_array(*args, **kwargs)

    def set_array(*args, **kwargs):
        set_pixel_array(*args, **kwargs)

    def get_pixel_array(*args, **kwargs): 
        return get_pixel_array(*args, **kwargs)

    def set_pixel_array(*args, **kwargs):
        set_pixel_array(*args, **kwargs)

    def ndarray(self, *args, **kwargs):
        return self.pixel_values(*args, **kwargs)

    def set_ndarray(self, *args, **kwargs):
        self.set_pixel_values(*args, **kwargs)


def _coords_shape(coords):
    if coords == {}:
        return (0,)
    shapes = [coords[tag].shape for tag in coords]
    shape = shapes[0]
    for s in shapes[1:]:
        if s != shape:
            msg = 'Coordinate shape is ambiguous - not all coordinates have the same shape.'
            raise ValueError(msg)
    return shapes[0]  

def _coords_size(coords):

    if coords == {}:
        return 0 

    # Coordinate values must a have the same size.
    sizes = np.unique([coords[tag].size for tag in coords])
    if len(sizes) > 1:
        msg = 'These are not proper coordinates. Each coordinate must have the same number of values.'
        raise ValueError(msg)
    return sizes[0]  

def _coords_vals(coords):
    values = [coords[tag].ravel() for tag in coords]
    values = np.stack(values)
    return values
 

def _check_if_coords(coords):

    # Check if coordinates are unique.
    size = _coords_size(coords)
    values = _coords_vals(coords)
    if size != np.unique(values, axis=1).shape[1]:
        msg = 'These are not proper coordinates. Coordinate values must be unique.'
        raise ValueError(msg)
    return coords

def _mesh_to_coords(coords):
    for c in coords:
        coords[c] = coords[c].ravel()
    return _check_if_coords(coords)
    

def _grid_to_meshcoords(gridcoords):

    grid = []
    for c in gridcoords:
        if not isinstance(gridcoords[c], np.ndarray):
            msg = 'Grid coordinates have to be numpy arrays.'
            raise TypeError(msg)
        if len(gridcoords[c].shape) != 1:
            msg = 'Grid coordinates have to be one-dimensionial.'
            raise ValueError(msg)
        if len(np.unique(gridcoords[c])) != len(gridcoords[c]):
            msg = 'Grid coordinates have to be unique.'
            raise ValueError(msg)
        grid.append(gridcoords[c])

    mesh = np.meshgrid(*tuple(grid), indexing='ij')
    meshcoords = {}
    for i, c in enumerate(gridcoords):
        meshcoords[c] = mesh[i]
    _check_if_coords(meshcoords)
    return meshcoords


def _grid_to_coords(grid):
    coords = _grid_to_meshcoords(grid)
    for c in coords:
        coords[c] = coords[c].flatten()
    return coords


def _as_meshcoords(coords):

    # First check that they are proper coordinates
    size = _coords_size(coords) 
    values = [coords[tag].ravel() for tag in coords]
    values = np.stack(values)
    if size != np.unique(values, axis=1).shape[1]:
        msg = 'These are not proper coordinates. Coordinate values must be unique.'
        raise ValueError(msg)
    
    values = _meshvals(values)
    meshcoords = {}
    for i, c in enumerate(coords):
        meshcoords[c] = values[i,...]
    return meshcoords
        

def _meshvals(values):
    # Input array shape: (d, f) with d = nr of dims and f = nr of frames
    # Output array shape: (d, f1,..., fd)
    vals, cnts = np.unique(values[0,:], return_counts=True)
    if len(np.unique(cnts)) > 1:
        msg = 'These are not mesh coordinates.'
        raise ValueError(msg) 
    if values.shape[0] == 1:
        return values
    mesh = []
    for v in vals:
        vind = np.where(values[0,:]==v)[0]
        vmesh = _meshvals(values[1:,vind])
        mesh.append(vmesh)
    mesh = np.stack(mesh, axis=1)
    a = [np.full(mesh.shape[2:], v) for v in vals]
    a = np.stack(a)
    a = np.expand_dims(a,0)
    mesh = np.concatenate((a, mesh))
    return mesh


def _concatenate_coords(coords:tuple, mesh=False):
    concat = {}
    for c in coords[0]:
        concat[c] = coords[0][c].flatten().copy()
    for coord in coords[1:]: 
        for c in coord:
            if c not in concat:
                msg = 'Cannot concatenate - all coordinates must have the same variables.'
                raise ValueError(msg)
            concat[c] = np.concatenate((concat[c], coord[c].flatten()))
    _check_if_coords(concat)
    if mesh:
        return _as_meshcoords(concat)
    else:
        return concat





def set_pixel_values(series, array, source=None, coords=None, **kwargs): 

    # If coordinates are given as 1D arrays, turn them into grids and flatten for iteration.
    if coords is not None:
        mesh_coords = {}
        v = list(coords.values())
        if v != []:
            v0 = v[0]
            if np.array(v0).ndim==1: # regular grid
                pos = tuple([coords[c] for c in coords])
                pos = np.meshgrid(*pos, indexing='ij')
                for i, c in enumerate(coords):
                    mesh_coords[c] = pos[i].ravel()

    # Flatten array for iterating
    nr_of_slices = int(np.prod(array.shape[2:]))
    array = array.reshape((array.shape[0], array.shape[1], nr_of_slices)) # shape (x,y,i)
    attr = {**series.attributes, **kwargs}
    if 'SliceLocation' in coords:
        affine = series.affine()
    for i, image in enumerate(source):
        series.progress(i+1, len(source), 'Saving array..')
        image.read()

        # Update any other header data provided
        for a, v in attr.items(): 
            setattr(image, a, v)
            # if isinstance(v, list):
            #     setattr(image, a, v[i])
            # else:
            #     setattr(image, a, v)

        # # If needed, use Defaults for geometry markers
        # if affine is not None:
        #     affine[2, 3] = i # not sufficiently general
        #     image.affine_matrix = affine

        # Set coordinates.
        if mesh_coords is not None:
            for c in mesh_coords:
                image[c] = mesh_coords[c][i] 
                if c == 'SliceLocation':
                    image['ImagePositionPatient'] = image_utils.image_position_from_slice_location(mesh_coords[c][i], affine) 

        image.set_pixel_array(array[:,:,i])
        image.clear()


# def slice_groups(series): # not yet in use
#     slice_groups = []
#     for orientation in series.ImageOrientationPatient:
#         sg = series.instances(ImageOrientationPatient=orientation)
#         slice_groups.append(sg)
#     return slice_groups


def subseries(record, move=False, **kwargs):
    series = record.new_sibling()
    instances = record.instances(**kwargs)
    for i, instance in enumerate(instances):
        record.progress(i+1, len(instances), 'Extracting subseries..')
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



def array(record, sortby=None, pixels_first=False, first_volume=False):
    if isinstance(record, list): # array of instances
        arr = np.empty(len(record), dtype=object)
        for i, rec in enumerate(record):
            arr[i] = rec
        return _get_pixel_array_from_instance_array(arr, sortby=sortby, pixels_first=pixels_first, first_volume=first_volume)
    elif isinstance(record, np.ndarray): # array of instances
        return _get_pixel_array_from_instance_array(record, sortby=sortby, pixels_first=pixels_first, first_volume=first_volume)
    else:
        return get_pixel_array(record, sortby=sortby, pixels_first=pixels_first, first_volume=first_volume)
    

def get_pixel_array(record, sortby=None, first_volume=False, pixels_first=False):
    source = instance_array(record, sortby)
    array, headers = _get_pixel_array_from_sorted_instance_array(source, pixels_first=pixels_first)
    if first_volume:
        return array[...,0], headers[...,0]
    else:
        return array, headers


def _get_pixel_array_from_instance_array(instance_array, sortby=None, pixels_first=False, first_volume=False):
    source = sort_instance_array(instance_array, sortby)
    array, headers = _get_pixel_array_from_sorted_instance_array(source, pixels_first=pixels_first) 
    if first_volume:
        return array[...,0], headers[...,0]
    else:
        return array, headers  


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


def set_pixel_array(series, array, source=None, pixels_first=False, **kwargs): 

    # Move pixels to the end (default)
    if pixels_first:    
        array = np.moveaxis(array, 0, -1)
        array = np.moveaxis(array, 0, -1)

    # if no header data are provided, use template headers.
    nr_of_slices = int(np.prod(array.shape[:-2]))
    if source is None:
        source = [series.new_instance(MRImage()) for _ in range(nr_of_slices)]
    if source.size == 0:
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
        image.set_pixel_array(array[i,...])
        image.clear()


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
        

def sort_instance_array(instance_array, sortby=None):
    if sortby is None:
        return instance_array
    else:
        if not isinstance(sortby, list):
            sortby = [sortby]
        df = read_dataframe_from_instance_array(instance_array, sortby + ['SOPInstanceUID'])
        df.sort_values(sortby, inplace=True) 
        return df_to_sorted_instance_array(instance_array[0], df, sortby)


def _instances(series, dims:tuple=None, inds:dict=None, select={}, **filters):

    # Use default dimensions if needed.
    if dims is None:
        dims = ('InstanceNumber',)

    # If indices are provided, check that they are compatible with dims.
    if inds is not None:
        for dim in inds:
            if dim not in dims:
                msg = 'Indices must be in the dimensions provided.'
                raise ValueError(msg)
    
    # Get the frames and sort by dim
    frames = instance_array(series, list(dims), report_none=True, select=select, **filters)
    if frames.size == 0:
        return frames.reshape(tuple([0]*len(dims)))
    if frames.shape[-1] > 1:
        d = ''.join(['('] + [str(v)+', ' for v in dims] + [')'])
        msg = 'series shape is ambiguous in dimensions ' + d
        msg += '\n--> Multiple frames exist at some or all locations.'
        msg += '\n--> Hint: use Series.unique() to list the values at all locations.'
        raise ValueError(msg)
    if None in frames:
        d = ''.join(['('] + [str(v)+', ' for v in dims] + [')'])
        msg = 'series shape is not well defined in dimensions ' + d
        msg += '\n--> There are no frames at some locations.'
        msg += '\n--> Hint: use Series.value() to find the values at all locations.'
        raise ValueError(msg)
    frames = frames[...,0]

    # Extract indices and coordinates if provided
    if inds is not None:
        for dim in inds:
            ind = inds[dim]
            d = dims.index(dim)
            frames = frames.take(ind, axis=d)
            if not isinstance(ind, np.ndarray):
                frames = np.expand_dims(frames, axis=d)
    if frames.size == 0:
        return frames.reshape(tuple([0]*len(dims)))
    else:
        return frames


def instance_array(record, sortby=None, report_none=False, select={}, **filters): 
    """Sort instances by a list of attributes.
    
    Args:
        sortby: 
            List of DICOM keywords by which the series is sorted
    Returns:
        An ndarray holding the instances sorted by sortby.
    """
    if sortby is None:
        instances = record.instances(**filters) # Note filter values here cant be arrays
        array = np.empty(len(instances), dtype=object)
        for i, instance in enumerate(instances): 
            array[i] = instance
        return array
    else:
        if not isinstance(sortby, list):
            sortby = [sortby]
        df = record.read_dataframe(sortby + ['SOPInstanceUID'], select=select, **filters) 
        df = df[df.SOPInstanceUID.values != None]
        if df.empty:
            return np.array([])
        if report_none:
            if None in df.values:
                d = ''.join(['('] + [str(v)+', ' for v in sortby] + [')'])
                msg = 'series shape is not well defined in dimensions ' + d
                msg += '\n--> Some of the dimensions are not defined in the header.'
                msg += '\n--> Hint: use Series.value() to find the undefined values.'                
                raise ValueError(msg)
        df.sort_values(sortby, inplace=True) 
        return df_to_sorted_instance_array(record, df, sortby)


def df_to_sorted_instance_array(record, df, sortby): 

    data = []
    vals = df[sortby[0]].unique()
    for i, c in enumerate(vals): 
        record.progress(i, len(vals), message='Sorting pixel data..')
        # if a type is not supported by np.isnan()
        # assume it is not a nan
        if c is None: # this happens when undefined keyword is used
            dfc = df[df[sortby[0]].isnull()]
        else:
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
            datac = df_to_sorted_instance_array(record, dfc, sortby[1:])
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
    #arrays = [a for a in arrays if a is not None]
    arrays = [a for a in arrays if a.size != 0]
    if arrays == []:
        return np.array([])
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







