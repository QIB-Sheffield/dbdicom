import os
import math
import pydicom
import numpy as np
from .. import utilities


class Record():

    def __init__(self, folder, UID=[], generation=0):

        objUID = [] + UID
#        for i in range(generation-len(UID)):
        while generation > len(objUID):
            newUID = pydicom.uid.generate_uid()
            objUID.append(newUID)    

        self.__dict__['UID'] = objUID
        self.__dict__['folder'] = folder
        self.__dict__['status'] = folder.status
        self.__dict__['dialog'] = folder.dialog
        self.__dict__['dicm'] = folder.dicm
        self.__dict__['ds'] = None

    @property
    def generation(self):
        return len(self.UID)

    @property
    def key(self):
        """The keywords describing the UID of the record"""

        key = ['PatientID', 'StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']
        return key[0:self.generation]

    def data(self):
        """Dataframe with current data - excluding those that were removed"""

        if self.folder.path is None:
            return self.folder.dataframe
        current = self.folder.dataframe.removed == False
        data = self.folder.dataframe[current]
        if self.UID == []: return data       
        rows = data[self.key[-1]] == self.UID[-1]
        return data[rows]

    def dataset(self, sortby=None): 
        """Sort instances by a list of attributes.
        
        Args:
            sortby: 
                List of DICOM keywords by which the series is sorted
        Returns:
            An ndarray holding the instances sorted by sortby.
        """
        if sortby is None:
            df = self.data()
            return self._dataset_from_df(df)
        else:
            df = utilities.dataframe(self.files, sortby, self.status)
            df.sort_values(sortby, inplace=True)
            return self._sorted_dataset_from_df(df, sortby)

    def _sorted_dataset_from_df(self, df, sortby): 

        data = []
        for c in df[sortby[0]].unique():
            self.status.message('Reading ' + str(sortby[0]) + ' ' + str(c))
            dfc = df[df[sortby[0]] == c]
            if len(sortby) == 1:
                datac = self._dataset_from_df(dfc)
            else:
                datac = self._sorted_dataset_from_df(dfc, sortby[1:])
            data.append(datac)
        return utilities._stack_arrays(data, align_left=True)

    def _dataset_from_df(self, df): 
        """Return datasets as numpy array of object type"""

        data = np.empty(df.shape[0], dtype=object)
        cnt = 0
        for file, _ in df.iterrows(): # just enumerate over df.index
            self.status.progress(cnt, df.shape[0])
            data[cnt] = self.folder.instance(file)
            cnt += 1
        self.status.hide()
        return data

    def array(self, sortby=None, pixels_first=False): 
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
        dataset = self.dataset(sortby)
        array = [instance.array() for instance in dataset.ravel()]
        array = utilities._stack_arrays(array)
        array = array.reshape(dataset.shape + array.shape[1:])
        if pixels_first:
            array = np.moveaxis(array, -1, 0)
            array = np.moveaxis(array, -1, 0)
        return array, dataset

    def set_array(self, array, dataset=None, pixels_first=False, inplace=True): 
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
        if dataset is None:
            dataset = self.dataset()
        # Return with error message if dataset and array do not match.
        nr_of_slices = math.prod(array.shape[:-2])
        if nr_of_slices != math.prod(dataset.shape):
            message = "Array and dataset do not match"
            message += '\n Array has ' + str(nr_of_slices) + ' elements'
            message += '\n dataset has ' + str(math.prod(dataset.shape)) + ' elements'
            self.dialog.error(message)
            return self
        # If self is not a series, create a new series.
        if self.generation != 3:
            series = self.new_series()
        else:
            series = self
        # Reshape, copy instances and save slices.
        array = array.reshape((nr_of_slices, array.shape[-2], array.shape[-1]))
        dataset = dataset.reshape(nr_of_slices)
        for i, instance in enumerate(dataset):
            instance.copy_to(series).set_array(array[i,...])
            if inplace: instance.remove()
        return series

#    def write_array(self, array, dataset): 
#        """
#        Set and array and write it to disk.
#        """
#        series = self.set_array(array, dataset)
#        series.write()
#        return series

    @property
    def _SOPClassUID(self):
        """The SOP Class UID of the first instance"""

        data = self.data()
        if data.empty: return None
        return self.data().iloc[0].SOPClassUID

    @property
    def files(self):
        """Returns the filepath to the instances in the object."""
 
        return self.data().index.tolist()

    def check(self):
        """Check all instances of the object."""

        self.set_checked(True)

    def uncheck(self):
        """Check all instances of the object."""

        self.set_checked(False)

    def set_checked(self, checked):
        """Set the checkstate of all instances of the object."""

        files = self.files
        self.folder.dataframe.at[files, 'checked'] = checked

    def is_checked(self):
        """Check if all instances of the object are checked."""

        return self.data().checked.all()

    def in_memory(self): # is_in_memory
        """Check if the object has been read into memory"""

        return self.ds is not None

    def on_disk(self): # is_on_disk

        return self.ds is None

    @property
    def parent(self):
        "Returns the parent object"

        return self.dicm.parent(self)
        
    def children(self, index=None, checked=None, **kwargs):
        """List of Patients"""

        if self.generation == 4: return []
        if self.in_memory():
            objects = utilities._filter(self.ds, **kwargs)
            if checked is not None:
                if checked:
                    objects = [obj for obj in objects if obj.is_checked()]
                else:
                    objects = [obj for obj in objects if not obj.is_checked()]
            if index is not None:
                if index >= len(objects): 
                    return
                else:
                    return objects[index]
            return objects
        return self.records(generation=self.generation+1, index=index, checked=checked, **kwargs)

    def records(self, generation=0, index=None, checked=None, **kwargs):
        """A list of all records of a given generation corresponding to the record.

        If generation is lower then that of the object, 
        all offspring of the given generation are returned.

        If the generation is higher than that of the object,
        the correspondong ancestor is return as a 1-element list.

        Optionally the list can be filtered by index, or by providing a 
        list of DICOM KeyWords and values. In that case only objects
        a returned that fulfill all criteria.
        
        Parameters
        ----------
        generation : int
            The generation to be returned (0 to 4)
        index : int
            Index of the single object to be return
        kwargs : (Key, Value)
            Conditions to filter the objects
        """
        objects = []
        if generation == 0:
            obj = self.dicm.object(self.folder, generation=0)
            objects.append(obj)
        else:
            key = self.folder._columns[0:generation]
            data = self.data()
            if data.empty: 
                if index is None:
                    return objects
                else:
                    return
            column = data[key[-1]]
            rec_list = column.unique()
            if index is not None:
                rec_list = [rec_list[index]]
            for rec in rec_list:
                rec_data = data[column == rec]
                if checked is not None:
                    if checked == True:
                        if not rec_data.checked.all():
                            continue
                    elif checked == False:
                        if rec_data.checked.all():
                            continue
                row = rec_data.iloc[0]
                obj = self.dicm.object(self.folder, row, generation)
                objects.append(obj)
        objects = utilities._filter(objects, **kwargs)
        if index is not None: return objects[0]
        return objects

    def patients(self, index=None, checked=None, **kwargs):
        """A list of patients of the object"""

        if self.generation==4: 
            return self.parent.parent.parent
        if self.generation==3:
            return self.parent.parent
        if self.generation==2:
            self.parent
        if self.generation==1:
            return
        return self.children(index=index, checked=checked, **kwargs)

    def studies(self, index=None, checked=None, **kwargs):
        """A list of studies of the object"""

        if self.generation==4: 
            return self.parent.parent
        if self.generation==3:
            return self.parent
        if self.generation==2:
            return
        if self.generation==1:
            return self.children(index=index, checked=checked, **kwargs)
        objects = []
        for child in self.children():
            inst = child.studies(checked=checked, **kwargs)
            objects.extend(inst)
        if index is not None:
            if index >= len(objects):
                return
            else:
                return objects[index]
        return objects

    def series(self, index=None, checked=None, **kwargs):
        """A list of series of the object"""

        if self.generation==4: 
            return self.parent
        if self.generation==3:
            return
        if self.generation==2:
            kids = self.children(index=index, checked=checked, **kwargs)
            return kids
        series = []
        for child in self.children():
            inst = child.series(checked=checked, **kwargs)
            series.extend(inst)
        if index is not None:
            if index >= len(series):
                return
            else:
                return series[index]
        return series

    def instances(self, index=None, checked=None, **kwargs): # VERY slow - needs optimizing
        """A list of instances of the object"""

        if self.generation==4: 
            return
        if self.generation==3:
            return self.children(index=index, checked=checked, **kwargs)
        instances = []
        for child in self.children():
            inst = child.instances(checked=checked, **kwargs)
            instances.extend(inst)
        if index is not None:
            if index >= len(instances):
                return
            else:
                return instances[index]
        return instances       

    def new_child(self):
        """Creates a new child object"""

        obj = self.dicm.new_child(self)
        obj.read()
        return obj

    def new_sibling(self):
        """
        Creates a new sibling under the same parent.
        """
        if self.generation == 0:
            return
        else:
            return self.parent.new_child()

    def new_pibling(self):
        """
        Creates a new sibling of parent.
        """
        if self.generation <= 1:
            return
        else:
            return self.parent.new_sibling()

    def new_series(self):
        """
        Creates a new series under the same parent
        """ 
        if self.generation <= 1: 
            return self.new_child().new_series()
        if self.generation == 2:
            return self.new_child()
        if self.generation == 3:
            return self.new_sibling()
        if self.generation == 4:
            return self.new_pibling() 

    def __getattr__(self, tag):
        """Gets the value of the data element with given tag.
        
        Arguments
        ---------
        tag : str
            DICOM KeyWord String

        Returns
        -------
        Value of the corresponding DICOM data element
        """
        return self[tag]

    def __setattr__(self, tag, value):
        """Sets the value of the data element with given tag."""

        if tag == 'folder':
            self.__dict__['folder'] = value
        elif tag == 'ds':
            self.__dict__['ds'] = value
        else:
            self[tag] = value

    def __getitem__(self, tags):
        """Gets the value of the data elements with specified tags.
        
        Arguments
        ---------
        tags : a string, hexadecimal tuple, or a list of strings and hexadecimal tuples

        Returns
        -------
        A value or a list of values
        """
        instance = self.instances(0)
        if instance is not None:
            return instance[tags]

    def __setitem__(self, tags, values):
        """Sets the value of the data element with given tag."""

        instances = self.instances()
        self.status.message('Writing DICOM tags..')
        for i, instance in enumerate(instances):
            instance[tags] = values
            self.status.progress(i, len(instances))
        self.status.hide()

    def remove(self):
        """Deletes the object. """ 

        files = self.files
        if files == []: 
            return
        self.folder.dataframe.loc[self.files,'removed'] = True

    def move_to(self, ancestor):
        """move object to a new parent.
        
        ancestor:any DICOM Class
            If the object is not a parent, the missing 
            intermediate generations are automatically created.
        """
#        self[ancestor.key] = ancestor.UID
        copy = self.copy_to(ancestor)
        self.remove()
        self = copy

    def move(self, child, ancestor):
        """Move a child object to a new parent"""

        if self.in_memory():
            if child in self.ds:
                self.ds.remove(child)
        child.move_to(ancestor)
    
    def copy(self):
        """Returns a copy in the same parent"""

        copy = self.copy_to(self.parent)
        if self.in_memory(): copy.read()
        return copy

    def copy_to(self, ancestor, message=None):
        """copy object to a new ancestor.
        
        ancestor: Root, Patient or Study
        If the object is not a study, the missing 
        intermediate generations are automatically created.
        """
        if self.generation == 0: return
#        if ancestor.generation == 0: return
        copy = self.__class__(self.folder, UID=ancestor.UID)
        if ancestor.in_memory():
            copy.read()
            ancestor.ds.append(copy)
        children = self.children()
        if message is None:
            message = "Copying " + self.__class__.__name__ + ' ' + self.label()
        self.status.message(message)
        for i, child in enumerate(children):
            child.copy_to(copy)
            self.status.progress(i, len(children))
        self.status.hide()

        return copy

    def export(self, path):
        """Export instances to an external folder.

        The instance itself will not be removed from the DICOM folder.
        Instead a copy of the file will be copied to the external folder.
        
        Arguments
        ---------
        path : str
            path to an external folder. If not provided,
            a window will prompt the user to select one.
        """
        instances = self.instances()
        self.status.message('Exporting..')
        for i, instance in enumerate(instances):
            instance.export(path)
            self.status.progress(i,len(instances))
        self.status.hide()

    def save(self):
        """Save all instances."""

        self.status.message("Saving all current instances..")
        instances = self.instances() 
        for i, instance in enumerate(instances):
            instance.save()
            self.status.progress(i, len(instances))
        self.status.hide()

        self.status.message("Deleting all removed instances..")
        if self.__class__.__name__ == 'Folder':
            data = self.folder.dataframe
        else:
            rows = self.folder.dataframe[self.key[-1]] == self.UID[-1]
            data = self.folder.dataframe[rows] 
        removed = data.removed[data.removed]
        files = removed.index.tolist()
        for i, file in enumerate(files): 
            os.remove(file)
            self.status.progress(i, len(files))
        self.folder.dataframe.drop(removed.index, inplace=True)
        self.status.hide()

    def restore(self, message = 'Restoring..'):
        """
        Restore all instances.
        """
        in_memory = self.in_memory() 
        self.clear()
        instances = self.instances()
        self.status.message(message)
        for i, instance in enumerate(instances):
            instance.restore()
            self.status.progress(i,len(instances))
        self.status.hide()
        if in_memory: self.read()
        return self

    def read_dataframe(self, tags):

        return utilities.dataframe(self.files, tags, self.status)

    def read(self, message = 'Reading..'):

        self.status.message(message)
        self.__dict__['ds'] = self.children()
        for i, child in enumerate(self.ds):
            child.read()
            self.status.progress(i, len(self.ds))
        self.status.hide()

    def write(self):

        if self.ds is None: 
            return
        for child in self.ds:
            child.write()

    def clear(self):

        if self.ds is None: 
            return
        for child in self.ds:
            child.clear()
        self.ds = None