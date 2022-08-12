import os

import dbdicom.methods.dbobject as dbobject
import dbdicom.utils.pydicom as pydcm


class DbObject():

    def __init__(self, folder, UID=[], generation=None, **attributes):

        objUID = [] + UID
        if generation is not None:
            while generation > len(objUID):
                newUID = pydcm.new_uid()
                objUID.append(newUID)    

        self.UID = objUID
        self.attributes = attributes
        self.dbindex = folder
        self.status = folder.status
        self.dialog = folder.dialog

    @property
    def generation(self):
        return len(self.UID)

    def type(self):
        return dbobject.type(self)

    @property
    def key(self):
        """The keywords describing the UID of the object"""

        key = ['PatientID', 'StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']
        return key[0:self.generation]

    @property
    def _SOPClassUID(self):
        """The SOP Class UID of the first instance"""

        data = self.data()
        if data.empty: return None
        return self.data().iloc[0].SOPClassUID

    def data(self):
        """Dataframe with current data - excluding those that were removed
        """

        if self.dbindex.path is None:
            return self.dbindex.dataframe
        current = self.dbindex.dataframe.removed == False
        data = self.dbindex.dataframe[current]
        if self.UID == []: 
            return data       
        rows = data[self.key[-1]] == self.UID[-1]
        return data[rows]

    @property
    def files(self):
        """Returns the filepath to the instances in the object."""
 
        relpaths = self.data().index.tolist()
        return [os.path.join(self.dbindex.path, p) for p in relpaths]

    @property
    def file(self):
        """Returns the filepath to the first file."""
 
        files = self.files
        if len(files) != 0:  
            return files[0]

    def empty(self):
        return self.data().empty

    @property
    def parent(self):
        """Objects in memory do not know their parents"""

        return None

    def read(self):
        """Convenience function - return the object itself"""
        return self

    def write(self, ds):
        """Convenience function - do nothing"""
        pass

    def __getattr__(self, tag):
        return self[tag]

    def __getitem__(self, tags):
        """Gets the value of the data elements with specified tags.
        
        Arguments
        ---------
        tags : a string, hexadecimal tuple, or a list of strings and hexadecimal tuples

        Returns
        -------
        A value or a list of values
        """
        # Read from self.UID, self.attributes or dataframe
        # if not isinstance(tags, list):
        #     tags = [tags]
        # key_tags = [t for t in tags if t in self.key]
        # save_tags = [t for t in tags if t not in self.key]
        
        if self.is_an_instance():
            ds = self.read()
            return pydcm.get_values(ds, tags)
        instances = self.instances()
        if instances == []:
            return
        if not isinstance(tags, list):
            values = []
            for instance in instances:
                ds = instance.read()._ds
                v = pydcm.get_values(ds, tags)
                values.append(v)
            return list(set(values))

        # For each tag, get a list of values, one for each instance
        # Return a list of unique items per tag
        values = [[] for _ in range(len(tags))]
        for instance in instances:
            ds = instance.read()._ds
            v = pydcm.get_values(ds, tags)
            for t in range(len(tags)):
                values[t].append(v[t])
        for v, value in enumerate(values):
            values[v] = list(set(value))
        return values

    def __setitem__(self, tags, values):
        """Sets the value of the data element with given tag."""

        # FASTER but needs testing
        # This also means __setitem__ can be removed from instance class
        #
        # db.set_value(self.instances(), dict(zip(tags, values)))
        
        # LAZY - SLOW
        if self.is_an_instance():
            instances = [self]
        else:
            instances = self.instances()
        for i, instance in enumerate(instances):
            # excludes tags in self.UID and attributes
            ds = instance.read()
            pydcm.set_values(ds, tags, values)
            instance.write(ds)
            self.status.progress(i, len(instances))
        self.status.hide()

    def is_an_instance(self):
        return self.generation == 4

    def new_instance(self, uid, **attributes):
        """Creates an instance of a dicm object from a row in the dataframe"""
        return self.__class__(self.dbindex, UID=uid, generation=4, **attributes) 

    def new_series(self, uid, **attributes):
        return self.__class__(self.dbindex, UID=uid, generation=3, **attributes) 

    def object(self, row, generation=4, **attributes):
        """Create a new dicom object from a row in the dataframe.
        
        Args:
            row: a row in the dataframe as a series
            generation: determines whether the object returned is 
                at level database (generation=0), patient (generation=1),
                study (generation=2), series (generation=3) 
                or instance (generation=4, default).
        Returns:
            An instance of one of the dicom classes defined in wedicom.classes.
        """

        if generation == 0: 
            return self.__class__(self.dbindex, UID=[], **attributes)
        key = self.dbindex.columns[0:generation]
        UID = row[key].values.tolist()
        return self.__class__(self.dbindex, UID=UID, **attributes)

    def instances(self, index=None, **kwargs): 
        """A list of instances of the object"""

        if self.generation == 4: 
            return [self]
        if self.generation == 3:
            return self.children(index=index, **kwargs)
        if self.generation < 3:
            objects = []
            for child in self.children():
                o = child.instances(**kwargs)
                objects.extend(o)
            if index is not None:
                if index >= len(objects):
                    return None
                else:
                    return objects[index]
            return objects
        
    def series(self, index=None, **kwargs): 
        """A list of series of the object"""

        if self.generation == 4:
            if self.parent is None:
                return None
            elif index is None:
                return [self.parent]
            elif index == 0:
                return self.parent
            else:
                return None
        if self.generation == 3: 
            return [self]
        if self.generation == 2:
            return self.children(index=index, **kwargs)
        if self.generation < 2:
            objects = []
            for child in self.children():
                o = child.series(**kwargs)
                objects.extend(o)
            if index is not None:
                if index >= len(objects):
                    return None
                else:
                    return objects[index]
            return objects

    def studies(self, index=None, **kwargs): 
        """A list of studies of the object"""

        if self.generation == 4:
            if self.parent is None:
                return None
            else:
                return self.parent.studies(index=index)
        if self.generation == 3:
            if self.parent is None:
                return None
            elif index is None:
                return [self.parent]
            elif index == 0:
                return self.parent
            else:
                return None
        if self.generation == 2: 
            return [self]
        if self.generation == 1:
            return self.children(index=index, **kwargs)
        if self.generation < 1:
            objects = []
            for child in self.children():
                o = child.studies(**kwargs)
                objects.extend(o)
            if index is not None:
                if index >= len(objects):
                    return None
                else:
                    return objects[index]
            return objects

    def patients(self, index=None, **kwargs): 
        """A list of patients of the object"""

        if self.generation >= 3:
            if self.parent is None:
                return None
            else:
                return self.parent.patients(index=index)
        if self.generation == 2:
            if self.parent is None:
                return None
            elif index is None:
                return [self.parent]
            elif index == 0:
                return self.parent
            else:
                return None
        if self.generation == 1: 
            return [self]
        if self.generation == 0:
            return self.children(index=index, **kwargs)

    def database(self): 
        if self.generation >= 2:
            if self.parent is None:
                return None
            else:
                return self.parent.database()
        if self.generation == 1:
            if self.parent is None:
                return None
            else:
                return self.parent
        if self.generation == 0: 
            return self

    def new_sibling(self, **attributes):
        """
        Creates a new sibling under the same parent.
        """
        if self.generation == 0:
            return
        if self.parent is None:
            return
        return self.parent.new_child(**attributes)

    def new_pibling(self, **attributes):
        """
        Creates a new sibling of parent.
        """
        if self.generation <= 1:
            return
        if self.parent is None:
            return
        return self.parent.new_sibling(**attributes)

    def new_cousin(self, **attributes):
        """
        Creates a new sibling of parent.
        """
        if self.generation <= 1:
            return
        pibling = self.new_pibling()
        if pibling is None:
            return
        return pibling.new_child(**attributes)

    def new_series(self, **attributes):
        """
        Creates a new series under the same parent
        """ 
        if self.generation == 0: 
            return self.new_child().new_child().new_series(**attributes)
        if self.generation == 1: 
            return self.new_child().new_series(**attributes)
        if self.generation == 2:
            return self.new_child(**attributes)
        if self.generation == 3:
            return self.new_sibling(**attributes)
        if self.generation == 4:
            return self.new_pibling(**attributes) 

    def new_child(self, **attributes):
        """Creates a new child object"""

        if self.generation == 4: 
            return None
        else:
            return self.__class__(self.dbindex, UID=self.UID, **attributes, generation=self.generation+1)

    def children(self, index=None, **kwargs):
        """List of children"""

        if self.generation == 4: 
            return []
        return self.records(generation=self.generation+1, index=index, **kwargs) 

    def records(self, generation=0, index=None, **kwargs):
        # This has become obsolete - used only in children()
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
            obj = self.__class__(self.dbindex, UID=[])
            objects.append(obj)
        else:
            key = self.dbindex.columns[0:generation]
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
                row = rec_data.iloc[0]
                obj = self.object(row, generation)
                objects.append(obj)
        objects = dbobject._filter(objects, **kwargs)
        if index is not None: return objects[0]
        return objects

    def print(self):
        """Prints a summary of the project folder to the terminal."""

        if self.generation != 0: # Still needs to be implemented for other generations
            return
        
        print(' ')
        print('---------- DICOM FOLDER --------------')
        print('FOLDER: ' + self.dbindex.path)
        for i, patient in enumerate(self.children()):
            print(' ')
            print('    PATIENT [' + str(i) + ']: ' + patient.label())
            print(' ')
            for j, study in enumerate(patient.children()):
                print('        STUDY [' + str(j) + ']: ' + study.label())
                print(' ')
                for k, series in enumerate(study.children()):
                    print('            SERIES [' + str(k) + ']: ' + series.label())
                    print('                Nr of instances: ' + str(len(series.children()))) 

    def label(self, row=None):

        if self.generation == 1:
            if row is None:
                data = self.data()
                if data.empty: return "New Patient"
                file = data.index[0]
                name = data.at[file, 'PatientName']
                id = data.at[file, 'PatientID']
            else:
                name = row.PatientName
                id = row.PatientID
                
            label = str(name)
            label += ' [' + str(id) + ']'
            return label

        if self.generation == 2:
            if row is None:
                data = self.data()
                if data.empty: return "New Study"
                file = data.index[0]
                descr = data.at[file, 'StudyDescription']
                date = data.at[file, 'StudyDate']
            else:
                descr = row.StudyDescription
                date = row.StudyDate

            label = str(descr)
            label += ' [' + str(date) + ']'
            return label  

        if self.generation == 3:

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

        if self.generation == 4:

            if row is None:
                data = self.data()
                if data.empty: return "New Instance"
                file = data.index[0]
                nr = data.at[file, 'InstanceNumber']
            else:
                nr = row.InstanceNumber

            return str(nr).zfill(6)    


    def merge_with(self, obj, message=None): 
        return dbobject.merge([self, obj], message=message)

    def copy_to(self, *args, **kwargs):
        return dbobject.copy_to(self, *args, **kwargs)

    def copy(self, *args, **kwargs): 
        return dbobject.copy_to(self, self.parent, *args, **kwargs)
        
    def copy_instances(self, *args, **kwargs):  
        return dbobject.copy_instances(self, *args, **kwargs)

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
        if self.is_an_instance():
            instances = [self]
        else:
            instances = self.instances()
        
        for i, instance in enumerate(instances):
            relpath = instance.data().index.tolist()[0]
            destination = os.path.join(path, relpath)
            ds = instance.read()._ds()
            pydcm.write(ds, destination, self.dialog)
            self.status.progress(i,len(instances), message='Exporting..')
        self.status.hide()

    def sort_instances(self, *args, **kwargs): 
        """Sort instances by a list of attributes.
        
        Args:
            sortby: 
                List of DICOM keywords by which the series is sorted
        Returns:
            An ndarray holding the instances sorted by sortby.
        """
        return dbobject.sort_instances(self, *args, **kwargs)

    def array(self, *args, **kwargs): 
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
        dbobject.array(self, *args, **kwargs)


    def set_array(self, *args, **kwargs): 
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
        return dbobject.set_array(self, *args, **kwargs)




