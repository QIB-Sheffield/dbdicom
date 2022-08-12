import os
import numpy as np
import pandas as pd

import dbdicom.methods.dbrecord as dbrecord
import dbdicom.utils.pydicom as pydcm
from dbdicom.dbindex import DbIndex
from dbdicom.dbobject import DbObject



class DbData(DbObject):

    def __init__(self, folder, UID=[], generation=0, **attributes):
        super().__init__(folder, UID, generation, **attributes)

        # For Database, Patient, Study or Series: list of pydicom Datasets instances
        # For Instance: pydicom DbData
        self._ds = None 

    def in_memory():
        return True

    def to_pydicom(self): # instances only

        return self._ds

    def __setattr__(self, tag, value):
        """Sets the value of the data element with given tag."""

        if tag in ['UID', 'folder', 'status', 'dialog', 'attributes', '_ds']:
            self.__dict__[tag] = value
        else:
            self[tag] = value

    def read(self, message=None):
        """Read the dataset from disk.
        
        This will throw an exception if the dataset exists in memory only.
        """
        self._ds = {}
        files, relpaths = self.dbindex.files()
        for i, file in enumerate(files):
            if message is not None:
                self.status.progress(i, len(files), message)
            self._ds[relpaths[i]] = pydcm.read(file, self.dialog)
        self.status.hide()

    def write(self, message = 'Writing data to disk..'):
        """Writing data from memory to disk.

        This will throw an exception if the dataset exists in memory only.
        """
        files, relpaths = self.dbindex.files()
        for i, file in enumerate(files):
            self.status.progress(i, len(relpaths), message)
            pydcm.write(self._ds[relpaths[i]], file, self.dialog)
        self.status.hide()


    def move_to(self, parent):

        if self.generation <= parent.generation:
            raise ValueError('Cannot write a ' + self.type() + ' into a ' + parent.type())
        while parent.generation > self.generation + 1:
            parent = parent.new_child()

        # Attributes that are inherited from the parent
        attr = {}
        for i, tag in enumerate(parent.key):
            attr[tag] = parent.UID[i]
        if parent.attributes is not None:
            attr.update(parent.attributes)

        # update datasets in memory
        for ds in self._ds:
            pydcm.set_values(ds, list(attr.keys()), list(attr.values()))

        # update the index dataframe
        relpaths = self.dbindex.update(self.SOPInstanceUID, attr)

        # Update UIDs of object
        self.UID[:-1] = parent.UID    


    def save(self, message = "Saving changes.."):
        """Save all instances of the record."""

        self.status.message(message)
        if self.generation == 0:
            data = self.dbindex.dataframe
        else:
            rows = self.dbindex.dataframe[self.key[-1]] == self.UID[-1]
            data = self.dbindex.dataframe[rows] 

        created = data.created[data.created]   
        removed = data.removed[data.removed]

        for p in removed.index.tolist():
            del self._ds[p]

        self.dbindex.dataframe.loc[created.index, 'created'] = False
        self.dbindex.dataframe.drop(removed.index, inplace=True)

        return self

    def restore(self, message = "Restoring saved state.."):
        """
        Restore all instances.
        """
        self.status.message(message)

        if self.generation == 0:
            data = self.dbindex.dataframe
        else:
            rows = self.dbindex.dataframe[self.key[-1]] == self.UID[-1]
            data = self.dbindex.dataframe[rows] 

        created = data.created[data.created]   
        removed = data.removed[data.removed]

        for p in created.index.tolist():
            del self._ds[p]

        self.dbindex.dataframe.loc[removed.index, 'removed'] = False
        self.dbindex.dataframe.drop(created.index, inplace=True)

        return self

    def get_dataframe(self, tags):
        return pydcm.get_dataframe(self._ds, tags)


class DbRecord(DbObject):

    def __setattr__(self, tag, value):
        """Sets the value of the data element with given tag."""

        if tag in ['UID', 'folder', 'status', 'dialog', 'attributes']:
        #if tag in list(self.__dict__.keys()): # does this work?
            self.__dict__[tag] = value
        else:
            self[tag] = value

    @property
    def parent(self):
        "Returns the parent object"

        if self.generation == 0: 
            return None
        else:
            return DbRecord(self.dbindex, UID=self.UID[:-1], generation=self.generation-1)

    def in_memory():
        return False

    def read(self, *args):

        dataset = DbData(self.dbindex, UID=self.UID, generation=self.generation, attributes=self.attributes)
        dataset.read(*args)
        return dataset

    def write(self, dataset): # dataset is a DbData instance
        """Write a DbData instance to the database."""

        parent = self
        if dataset.generation <= parent.generation:
            raise ValueError('Cannot write a ' + dataset.type() + ' into a ' + parent.type())
        while parent.generation > self.generation + 1:
            parent = parent.new_child()

        # If the dataset already exists in the database
        if dataset.dbindex.path == self.dbindex.path:
            dataset.move_to(parent) # Needs writing
            dataset.write()
            return

        # Move the dataset to the correct parent.
        df = dataset.data()
        relpaths = df.index.tolist()
        
        dataset.UID[:-1] = parent.UID

        # If the dataset exists in memory only or is linked to another database.
        if dataset.dbindex.path != self.dbindex.path:
            df['removed'] = False
            df['created'] = True
            self.dbindex.dataframe = pd.concat([self.dbindex.dataframe, df])
            dataset.dbindex = self.dbindex

        dataset.write()

        # if dataset.is_an_instance():
        #     instances = [dataset]
        # else:
        #     instances = dataset.instances()
        # for instance in instances:
        #     self._copy_attributes_to(instance)
        # self.status.message('Updating database..')
        # self.dbindex._add(instances)
        # # Writing the data to file
        # df = self.data()
        # for cnt, instance in enumerate(instances):
        #     self.status.progress(cnt, len(instances), 'Writing files to disk..')
        #     uid = instance.SOPInstanceUID
        #     filename = df.index[df.SOPInstanceUID == uid].tolist()[0]
        #     file = os.path.join(self.dbindex.path, filename)
        #     pydcm.write(instance._ds, file, self.dialog)
        # self.status.hide()


    def _copy_attributes_to(self, ds): # ds is an instance DbData

        if self.generation == 0:
            pass
        elif self.generation == 1:
            ds.PatientID = self.UID[0]
        elif self.generation == 2:
            ds.PatientID = self.UID[0]
            ds.StudyInstanceUID = self.UID[1]
        elif self.generation == 3:
            ds.PatientID = self.UID[0]
            ds.StudyInstanceUID = self.UID[1]
            ds.SeriesInstanceUID = self.UID[2]   
        elif self.generation == 4:
            ds.PatientID = self.UID[0]
            ds.StudyInstanceUID = self.UID[1]
            ds.SeriesInstanceUID = self.UID[2]   
            ds.SOPInstanceUID = self.UID[3]  

        pydcm.set_values(ds, list(self.attributes.keys()), list(self.attributes.values()))

    def save(self, message = "Saving changes.."):
        """Save all instances of the record."""

        self.status.message(message)
        if self.generation == 0:
            data = self.dbindex.dataframe
        else:
            rows = self.dbindex.dataframe[self.key[-1]] == self.UID[-1]
            data = self.dbindex.dataframe[rows] 

        created = data.created[data.created]   
        removed = data.removed[data.removed]

        files = [os.path.join(self.dbindex.path, p) for p in removed.index.tolist()]
        for i, file in enumerate(files): 
            self.status.progress(i, len(files), message='Deleting removed files..')
            if os.path.exists(file): 
                os.remove(file)
        #self.status.message('Clearing rapid access storage..')
        #npyfile = self.npy()
        #if os.path.exists(npyfile): os.remove(npyfile)
        self.status.message('Done saving..')
        self.dbindex.dataframe.loc[created.index, 'created'] = False
        self.dbindex.dataframe.drop(removed.index, inplace=True)
        self.dbindex._write_df()

    def restore(self, message = "Restoring saved state.."):
        """
        Restore all instances.
        """
        self.status.message(message)

        if self.generation == 0:
            data = self.dbindex.dataframe
        else:
            rows = self.dbindex.dataframe[self.key[-1]] == self.UID[-1]
            data = self.dbindex.dataframe[rows] 

        created = data.created[data.created]   
        removed = data.removed[data.removed]

        files = [os.path.join(self.dbindex.path, p) for p in created.index.tolist()]
        for i, file in enumerate(files): 
            self.status.progress(i, len(files), message='Deleting new files..')
            if os.path.exists(file): os.remove(file)
        self.status.hide()
        self.dbindex.dataframe.loc[removed.index, 'removed'] = False
        self.dbindex.dataframe.drop(created.index, inplace=True)
        self.dbindex._write_df()

        return self

    def remove(self):
        """Deletes the object. """ 

        files = self.files
        if files == []: 
            return
        self.dbindex.dataframe.loc[self.data().index,'removed'] = True 

    def npy(self):

        path = os.path.join(self.dbindex.path, "dbdicom_npy")
        if not os.path.isdir(path): os.mkdir(path)
        file = os.path.join(path, self.UID[-1] + '.npy') 
        return file

    def load_npy(self):

        file = self.npy()
        if not os.path.exists(file):
            return
        with open(file, 'rb') as f:
            array = np.load(f)
        return array

    def save_npy(self, array=None, sortby=None, pixels_first=False):

        if array is None:
            array = self.array(sortby=sortby, pixels_first=pixels_first)
        file = self.npy() 
        with open(file, 'wb') as f:
            np.save(f, array)

    def move_to(self, ancestor):
        """move object to a new parent.
        
        ancestor:any DICOM Class
            If the object is not a parent, the missing 
            intermediate generations are automatically created.
        """
        # This edits the dataframe twice
        copy = self.copy_to(ancestor)
        self.remove()
        return copy

    def _initialize(self, ds, ref=None): # ds is a pydicom dataset

        if self.generation == 4:
            ds = pydcm._initialize(ds, UID=self.UID, ref=ref)
            if dbrecord.type(self) == 'MRImage':
                self.ImageType.insert(0, "DERIVED")
        else:
            for i, obj in enumerate(ds):
                if ref is not None:
                    obj._initialize(ds[i], ref=ref)
                else:
                    obj._initialize()

        
    def close(self):
        """Close an open record.
        
        This method checks if the changes have been saved and prompts the 
        user to save or restore them if not. The user also has the option 
        to cancel closing, in which case the function does nothing. 
        
        `close()` also resets the dataframe and path to default values.

        Returns: 
            True if the user has agreed to close the folder 
            (possible after save or restore) and False if the user 
            has cancelled the closing of the folder.
        """
        if not self.dbindex.is_open(): 
            return True
        if not self.dbindex.is_saved():
            reply = self.dialog.question( 
                title = 'Closing DICOM folder', 
                message = 'Save changes before closing?',
                cancel = True, 
            )
            if reply == "Cancel": 
                return False
            if reply == "Yes":
                self.save()
            elif reply == "No":
                self.restore()

        self.dbindex._write_df()
        self.dbindex.dataframe = pd.DataFrame([]*len(self.dbindex.columns), columns=self.dbindex.columns)            
        self.dbindex.path = None

        return True


    def get_dataframe(self, tags):
        return pydcm.read_dataframe(self.dbindex.path, self.files, tags, 
            self.status, message='Reading DICOM folder..')

    def instance(self, file):
        # generalise - from dataframe index value
        """Create an instance from a filepath"""

        row = self.dbindex.dataframe.loc[file]
        return self.object(row, generation=4) 

    def map_onto(self, target):
        if self.generation == 4:
            return dbrecord._image_map_onto(self, target)
        return dbrecord.map_onto(self, target)

    def export_as_nifti(self, *args, **kwargs): # use args and kwargs
        return dbrecord.export_as_nifti(self, *args, **kwargs)

    def export_as_csv(self, *args, **kwargs):
        return dbrecord.export_as_csv(self, *args, **kwargs)

    def export_as_png(self, *args, **kwargs):
        return dbrecord.export_as_png(self, *args, **kwargs)

    def zeros(self, *args, **kwargs):
        return dbrecord.zeros(self, *args, **kwargs)

    def get_colormap(self):
        return dbrecord.get_colormap(self)

    def set_colormap(self, **kwargs):
        return dbrecord.get_colormap(self, **kwargs)

    def get_lut(self):
        return dbrecord.get_lut(self)

    def image_type(self):
        return dbrecord.image_type(self)


