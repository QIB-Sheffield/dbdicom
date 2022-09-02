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


