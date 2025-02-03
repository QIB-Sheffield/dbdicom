# Importing annotations to handle or sign in import type hints
from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm
import vreg

from dbdicom.record import Record
from dbdicom.utils.files import gif2numpy
from dbdicom.ds.create import read_dataset



class Database(Record):

    name = 'Database'

    def loc(self):
        return self.manager._dbloc()
        # df = self.manager.register
        # return df.removed==False

    def _set_key(self):
        #if not self.manager.register.empty:
        if not self.manager._empty():
            self._key = self.manager._keys(0)
            #self._key = self.manager.register.index[0]
        else:
            self._key = None

    def close(self):
        return self.manager.close()

    def set_path(self,path):
        # Used in example of clear
        self.manager.path=path

    def parent(self):
        return

    def children(self, **kwargs):
        return self.patients(**kwargs)

    def new_child(self, dataset=None, **kwargs): 
        attr = {**kwargs, **self.attributes}
        return self.new_patient(**attr)
    
    def new_sibling(self, suffix=None, **kwargs):
        msg = 'You cannot create a sibling from a database \n'
        msg += 'You can start a new database with db.database()'
        raise RuntimeError(msg)

    def save(self, path=None):
        #self.manager.save('Database')
        self.manager.save()
        self.write(path)
        return self

    def restore(self, path=None):
        self.manager.restore()
        self.write(path)

    def open(self, path):
        self.manager.open(path)

    def close(self):
        return self.manager.close()

    def scan(self):
        self.manager.scan()

    def import_dicom(self, files):
        uids = self.manager.import_datasets(files)
        return uids is not None

    def import_nifti(self, files):
        self.manager.import_datasets_from_nifti(files)

    def import_gif(self, files):
        study = self.new_patient().new_study()
        for file in files:
            array = gif2numpy(file)
            series = study.new_series()
            series.set_array(array)
        return study

    def _copy_from(self, record):
        uids = self.manager.copy_to_database(record.uid, **self.attributes)
        if isinstance(uids, list):
            return [self.record('Patient', uid, **self.attributes) for uid in uids]
        else:
            return self.record('Patient', uids, **self.attributes)

    def zeros(*args, **kwargs): # OBSOLETE - remove
        return zeros(*args, **kwargs)

    # def export_as_dicom(self, path): 
    #     for child in self.children():
    #         child.export_as_dicom(path)

    # def export_as_png(self, path): 
    #     for child in self.children():
    #         child.export_as_png(path)

    # def export_as_csv(self, path): 
    #     for child in self.children():
    #         child.export_as_csv(path)

    def pixel_values(self, series, index=None, 
                     dims=('InstanceNumber', ), return_vals=None):
        s = _series(self, series, index)
        return s.pixel_values(dims=dims, return_vals=return_vals)


    def volume(self, series, index=None, dims=None, return_coords=False) -> vreg.Volume3D:
        s =  _series(self, series, index)
        return s.volume(dims, return_coords=return_coords)

    
    def write_volume(self, vol, series='Series', ref=None, ref_index=None, 
                     coords=None, study=None, **kwargs):
    
        # Find study and reference series
        study = _study(self, study) 
        ref = _series(self, ref, ref_index)  

        # Create new series
        if study is not None:
            series = study.new_series(SeriesDescription=series, **kwargs)
        elif ref is not None:
            series = ref.new_sibling(SeriesDescription=series, **kwargs)
        else:
            series = self.new_series(SeriesDescription=series, **kwargs)

        # Write the volume
        series.write_volume(vol, ref=ref, coords=coords)
        return self


    def merge_series(self, desc, merged='Merged Series', study=None):

        # Get series to merge
        if isinstance(desc, list):
            series = []
            for d in desc:
                series += self.series(SeriesDescription=d)
        else:
            series = self.series(SeriesDescription=desc)

        # Check if all valid
        if series == []:
            raise ValueError(
                "Cannot merge series " + str(desc) + ". No "
                "series found with that SeriesDescription."
            )
        for s in series:
            if s.type() != 'Series':
                raise ValueError(
                    "Cannot merge series " + str(desc) + ". These "
                    "are not all valid series."
                )
            
        # Get study for new series
        if study is None:
            study = series[0].parent()
        else:
            study = _study(self, study) 
        
        # Merge series
        uid, key = self.manager.merge_series(
            [s.uid for s in series], 
            study.uid, 
            SeriesDescription=merged,
        )
        return self.record('Series', uid, key)





def zeros(database, shape, dtype='mri'): # OBSOLETE - remove
    study = database.new_study()
    return study.zeros(shape, dtype=dtype)


def _study(database, study):
        
    if isinstance(study, str):
        studies = database.studies(StudyDescription=study)
        if studies == []:
            study = database.new_study(StudyDescription=study)
        elif len(studies) == 1:
            study = studies[0]
        else:
            raise ValueError(
                "Multiple studies found with the same "
                "StudyDescription. Use studies() to list them all and "
                "select one.")  
    return study


def _series(database, series, index=None):

    if isinstance(series, str):
        all_series = database.series(SeriesDescription=series)
        if all_series == []:
            raise ValueError(
                "No series found with the SeriesDescription " + series)  
        elif len(all_series) == 1:
            series = all_series[0]
        elif index is not None:
            series = all_series[index]
        else:
            raise ValueError(
                "Multiple series found with the "
                "SeriesDescription " + series + ". Use series() to select"
                "a single one.") 
        
    return series



