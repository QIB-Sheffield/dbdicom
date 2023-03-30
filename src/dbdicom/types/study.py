import os
import numpy as np
from dbdicom.record import DbRecord

class Study(DbRecord):

    name = 'StudyInstanceUID'

    def remove(self):
        self.manager.delete_studies([self.uid])

    def new_series(self, **kwargs):
        attr = {**kwargs, **self.attributes}
        uid, key = self.manager.new_series(parent=self.uid, key=self.key(), **attr)
        return self.record('Series', uid, key, **attr)

    def parent(self):
        #uid = self.manager.register.at[self.key(), 'PatientID']
        uid = self.manager._at(self.key(), 'PatientID')
        return self.record('Patient', uid, key=self.key())

    def children(self, **kwargs):
        return self.series(**kwargs)

    def new_child(self, dataset=None, **kwargs): 
        attr = {**kwargs, **self.attributes}
        return self.new_series(**attr)
    
    def new_sibling(self, suffix=None, **kwargs):
        if suffix is not None:
            desc = self.instance().StudyDescription 
            kwargs['StudyDescription'] = desc + ' [' + suffix + ']'
        return self.parent().new_child(**kwargs)

    def _copy_from(self, record, **kwargs):
        attr = {**kwargs, **self.attributes}
        uids = self.manager.copy_to_study(record.uid, self.uid, **attr)
        if isinstance(uids, list):
            return [self.record('Series', uid) for uid in uids]
        else:
            return self.record('Series', uids)

    def zeros(*args, **kwargs):
        return zeros(*args, **kwargs)


def zeros(study, shape, dtype='mri'):
    series = study.new_series()
    array = np.zeros(shape, dtype=np.float32)
    if dtype not in ['mri', 'MRImage']:
        message = 'dbdicom can only create images of type MRImage at this stage'
        raise ValueError(message)
    series.set_pixel_array(array)
    return series
