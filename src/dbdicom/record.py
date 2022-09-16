__all__ = ['open_database']

import os
import numpy as np

from dbdicom.register import DbRegister
import dbdicom.dataset as dbdataset

class DbRecord():

    def __init__(self, register, uid='Database', **kwargs):   

        self.uid = uid
        self.attributes = kwargs
        self.register = register
    
    def __eq__(self, other):
        return self.uid == other.uid

    def __getattr__(self, attribute):
        return self.get_values(attribute)

    def __getitem__(self, attributes):
        return self.get_values(attributes)

    def __setattr__(self, attribute, value):
        if attribute in ['uid', 'register', 'attributes']:
            self.__dict__[attribute] = value
        else:
            self.set_values(attribute, value)
        
    def __setitem__(self, attributes, values):
        self.set_values(attributes, values)

    @property
    def status(self):
        return self.register.status

    @property
    def dialog(self):
        return self.register.dialog

    def type(self):
        return self.register.type(self.uid)

    @property
    def generation(self): # Obsolete
        type = self.type()
        if type == 'Database':
            return 0
        elif type == 'Patient':
            return 1
        elif type == 'Study':
            return 2
        elif type == 'Series':
            return 3
        elif type == 'Instance':
            return 4

    def series_data(self):
        attr = dbdataset.module_series()
        vals = self[attr]
        return attr, vals

    def study_data(self):
        attr = dbdataset.module_study()
        vals = self[attr]
        return attr, vals

    def patient_data(self):
        attr = dbdataset.module_patient()
        vals = self[attr]
        return attr, vals

    def parent(self):
        uid = self.register.parent(self.uid)
        return self.__class__(self.register, uid)

    def children(self, **kwargs):
        return children([self], **kwargs)

    def instances(self, **kwargs):
        return instances([self], **kwargs)

    def series(self, **kwargs):
        return series([self], **kwargs)

    def studies(self, **kwargs):
        return studies([self], **kwargs)

    def patients(self, **kwargs):
        return patients([self], **kwargs)

    def siblings(self, **kwargs):
        uids = self.register.siblings(self.uid, **kwargs)
        return [self.__class__(self.register, uid) for uid in uids]

    def database(self):
        return self.__class__(self.register)

    def new_patient(self, **kwargs):
        attr = {**kwargs, **self.attributes}
        uid = self.register.new_patient(parent=self.uid, 
            PatientName = attr['PatientName'] if 'PatientName' in attr else 'New Patient',
        )
        return self.__class__(self.register, uid, **attr)

    def new_study(self, **kwargs):
        attr = {**kwargs, **self.attributes}
        uid = self.register.new_study(parent=self.uid, 
            StudyDescription = attr['StudyDescription'] if 'StudyDescription' in attr else 'New Study',
        )
        return self.__class__(self.register, uid, **attr)

    def new_series(self, **kwargs):
        attr = {**kwargs, **self.attributes}
        uid = self.register.new_series(parent=self.uid,
            SeriesDescription = attr['SeriesDescription'] if 'SeriesDescription' in attr else 'New Series',
        )
        return self.__class__(self.register, uid, **attr)

    def new_instance(self, dataset=None, **kwargs):
        attr = {**kwargs, **self.attributes}
        uid = self.register.new_instance(parent=self.uid, dataset=dataset, **attr)
        return self.__class__(self.register, uid, **attr)

    def new_child(self, **kwargs): 
        attr = {**kwargs, **self.attributes}
        uid = self.register.new_child(uid=self.uid, **attr)
        return self.__class__(self.register, uid, **attr)

    def new_sibling(self):
        uid = self.register.new_sibling(uid=self.uid)
        return self.__class__(self.register, uid)

    def new_pibling(self):
        uid = self.register.new_pibling(uid=self.uid)
        return self.__class__(self.register, uid)
    
    def label(self):
        return self.register.label(self.uid)

    def print(self):
        self.register.print() # print self.uid only

    def read(self):
        self.register.read(self.uid)

    def write(self, path=None):
        if path is not None:
            self.register.path = path
        self.register.write(self.uid)
        self.register._write_df()

    def clear(self):
        self.register.clear(self.uid)

    def remove(self):
        self.register.delete(self.uid)

    def copy_to(self, target):
        return copy_to([self], target)[0]
    
    def move_to(self, target):
        move_to([self], target)
        return self

    def set_values(self, attributes, values):
        set_values([self], attributes, values)

    def get_values(self, attributes):
        return get_values([self], attributes)[0]

    def get_dataset(self):
        return self.register.get_dataset(self.uid)

    def set_dataset(self, dataset):
        self.register.set_dataset(self.uid, dataset)

    def save(self, path=None):
        self.register.save(self.uid)
        self.write(path)
        
    def restore(self):
        self.register.restore(self.uid)
        self.write()

    def import_datasets(self, files):
        self.register.import_datasets(files)

    def export_datasets(self, records, database):
        uids = [rec.uid for rec in records]
        self.register.export_datasets(uids, database.register)

    def open(self, path):
        self.register.open(path)

    def close(self):
        if self.uid == 'Database':
            self.register.close()


def new_database(path=None, **kwargs):
    if path is not None:
        return open_database(path, **kwargs)
    dbr = DbRegister()
    return DbRecord(dbr, **kwargs)

def open_database(path, **kwargs):

    dbr = DbRegister(path, **kwargs)
    dbr.open(path)
    return DbRecord(dbr) 



#
# Functions on a list of records of the same database
#

def get_values(records, attributes):

    uids = [rec.uid for rec in records]
    dbr = records[0].register
    return dbr.get_values(uids, attributes)

def set_values(records, attributes, values):

    uids = [rec.uid for rec in records]
    dbr = records[0].register
    dbr.set_values(uids, attributes, values)

def children(records, **kwargs):

    dbr = records[0].register
    uids = [rec.uid for rec in records]
    uids = dbr.children(uids, **kwargs)
    return [DbRecord(dbr, uid) for uid in uids]

def instances(records, **kwargs):

    dbr = records[0].register
    uids = [rec.uid for rec in records]
    uids = dbr.instances(uids, **kwargs)
    return [DbRecord(dbr, uid) for uid in uids]

def series(records, **kwargs):

    dbr = records[0].register
    uids = [rec.uid for rec in records]
    uids = dbr.series(uids, **kwargs)
    return [DbRecord(dbr, uid) for uid in uids]

def studies(records, **kwargs):

    dbr = records[0].register
    uids = [rec.uid for rec in records]
    uids = dbr.studies(uids, **kwargs)
    return [DbRecord(dbr, uid) for uid in uids]

def patients(records, **kwargs):

    dbr = records[0].register
    uids = [rec.uid for rec in records]
    uids = dbr.patients(uids, **kwargs)
    return [DbRecord(dbr, uid) for uid in uids]

def copy_to(records, target):

    dbr = records[0].register
    uids = [rec.uid for rec in records]
    uids = dbr.copy_to(uids, target.uid, **target.attributes)
    if isinstance(uids, list):
        return [DbRecord(dbr, uid) for uid in uids]
    else:
        return [DbRecord(dbr, uids)]

def move_to(records, target):

    dbr = records[0].register
    uids = [rec.uid for rec in records]
    dbr.move_to(uids, target.uid, **target.attributes)
    return records

def group(records, into=None):

    if into is None:
        into = records[0].new_pibling()
    copy_to(records, into)
    return into

def merge(records, into=None):

    return group(children(records), into=into)

