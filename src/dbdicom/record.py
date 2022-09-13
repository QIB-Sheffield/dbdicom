__all__ = ['open']

import os
import numpy as np

from dbdicom.register import DbRegister

class DbRecord():

    def __init__(self, register, uid='Database', **attributes):   

        self.uid = uid
        self.attributes = attributes
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
        uid = self.register.new_patient(parent=self.uid, **kwargs)
        return self.__class__(self.register, uid)

    def new_study(self, **kwargs):
        uid = self.register.new_study(parent=self.uid, **kwargs)
        return self.__class__(self.register, uid)

    def new_series(self, **kwargs):
        uid = self.register.new_series(parent=self.uid, **kwargs)
        return self.__class__(self.register, uid)

    def new_instance(self, **kwargs):
        uid = self.register.new_instance(parent=self.uid, **kwargs)
        return self.__class__(self.register, uid)

    def new_child(self, **kwargs): # inherit attributes
        uid = self.register.new_child(uid=self.uid, **kwargs)
        return self.__class__(self.register, uid)

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

    def write(self):
        self.register.write(self.uid)

    def clear(self):
        self.register.clear(self.uid)

    def close(self):
        close(self) 

    def delete(self):
        self.register.delete(self.uid)

    def copy_to(self, target):
        return copy_to([self], target)
    
    def move_to(self, target):
        return move_to([self], target)

    def set_values(self, attributes, values):
        set_values([self], attributes, values)

    def get_values(self, attributes):
        return get_values([self], attributes)[0]

    def get_dataset(self):
        return self.register.get_dataset(self.uid)

    def set_dataset(self, dataset):
        self.register.set_dataset(self.uid, dataset)

    def save(self):
        self.register.save(self.uid)

    def restore(self):
        self.register.restore(self.uid)

    def import_datasets(self, files):
        self.register.import_datasets(files)

    def export_datasets(self, records, database):
        uids = [rec.uid for rec in records]
        self.register.export_datasets(uids, database.register)

    

def open(path, **kwargs):

    dbr = DbRegister(path, **kwargs)
    dbr.open(path)
    return DbRecord(dbr) 

def close(dbr): # move to subclass only ?
    if dbr.uid == 'Database':
        dbr.register.close()


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
    uids = dbr.copy_to(uids, target.uid)
    if isinstance(uids, list):
        return [DbRecord(dbr, uid) for uid in uids]
    else:
        return DbRecord(dbr, uids)

def move_to(records, target):

    dbr = records[0].register
    uids = [rec.uid for rec in records]
    uids = dbr.move_to(uids, target.uid)
    if isinstance(uids, list):
        return [DbRecord(dbr, uid) for uid in uids]
    else:
        return DbRecord(dbr, uids)

def group(records, into=None):

    if into is None:
        into = records[0].new_pibling()
    copy_to(records, into)
    return into

def merge(records, into=None):

    return group(children(records), into=into)



def load_npy(record):
    # Not in use - loading of temporary numpy files
    file = record.register.npy()
    if not os.path.exists(file):
        return
    with open(file, 'rb') as f:
        array = np.load(f)
    return array

def save_npy(record, array=None, sortby=None, pixels_first=False):
    # Not in use - saving of temporary numpy files
    if array is None:
        array = record.array(sortby=sortby, pixels_first=pixels_first)
    file = record.register.npy()
    with open(file, 'wb') as f:
        np.save(f, array)