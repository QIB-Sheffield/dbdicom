import numpy as np
import pandas as pd
import dbdicom.ds.dataset as dbdataset
from dbdicom.utils.files import export_path



class DbRecord():

    def __init__(self, create, manager, uid='Database', key=None, **kwargs):   

        self._key = key
        self._mute = False
        self.uid = uid
        self.attributes = kwargs
        self.manager = manager
        self.new = create
    
    def __eq__(self, other):
        if other is None:
            return False
        return self.uid == other.uid

    def __getattr__(self, attribute):
        return self.get_values(attribute)

    def __getitem__(self, attributes):
        return self.get_values(attributes)

    def __setattr__(self, attribute, value):
        if attribute in ['_key','_mute', 'uid', 'manager', 'attributes', 'new']:
            self.__dict__[attribute] = value
        else:
            self.set_values([attribute], [value])
           
    def __setitem__(self, attributes, values):
        self.set_values(attributes, values)

    def path(self):
        return self.manager.path

    def loc(self):
        return self.manager._loc(self.name, self.uid)
        # df = self.manager.register
        # return (df.removed==False) & (df[self.name]==self.uid)

    def keys(self):
        keys = self.manager._keys(self.loc())
#        keys = self.manager.register.index[self.loc()]
        if len(keys) == 0:
            raise Exception("DICOM record has been deleted")
        else:
            self._key = keys[0]
            return keys

    def _set_key(self):
        self._key = self.manager._keys(self.loc())[0]
        #self._key = self.manager.register.index[self.loc()][0]

    def key(self):
        try:
            key_removed = self.manager._at(self._key, 'removed')
#         key_removed = self.manager.register.at[self._key, 'removed']
        except:
            self._set_key()
        else:
            if key_removed:
                self._set_key()
        return self._key

    @property
    def status(self): 
        return self.manager.status

    @property
    def dialog(self):
        return self.manager.dialog

    def mute(self):
        self._mute = True
        
    def unmute(self):
        self._mute = False

    def progress(self, *args, **kwargs):
        if not self._mute:
            self.manager.status.progress(*args, **kwargs)

    def message(self, *args, **kwargs):
        if not self._mute:
            self.manager.status.message(*args, **kwargs)

    def type(self):
        return self.__class__.__name__

    def files(self):
        return [self.manager.filepath(key) for key in self.keys()]

    def exists(self):
        #if self.manager.register is None:
        if not self.manager.is_open():
            return False
        try:
            keys = self.keys().tolist()
        except:
            return False
        return keys != []

    def empty(self):
        return not self.loc().any()

    def record(self, type, uid='Database', key=None, **kwargs):
        return self.new(self.manager, uid, type, key=key, **kwargs)

    def register(self):
        return self.manager._extract(self.keys())
        #return self.manager.register.loc[self.keys(),:]
    
    def label(self):
        return self.manager.label(self.uid, key=self.key(), type=self.__class__.__name__)

    def instances(self, sort=True, sortby=None, **kwargs): 
        inst = self.manager.instances(keys=self.keys(), sort=sort, sortby=sortby, **kwargs)
        return [self.record('Instance', uid, key) for key, uid in inst.items()]

    def images(self, sort=True, sortby=None, **kwargs): 
        inst = self.manager.instances(keys=self.keys(), sort=sort, sortby=sortby, images=True, **kwargs)
        return [self.record('Instance', uid, key) for key, uid in inst.items()]

    def series(self, sort=True, **kwargs):
        series = self.manager.series(keys=self.keys(), sort=sort, **kwargs)
        return [self.record('Series', uid) for uid in series]

    def studies(self, sort=True, **kwargs):
        studies = self.manager.studies(keys=self.keys(), sort=sort, **kwargs)
        return [self.record('Study', uid) for uid in studies]

    def patients(self, sort=True, **kwargs):
        patients = self.manager.patients(keys=self.keys(), sort=sort, **kwargs)
        return [self.record('Patient', uid) for uid in patients]

    def siblings(self, **kwargs):
        siblings = self.parent().children(**kwargs)
        siblings.remove(self)
        return siblings
        # uids = self.manager.siblings(self.uid, **kwargs)
        # return [self.__class__(self.new, self.manager, uid) for uid in uids]

    def read(self):
        self.manager.read(self.uid, keys=self.keys())
        return self

    def write(self, path=None):
        if path is not None:
            self.manager.path = path
        self.manager.write(self.uid, keys=self.keys())
        self.manager._write_df()

    def clear(self):
        self.manager.clear(self.uid, keys=self.keys())

    def remove(self):
        self.manager.delete(self.uid, keys=self.keys())

    def new_patient(self, **kwargs):
        attr = {**kwargs, **self.attributes}
        uid, key = self.manager.new_patient(parent=self.uid, **attr)
        return self.record('Patient', uid, key, **attr)

    def new_study(self, **kwargs):
        attr = {**kwargs, **self.attributes}
        uid, key = self.manager.new_study(parent=self.uid, key=self.key(),**attr)
        return self.record('Study', uid, key, **attr)

    def new_series(self, **kwargs):
        attr = {**kwargs, **self.attributes}
        uid, key = self.manager.new_series(parent=self.uid, **attr)
        return self.record('Series', uid, key, **attr)

    def new_instance(self, dataset=None, **kwargs):
        attr = {**kwargs, **self.attributes}
        uid, key = self.manager.new_instance(parent=self.uid, dataset=dataset, **attr)
        return self.record('Instance', uid, key, **attr)

    def new_sibling(self, **kwargs):
        type = self.__class__.__name__
        if type == 'Database':
            return None
        return self.parent().new_child(**kwargs)

    def new_pibling(self, **kwargs):
        type = self.__class__.__name__
        if type == 'Database':
            return None
        if type == 'Patient':
            return None
        return self.parent().new_sibling(**kwargs)

    def print(self):
        self.manager.print() # print self.uid only

    def copy(self, **kwargs):
        return self.copy_to(self.parent(), **kwargs)

    def copy_to(self, target, **kwargs):
        return target._copy_from(self, **kwargs)
    
    def move_to(self, target):
        move_to(self, target)
        return self

    def set_values(self, attributes, values):
        self._key = self.manager.set_values(attributes, values, self.keys())

    def get_values(self, attributes):
        return self.manager.get_values(attributes, self.keys())

    def get_dataset(self):
        ds = self.manager.get_dataset(self.uid, self.keys())
        return ds

    def set_dataset(self, dataset):
        self.manager.set_dataset(self.uid, dataset, self.keys())

    def save(self, path=None):
        #rows = self.manager.register[self.name] == self.uid
        rows = self.manager._extract_record(self.name, self.uid)
        self.manager.save(rows)
        self.write(path)
        
    def restore(self):
        #rows = self.manager.register[self.name] == self.uid
        rows = self.manager._extract_record(self.name, self.uid)
        self.manager.restore(rows)
        self.write()

    # Needs a unit test
    def instance(self, uid=None, key=None):
        if key is not None:
            #uid = self.manager.register.at[key, 'SOPInstanceUID']
            uid = self.manager._at(key, 'SOPInstanceUID')
            if uid is None:
                return
            return self.record('Instance', uid, key=key)
        if uid is not None:
            return self.record('Instance', uid)
        key = self.key()
        #uid = self.manager.register.at[key, 'SOPInstanceUID']
        uid = self.manager._at(key, 'SOPInstanceUID')
        return self.record('Instance', uid, key=key)

    # This needs a test whether the instance is an image - else move to the next
    def image(self, **kwargs):
        return self.instance(**kwargs)


    # Needs a unit test
    def sery(self, uid=None, key=None):
        if key is not None:
            #uid = self.manager.register.at[key, 'SeriesInstanceUID']
            uid = self.manager._at(key, 'SeriesInstanceUID')
            if uid is None:
                return
            return self.record('Series', uid, key=key)
        if uid is not None:
            return self.record('Series', uid)
        key = self.key()
        #uid = self.manager.register.at[key, 'SeriesInstanceUID']
        uid = self.manager._at(key, 'SeriesInstanceUID')
        return self.record('Series', uid, key=key)

    # Needs a unit test
    def study(self, uid=None, key=None):
        if key is not None:
            #uid = self.manager.register.at[key, 'StudyInstanceUID']
            uid = self.manager._at(key, 'StudyInstanceUID')
            if uid is None:
                return
            return self.record('Study', uid, key=key)
        if uid is not None:
            return self.record('Study', uid)
        key = self.key()
        #uid = self.manager.register.at[key, 'StudyInstanceUID']
        uid = self.manager._at(key, 'StudyInstanceUID')
        return self.record('Study', uid, key=key)

    # Needs a unit test
    def patient(self, uid=None, key=None):
        if key is not None:
            #uid = self.manager.register.at[key, 'PatientID']
            uid = self.manager._at(key, 'PatientID')
            if uid is None:
                return
            return self.record('Patient', uid, key=key)
        if uid is not None:
            return self.record('Patient', uid)
        key = self.key()
        #uid = self.manager.register.at[key, 'PatientID']
        uid = self.manager._at(key, 'PatientID')
        return self.record('Patient', uid, key=key)

    def database(self):
        return self.record('Database')

    def export_as_dicom(self, path): 
        folder = self.label()
        path = export_path(path, folder)
        for child in self.children():
            child.export_as_dicom(path)

    def export_as_png(self, path): 
        folder = self.label()
        path = export_path(path, folder)
        for child in self.children():
            child.export_as_png(path)

    def export_as_csv(self, path):
        folder = self.label()
        path = export_path(path, folder)
        for child in self.children():
            child.export_as_csv(path)

    def export_as_nifti(self, path):
        folder = self.label()
        path = export_path(path, folder)
        for child in self.children():
            child.export_as_nifti(path)

    # def sort(self, sortby=['StudyDate','SeriesNumber','InstanceNumber']):
    #     self.manager.register.sort_values(sortby, inplace=True)

    def read_dataframe(*args, **kwargs):
        return read_dataframe(*args, **kwargs)

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

    # def tree(*args, **kwargs):
    #     return tree(*args, **kwargs)



#
# Functions on a list of records of the same database
#


def copy_to(records, target):
    if not isinstance(records, list):
        return records.copy_to(target)
    copy = []
    desc = target.label()
    for r, record in enumerate(records):
        record.status.progress(r+1, len(records), 'Copying ' + desc)
        copy_record = record.copy_to(target)
        if isinstance(copy_record, list):
            copy += copy_record
        else:
            copy.append(copy_record)
    record.status.hide()
    return copy

def move_to(records, target):
    #if type(records) is np.ndarray:
    #    records = records.tolist()
    if not isinstance(records, list):
        records = [records]
    mgr = records[0].manager
    uids = [rec.uid for rec in records]
    mgr.move_to(uids, target.uid, **target.attributes)
    return records

def group(records, into=None, inplace=False):
    if not isinstance(records, list):
        records = [records]
    if into is None:
        into = records[0].new_pibling()
    if inplace:
        move_to(records, into)
    else:
        copy_to(records, into)
    return into

def merge(records, into=None, inplace=False):
    if not isinstance(records, list):
        records = [records]
    children = []
    for record in records:
        children += record.children()
    new_series = group(children, into=into, inplace=inplace)
    if inplace:
        for record in records:
            record.remove()
    return new_series


# 
# Read and write
#




def read_dataframe(record, tags):
    if set(tags) <= set(record.manager.columns):
        return record.register()[tags]  
    instances = record.instances()
    return _read_dataframe_from_instance_array_values(instances, tags)


def read_dataframe_from_instance_array(instances, tags):
    mgr = instances[0].manager
    if set(tags) <= set(mgr.columns):
        keys = [i.key() for _, i in np.ndenumerate(instances)]
        return mgr._extract(keys)[tags]
    return _read_dataframe_from_instance_array_values(instances, tags)

    
def _read_dataframe_from_instance_array_values(instances, tags):
    indices = []
    data = []
    for i, instance in enumerate(instances):
        index = instance.key()
        values = instance.get_values(tags)
        indices.append(index)
        data.append(values)
        instance.progress(i+1, len(instances), 'Reading dataframe..')
    return pd.DataFrame(data, index=indices, columns=tags)





