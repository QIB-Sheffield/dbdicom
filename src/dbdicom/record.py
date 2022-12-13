import timeit
import pandas as pd
import dbdicom.ds.dataset as dbdataset
from dbdicom.manager import Manager


class DbRecord():

    def __init__(self, create, manager, uid='Database', key=None, **kwargs):   

        self._key = key
        self._mute = False
        self.uid = uid
        self.attributes = kwargs
        self.manager = manager
        self.new = create
    
    def __eq__(self, other):
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
        df = self.manager.register
        return (df.removed==False) & (df[self.name]==self.uid)

    def keys(self):
        keys = self.manager.register.index[self.loc()]
        if len(keys) == 0:
            raise Exception("DICOM record has been deleted")
        else:
            self._key = keys[0]
            return keys

    def _set_key(self):
        self._key = self.manager.register.index[self.loc()][0]

    def key(self):
        try:
            key_removed = self.manager.register.at[self._key, 'removed']
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
        if self.manager.register is None:
            return False
        keys = self.keys().tolist()
        return keys != []

    def empty(self):
        return not self.loc().any()

    def record(self, type, uid='Database', key=None, **kwargs):
        return self.new(self.manager, uid, type, key=key, **kwargs)

    def register(self):
        return self.manager.register.loc[self.keys(),:]
    
    def label(self):
        return self.manager.label(self.uid, key=self.key(), type=self.__class__.__name__)

    def instances(self, sort=True, **kwargs):
        inst = self.manager.instances(keys=self.keys(), sort=sort, **kwargs)
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

    def copy(self):
        return self.copy_to(self.parent())

    def copy_to(self, target):
        return target._copy_from(self)
    
    def move_to(self, target):
        move_to(self, target)
        return self

    def set_values(self, attributes, values):
        self._key = self.manager.set_values(attributes, values, self.keys())

    def get_values(self, attributes):
        return self.manager.get_values(attributes, self.keys())

    def get_dataset(self):
        return self.manager.get_dataset(self.uid, self.keys())

    def set_dataset(self, dataset):
        self.manager.set_dataset(self.uid, dataset, self.keys())

    def save(self, path=None):
        rows = self.manager.register[self.name] == self.uid
        self.manager.save(rows)
        self.write(path)
        
    def restore(self):
        rows = self.manager.register[self.name] == self.uid
        self.manager.restore(rows)
        self.write()

    # Needs a unit test
    def instance(self, uid=None, key=None):
        if key is not None:
            uid = self.manager.register.at[key, 'SOPInstanceUID']
            if uid is None:
                return
            return self.record('Instance', uid, key=key)
        if uid is not None:
            return self.record('Instance', uid)
        key = self.key()
        uid = self.manager.register.at[key, 'SOPInstanceUID']
        return self.record('Instance', uid, key=key)

    # Needs a unit test
    def sery(self, uid=None, key=None):
        if key is not None:
            uid = self.manager.register.at[key, 'SeriesInstanceUID']
            if uid is None:
                return
            return self.record('Series', uid, key=key)
        if uid is not None:
            return self.record('Series', uid)
        key = self.key()
        uid = self.manager.register.at[key, 'SeriesInstanceUID']
        return self.record('Series', uid, key=key)

    # Needs a unit test
    def study(self, uid=None, key=None):
        if key is not None:
            uid = self.manager.register.at[key, 'StudyInstanceUID']
            if uid is None:
                return
            return self.record('Study', uid, key=key)
        if uid is not None:
            return self.record('Study', uid)
        key = self.key()
        uid = self.manager.register.at[key, 'StudyInstanceUID']
        return self.record('Study', uid, key=key)

    # Needs a unit test
    def patient(self, uid=None, key=None):
        if key is not None:
            uid = self.manager.register.at[key, 'PatientID']
            if uid is None:
                return
            return self.record('Patient', uid, key=key)
        if uid is not None:
            return self.record('Patient', uid)
        key = self.key()
        uid = self.manager.register.at[key, 'PatientID']
        return self.record('Patient', uid, key=key)

    def database(self):
        return self.record('Database')

    def export_as_dicom(self, path): 
        files = [self.manager.filepath(key) for key in self.keys()]
        mgr = Manager(path)
        mgr.open(path)
        mgr.import_datasets(files)

    def export_as_csv(*args, **kwargs):
        export_as_csv(*args, **kwargs)

    def export_as_png(*args, **kwargs):
        export_as_png(*args, **kwargs)

    def export_as_nifti(*args, **kwargs):
        export_as_nifti(*args, **kwargs)

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
    for record in records:
        copy_record = record.copy_to(target)
        if isinstance(copy_record, list):
            copy += copy_record
        else:
            copy.append(copy_record)
    return copy

def move_to(records, target):
    if not isinstance(records, list):
        records = [records]
    mgr = records[0].manager
    uids = [rec.uid for rec in records]
    mgr.move_to(uids, target.uid, **target.attributes)
    return records

def group(records, into=None):
    if not isinstance(records, list):
        records = [records]
    if into is None:
        into = records[0].new_pibling()
    copy_to(records, into)
    return into

def merge(records, into=None):
    if not isinstance(records, list):
        records = [records]
    children = []
    for record in records:
        children += record.children()
    return group(children, into=into)


# 
# Read and write
#


def read_dataframe(record, tags):
    if set(tags) <= set(record.manager.columns):
        return record.register()[tags]  
    indices = []
    data = []
    instances = record.instances()
    for i, instance in enumerate(instances):
        #index = record.manager.keys(instance=instance.uid)[0]
        index = instance.key()
        values = instance.get_values(tags)
        indices.append(index)
        data.append(values)
        record.progress(i+1, len(instances), 'Reading dataframe..')
    return pd.DataFrame(data, index=indices, columns=tags)


def export_as_csv(record, directory=None, filename=None, columnHeaders=None):
    """Export all images as csv files"""

    if directory is None: 
        directory = record.dialog.directory(message='Please select a folder for the csv data')
    if filename is None:
        filename = record.SeriesDescription
    instances = record.instances()
    for i, instance in enumerate(instances):
        instance.status.progress(i+1, len(instances))
        instance.export_as_csv( 
            directory = directory, 
            filename = filename + ' [' + str(i) + ']', 
            columnHeaders = columnHeaders)

def export_as_png(record, directory=None, filename=None):
    """Export all images as png files"""

    if directory is None: 
        directory = record.dialog.directory(message='Please select a folder for the png data')
    if filename is None:
        filename = record.SeriesDescription
    instances = record.instances()
    for i, instance in enumerate(instances):
        instance.status.progress(i+1, len(instances))
        instance.export_as_png( 
            directory = directory, 
            filename = filename + ' [' + str(i) + ']')

def export_as_nifti(record, directory=None, filename=None):
    """Export all images as nifti files"""

    if directory is None: 
        directory = record.dialog.directory(message='Please select a folder for the png data')
    if filename is None:
        filename = record.SeriesDescription
    instances = record.instances()
    for i, instance in enumerate(instances):
        instance.status.progress(i+1, len(instances))
        instance.export_as_nifti( 
            directory = directory, 
            filename = filename + ' [' + str(i) + ']')


