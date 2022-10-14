import pandas as pd
import dbdicom.ds.dataset as dbdataset
from dbdicom.manager import Manager


class DbRecord():

    def __init__(self, create, manager, uid='Database', **kwargs):   

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
        if attribute in ['uid', 'manager', 'attributes', 'new']:
            self.__dict__[attribute] = value
        else:
            #self.set_values(attribute, value)
            self.set_values([attribute], [value])
        
    def __setitem__(self, attributes, values):
        self.set_values(attributes, values)

    @property
    def status(self):
        return self.manager.status

    @property
    def dialog(self):
        return self.manager.dialog

    def mute(self):
        self.status.mute()
        
    def unmute(self):
        self.status.unmute()

    def files(self):
        return self.manager.filepaths(self.uid)

    def type(self):
        return self.manager.type(self.uid)

    def in_database(self):
        return self.manager.in_database(self.uid)

    def empty(self):
        return self.manager.instances(self.uid) == []

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
        type = self.__class__.__name__
        if type == 'Database':
            return None
        if type == 'Patient':
            return self.new(self.manager, 'Database')
        uid = self.manager.parent(self.uid)
        if type == 'Study':
            return self.new(self.manager, uid, 'Patient')
        if type == 'Series':
            return self.new(self.manager, uid, 'Study')
        if type == 'Instance':
            return self.new(self.manager, uid, 'Series')

    def children(self, **kwargs):
        return children(self, **kwargs)
    def instances(self, **kwargs):
        return instances(self, **kwargs)
    def series(self, **kwargs):
        return series(self, **kwargs)
    def studies(self, **kwargs):
        return studies(self, **kwargs)
    def patients(self, **kwargs):
        return patients(self, **kwargs)

    def siblings(self, **kwargs):
        uids = self.manager.siblings(self.uid, **kwargs)
        return [self.__class__(self.new, self.manager, uid) for uid in uids]

    def new_patient(self, **kwargs):
        attr = {**kwargs, **self.attributes}
        #desc = attr['PatientName'] if 'PatientName' in attr else 'New Patient'
        #uid = self.manager.new_patient(parent=self.uid, PatientName = desc)
        uid = self.manager.new_patient(parent=self.uid, **attr)
        return self.new(self.manager, uid, 'Patient', **attr)

    def new_study(self, **kwargs):
        attr = {**kwargs, **self.attributes}
        #desc = attr['StudyDescription'] if 'StudyDescription' in attr else 'New Study'
        #uid = self.manager.new_study(parent=self.uid, StudyDescription = desc)
        uid = self.manager.new_study(parent=self.uid, **attr)
        return self.new(self.manager, uid, 'Study', **attr)

    def new_series(self, **kwargs):
        attr = {**kwargs, **self.attributes}
        #desc = attr['SeriesDescription'] if 'SeriesDescription' in attr else 'New Series'
        #uid = self.manager.new_series(parent=self.uid, SeriesDescription = desc)
        uid = self.manager.new_series(parent=self.uid, **attr)
        return self.new(self.manager, uid, 'Series', **attr)

    def new_instance(self, dataset=None, **kwargs):
        attr = {**kwargs, **self.attributes}
        uid = self.manager.new_instance(parent=self.uid, dataset=dataset, **attr)
        return self.new(self.manager, uid, 'Instance', **attr)

    def new_child(self, dataset=None, **kwargs): 
        attr = {**kwargs, **self.attributes}
        uid = self.manager.new_child(uid=self.uid, dataset=dataset, **attr)
        return self.new(self.manager, uid, **attr)

    def new_sibling(self, **kwargs):
        uid = self.manager.new_sibling(uid=self.uid, **kwargs)
        return self.__class__(self.new, self.manager, uid)

    def new_pibling(self):
        type = self.__class__.__name__
        if type == 'Database':
            return None
        if type == 'Patient':
            return None
        uid = self.manager.new_pibling(uid=self.uid)
        if type == 'Study':
            return self.new(self.manager, uid, 'Patient')
        if type == 'Series':
            return self.new(self.manager, uid, 'Study')
        if type == 'Instance':
            return self.new(self.manager, uid, 'Series')

    def record(self, type, uid):
        return self.new(self.manager, uid, type)

    def register(self):
        keys = self.manager.keys(self.uid)
        return self.manager.register.loc[keys,:]
    
    def label(self):
        return self.manager.label(self.uid)

    def print(self):
        self.manager.print() # print self.uid only

    def read(self):
        self.manager.read(self.uid)
        return self

    def write(self, path=None):
        if path is not None:
            self.manager.path = path
        self.manager.write(self.uid)
        self.manager._write_df()

    def clear(self):
        self.manager.clear(self.uid)

    def remove(self):
        self.manager.delete(self.uid)

    def copy(self):
        return self.copy_to(self.parent())

    def copy_to(self, target):
        return copy_to(self, target)[0]
    
    def move_to(self, target):
        move_to(self, target)
        return self

    def set_values(self, attributes, values):
        set_values(self, attributes, values)

    def get_values(self, attributes):
        return get_values(self, attributes)

    def get_dataset(self):
        return self.manager.get_dataset(self.uid)

    def set_dataset(self, dataset):
        self.manager.set_dataset(self.uid, dataset)

    def save(self, path=None):
        self.manager.save(self.uid)
        self.write(path)
        
    def restore(self):
        self.manager.restore(self.uid)
        self.write()

    def instance(self, uid):
        return self.new(self.manager, uid, 'Instance')
    def sery(self, uid):
        return self.new(self.manager, uid, 'Series')
    def study(self, uid):
        return self.new(self.manager, uid, 'Study')
    def patient(self, uid):
        return self.new(self.manager, uid, 'Patient')
    def database(self):
        return self.new(self.manager, 'Database')

    def export_as_dicom(self, path): 
        mgr = Manager(path)
        mgr.open(path)
        uids = self.manager.instances(self.uid)
        self.manager.export_datasets(uids, mgr)

    def export_as_csv(*args, **kwargs):
        export_as_csv(*args, **kwargs)

    def export_as_png(*args, **kwargs):
        export_as_png(*args, **kwargs)

    def export_as_nifti(*args, **kwargs):
        export_as_nifti(*args, **kwargs)

    def sort(self, sortby=['StudyDate','SeriesNumber','InstanceNumber']):
        self.manager.register.sort_values(sortby, inplace=True)

    def read_dataframe(*args, **kwargs):
        return read_dataframe(*args, **kwargs)

    # def tree(*args, **kwargs):
    #     return tree(*args, **kwargs)


def read_dataframe(record, tags):
    if set(tags) <= set(record.manager.columns):
        return record.register()[tags]  
    data = []
    indices = []
    instances = record.instances()
    for i, instance in enumerate(instances):
        index = record.manager.keys(instance=instance.uid)[0]
        indices.append(index)
        row = get_values(instance, tags)
        data.append(row)
        #record.status.progress(i+1, len(instances), 'Reading dataframe..')
    return pd.DataFrame(data, index=indices, columns=tags)


def export_as_csv(record, directory=None, filename=None, columnHeaders=None):
    """Export all images as csv files"""

    if directory is None: 
        directory = record.dialog.directory(message='Please select a folder for the csv data')
    if filename is None:
        filename = record.SeriesDescription
    for i, instance in enumerate(record.instances()):
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
    for i, instance in enumerate(record.instances()):
        instance.export_as_png( 
            directory = directory, 
            filename = filename + ' [' + str(i) + ']')

def export_as_nifti(record, directory=None, filename=None):
    """Export all images as nifti files"""

    if directory is None: 
        directory = record.dialog.directory(message='Please select a folder for the png data')
    if filename is None:
        filename = record.SeriesDescription
    for i, instance in enumerate(record.instances()):
        instance.export_as_nifti( 
            directory = directory, 
            filename = filename + ' [' + str(i) + ']')




#
# Functions on a list of records of the same database
#

def get_values(records, attributes):

    if not isinstance(records, list):
        mgr = records.manager
        return mgr.get_values(records.uid, attributes)
    uids = [rec.uid for rec in records]
    mgr = records[0].manager
    return mgr.get_values(uids, attributes)

def set_values(records, attributes, values):

    if not isinstance(records, list):
        records = [records]
    uids = [rec.uid for rec in records]
    mgr = records[0].manager
    mgr.set_values(uids, attributes, values)

def children(records, **kwargs):

    if not isinstance(records, list):
        records = [records]
    mgr = records[0].manager
    uids = [rec.uid for rec in records]
    uids = mgr.children(uids, **kwargs)
    return [records[0].new(mgr, uid) for uid in uids]

def instances(records, **kwargs):

    if not isinstance(records, list):
        records = [records]
    mgr = records[0].manager
    uids = [rec.uid for rec in records]
    uids = mgr.instances(uids, **kwargs)
    return [records[0].new(mgr, uid, 'Instance') for uid in uids]

def series(records, **kwargs):

    if not isinstance(records, list):
        records = [records]
    mgr = records[0].manager
    uids = [rec.uid for rec in records]
    uids = mgr.series(uids, **kwargs)
    return [records[0].new(mgr, uid, 'Series') for uid in uids]

def studies(records, **kwargs):

    if not isinstance(records, list):
        records = [records]
    mgr = records[0].manager
    uids = [rec.uid for rec in records]
    uids = mgr.studies(uids, **kwargs)
    return [records[0].new(mgr, uid, 'Study') for uid in uids]

def patients(records, **kwargs):

    if not isinstance(records, list):
        records = [records]
    mgr = records[0].manager
    uids = [rec.uid for rec in records]
    uids = mgr.patients(uids, **kwargs)
    return [records[0].new(mgr, uid, 'Patient') for uid in uids]

def copy_to(records, target):

    if not isinstance(records, list):
        records = [records]
    mgr = records[0].manager
    uids = [rec.uid for rec in records]
    uids = mgr.copy_to(uids, target.uid, **target.attributes)
    if isinstance(uids, list):
        return [records[0].new(mgr, uid) for uid in uids]
    else:
        return [records[0].new(mgr, uids)]

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
    return group(children(records), into=into)
