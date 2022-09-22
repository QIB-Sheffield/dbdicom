
import dbdicom.dsdicom.dataset as dbdataset


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
            self.set_values(attribute, value)
        
    def __setitem__(self, attributes, values):
        self.set_values(attributes, values)

    @property
    def status(self):
        return self.manager.status

    @property
    def dialog(self):
        return self.manager.dialog

    def type(self):
        return self.manager.type(self.uid)

    @property
    def generation(self): # Obsolete
        type = self.__class__.__name__
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
        uids = self.manager.siblings(self.uid, **kwargs)
        return [self.__class__(self.new, self.manager, uid) for uid in uids]

    def new_patient(self, **kwargs):
        attr = {**kwargs, **self.attributes}
        uid = self.manager.new_patient(parent=self.uid, 
            PatientName = attr['PatientName'] if 'PatientName' in attr else 'New Patient',
        )
        return self.new(self.manager, uid, 'Patient', **attr)

    def new_study(self, **kwargs):
        attr = {**kwargs, **self.attributes}
        uid = self.manager.new_study(parent=self.uid, 
            StudyDescription = attr['StudyDescription'] if 'StudyDescription' in attr else 'New Study',
        )
        return self.new(self.manager, uid, 'Study', **attr)

    def new_series(self, **kwargs):
        attr = {**kwargs, **self.attributes}
        uid = self.manager.new_series(parent=self.uid,
            SeriesDescription = attr['SeriesDescription'] if 'SeriesDescription' in attr else 'New Series',
        )
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
    
    def label(self):
        return self.manager.label(self.uid)

    def print(self):
        self.manager.print() # print self.uid only

    def read(self):
        self.manager.read(self.uid)

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
        return copy_to([self], target)[0]
    
    def move_to(self, target):
        move_to([self], target)
        return self

    def set_values(self, attributes, values):
        set_values([self], attributes, values)

    def get_values(self, attributes):
        return get_values([self], attributes)[0]

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



#
# Functions on a list of records of the same database
#

def get_values(records, attributes):

    uids = [rec.uid for rec in records]
    dbr = records[0].manager
    return dbr.get_values(uids, attributes)

def set_values(records, attributes, values):

    uids = [rec.uid for rec in records]
    dbr = records[0].manager
    dbr.set_values(uids, attributes, values)

def children(records, **kwargs):

    dbr = records[0].manager
    uids = [rec.uid for rec in records]
    uids = dbr.children(uids, **kwargs)
    return [records[0].new(dbr, uid) for uid in uids]

def instances(records, **kwargs):

    dbr = records[0].manager
    uids = [rec.uid for rec in records]
    uids = dbr.instances(uids, **kwargs)
    return [records[0].new(dbr, uid, 'Instance') for uid in uids]

def series(records, **kwargs):

    dbr = records[0].manager
    uids = [rec.uid for rec in records]
    uids = dbr.series(uids, **kwargs)
    return [records[0].new(dbr, uid, 'Series') for uid in uids]

def studies(records, **kwargs):

    dbr = records[0].manager
    uids = [rec.uid for rec in records]
    uids = dbr.studies(uids, **kwargs)
    return [records[0].new(dbr, uid, 'Study') for uid in uids]

def patients(records, **kwargs):

    dbr = records[0].manager
    uids = [rec.uid for rec in records]
    uids = dbr.patients(uids, **kwargs)
    return [records[0].new(dbr, uid, 'Patient') for uid in uids]

def copy_to(records, target):

    dbr = records[0].manager
    uids = [rec.uid for rec in records]
    uids = dbr.copy_to(uids, target.uid, **target.attributes)
    if isinstance(uids, list):
        return [records[0].new(dbr, uid) for uid in uids]
    else:
        return [records[0].new(dbr, uids)]

def move_to(records, target):

    dbr = records[0].manager
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
