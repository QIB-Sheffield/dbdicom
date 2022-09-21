from dbdicom.manager import Manager
from dbdicom.methods.database import Database
from dbdicom.methods.patient import Patient
from dbdicom.methods.study import Study
from dbdicom.methods.series import Series
from dbdicom.methods.instance import Instance


def create(manager, uid='Database', type=None, **kwargs):

    if uid == 'Database':
        return Database(create, manager, **kwargs)
    if type is None:
        type = manager.type(uid)
    if type == 'Patient':
        return Patient(create, manager, uid, **kwargs)
    if type == 'Study':
        return Study(create, manager, uid, **kwargs)
    if type == 'Series':
        return Series(create, manager, uid, **kwargs)
    if type == 'Instance':
        return Instance(create, manager, uid, **kwargs)


def new_database(path=None, **kwargs):
    if path is not None:
        return open(path, **kwargs)
    dbr = Manager()
    return Database(create, dbr, **kwargs)

def open_database(path, **kwargs):

    dbr = Manager(path, **kwargs)
    dbr.open(path)
    return Database(create, dbr, **kwargs) 