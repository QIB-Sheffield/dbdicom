
from dbdicom.manager import Manager
from dbdicom.types.database import Database
from dbdicom.types.patient import Patient
from dbdicom.types.study import Study
from dbdicom.types.series import Series
from dbdicom.types.instance import Instance


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


def database(path=None, **kwargs):

    if path is None:
        dbr = Manager()
    else:
        dbr = Manager(path, **kwargs)
        dbr.open(path)
    return Database(create, dbr, **kwargs) 