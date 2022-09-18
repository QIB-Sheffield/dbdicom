from dbdicom.register import DbRegister
from dbdicom.methods.database import Database
from dbdicom.methods.patient import Patient
from dbdicom.methods.study import Study
from dbdicom.methods.series import Series
from dbdicom.methods.instance import Instance


def create(register, uid='Database', type=None, **kwargs):

    if uid == 'Database':
        return Database(create, register, **kwargs)
    if type is None:
        type = register.type(uid)
    if type == 'Patient':
        return Patient(create, register, uid, **kwargs)
    if type == 'Study':
        return Study(create, register, uid, **kwargs)
    if type == 'Series':
        return Series(create, register, uid, **kwargs)
    if type == 'Instance':
        return Instance(create, register, uid, **kwargs)


def new_database(path=None, **kwargs):
    if path is not None:
        return open(path, **kwargs)
    dbr = DbRegister()
    return Database(create, dbr, **kwargs)

def open_database(path, **kwargs):

    dbr = DbRegister(path, **kwargs)
    dbr.open(path)
    return Database(create, dbr, **kwargs) 