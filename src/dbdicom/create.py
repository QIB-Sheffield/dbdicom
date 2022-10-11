import numpy as np
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
        mgr = Manager()
    else:
        mgr = Manager(path, **kwargs)
        mgr.open(path)
    return Database(create, mgr, **kwargs) 


def series(array, pixels_first=False, path=None):
    db = database(path)
    patient = db.new_patient()
    study = patient.new_study()
    series = study.new_series()
    series.set_pixel_array(array, pixels_first=pixels_first)
    return series

def zeros(shape, dtype='mri', path=None):
    db = database(path)
    patient = db.new_patient()
    study = patient.new_study()
    return study.zeros(shape, dtype=dtype)


