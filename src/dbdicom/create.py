import numpy as np
from dbdicom.manager import Manager
from dbdicom.types.database import Database
from dbdicom.types.patient import Patient
from dbdicom.types.study import Study
from dbdicom.types.series import Series
from dbdicom.types.instance import Instance


def create(manager, uid='Database', type=None, key=None, **kwargs):

    if uid is None:
        return
    if uid == 'Database':
        return Database(create, manager, **kwargs)

    # This case is included for convenience but should be avoided 
    # at all costs because the lookup of type at creation is very expensive.
    # Considering removing and make type a requirement
    if type is None:
        type = manager.type(uid)

    if type == 'Patient':
        return Patient(create, manager, uid, key=key, **kwargs)
    if type == 'Study':
        return Study(create, manager, uid, key=key, **kwargs)
    if type == 'Series':
        return Series(create, manager, uid, key=key, **kwargs)
    if type == 'Instance':
        return Instance(create, manager, uid, key=key, **kwargs)


def database(path=None, **kwargs):
    """_summary_

    Args:
        path (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if path is None:
        mgr = Manager()
    else:
        mgr = Manager(path, **kwargs)
        mgr.open(path)
    return Database(create, mgr, **kwargs) 


def series(array, pixels_first=False, path=None):
    """_summary_

    Args:
        array (_type_): _description_
        pixels_first (bool, optional): _description_. Defaults to False.
        path (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    db = database(path)
    patient = db.new_patient()
    study = patient.new_study()
    series = study.new_series()
    series.set_pixel_array(array, pixels_first=pixels_first)
    return series


def zeros(shape, dtype='mri', path=None):
    """_summary_

    Args:
        shape (_type_): _description_
        dtype (str, optional): _description_. Defaults to 'mri'.
        path (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    db = database(path)
    patient = db.new_patient()
    study = patient.new_study()
    return study.zeros(shape, dtype=dtype)


