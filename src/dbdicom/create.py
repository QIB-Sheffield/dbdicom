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


def database(path:str=None, **kwargs) -> Database:
    """Open an existing database or create a new one.

    Args:
        path (str, optional): path to an existing database. In case none is provided, this will create a new empty database.
        kwargs: any valid DICOM (tag, value) pair can be provided as keyword argument. These attributes will be assigned to the database and inherited by all DICOM objects later saved in the database.

    Returns:
        Database: Instance of the Database class.

    Note:
        If no path is provided, a new database is created in memory. Any changes or additions to that database will only exist in memory until the new database is saved with .save().
    
    Example:

        Create a new database in memory and print the contents:

        >>> database = db.database()
        >>> database.print()
        ---------- DICOM FOLDER --------------
        DATABASE:  new
        --------------------------------------

        Open an existing DICOM database and check the contents:

        >>> database = db.database('path\\to\\DICOM\\database')
        >>> database.print()
        ---------- DICOM FOLDER --------------
        DATABASE:  path\\to\\DICOM\\database
        PATIENT [0]: Patient Al Pacino
        PATIENT [1]: Patient Sean Connery
        --------------------------------------
    """
    if path is None:
        mgr = Manager()
    else:
        mgr = Manager(path, **kwargs)
        mgr.open(path)
    return Database(create, mgr, **kwargs) 


# THESE SHOULD MOVE TO SERIES MODULE

def as_series(array:np.ndarray, dtype='mri', pixels_first=False, in_study:Study=None, in_database:Database=None, path:str=None)->Series:
    """Create a DICOM series from a numpy array.

    Args:
        array (np.ndarray): Array with image data
        dtype (str, optional): The type of the series to create. Defaults to 'mri'.
        pixels_first (bool, optional): Flag to specify whether the pixel indices are first or last. Defaults to False.
        in_study (Study, optional): If provided, the series is created in this study. Defaults to None.
        in_database (Database, optional): If provided, the series is created in this database. Defaults to None.
        path (str, optional): if provided, a database is created at this location. Otherwise the series is created within a new database in memory.

    Returns:
        Series: DICOM series containing the provided array as image data and defaults for all other parameters.
 
    Raises:
        ValueError: if a dtype is requested that is currently not yet implemented

    See Also:
        :func:`~zeros`

    Example:
        Create a series containing a 3-dimensional array:

        >>> array = np.zeros((3, 128, 128))
        >>> zeros = db.as_series(array)
        >>> zeros.print()
        ---------- DICOM FOLDER --------------
        DATABASE:  new
        PATIENT [0]: Patient New Patient
            STUDY [0]: Study New Study [None]
            SERIES [0]: Series 001 [New Series]
                Nr of instances: 3
        --------------------------------------

        Note since no Patient and Study objects are provided, a default hierarchy is created automatically.
    """
    if dtype not in ['mri', 'MRImage']:
        message = 'dbdicom can only create images of type MRImage at this stage'
        raise ValueError(message)
    
    if in_study is not None:
        series = in_study.new_series()
    else:
        if in_database is None:
            db = database(path)
        else:
            db = in_database
        patient = db.new_patient()
        study = patient.new_study()
        series = study.new_series()
    series.set_pixel_array(array, pixels_first=pixels_first)
    return series


def zeros(shape:tuple, **kwargs) -> Series:
    """Create a DICOM series populated with zeros.

    This is a convenience wrapper providing a numpy-like interface for :func:`~as_series`.

    Args:
        shape (tuple): shape of the array
        kwargs: see :func:`~as_series`
        
    Returns:
        Series: DICOM series with zero values

    See Also:
        :func:`~as_series`

    Example:
        Create a series containing a 3-dimensional array of zeros:

        >>> zeros = db.zeros((3, 128, 128))
        >>> zeros.print()
        ---------- DICOM FOLDER --------------
        DATABASE:  new
        PATIENT [0]: Patient New Patient
            STUDY [0]: Study New Study [None]
            SERIES [0]: Series 001 [New Series]
                Nr of instances: 3
        --------------------------------------

        Note since no Patient and Study objects are provided, a default hierarchy is created automatically.
    """
    array = np.zeros(shape, dtype=np.float32)
    return as_series(array, **kwargs)



def series(*args, **kwargs): # OBSOLETE - remove
    return as_series(*args, **kwargs)











