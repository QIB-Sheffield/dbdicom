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

    See Also:
        :func:`~patient`
        :func:`~study`
        :func:`~series`
    
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


def database_hollywood()->Database:
    """Create an empty toy database for demonstration purposes.

    Returns:
        Database: Database with two patients, two studies per patient and two empty series per study.

    See Also:
        :func:`~database`

    Example:
        >>> database = db.database_hollywood()
        >>> database.print()
        ---------- DICOM FOLDER --------------
        DATABASE:  new
        PATIENT [0]: Patient James Bond
            STUDY [0]: Study MRI [None]  
                SERIES [0]: Series 001 [Localizer]
                    Nr of instances: 0        
                SERIES [1]: Series 002 [T2w]
                    Nr of instances: 0      
            STUDY [1]: Study Xray [None]
                SERIES [0]: Series 001 [Chest]
                    Nr of instances: 0
                SERIES [1]: Series 002 [Head]
                    Nr of instances: 0
        PATIENT [1]: Patient Scarface
            STUDY [0]: Study MRI [None]
                SERIES [0]: Series 001 [Localizer]
                    Nr of instances: 0
                SERIES [1]: Series 002 [T2w]
                    Nr of instances: 0
            STUDY [1]: Study Xray [None]
                SERIES [0]: Series 001 [Chest]
                    Nr of instances: 0
                SERIES [1]: Series 002 [Head]
                    Nr of instances: 0
        --------------------------------------
    """
    hollywood = database()

    james_bond = hollywood.new_patient(PatientName='James Bond')
    james_bond_mri = james_bond.new_study(StudyDescription='MRI')
    james_bond_mri_localizer = james_bond_mri.new_series(SeriesDescription='Localizer')
    james_bond_mri_T2w = james_bond_mri.new_series(SeriesDescription='T2w')
    james_bond_xray = james_bond.new_study(StudyDescription='Xray')
    james_bond_xray_chest = james_bond_xray.new_series(SeriesDescription='Chest')
    james_bond_xray_head = james_bond_xray.new_series(SeriesDescription='Head')

    scarface = hollywood.new_patient(PatientName='Scarface')
    scarface_mri = scarface.new_study(StudyDescription='MRI')
    scarface_mri_localizer = scarface_mri.new_series(SeriesDescription='Localizer')
    scarface_mri_T2w = scarface_mri.new_series(SeriesDescription='T2w')
    scarface_xray = scarface.new_study(StudyDescription='Xray')
    scarface_xray_chest = scarface_xray.new_series(SeriesDescription='Chest')
    scarface_xray_head = scarface_xray.new_series(SeriesDescription='Head')

    return hollywood


# THESE SHOULD MOVE TO SERIES MODULE

def series(dtype='mri', in_study:Study=None, in_database:Database=None)->Series: 
    """Create an empty DICOM series.

    Args:
        dtype (str, optional): The type of the series to create. Defaults to 'mri'.
        in_study (Study, optional): If provided, the series is created in this study. Defaults to None.
        in_database (Database, optional): If provided, the series is created in this database. Defaults to None.

    Returns:
        Series: DICOM series with defaults for all attributes.
 
    Raises:
        ValueError: if a dtype is requested that is currently not yet implemented

    See Also:
        :func:`~database`
        :func:`~patient`
        :func:`~study`
        :func:`~as_series`
        :func:`~zeros`

    Example:
        Create an empty series in memory:

        >>> sery = db.series()
        >>> sery.print()
        ---------- DICOM FOLDER --------------
        DATABASE:  new
        PATIENT [0]: Patient New Patient
            STUDY [0]: Study New Study [None]
            SERIES [0]: Series 001 [New Series]
                Nr of instances: 0
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
            db = database()
        else:
            db = in_database
        patient = db.new_patient()
        study = patient.new_study()
        series = study.new_series()
    return series


def as_series(array:np.ndarray, pixels_first=False, dtype='mri', in_study:Study=None, in_database:Database=None)->Series:
    """Create a DICOM series from a numpy array.

    Args:
        array (np.ndarray): Array with image data
        pixels_first (bool, optional): Flag to specify whether the pixel indices are first or last. Defaults to False.
        dtype (str, optional): The type of the series to create. Defaults to 'mri'.
        in_study (Study, optional): If provided, the series is created in this study. Defaults to None.
        in_database (Database, optional): If provided, the series is created in this database. Defaults to None.

    Returns:
        Series: DICOM series containing the provided array as image data and defaults for all other parameters.
 
    Raises:
        ValueError: if a dtype is requested that is currently not yet implemented

    See Also:
        :func:`~series`
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
    sery = series(dtype=dtype, in_study=in_study, in_database=in_database)
    sery.mute()
    sery.set_pixel_array(array, pixels_first=pixels_first)
    sery.unmute()
    return sery


def zeros(shape:tuple, **kwargs) -> Series:
    """Create a DICOM series populated with zeros.

    This is a convenience wrapper providing a numpy-like interface for :func:`~as_series`.

    Args:
        shape (tuple): shape of the array
        kwargs: see :func:`~series`
        
    Returns:
        Series: DICOM series with zero values

    See Also:
        :func:`~series`
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


# THESE SHOULD MOVE TO STUDY MODULE

def study(in_patient:Patient=None, in_database:Database=None)->Study: 
    """Create an empty DICOM study record.

    Args:
        in_patient (Patient, optional): If provided, the study is created in this Patient. Defaults to None.
        in_database (Database, optional): If provided, the study is created in this database. Defaults to None.

    Returns:
        Study: DICOM study with defaults for all attributes.

    See Also:
        :func:`~database`
        :func:`~patient`
        :func:`~series`

    Example:
        Create an empty study in memory:

        >>> study = db.study()
        >>> study.print()
        ---------- DICOM FOLDER --------------
        DATABASE:  new
        PATIENT [0]: Patient New Patient
            STUDY [0]: Study New Study [None]
        --------------------------------------

        Note since no Patient object is provided, a default hierarchy is created automatically.
    """
    
    if in_patient is not None:
        study = in_patient.new_study()
    else:
        if in_database is None:
            db = database()
        else:
            db = in_database
        patient = db.new_patient()
        study = patient.new_study()
    return study


# THESE SHOULD MOVE TO Patient MODULE

def patient(in_database:Database=None)->Patient: 
    """Create an empty DICOM patient record.

    Args:
        in_database (Database, optional): If provided, the patient is created in this database. Defaults to None.

    Returns:
        Study: DICOM patient with defaults for all attributes.

    See Also:
        :func:`~database`
        :func:`~study`
        :func:`~series`

    Example:
        Create an empty patient in memory:

        >>> patient = db.patient()
        >>> patient.print()

        ---------- DICOM FOLDER --------------
        DATABASE:  new
        PATIENT [0]: Patient New Patient
        --------------------------------------

        Note since no Patient object is provided, a default hierarchy is created automatically.
    """
    if in_database is None:
        db = database()
    else:
        db = in_database
    patient = db.new_patient()
    return patient














