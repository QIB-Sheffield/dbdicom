import numpy as np
import pandas as pd
import dbdicom.ds.dataset as dbdataset
from dbdicom.utils.files import export_path



class Record():

    name = 'Record'

    def __init__(self, create, manager, uid='Database', key=None, **kwargs):   

        self._key = key
        self._mute = False
        self.uid = uid
        self.attributes = kwargs
        self.manager = manager
        self.new = create
    
    def __eq__(self, other):
        if other is None:
            return False
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

    def loc(self):
        return self.manager._loc(self.name, self.uid)
        # df = self.manager.register
        # return (df.removed==False) & (df[self.name]==self.uid)

    def keys(self):
        keys = self.manager._keys(self.loc())
#        keys = self.manager.register.index[self.loc()]
        if len(keys) == 0:
            if self.name == 'Database':
                return keys
            else:
                raise Exception("This record has no data")
        else:
            self._key = keys[0]
            return keys

    def _set_key(self):
        self._key = self.manager._keys(self.loc())[0]
        #self._key = self.manager.register.index[self.loc()][0]

    def key(self):
        try:
            key_removed = self.manager._at(self._key, 'removed')
#         key_removed = self.manager.register.at[self._key, 'removed']
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
    




    
    def print(self):
        """Print the contents of the DICOM database containing this object.

        See Also:
            :func:`~path`

        Example:

            >>> database = db.database('path\\to\\DICOM\\database')
            >>> database.print()
            ---------- DICOM FOLDER --------------
            DATABASE:  path\\to\\DICOM\\database
            PATIENT [0]: Patient Al Pacino
            PATIENT [1]: Patient Sean Connery
            --------------------------------------      
        """
        self.manager.print() 

    
    def path(self) -> str:
        """Directory of the DICOM database

        Returns:
            str: full path to the directory

        See Also:
            :func:`~print`

        Example:
            Create a new database in memory:

            >>> database = db.database()
            >>> print(database.path())
            None

            Open an existing DICOM database:

            >>> database = db.database('path\\to\\DICOM\\database')
            >>> print(database.path())

            path\to\DICOM\database
        """
        return self.manager.path
    

    def empty(self)->bool:
        """Check if the record has data.

        Returns:
            bool: False if the record has data, True if not

        See Also:
            :func:`~print`
            :func:`~path`

        Example:

            Check if a database on disk is empty:

            >>> database = db.database('path\\to\\database')
            >>> print(database.empty)
            False

            A new database is empty by default:

            >>> database = db.database()
            >>> print(database.empty())
            True

            A new series is empty by default, but since this automatically generates a default study and patient, the database is no longer empty:

            >>> database = db.database()
            >>> series = database.new_series()
            >>> print(series.empty(), database.empty())
            True, False            

            If the new series is populated with data on creation, it is not empty:
            >>> database = db.database()
            >>> series = db.zeros((3,128,128), in_database=database)
            >>> print(series.empty(), database.empty())
            False, False 
        """
        if self.manager.register.empty:
            return True
        return self.children() == []
        return not self.loc().any() # This gives the incorrect result for empty series, study, patients
    

    def files(self) -> list:
        """Return a list of all DICOM files saved in the database

        Returns:
            list: A list of absolute filepaths to valid DICOM files

        See Also:
            :func:`~print`
            :func:`~path`
            :func:`~empty`

        Example:

            A new database in memory has no files on disk:
    
            >>> database = db.database()
            >>> print(database.files())
            []

            If a series is created in memory, there are no files on disk:

            >>> series = db.zeros((3,128,128))
            >>> print(series.files())
            []

            If a series is created in memory, then written to disk, there are files associated. Since the default format is single-frame DICOM, there are 3 files in this case:

            >>> series = db.zeros((3,128,128))
            >>> series.write('path\\to\\DICOM\\database')
            >>> print(series.files())
            ['path\\to\\DICOM\\database\\dbdicom\\1.2.826.0.1.3680043.8.498.10200622747714198480020099226433338888.dcm', 'path\\to\\DICOM\\database\\dbdicom\\1.2.826.0.1.3680043.8.498.95074529334441498207488699470663781148.dcm', 'path\\to\\DICOM\\database\\dbdicom\\1.2.826.0.1.3680043.8.498.30452523525370800574103459899629273584.dcm']
        """
        files = [self.manager.filepath(key) for key in self.keys()]
        files = [f for f in files if f is not None] # Added 29/05/23 - check if this creates issues
        return files
    
    def label(self)->str:
        """Return a human-readable label describing the record.

        Returns:
            str: label with descriptive information.

        See Also:
            :func:`~print`

        Example:
            Print the label of a default series:

            >>> series = db.zeros((3,128,128))
            >>> print(series.label())
            Series 001 [New Series]
        """
        return self.manager.label(self.uid, key=self.key(), type=self.__class__.__name__)
    





    def progress(self, value: float, maximum: float, message: str=None):
        """Print progress message to the terminal..

        Args:
            value (float): current status
            maximum (float): maximal value
            message (str, optional): Message to include in the update. Defaults to None.

        Note:
            When working through a terminal this could easily be replicated with a print statement. The advantage of using the progress interface is that the code does not need to be changed when the computation is run through a graphical user interface (assuming this uses a compatible API). 
            
            Another advantage is that messaging can be muted/unmuted using .mute() and .unmute(), for instance when the object is passed to a subroutine.

        See Also:
            :func:`~message`
            :func:`~mute`
            :func:`~unmute`

        Example:
            >>> nr_of_slices = 3
            >>> series = db.zeros((nr_of_slices,128,128))
            >>> for slice in range(nr_of_slices):
                series.progress(1+slice, nr_of_slices, 'Looping over slices')
            Looping over slices [33 %]
            Looping over slices [67 %]
            Looping over slices [100 %]  
        """
        if not self._mute:
            self.manager.status.progress(value, maximum, message=message)


    def message(self, message: str):
        """Print a message to the user.

        Args:
            message (str): Message to be printed.

        Note:
            When working through a terminal a print statement would have exactly the same effect. The advantage of using the message interface is that the code does not need to be changed when the computation is run through a graphical user interface (assuming this uses a compatible API). 
            
            Another advantage is that messaging can be muted/unmuted using .mute() and .unmute() for instance when the object is passed to a subroutine.

        See Also:
            :func:`~progress`
            :func:`~mute`
            :func:`~unmute`

        Example:

            >>> series.message('Starting computation..')
            Starting computation..
            
            After muting the same statment does not send a message:

            >>> series.mute()
            >>> series.message('Starting computation..')

            Unmute to reactivate sending messages:

            >>> series.unmute()
            >>> series.message('Starting computation..')
            Starting computation..
        """
        if not self._mute:
            self.manager.status.message(message)

    def mute(self):
        """Prevent the object from sending status updates to the user

        See Also:
            :func:`~unmute`
            :func:`~message`
            :func:`~progress`
        
        Example:
            >>> series = db.zeros((3,128,128))
            >>> print('My message: ')
            >>> series.message('Hello World')
            >>> series.mute()
            >>> print('My message: ')
            >>> series.message('Hello World')

            My message: 
            Hello World
            My message:
        """
        self._mute = True
        
    def unmute(self):
        """Allow the object from sending status updates to the user

        Note:
            Records are unmuted by default, so unmuting is only necessary after a previouse call to mute(). Unmuting has no effect when the record is already unmuted.

        See Also:
            :func:`~mute`
            :func:`~message`
            :func:`~progress`
        
        Example:
            >>> series = db.zeros((3,128,128))
            >>> print('My message: ')
            >>> series.message('Hello World')
            >>> series.mute()
            >>> print('My message: ')
            >>> series.message('Hello World')
            >>> series.unmute()
            >>> print('My message: ')
            >>> series.message('Hello World')

            My message: 
            Hello World
            My message:
            My message:
            Hello World
        """
        self._mute = False

    def type(self):
        return self.__class__.__name__

    def exists(self):
        #if self.manager.register is None:
        if not self.manager.is_open():
            return False
        try:
            keys = self.keys().tolist()
        except:
            return False
        return keys != []

    def record(self, type, uid='Database', key=None, **kwargs):
        return self.new(self.manager, uid, type, key=key, **kwargs)

    def register(self):
        return self.manager._extract(self.keys())
        #return self.manager.register.loc[self.keys(),:]
    
    def instances(self, sort=True, sortby=None, **kwargs): 
        inst = self.manager.instances(keys=self.keys(), sort=sort, sortby=sortby, **kwargs)
        return [self.record('Instance', uid, key) for key, uid in inst.items()]

    def images(self, sort=True, sortby=None, **kwargs): 
        inst = self.manager.instances(keys=self.keys(), sort=sort, sortby=sortby, images=True, **kwargs)
        return [self.record('Instance', uid, key) for key, uid in inst.items()]

    def parent(self):
        """Return the parent of the record.

        Returns:
            Record: The parent object.

        See Also:
            :func:`~children`
            :func:`~siblings`
            :func:`~series`
            :func:`~studies`
            :func:`~patients`
            :func:`~database`
            
        Example:
            Find the parent of a given study:

            >>> study = db.study()
            >>> patient = study.parent()
            >>> print(patient.PatientName)
            New Patient
        """
        # Note this function is reimplemented in all subclasses. 
        # It is included in the Record class only for documentation purposes.
        return None

    def children(self, **kwargs)->list:
        """Return all children of the record.

        Args:
            kwargs: Provide any number of valid DICOM (tag, value) pair as keywords to filter the list.

        Returns:
            list: A list of all children.

        See Also:
            :func:`~parent`
            :func:`~siblings`
            :func:`~series`
            :func:`~studies`
            :func:`~patients`
            :func:`~database`
            
        Example:
            Find the patients of a given database:

            >>> database = db.database_hollywood()
            >>> patients = database.children()
            >>> print([p.PatientName for p in patients])
            ['James Bond', 'Scarface']

            Find all patients with a given name:

            >>> database = db.database_hollywood()
            >>> patients = database.children(PatientName='James Bond')
            >>> print([p.PatientName for p in patients])
            ['James Bond']
        """
        # Note this function is reimplemented in all subclasses. 
        # It is included in the Record class for documentation purposes.
        return []
    
    def siblings(self, **kwargs)->list:
        """Return all siblings of the record.

        Args:
            kwargs: Provide any number of valid DICOM (tag, value) pair as keywords to filter the list.

        Returns:
            list: A list of all siblings.

        See Also:
            :func:`~parent`
            :func:`~children`
            :func:`~series`
            :func:`~studies`
            :func:`~patients`
            :func:`~database`
            
        Example:
            For a given study, find all other studies of the same patient:

            >>> database = db.database_hollywood()
            >>> study = database.studies()[0]
            >>> siblings = study.siblings()
            >>> print([s.StudyDescription for s in siblings])
            ['Xray']
        """
        siblings = self.parent().children(**kwargs)
        siblings.remove(self)
        return siblings
        # uids = self.manager.siblings(self.uid, **kwargs)
        # return [self.__class__(self.new, self.manager, uid) for uid in uids]

    def series(self, sort=True, sortby=['PatientName', 'StudyDescription', 'SeriesNumber'], **kwargs)->list:
        """Return a list of series under the record.

        If the record is a study, this returns the record's children. If it is a patient, this returns a list the record's grand children.

        Args:
            sort (bool, optional): Set to False to return an unsorted list (faster). Defaults to True.
            sortby (list, optional):  list of DICOM keywords to sort the result. This argument is ignored if sort=False. Defaults to ['PatientName', 'StudyDescription', 'SeriesNumber'].
            kwargs (keyword arguments, optional): Set any number of valid DICOM (tag, value) pairs as keywords to filer the list. The result will only contain series with the appropriate values

        Returns:
            list: A list of dbdicom Series objects.

        See Also:
            :func:`~parent`
            :func:`~children`
            :func:`~siblings`
            :func:`~studies`
            :func:`~patients`
            :func:`~database`

        Example:
            Find all series in a database:

            >>> database = db.database_hollywood()
            >>> series_list = database.series()
            >>> print([s.label() for s in series_list])
            ['Series 001 [Localizer]', 'Series 002 [T2w]', 'Series 001 [Chest]', 'Series 002 [Head]', 'Series 001 [Localizer]', 'Series 002 [T2w]', 'Series 001 [Chest]', 'Series 002 [Head]']

            Find all series with a given SeriesDescription:

            >>> database = db.database_hollywood()
            >>> series_list = database.series(SeriesDescription='Chest')
            >>> print([s.label() for s in series_list])
            ['Series 001 [Chest]', 'Series 001 [Chest]']

            Find all series with a given SeriesDescription of a given Patient:

            >>> database = db.database_hollywood()
            >>> series_list = database.series(SeriesDescription='Chest', PatientName='James Bond')
            >>> print([s.label() for s in series_list])
            ['Series 001 [Chest]']
        """
        series = self.manager.series(keys=self.keys(), sort=sort, sortby=sortby, **kwargs)
        return [self.record('Series', uid) for uid in series]

    def studies(self, sort=True, sortby=['PatientName', 'StudyDescription'], **kwargs)->list:
        """Return a list of studies under the record.

        If the record is a patient, this returns the record's children. If it is a series, this returns the parent study.

        Args:
            sort (bool, optional): Set to False to return an unsorted list (faster). Defaults to True.
            sortby (list, optional):  list of DICOM keywords to sort the result. This argument is ignored if sort=False. Defaults to ['PatientName', 'StudyDescription'].
            kwargs (keyword arguments, optional): Set any number of valid DICOM (tag, value) pairs as keywords to filer the list. The result will only contain studies with the appropriate values.

        Returns:
            list: A list of dbdicom Study objects.

        See Also:
            :func:`~parent`
            :func:`~children`
            :func:`~siblings`
            :func:`~series`
            :func:`~patients`
            :func:`~database`
        
        Example:
            Find all studies in a database:

            >>> database = db.database_hollywood()
            >>> studies_list = database.studies()
            >>> print([s.label() for s in studies_list])
            ['Study MRI [None]', 'Study Xray [None]', 'Study MRI [None]', 'Study Xray [None]']

            Find all studies of a given Patient:

            >>> database = db.database_hollywood()
            >>> studies_list = database.studies(PatientName='James Bond')
            >>> print([s.label() for s in studies_list])
            ['Study MRI [None]', 'Study Xray [None]']
        """
        studies = self.manager.studies(keys=self.keys(), sort=sort, sortby=sortby, **kwargs)
        return [self.record('Study', uid) for uid in studies]


    def patients(self, sort=True, sortby=['PatientName'], **kwargs)->list:
        """Return a list of patients under the record.

        If the record is a database, this returns the children. If it is a series or a study, this returns the parent patient.

        Args:
            sort (bool, optional): Set to False to return an unsorted list (faster). Defaults to True.
            sortby (list, optional):  list of DICOM keywords to sort the result. This argument is ignored if sort=False. Defaults to ['PatientName'].
            kwargs (keyword arguments, optional): Set any number of valid DICOM (tag, value) pairs as keywords to filer the list. The result will only contain patients with the appropriate values.

        Returns:
            list: A list of dbdicom Patient objects.

        See Also:
            :func:`~parent`
            :func:`~children`
            :func:`~siblings`
            :func:`~series`
            :func:`~studies`
            :func:`~database`

        Example:
            Find all patients in a database:

            >>> database = db.database_hollywood()
            >>> patients_list = database.patients()
            >>> print([s.label() for s in patients_list])
            ['Patient James Bond', 'Patient Scarface']

            Find all patients with a given name:

            >>> database = db.database_hollywood()
            >>> patients_list = database.patients(PatientName='James Bond')
            >>> print([s.label() for s in patients_list])
            ['Patient James Bond']
        """
        patients = self.manager.patients(keys=self.keys(), sort=sort, sortby=sortby, **kwargs)
        return [self.record('Patient', uid) for uid in patients]
    
    def database(self):
        """Return the database of the record.

        Returns:
            Database: Database of the record

        See Also:
            :func:`~parent`
            :func:`~children`
            :func:`~siblings`
            :func:`~series`
            :func:`~studies`

        Example:
            >>> study = db.study()
            >>> database = study.database()
            >>> print(database.label())
            Database [in memory]
        """
        return self.record('Database')
    

    # This needs a test whether the instance is an image - else move to the next
    def image(self, **kwargs):
        return self.instance(**kwargs)
    
    # Needs a unit test
    def instance(self, uid=None, key=None):
        if key is not None:
            #uid = self.manager.register.at[key, 'SOPInstanceUID']
            uid = self.manager._at(key, 'SOPInstanceUID')
            if uid is None:
                return
            return self.record('Instance', uid, key=key)
        if uid is not None:
            return self.record('Instance', uid)
        key = self.key()
        #uid = self.manager.register.at[key, 'SOPInstanceUID']
        uid = self.manager._at(key, 'SOPInstanceUID')
        return self.record('Instance', uid, key=key)

    # Needs a unit test
    def sery(self, uid=None, key=None):
        if key is not None:
            #uid = self.manager.register.at[key, 'SeriesInstanceUID']
            uid = self.manager._at(key, 'SeriesInstanceUID')
            if uid is None:
                return
            return self.record('Series', uid, key=key)
        if uid is not None:
            return self.record('Series', uid)
        key = self.key()
        #uid = self.manager.register.at[key, 'SeriesInstanceUID']
        uid = self.manager._at(key, 'SeriesInstanceUID')
        return self.record('Series', uid, key=key)

    # Needs a unit test
    def study(self, uid=None, key=None):
        if key is not None:
            #uid = self.manager.register.at[key, 'StudyInstanceUID']
            uid = self.manager._at(key, 'StudyInstanceUID')
            if uid is None:
                return
            return self.record('Study', uid, key=key)
        if uid is not None:
            return self.record('Study', uid)
        key = self.key()
        #uid = self.manager.register.at[key, 'StudyInstanceUID']
        uid = self.manager._at(key, 'StudyInstanceUID')
        return self.record('Study', uid, key=key)

    # Needs a unit test
    def patient(self, uid=None, key=None):
        if key is not None:
            #uid = self.manager.register.at[key, 'PatientID']
            uid = self.manager._at(key, 'PatientID')
            if uid is None:
                return
            return self.record('Patient', uid, key=key)
        if uid is not None:
            return self.record('Patient', uid)
        key = self.key()
        #uid = self.manager.register.at[key, 'PatientID']
        uid = self.manager._at(key, 'PatientID')
        return self.record('Patient', uid, key=key)



    def save(self, path=None):
        #rows = self.manager.register[self.name] == self.uid
        rows = self.manager._extract_record(self.name, self.uid)
        self.manager.save(rows)
        self.write(path)
        
    def restore(self):
        #rows = self.manager.register[self.name] == self.uid
        rows = self.manager._extract_record(self.name, self.uid)
        self.manager.restore(rows)
        self.write()

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



    def new_patient(self, **kwargs):
        """Create a new patient.

        Args:
            kwargs (optional): Any valid DICOM (tag, value) pair can be assigned up front as properties of the new patient.

        Returns:
            Patient: instance of the new patient

        See Also:
            :func:`~new_study`
            :func:`~new_series`
            :func:`~new_pibling`

        Example:
            Create a new patient in a database:

            >>> database = db.database()
            >>> database.print()
            >>> patient = database.new_patient(PatientName='John Dory')
            >>> database.print()
            ---------- DICOM FOLDER --------------
            DATABASE:  new
            --------------------------------------
            ---------- DICOM FOLDER --------------
            DATABASE:  new
            PATIENT [0]: Patient John Dory      
            --------------------------------------

            A lower-level record can also create a new patient:

            >>> series = db.series()
            >>> series.print()
            >>> patient = series.new_patient(PatientName='John Dory')
            >>> series.print()
            ---------- DICOM FOLDER --------------
            DATABASE:  new
            PATIENT [0]: Patient New Patient
                STUDY [0]: Study New Study [None]
                SERIES [0]: Series 001 [New Series]
                    Nr of instances: 0
            --------------------------------------
            ---------- DICOM FOLDER --------------
            DATABASE:  new
            PATIENT [0]: Patient John Dory
            PATIENT [1]: Patient New Patient
                STUDY [0]: Study New Study [None]
                SERIES [0]: Series 001 [New Series]
                    Nr of instances: 0
            --------------------------------------
        """
        attr = {**kwargs, **self.attributes}
        uid, key = self.manager.new_patient(parent=self.uid, **attr)
        return self.record('Patient', uid, key, **attr)

    def new_study(self, **kwargs):
        """Create a new study.

        Args:
            kwargs (optional): Any valid DICOM (tag, value) pair can be assigned up front as properties of the new study.

        Returns:
            Study: instance of the new study

        See Also:
            :func:`~new_patient`
            :func:`~new_series`
            :func:`~new_pibling`

        Example:
            Create a new study in a patient:

            >>> patient = db.patient()
            >>> patient.print()
            >>> study = patient.new_study(StudyDescription='Manager protocol')
            >>> patient.print()
            ---------- DICOM FOLDER --------------
            DATABASE:  new
            PATIENT [0]: Patient New Patient
            --------------------------------------
            ---------- DICOM FOLDER --------------
            DATABASE:  new
            PATIENT [0]: Patient New Patient
                STUDY [0]: Study Manager protocol [None]
            --------------------------------------

            Any other record can also create a new study. Missing intermediate generations are created automatically:

            >>> database = db.database()
            >>> database.print()
            >>> study = database.new_study(StudyDescription='Manager protocol')
            >>> database.print()
            ---------- DICOM FOLDER --------------
            DATABASE:  new
            --------------------------------------
            ---------- DICOM FOLDER --------------
            DATABASE:  new
            PATIENT [0]: Patient New Patient
                STUDY [0]: Study Manager protocol [None]
            --------------------------------------
        """
        attr = {**kwargs, **self.attributes}
        uid, key = self.manager.new_study(parent=self.uid, key=self.key(),**attr)
        return self.record('Study', uid, key, **attr)

    def new_series(self, **kwargs):
        """Create a new series.

        Args:
            kwargs (optional): Any valid DICOM (tag, value) pair can be assigned up front as properties of the new series.

        Returns:
            Series: instance of the new series

        See Also:
            :func:`~new_patient`
            :func:`~new_study`
            :func:`~new_pibling`

        Example:
            Create a new series in a study:

            >>> study = db.study()
            >>> study.print()
            >>> study = study.new_series(SeriesDescription='MRI T2w')
            >>> study.print()
            ---------- DICOM FOLDER --------------
            DATABASE:  new
            PATIENT [0]: Patient New Patient
                STUDY [0]: Study New Study [None]
            --------------------------------------
            ---------- DICOM FOLDER --------------
            DATABASE:  new
            PATIENT [0]: Patient New Patient
                STUDY [0]: Study New Study [None]
                SERIES [0]: Series 001 [MRI T2w]
                    Nr of instances: 0
            --------------------------------------

            Any other record can also create a new series. Missing intermediate generations are created automatically:

            >>> database = db.database()
            >>> database.print()
            >>> series = database.new_series(SeriesDescription='MRI T2w')
            >>> database.print()
            ---------- DICOM FOLDER --------------
            DATABASE:  new
            --------------------------------------
            ---------- DICOM FOLDER --------------
            DATABASE:  new
            PATIENT [0]: Patient New Patient   
                STUDY [0]: Study New Study [None]
                SERIES [0]: Series 001 [MRI T2w]
                    Nr of instances: 0
            --------------------------------------
        """
        attr = {**kwargs, **self.attributes}
        uid, key = self.manager.new_series(parent=self.uid, **attr)
        return self.record('Series', uid, key, **attr)

    def new_instance(self, dataset=None, **kwargs):
        attr = {**kwargs, **self.attributes}
        uid, key = self.manager.new_instance(parent=self.uid, dataset=dataset, **attr)
        return self.record('Instance', uid, key, **attr)

    # def new_sibling(self, suffix=None, **kwargs):
    #     # type = self.__class__.__name__
    #     # if type == 'Database':
    #     #     return None
    #     return self.parent().new_child(**kwargs)

    def new_pibling(self, **kwargs):
        """Create a new sibling of the parent record (pibling).

        Args:
            kwargs (optional): Any valid DICOM (tag, value) pair can be assigned up front as properties of the new pibling.

        Returns:
            Record: instance of the new parent

        See Also:
            :func:`~new_patient`
            :func:`~new_study`
            :func:`~new_series`

        Example:
            Create a new study as sibling of the parent study:

            >>> series = db.series()
            >>> series.print()
            >>> study = series.new_pibling(StudyDescription='Manager protocol')
            >>> series.print()
            ---------- DICOM FOLDER --------------
            DATABASE:  new
            PATIENT [0]: Patient New Patient
                STUDY [0]: Study New Study [None]
                SERIES [0]: Series 001 [New Series]
                    Nr of instances: 0
            --------------------------------------
            ---------- DICOM FOLDER --------------
            DATABASE:  new
            PATIENT [0]: Patient New Patient
                STUDY [0]: Study Manager protocol [None]
                STUDY [1]: Study New Study [None]
                SERIES [0]: Series 001 [New Series]
                    Nr of instances: 0
            --------------------------------------
        """
        type = self.__class__.__name__
        if type == 'Database':
            return None
        if type == 'Patient':
            return None
        return self.parent().new_sibling(**kwargs)

    def remove(self):
        """Remove a record from the database.

        See Also:
            :func:`~copy`
            :func:`~copy_to`
            :func:`~move_to`

        Example:
            Create a new study and remove it again:

            >>> database = db.database()
            >>> study = database.new_study(StudyDescription='Demo Study')
            >>> database.print()
            >>> study.remove()
            >>> database.print()
            ---------- DICOM FOLDER --------------
            DATABASE:  new
            PATIENT [0]: Patient New Patient
                STUDY [0]: Study Demo Study [None]
            --------------------------------------
            ---------- DICOM FOLDER --------------
            DATABASE:  new
            PATIENT [0]: Patient New Patient
            --------------------------------------
        """
        self.manager.delete(self.uid, keys=self.keys())


    def copy(self, **kwargs):
        """Return a copy of the record under the same parent.

        Args:
            kwargs (optional): Any valid DICOM (tag, value) pair can be assigned up front as properties of the copy.

        Returns:
            Record: copy of the same type.

        See Also:
            :func:`~remove`
            :func:`~copy_to`
            :func:`~move_to`

        Example:
            Copy a DICOM series:

            >>> series = db.series()
            >>> series.print()
            >>> copy_series = series.copy(SeriesDescription='Clone')
            >>> series.print()
            ---------- DICOM FOLDER --------------
            DATABASE:  new
            PATIENT [0]: Patient New Patient
                STUDY [0]: Study New Study [None]
                SERIES [0]: Series 001 [New Series]
                    Nr of instances: 0
            --------------------------------------
            Copying series New Series (1/1) [100 %]
            ---------- DICOM FOLDER --------------
            DATABASE:  new
            PATIENT [0]: Patient New Patient
                STUDY [0]: Study New Study [None]
                SERIES [0]: Series 001 [New Series]
                    Nr of instances: 0
                SERIES [1]: Series 001 [2]
                    Nr of instances: 1
            --------------------------------------
        """
        return self.copy_to(self.parent(), **kwargs)

    def copy_to(self, parent, **kwargs):
        """Return a copy of the record under another parent.

        Args:
            parent: parent where the copy will be placed.
            kwargs (optional): Any valid DICOM (tag, value) pair can be assigned up front as properties of the copy.

        Returns:
            Record: copy of the same type.

        See Also:
            :func:`~remove`
            :func:`~copy`
            :func:`~move_to`

        Example:
            Copy a DICOM series to a new study:

            >>> series = db.series()
            >>> series.print()
            >>> new_study = series.new_pibling(StudyDescription='Study of clones')
            >>> copy_series = series.copy_to(new_study, SeriesDescription='Clone')
            >>> series.print()
            ---------- DICOM FOLDER --------------
            DATABASE:  new
            PATIENT [0]: Patient New Patient
                STUDY [0]: Study New Study [None]
                SERIES [0]: Series 001 [New Series]
                    Nr of instances: 0
            --------------------------------------
            Copying series New Series (1/1) [100 %]
            ---------- DICOM FOLDER --------------
            DATABASE:  new
            PATIENT [0]: Patient New Patient
                STUDY [0]: Study New Study [None]
                SERIES [0]: Series 001 [New Series]
                    Nr of instances: 0
                STUDY [1]: Study Study of clones [None]
                SERIES [0]: Series 001 [1]
                    Nr of instances: 1
            --------------------------------------
        """
        return parent._copy_from(self, **kwargs)
    
    def move_to(self, parent):
        """Move the record to another parent.

        Args:
            parent: parent where the record will be moved to.

        See Also:
            :func:`~remove`
            :func:`~copy`
            :func:`~copy_to`

        Example:
            Move a DICOM series to a new study:

            >>> series = db.series()
            >>> series.print()
            >>> new_study = series.new_pibling(StudyDescription='Study of clones')
            >>> series.move_to(new_study)
            >>> series.print()

        """
        move_to(self, parent)
        return self



    def set_values(self, attributes, values):
        self._key = self.manager.set_values(attributes, values, self.keys())

    def get_values(self, attributes):
        return self.manager.get_values(attributes, self.keys())

    def get_dataset(self):
        ds = self.manager.get_dataset(self.uid, self.keys())
        return ds

    def set_dataset(self, dataset):
        self.manager.set_dataset(self.uid, dataset, self.keys())





  




    def export_as_dicom(self, path): 
        folder = self.label()
        path = export_path(path, folder)
        for child in self.children():
            child.export_as_dicom(path)

    def export_as_png(self, path): 
        folder = self.label()
        path = export_path(path, folder)
        for child in self.children():
            child.export_as_png(path)

    def export_as_csv(self, path):
        folder = self.label()
        path = export_path(path, folder)
        for child in self.children():
            child.export_as_csv(path)

    def export_as_nifti(self, path):
        folder = self.label()
        path = export_path(path, folder)
        for child in self.children():
            child.export_as_nifti(path)

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
    desc = target.label()
    for r, record in enumerate(records):
        record.status.progress(r+1, len(records), 'Copying ' + desc)
        copy_record = record.copy_to(target)
        if isinstance(copy_record, list):
            copy += copy_record
        else:
            copy.append(copy_record)
    record.status.hide()
    return copy

def move_to(records, target):
    #if type(records) is np.ndarray:
    #    records = records.tolist()
    if not isinstance(records, list):
        records = [records]
    mgr = records[0].manager
    uids = [rec.uid for rec in records]
    mgr.move_to(uids, target.uid, **target.attributes)
    return records

def group(records, into=None, inplace=False):
    if not isinstance(records, list):
        records = [records]
    if into is None:
        into = records[0].new_pibling()
    if inplace:
        move_to(records, into)
    else:
        copy_to(records, into)
    return into

def merge(records, into=None, inplace=False):
    if not isinstance(records, list):
        records = [records]
    children = []
    for record in records:
        children += record.children()
    new_series = group(children, into=into, inplace=inplace)
    if inplace:
        for record in records:
            record.remove()
    return new_series


# 
# Read and write
#




def read_dataframe(record, tags):
    if set(tags) <= set(record.manager.columns):
        return record.register()[tags]  
    instances = record.instances()
    return _read_dataframe_from_instance_array_values(instances, tags)


def read_dataframe_from_instance_array(instances, tags):
    mgr = instances[0].manager
    if set(tags) <= set(mgr.columns):
        keys = [i.key() for _, i in np.ndenumerate(instances)]
        return mgr._extract(keys)[tags]
    return _read_dataframe_from_instance_array_values(instances, tags)

    
def _read_dataframe_from_instance_array_values(instances, tags):
    indices = []
    data = []
    for i, instance in enumerate(instances):
        index = instance.key()
        values = instance.get_values(tags)
        indices.append(index)
        data.append(values)
        instance.progress(i+1, len(instances), 'Reading dataframe..')
    return pd.DataFrame(data, index=indices, columns=tags)





