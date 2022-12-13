"""
Maintains an index of all files on disk.
"""

import os
import copy
import timeit
#from tkinter import N
import pandas as pd
import numpy as np

from dbdicom.message import StatusBar, Dialog
import dbdicom.utils.files as filetools
import dbdicom.utils.dcm4che as dcm4che
import dbdicom.ds.dataset as dbdataset
from dbdicom.ds.create import read_dataset, SOPClass, new_dataset

class DatabaseCorrupted(Exception):
    pass


class Manager(): 
    """Programming interface for reading and writing a DICOM folder."""

    # The column labels of the register
    columns = [    
        'PatientID', 'StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID', 'SOPClassUID', 
        'PatientName', 'StudyDescription', 'StudyDate', 'SeriesDescription', 'SeriesNumber', 'InstanceNumber', 
        'ImageOrientationPatient', 'ImagePositionPatient', 'PixelSpacing', 'SliceThickness', 'SliceLocation', 'AcquisitionTime',
    ]

    def default(self):
        return [None, None, None, None, None,
            None, None, None, None, int(-1), int(-1),
            None, None, None, float(-1.0), float(-1.0), None,
        ]


    def __init__(self, path=None, dataframe=None, status=StatusBar(), dialog=Dialog()):
        """Initialise the folder with a path and objects to message to the user.
        
        When used inside a GUI, status and dialog should be instances of the status bar and 
        dialog class defined in `wezel`.

        path = None: The index manages data in memory
        dataframe = None: no database open
        """  
        if dataframe is None:
            dataframe = pd.DataFrame(index=[], columns=self.columns)
        # THIS NEEDS A MECHANISM TO PREVENT ANOTHER Manager to open the same database.
        self.status = status
        self.dialog = dialog 
        self.path = path
        self.register = dataframe
        self.dataset = {}
        self._pause_extensions = False
        self._new_keys = []
        self._new_data = []

    def scan(self, unzip=False):
        """
        Reads all files in the folder and summarises key attributes in a table for faster access.
        """
#       Take unzip out until test is developed - less essential feature
#        if unzip:
#            filetools._unzip_files(self.path, self.status)

        #self.read_dataframe()

        if self.path is None:
            self.register = pd.DataFrame(index=[], columns=self.columns)
            self.dataset = {}
            return
        files = filetools.all_files(self.path)
        self.register = dbdataset.read_dataframe(files, self.columns+['NumberOfFrames'], self.status, path=self.path, message='Reading database..')
        self.register['removed'] = False
        self.register['created'] = False
        self.register['placeholder'] = False
        # No support for multiframe data at the moment
        self._multiframe_to_singleframe()
        self.register.drop('NumberOfFrames', axis=1, inplace=True)
        # Create placeholder rows for unsaved changes
        df = self.register.copy(deep=True)
        df.index = [self.new_key() for _ in range(df.shape[0])]
        df.removed = True
        df.created = True
        df.placeholder = True
        self.register = pd.concat([self.register, df])
        return self


    def _multiframe_to_singleframe(self):
        """Converts all multiframe files in the folder into single-frame files.
        
        Reads all the multi-frame files in the folder,
        converts them to singleframe files, and delete the original multiframe file.
        """
        if self.path is None:
            # Low priority - we are not creating multiframe data from scratch yet
            # So will always be loaded from disk initially where the solution exists. 
            # Solution: save data in a temporary file, use the filebased conversion, 
            # the upload the solution and delete the temporary file.
            raise ValueError('Multi-frame to single-frame conversion does not yet exist from data in memory')
        singleframe = self.register.NumberOfFrames.isnull() 
        multiframe = singleframe == False
        nr_multiframe = multiframe.sum()
        if nr_multiframe != 0: 
            cnt=0
            for relpath in self.register[multiframe].index.values:
                cnt+=1
                msg = "Converting multiframe file " + relpath
                self.status.progress(cnt, nr_multiframe, message=msg)
                #
                # Create these in the dbdicom folder, not in the original folder.
                #
                filepath = os.path.join(self.path, relpath)
                singleframe_files = dcm4che.split_multiframe(filepath) 
                if singleframe_files != []:            
                    # add the single frame files to the dataframe
                    df = dbdataset.read_dataframe(singleframe_files, self.columns, path=self.path)
                    df['removed'] = False
                    df['created'] = False
                    df['placeholder'] = False
                    self.register = pd.concat([self.register, df])
                    # delete the original multiframe 
                    os.remove(filepath)
                # drop the file also if the conversion has failed
                self.register.drop(index=relpath, inplace=True)

    def _pkl(self):
        """ Returns the file path of the .pkl file"""
        if self.path is None:
            return None
        filename = os.path.basename(os.path.normpath(self.path)) + ".pkl"
        return os.path.join(self.path, filename) 

    def npy(self, uid):
        # Not in use - default path for temporary storage in numoy format
        path = os.path.join(self.path, "dbdicom_npy")
        if not os.path.isdir(path): 
            os.mkdir(path)
        file = os.path.join(path, uid + '.npy') 
        return file

    def _write_df(self):
        """ Writes the dataFrame as a .pkl file"""
        if self.path is None:
            return
        file = self._pkl()
        self.register.to_pickle(file)

    def _read_df(self):
        """Reads the dataFrame from a .pkl file """
        if self.path is None:
            return
        file = self._pkl()
        self.register = pd.read_pickle(file)

    def write_csv(self, file):
        """ Writes the dataFrame as a .csv file for visual inspection"""
        self.register.to_csv(file)

    def filepath(self, key):
        """Return the full filepath for a given relative path.
        
        Returns None for data that live in memory only."""
        # Needs a formal test for completeness
        if self.path is None:
            return None
        return os.path.join(self.path, key)

    def filepaths(self, *args, **kwargs):
        """Return a list of full filepaths for all dicom files in the folder"""
        # Needs a formal test for completeness
        return [self.filepath(key) for key in self.keys(*args, **kwargs)]

    def open(self, path=None, unzip=False):
        """Opens a DICOM folder for read and write.
        
        Reads the contents of the folder and summarises all DICOM files
        in a dataframe for faster access next time. The dataframe is saved 
        as a pkl file when the folder is closed with `.close()`. 
        All non-DICOM files in the folder are ignored.
        
        Args:
            path: The full path to the directory that is to be opened.

        """
        if path is not None:
            self.path = path
        if self.path is None:
            raise ValueError('Cannot open database - no path is specified')
        if os.path.exists(self._pkl()):
            self._read_df()
        else:
            self.scan(unzip=unzip)
        return self


    def type(self, uid=None, key=None):
        """Is the UID a patient, study, series or dataset"""

        if uid is None:
            return None
        if uid == 'Database':
            return uid

        if key is None:
            df = self.register
            type = df.columns[df.isin([uid]).any()].values
            if type.size == 0: # uid does not exists in the database
                return None
            else:
                type = type[0]
        else:
            df = self.register.loc[key,:]
            type = df[df.isin([uid])].index[0]

        if type == 'PatientID':
            return 'Patient'
        if type == 'StudyInstanceUID':
            return 'Study'
        if type == 'SeriesInstanceUID':
            return 'Series'
        if type == 'SOPInstanceUID':
            return 'Instance'


    def tree(self, depth=3):

        df = self.register
        if df is None:
            raise ValueError('Cannot build tree - no database open')
        df = df[df.removed == False]
        df.sort_values(['PatientName','StudyDate','SeriesNumber','InstanceNumber'], inplace=True)
        
        database = {'uid': self.path}
        database['patients'] = []
        for uid_patient in df.PatientID.dropna().unique():
            patient = {'uid': uid_patient}
            database['patients'].append(patient)
            if depth >= 1:
                df_patient = df[df.PatientID == uid_patient]
                patient['key'] = df_patient.index[0]
                patient['studies'] = []
                for uid_study in df_patient.StudyInstanceUID.dropna().unique():
                    study = {'uid': uid_study}
                    patient['studies'].append(study)
                    if depth >= 2:
                        df_study = df_patient[df_patient.StudyInstanceUID == uid_study]
                        study['key'] = df_study.index[0]
                        study['series'] = []
                        for uid_sery in df_study.SeriesInstanceUID.dropna().unique():
                            series = {'uid': uid_sery}
                            study['series'].append(series)
                            if depth == 3:
                                df_series = df_study[df_study.SeriesInstanceUID == uid_sery]
                                series['key'] = df_series.index[0]
        return database


    def keys(self,
        uid = None, 
        patient = None,
        study = None,
        series = None,
        instance = None, 
        dropna = False): 
        """Return a list of indices for all dicom datasets managed by the index.
        
        These indices are strings with unique relative paths 
        that either link to an existing file in the database or can be used for 
        writing a database that is in memory.
        """

        df = self.register
        if df is None:
            raise ValueError('Cant return dicom files - no database open')

        # If no arguments are provided
        if (uid is None) & (patient is None) & (study is None) & (series is None) & (instance is None):
            return []

        if isinstance(uid, list):
            if 'Database' in uid:
                return self.keys('Database', dropna=dropna)

        not_deleted = df.removed == False

        if uid == 'Database':
            keys = not_deleted[not_deleted].index.tolist()
            if dropna:
                keys = [key for key in keys if self.register.at[key,'SOPInstanceUID'] is not None]
            return keys

        # If arguments are provided, create a list of unique datasets
        # keys = []
        if uid is not None:
            if not isinstance(uid, list):
                uid = [uid]
            uid = [i for i in uid if i is not None]
            rows = np.isin(df, uid).any(axis=1) & not_deleted
        if patient is not None:
            if not isinstance(patient, list):
                rows = (df.PatientID==patient) & not_deleted
            else:
                patient = [i for i in patient if i is not None]
                rows = df.PatientID.isin(patient) & not_deleted
        if study is not None:
            if not isinstance(study, list):
                rows = (df.StudyInstanceUID==study) & not_deleted
            else:
                study = [i for i in study if i is not None]
                rows = df.StudyInstanceUID.isin(study) & not_deleted
        if series is not None:
            if not isinstance(series, list):
                rows = (df.SeriesInstanceUID==series) & not_deleted
            else:
                series = [i for i in series if i is not None]
                rows = df.SeriesInstanceUID.isin(series) & not_deleted
        if instance is not None: 
            if not isinstance(instance, list):
                rows = (df.SOPInstanceUID==instance) & not_deleted
            else:
                instance = [i for i in instance if i is not None]
                rows = df.SOPInstanceUID.isin(instance) & not_deleted

        keys = df.index[rows].tolist()
        if dropna:
            keys = [key for key in keys if self.register.at[key,'SOPInstanceUID'] is not None]
        return keys

    def value(self, key, column):
        try:
            if isinstance(key, pd.Index):
                return self.register.loc[key, column].values
            if not isinstance(key, list) and not isinstance(column, list):
                return self.register.at[key, column]
            else:
                return self.register.loc[key, column].values
        except:
            return None

    def parent(self, uid=None):
        # For consistency with other definitions
        # Allow uid to be list and return list if multiple parents are found
        """Returns the UID of the parent object"""

        keys = self.keys(uid)
        if keys == []:
            return None
        row = self.register.loc[keys[0]].values.tolist()
        i = row.index(uid)
        if self.columns[i] == 'PatientID':
            return 'Database'
        else:
            return row[i-1]


    def filter(self, uids=None, **kwargs):
        uids = [id for id in uids if id is not None]
        if not kwargs:
            return uids
        vals = list(kwargs.values())
        attr = list(kwargs.keys())
        return [id for id in uids if self.get_values(attr, uid=id) == vals]
        #return [id for id in uids if function(self.get_values(attr, uid=id), vals)]


    def filter_instances(self, df, **kwargs):
        df.dropna(inplace=True)
        if not kwargs:
            return df
        vals = list(kwargs.values())
        attr = list(kwargs.keys())
        keys = [key for key in df.index if self.get_values(attr, [key]) == vals]
        return df[keys]


    def instances(self, uid=None, keys=None, sort=True, **kwargs):
        if keys is None:
            keys = self.keys(uid)
        if sort:
            sortby = ['PatientName', 'StudyDescription', 'SeriesNumber', 'InstanceNumber']
            df = self.register.loc[keys, sortby + ['SOPInstanceUID']]
            df.sort_values(sortby, inplace=True)
            df = df.SOPInstanceUID
        else:
            df = self.register.loc[keys,'SOPInstanceUID']
        return self.filter_instances(df, **kwargs)


    def series(self, uid=None, keys=None, sort=True, **kwargs):
        if keys is None:
            keys = self.keys(uid)
        if sort:
            sortby = ['PatientName', 'StudyDescription', 'SeriesNumber']
            df = self.register.loc[keys, sortby + ['SeriesInstanceUID']]
            df.sort_values(sortby, inplace=True)
            df = df.SeriesInstanceUID
        else:
            df = self.register.loc[keys,'SeriesInstanceUID']
        uids = df.unique().tolist()
        return self.filter(uids, **kwargs)


    def studies(self, uid=None, keys=None, sort=True, **kwargs):
        if keys is None:
            keys = self.keys(uid)
        if sort:
            sortby = ['PatientName', 'StudyDescription']
            df = self.register.loc[keys, sortby + ['StudyInstanceUID']]
            df.sort_values(sortby, inplace=True)
            df = df.StudyInstanceUID
        else:
            df = self.register.loc[keys,'StudyInstanceUID']
        uids = df.unique().tolist()
        return self.filter(uids, **kwargs)


    def patients(self, uid=None, keys=None, sort=True, **kwargs):
        if keys is None:
            keys = self.keys(uid)
        if sort:
            sortby = ['PatientName']
            df = self.register.loc[keys, sortby + ['PatientID']]
            df.sort_values(sortby, inplace=True)
            df = df.PatientID
        else:
            df = self.register.loc[keys,'PatientID']
        uids = df.unique().tolist()
        return self.filter(uids, **kwargs)


    def pause_extensions(self):
        self._pause_extensions = True


    def resume_extensions(self):
        self._pause_extensions = False
        self.extend()


    def extend(self):
        if self._pause_extensions:
            return
        if self._new_keys == []:
            return
        df = pd.DataFrame(self._new_data, index=self._new_keys, columns=self.columns)
        df['removed'] = False
        df['created'] = True
        df['placeholder'] = False
        self.register = pd.concat([self.register, df])
        self._new_data = []
        self._new_keys = []


    def new_patient(self, parent='Database', **kwargs):
        # Allow multiple to be made at the same time

        PatientName = kwargs['PatientName'] if 'PatientName' in kwargs else 'New Patient'

        data = self.default()
        data[0] = dbdataset.new_uid()
        data[5] = PatientName

        key = self.new_key()
        self._new_data.append(data)
        self._new_keys.append(key)
        self.extend()

        return data[0], key


    def new_study(self, parent=None, key=None, **kwargs):
        # Allow multiple to be made at the same time

        StudyDescription = kwargs['StudyDescription'] if 'StudyDescription' in kwargs else 'New Study'

        if key is None:
            if parent is None:
                parent, key = self.new_patient()
            elif self.type(parent) != 'Patient':
                parent, key = self.new_patient(parent)
            else:
                key = self.keys(patient=parent)[0]

        data = self.default()
        data[0] = self.value(key, 'PatientID')
        data[1] = dbdataset.new_uid()
        data[5] = self.value(key, 'PatientName')
        data[6] = StudyDescription

        if self.value(key, 'StudyInstanceUID') is None:
            # New patient without studies - use existing row
            self.register.loc[key, self.columns] = data
        else:
            # Patient with existing study - create new row
            key = self.new_key()
            self._new_data.append(data)
            self._new_keys.append(key)
            self.extend()

        return data[1], key


    def new_series(self, parent=None, key=None, **kwargs):
        # Allow multiple to be made at the same time

        SeriesDescription = kwargs['SeriesDescription'] if 'SeriesDescription' in kwargs else 'New Series'

        if key is None:
            if parent is None:
                parent, key = self.new_study()
            elif self.type(parent) != 'Study':
                #parent = self.studies(parent)[0]
                parent, key = self.new_study(parent)
            else:
                key = self.keys(study=parent)[0]

        # data = self.value(key, self.columns)
        data = self.register.loc[key, self.columns].values.tolist()
        data[2] = dbdataset.new_uid()
        data[3] = self.default()[3]
        data[4] = self.default()[4]
        data[8] = SeriesDescription
        data[9] = 1 + len(self.series(parent))
        data[10] = self.default()[10]

        if self.value(key, 'SeriesInstanceUID') is None:
            # New study without series - use existing row
            self.register.loc[key, self.columns] = data
        else:
            # Study with existing series - create new row
            key = self.new_key()
            self._new_data.append(data)
            self._new_keys.append(key)
            self.extend()

        return data[2], key

    
    def new_instance(self, parent=None, dataset=None, key=None, **kwargs):
        # Allow multiple to be made at the same time

        if key is None:
            if parent is None:
                parent, key = self.new_series()
            elif self.type(parent) != 'Series':
                # parent = self.series(parent)[0] 
                parent, key = self.new_series(parent)
            else:
                key = self.keys(series=parent)[0]

        data = self.value(key, self.columns)
        data[3] = dbdataset.new_uid()
        data[4] = self.default()[4]
        #data[10] = 1 + len(self.instances(parent))
        data[10] = 1 + len(self.instances(keys=self.keys(series=parent)))

        if self.value(key, 'SOPInstanceUID') is None:
            # New series without instances - use existing row
            self.register.loc[key, self.columns] = data
        else:
            # Series with existing instances - create new row
            key = self.new_key()
            self._new_data.append(data)
            self._new_keys.append(key)
            self.extend()

        if dataset is not None:
            self.set_dataset(data[3], dataset)

        return data[3], key


    # def in_database(self, uid):
    #     keys = self.keys(uid)
    #     return keys != []


    # def is_empty(self, instance):
    #     # Needs a unit test
    #     key = self.keys(instance)[0]
    #     if key in self.dataset:
    #         return False
    #     else:
    #         file = self.filepath(key)
    #         if file is None:
    #             return True
    #         elif not os.path.exists(file):
    #             return True
    #         else:
    #             return False

    def get_instance_dataset(self, key):
    
        """Gets a datasets for a single instance
        
        Datasets in memory will be returned.
        If they are not in memory, and the database exists on disk, they will be read from disk.
        If they are not in memory, and the database does not exist on disk, an exception is raised.
        """
        if key in self.dataset:
            # If in memory, get from memory
            return self.dataset[key]
        # If not in memory, read from disk
        file = self.filepath(key)
        if file is None: # No dataset assigned yet
            return
        if not os.path.exists(file):  # New instance, series, study or patient 
            return 
        return read_dataset(file, self.dialog)  


    def get_dataset(self, uid, keys=None, message=None):
        """Gets a list of datasets for a single record
        
        Datasets in memory will be returned.
        If they are not in memory, and the database exists on disk, they will be read from disk.
        If they are not in memory, and the database does not exist on disk, an exception is raised.
        """
        if uid is None: # empty record
            return
        if keys is None:
            keys = self.keys(uid)
        dataset = []
        for key in keys:
            ds = self.get_instance_dataset(key) 
            dataset.append(ds)
        if self.type(uid, keys[0]) == 'Instance':
            if dataset == []:
                return
            else:
                return dataset[0]
        else:
            return dataset


    # def _get_dataset(self, instances):
    #     """Helper function"""

    #     for key, uid in instances.items():
    #         ds = self.get_dataset(uid, [key])
    #         if ds is not None:
    #             return ds
    #     return None


    # def _get_values(self, instances, attr):
    #     """Helper function"""

    #     #ds = self._get_dataset(instances)
    #     ds = None
    #     for key, uid in instances.items():
    #         ds = self.get_dataset(uid, [key])
    #         if ds is not None:
    #             break
    #     if ds is None:
    #         return [None] * len(attr)
    #     else:
    #         return ds.get_values(attr)


    def _get_values(self, keys, attr):
        """Helper function"""

        #ds = self._get_dataset(instances)
        ds = None
        for key in keys:
            ds = self.get_instance_dataset(key)
            if ds is not None:
                break
        if ds is None:
            return [None] * len(attr)
        else:
            return ds.get_values(attr)


    def series_header(self, key):
        """Attributes and values inherited from series, study and patient"""

        attr_patient = ['PatientID', 'PatientName']
        attr_study = ['StudyInstanceUID', 'StudyDescription', 'StudyDate']
        attr_series = ['SeriesInstanceUID', 'SeriesDescription', 'SeriesNumber'] 

        parent = self.register.at[key, 'SeriesInstanceUID']
        keys = self.keys(series=parent, dropna=True)
        if keys != []:
            attr = list(set(dbdataset.module_patient() + dbdataset.module_study() + dbdataset.module_series()))
            vals = self._get_values(keys, attr)
        else:
            parent = self.register.at[key, 'StudyInstanceUID']
            keys = self.keys(study=parent, dropna=True)
            if keys != []:
                attr = list(set(dbdataset.module_patient() + dbdataset.module_study()))
                vals = self._get_values(keys, attr)
                attr += attr_series
                vals += self.value(key, attr_series).tolist()
            else:
                parent = self.register.at[key, 'PatientID']
                keys = self.keys(patient=parent, dropna=True)
                if keys != []:
                    attr = dbdataset.module_patient()
                    vals = self._get_values(keys, attr)
                    attr += attr_study + attr_series
                    vals += self.value(key, attr_study + attr_series).tolist()
                else:
                    attr = attr_patient + attr_study + attr_series
                    vals = self.value(key, attr).tolist()
        return attr, vals


    def study_header(self, key):
        """Attributes and values inherited from series, study and patient"""

        attr_patient = ['PatientID', 'PatientName']
        attr_study = ['StudyInstanceUID', 'StudyDescription', 'StudyDate']

        parent = self.register.at[key, 'StudyInstanceUID']
        keys = self.keys(study=parent, dropna=True)
        if keys != []:
            attr = list(set(dbdataset.module_patient() + dbdataset.module_study()))
            vals = self._get_values(keys, attr)
        else:
            parent = self.register.at[key, 'PatientID']
            keys = self.keys(patient=parent, dropna=True)
            if keys != []:
                attr = dbdataset.module_patient()
                vals = self._get_values(keys, attr)
                attr += attr_study
                vals += self.value(key, attr_study).tolist()
            else:
                attr = attr_patient + attr_study
                vals = self.value(key, attr).tolist()
        return attr, vals


    def patient_header(self, key):
        """Attributes and values inherited from series, study and patient"""

        attr_patient = ['PatientID', 'PatientName']

        parent = self.register.at[key, 'PatientID']
        keys = self.keys(patient=parent, dropna=True)
        if keys != []:
            attr = dbdataset.module_patient()
            vals = self._get_values(keys, attr)
        else:
            attr = attr_patient
            vals = self.value(key, attr).tolist()
        return attr, vals


    def set_instance_dataset(self, instance, ds, key=None):

        if isinstance(ds, list):
            if len(ds) > 1:
                raise ValueError('Cannot set multiple datasets to a single instance')
            else:
                ds = ds[0]
        if key is None:
            keys = self.keys(instance)
            if keys == []: # instance does not exist
                return
            key = keys[0]

        data = self.register.loc[key, self.columns]
        data[4] = ds.SOPClassUID
        ds.set_values(self.columns, data)
        if self.value(key, 'created'):
            for i, c in enumerate(self.columns):
                self.register.at[key, c] = data[i]
            self.dataset[key] = ds
            return key
        else:
            # If the dataset has not yet been edited
            # find the placeholder row and copy the data
            placeholder = (self.register.SOPInstanceUID == instance) & (self.register.placeholder)
            placeholder = placeholder[placeholder]
            placeholder = placeholder.index[0]
            #self.register.loc[placeholder, self.columns] = data
            for i, c in enumerate(self.columns):
                self.register.at[placeholder, c] = data[i]
            # remove the current row and make the placeholder row current
            self.register.at[key, 'removed'] = True
            self.register.at[placeholder, 'removed'] = False
            self.register.at[placeholder, 'placeholder'] = False
            self.dataset[placeholder] = ds
            return placeholder
            # new_key = self.new_key()
            # self.dataset[new_key] = ds

            # self._new_data.append(data)
            # self._new_keys.append(new_key)
            # self.extend()


    def set_dataset(self, uid, dataset, keys=None):

        if keys is None:
            parent_keys = self.keys(uid)
        else:
            parent_keys = keys

        # LOOKUP!!!
        # ELIMINATE
        if self.type(uid, parent_keys[0]) == 'Instance': 
            self.set_instance_dataset(uid, dataset, parent_keys[0])
            return

        if not isinstance(dataset, list):
           dataset = [dataset]
         
        attr, vals = self.series_header(parent_keys[0])

        new_data = []
        new_keys = []
        instances = self.value(parent_keys, 'SOPInstanceUID').tolist()

        for ds in dataset:
            try:
                ind = instances.index(ds.SOPInstanceUID)
            except:  # Save dataset in new instance

                # Set parent modules
                ds.set_values(attr, vals)

                # Set values in manager
                key = parent_keys[0]
                data = self.value(key, self.columns)
                data[3] = dbdataset.new_uid()
                data[4] = ds.SOPClassUID
                nrs = self.value(parent_keys, 'InstanceNumber')
                nrs = [n for n in nrs if n != -1]
                if nrs == []:
                    data[10] = 1
                else:
                    data[10] = 1 + max(nrs)
                ds.set_values(self.columns, data)

                # Add to database in memory
                new_key = self.new_key()
                self.dataset[new_key] = ds
                new_data.append(data)
                new_keys.append(new_key)

            else: # If the dataset is already in the object

                #key = self.keys(instances[ind])[0]
                key = parent_keys[ind]
                data = self.value(key, self.columns)
                data[4] = ds.SOPClassUID
                if self.value(key, 'created'):
                    self.dataset[key] = ds
                else:
                    # If the dataset has not yet been edited
                    # find the placeholder row and copy the data
                    instance = ds.SOPInstanceUID
                    placeholder = (self.register.SOPInstanceUID == instance) & (self.register.placeholder==True)
                    placeholder = placeholder[placeholder]
                    new_key = placeholder.index[0]
                    #self.register.loc[new_key, self.columns] = data
                    for i, c in enumerate(self.columns):
                        self.register.at[new_key, c] = data[i]
                    # remove the current row and make the placeholder row current
                    self.register.at[key, 'removed'] = True
                    self.register.at[new_key, 'removed'] = False
                    self.register.at[new_key, 'placeholder'] = False
                    self.dataset[new_key] = ds

                    # self.register.at[key,'removed'] = True

                    #  # Add to database in memory
                    # new_key = self.new_key()
                    # self.dataset[new_key] = ds
                    # new_data.append(data)
                    # new_keys.append(new_key)

        # Update the dataframe in the index

        # If the series is empty and new instances have been added
        # then delete the row 
        if self.value(parent_keys[0], 'SOPInstanceUID') is None:
            if new_keys != []:
                if self.register.at[parent_keys[0], 'created']:
                    self.register.drop(index=parent_keys[0], inplace=True)
                else:
                    self.register.at[parent_keys[0], 'removed'] == True

        self._new_keys += new_keys
        self._new_data += new_data
        self.extend() 


    # def in_memory(self, uid): # needs a test

    #     key = self.keys(uid)[0]
    #     return key in self.dataset


    def label(self, uid=None, key=None, type=None):
        """Return a label to describe a row as Patient, Study, Series or Instance"""

        if self.register is None:
            raise ValueError('Cant provide labels - no database open')

        if uid is None:
            if key is None:
                return ''
    
        if uid == 'Database':
            return 'Database: ' + self.path

        if type is None:
            type = self.type(uid)

        if type == 'Patient':
            if key is None:
                key = self.keys(patient=uid)[0]
            row = self.register.loc[key]
            name = row.PatientName
            #id = row.PatientID
            label = str(name)
            #label += ' [' + str(id) + ']'
            return type + " {}".format(label)
        if type == 'Study':
            if key is None:
                key = self.keys(study=uid)[0]
            row = self.register.loc[key]
            descr = row.StudyDescription
            date = row.StudyDate
            label = str(descr)
            label += ' [' + str(date) + ']'
            return type + " {}".format(label)
        if type == 'Series':
            if key is None:
                key = self.keys(series=uid)[0]
            row = self.register.loc[key]
            descr = row.SeriesDescription
            nr = row.SeriesNumber
            label = str(nr).zfill(3)  
            label += ' [' + str(descr) + ']'
            return type + " {}".format(label)
        if type == 'Instance':
            if key is None:
                key = self.keys(instance=uid)[0]
            row = self.register.loc[key]
            nr = row.InstanceNumber
            label = str(nr).zfill(6)
            return SOPClass(row.SOPClassUID) + " {}".format(label)


    def print(self):
        """Prints a summary of the project folder to the terminal."""
        
        print('---------- DICOM FOLDER --------------')
        print('DATABASE: ' + self.path)
        for i, patient in enumerate(self.patients('Database')):
            print('  PATIENT [' + str(i) + ']: ' + self.label(patient))
            for j, study in enumerate(self.studies(patient)):
                print('    STUDY [' + str(j) + ']: ' + self.label(study))
                for k, series in enumerate(self.series(study)):
                    print('      SERIES [' + str(k) + ']: ' + self.label(series))
                    print('        Nr of instances: ' + str(len(self.instances(series)))) 


    def read(self, *args, keys=None, message=None, **kwargs):
        """Read the dataset from disk.
        """
        if keys is None:
            keys = self.keys(*args, **kwargs)
        for i, key in enumerate(keys):
            #if message is not None:
            #    self.status.progress(i, len(keys), message)
            # do not read if they are already in memory
            # this could overwrite changes made in memory only
            if not key in self.dataset:
                instance_uid = self.value(key, 'SOPInstanceUID')
                ds = self.get_dataset(instance_uid, [key])
                if ds is not None:
                    self.dataset[key] = ds

    def write(self, *args, keys=None, message=None, **kwargs):
        """Writing data from memory to disk.

        This does nothing if the data are not in memory.
        """
        if keys is None:
            keys = self.keys(*args, **kwargs)
        for i, key in enumerate(keys):
            # if message is not None:
            #     self.status.progress(i, len(keys), message)
            if key in self.dataset:
                file = self.filepath(key)
                self.dataset[key].write(file, self.status)
        #self.status.hide()

    def clear(self, *args, keys=None, **kwargs):
        """Clear all data from memory"""
        if keys is None:
            keys = self.keys(*args, **kwargs)
        # write to disk first so that any changes made in memory are not lost
        self.write(*args, keys=keys, **kwargs)
        # then delete the instances from memory
        for key in keys:
            self.dataset.pop(key, None) 

    def close(self):
        """Close an open database.
        """

        #if not self.is_open(): 
        if self.register is None:
            return True
        # This is the case where the database exists in memory only
        # Needs testing..
        if self.path is None: 
            reply = self.dialog.question( 
                title = 'Closing DICOM folder', 
                message = 'Save changes before closing?',
                cancel = True, 
            )
            if reply == "Cancel": 
                return False
            elif reply == "Yes":
                path = self.dialog.directory('Please enter the full path to an existing folder')
                if path is None:
                    return False
                self.path = path
                self.save()
                #self.save('Database')
                return self.close()
            elif reply == "No":
                return True
            
        if not self.is_saved():
            reply = self.dialog.question( 
                title = 'Closing DICOM folder', 
                message = 'Save changes before closing?',
                cancel = True, 
            )
            if reply == "Cancel": 
                return False
            if reply == "Yes":
                self.save()
                #self.save('Database')
            elif reply == "No":
                self.restore()

        self._write_df()
        self.write()
        self.register = None            
        self.path = None
        return True


    def is_saved(self):
        """Check if the folder is saved.
        
        Returns: 
            True if the folder is saved and False otherwise.
        """
        # Needs a formal test for completeness
        if ((self.register.removed==True) & (self.register.created==False)).any():
            return False
        if ((self.register.removed==False) & (self.register.created==True)).any():
            return False
        if ((self.register.removed==True) & (self.register.created==True) & (self.register.placeholder==False)).any():
            return False
        return True

    def is_open(self):
        """Check if a database is currently open, either in memory or on disk
        
        Returns: 
            True if a database is open and False otherwise.
        """
        # Needs a formal test for completeness
        return self.register is not None


    def delete(self, *args, keys=None, **kwargs):
        """Deletes some datasets
        
        Deleted datasets are stashed and can be recovered with restore()
        Using save() will delete them permanently
        """
        if keys is None:
            keys = self.keys(*args, **kwargs)
        self.register.loc[keys,'removed'] = True


    def save(self, rows=None): 

        no_placeholder = self.register.placeholder==False
        created = self.register.created & (self.register.removed==False) & no_placeholder
        removed = self.register.removed & no_placeholder
        if rows is not None:
            created = created & rows
            removed = removed & rows
        created = created[created].index
        removed = removed[removed].index

        # delete datasets marked for removal
        for key in removed.tolist():
            # delete in memory
            if key in self.dataset:
                del self.dataset[key]
            # delete on disk
            file = self.filepath(key) 
            if os.path.exists(file): 
                os.remove(file)
        # and drop then from the dataframe
        self.register.drop(index=removed, inplace=True)

        # for new or edited data, mark as saved and create placeholders
        self.register.loc[created, 'created'] = False
        #df = self.register.loc[created, :].copy(deep=True)
        df = self.register.loc[created, :]
        df.index = [self.new_key() for _ in range(df.shape[0])]
        df.removed = True
        df.created = True
        df.placeholder = True
        self.register = pd.concat([self.register, df])

        self._write_df()
        self.write()


    def restore(self, rows=None):  

        no_placeholder = self.register.placeholder==False
        created = self.register.created & no_placeholder
        removed = self.register.removed & (self.register.created==False) & no_placeholder
        if rows is not None:
            created = created & rows
            removed = removed & rows
        created = created[created].index
        removed = removed[removed].index

        # permanently delete newly created datasets
        for key in created.tolist():
            # delete in memory
            if key in self.dataset:
                del self.dataset[key]
            # delete on disk
            file = self.filepath(key) 
            if os.path.exists(file): 
                os.remove(file)
        self.register.drop(index=created, inplace=True)

        # For those that are restored, create placeholders.
        self.register.loc[removed, 'removed'] = False
        df = self.register.loc[removed, :].copy(deep=True)
        df.index = [self.new_key() for _ in range(df.shape[0])]
        df.removed = True
        df.created = True
        df.placeholder = True
        self.register = pd.concat([self.register, df])

        self._write_df()
        # self.write()      


    def delete_studies(self, studies: list):
        """Delete a list of studies"""
        copy_data = []
        copy_keys = []
        for study in studies:
            keys = self.keys(study=study)
            self.register.loc[keys,'removed'] = True
            # If this was the last study in the patient
            # keep the patient as an empty patient
            patient = self.register.at[keys[0], 'PatientID']
            patient = (self.register.removed == False) & (self.register.PatientID == patient)
            patient_studies = self.register.StudyInstanceUID[patient]
            patient_studies_cnt = len(patient_studies.unique())
            if patient_studies_cnt == 0:
                row = self.default()
                row[0] = self.register.at[keys[0], 'PatientID']
                row[5] = self.register.at[keys[0], 'PatientName']
                copy_data.append(row)
                copy_keys.append(self.new_key())
        self._new_keys += copy_keys
        self._new_data += copy_data
        self.extend()

      
    def delete_series(self, series: list):
        """Delete a list of series"""
        copy_data = []
        copy_keys = []
        for sery in series:
            keys = self.keys(series=sery)
            self.register.loc[keys,'removed'] = True
            # If this was the last series in the study
            # keep the study as an empty study
            study = self.register.at[keys[0], 'StudyInstanceUID']
            study = (self.register.removed == False) & (self.register.StudyInstanceUID == study)
            study_series = self.register.SeriesInstanceUID[study]
            study_series_cnt = len(study_series.unique())
            if study_series_cnt == 0:
                row = self.default()
                row[0] = self.register.at[keys[0], 'PatientID']
                row[1] = self.register.at[keys[0], 'StudyInstanceUID']
                row[5] = self.register.at[keys[0], 'PatientName']
                row[6] = self.register.at[keys[0], 'StudyDescription']
                row[7] = self.register.at[keys[0], 'StudyDate']
                copy_data.append(row)
                copy_keys.append(self.new_key())
        self._new_keys += copy_keys
        self._new_data += copy_data
        self.extend()


    def new_key(self):
        """Generate a new key"""
        return os.path.join('dbdicom', dbdataset.new_uid() + '.dcm') 


    def copy_instance_to_series(self, instance_key, target_keys, tmp, **kwargs):
        """Copy instances to another series"""

        attributes, values = self.series_header(target_keys[0])
        
        for key in kwargs:
            try:
                ind = attributes.index(key)
            except:
                attributes.append(key)
                values.append(kwargs[key])
            else:
                values[ind] = kwargs[key]

        n = self.register.loc[target_keys,'InstanceNumber'].values
        n = n[n != -1]
        max_number=0 if n.size==0 else np.amax(n)
        
        #copy_data = []
        #copy_keys = []

        new_instance = dbdataset.new_uid()
        new_key = self.new_key()
        ds = self.get_instance_dataset(instance_key)

        if ds is None:
            row = self.value(instance_key, self.columns).tolist()
            row[0] = self.value(target_keys[0], 'PatientID')
            row[1] = self.value(target_keys[0], 'StudyInstanceUID')
            row[2] = self.value(target_keys[0], 'SeriesInstanceUID')
            row[3] = new_instance
            row[5] = self.value(target_keys[0], 'PatientName')
            row[6] = self.value(target_keys[0], 'StudyDescription')
            row[7] = self.value(target_keys[0], 'StudyDate')
            row[8] = self.value(target_keys[0], 'SeriesDescription')
            row[9] = self.value(target_keys[0], 'SeriesNumber')
            row[10] = 1 + max_number
        else:
            if instance_key in self.dataset:
                ds = copy.deepcopy(ds)
                self.dataset[new_key] = ds
            ds.set_values( 
                attributes + ['SOPInstanceUID', 'InstanceNumber'], 
                values + [new_instance, 1+max_number])
            if not instance_key in self.dataset:
                ds.write(self.filepath(new_key), self.status)
            row = ds.get_values(self.columns)

        # Get new data for the dataframe
        #copy_data.append(row)
        #copy_keys.append(new_key)
        copy_data = [row]
        copy_keys = [new_key]

        # Update the dataframe in the index

        # If the series is empty and new instances have been added
        # then delete the row 
        if self.value(target_keys[0], 'SOPInstanceUID') is None:
            if copy_keys != []:
                if self.register.at[target_keys[0], 'created']:
                    self.register.drop(index=target_keys[0], inplace=True)
                else:
                    self.register.at[target_keys[0], 'removed'] == True

        self._new_keys += copy_keys
        self._new_data += copy_data
        self.extend()
        
        return new_instance


    def copy_to_series(self, uids, target, **kwargs):
        """Copy instances to another series"""

        target_keys = self.keys(series=target)

        attributes, values = self.series_header(target_keys[0])
        for key in kwargs:
            try:
                ind = attributes.index(key)
            except:
                attributes.append(key)
                values.append(kwargs[key])
            else:
                values[ind] = kwargs[key]

        n = self.register.loc[target_keys,'InstanceNumber'].values
        n = n[n != -1]
        max_number=0 if n.size==0 else np.amax(n)
        
        copy_data = []
        copy_keys = []

        keys = self.keys(uids)
        new_instances = dbdataset.new_uid(len(keys))

        for i, key in enumerate(keys):

            #self.status.progress(i+1, len(keys), message='Copying..')

            new_key = self.new_key()
            instance_uid = self.value(key, 'SOPInstanceUID')
            ds = self.get_dataset(instance_uid, [key])
            if ds is None:
                row = self.value(key, self.columns).tolist()
                row[0] = self.value(target_keys[0], 'PatientID')
                row[1] = self.value(target_keys[0], 'StudyInstanceUID')
                row[2] = self.value(target_keys[0], 'SeriesInstanceUID')
                row[3] = new_instances[i]
                row[5] = self.value(target_keys[0], 'PatientName')
                row[6] = self.value(target_keys[0], 'StudyDescription')
                row[7] = self.value(target_keys[0], 'StudyDate')
                row[8] = self.value(target_keys[0], 'SeriesDescription')
                row[9] = self.value(target_keys[0], 'SeriesNumber')
                row[10] = i+1+max_number
            else:
                if key in self.dataset:
                    ds = copy.deepcopy(ds)
                    self.dataset[new_key] = ds
                ds.set_values( 
                    attributes + ['SOPInstanceUID', 'InstanceNumber'], 
                    values + [new_instances[i], i+1+max_number])
                if not key in self.dataset:
                    ds.write(self.filepath(new_key), self.status)
                row = ds.get_values(self.columns)

            # Get new data for the dataframe
            copy_data.append(row)
            copy_keys.append(new_key)

        # Update the dataframe in the index

        # If the series is empty and new instances have been added
        # then delete the row 
        if self.value(target_keys[0], 'SOPInstanceUID') is None:
            if copy_keys != []:
                if self.register.at[target_keys[0], 'created']:
                    self.register.drop(index=target_keys[0], inplace=True)
                else:
                    self.register.at[target_keys[0], 'removed'] == True

        self._new_keys += copy_keys
        self._new_data += copy_data
        self.extend()

        if len(new_instances) == 1:
            return new_instances[0]
        else:
            return new_instances


    def copy_to_study(self, uid, target, **kwargs):
        """Copy series to another study"""

        target_keys = self.keys(study=target)
        target_key = target_keys[0]

        attributes, values = self.study_header(target_key)
        for key in kwargs:
            try:
                ind = attributes.index(key)
            except:
                attributes.append(key)
                values.append(kwargs[key])
            else:
                values[ind] = kwargs[key]

        n = self.value(target_keys, 'SeriesNumber')
        n = n[n != -1]
        max_number=0 if n.size==0 else np.amax(n)

        copy_data = []
        copy_keys = []

        all_series = self.series(uid)
        new_series = dbdataset.new_uid(len(all_series))

        for s, series in enumerate(all_series):

            #self.status.progress(s+1, len(all_series), message='Copying..')
            new_number = s + 1 + max_number

            for key in self.keys(series=series):

                new_key = self.new_key()
                instance_uid = self.value(key, 'SOPInstanceUID')
                ds = self.get_dataset(instance_uid, [key])
                if ds is None:
                    row = self.value(key, self.columns).tolist()
                    row[0] = self.value(target_key, 'PatientID')
                    row[1] = self.value(target_key, 'StudyInstanceUID')
                    row[2] = new_series[s]
                    row[3] = dbdataset.new_uid()
                    row[5] = self.value(target_key, 'PatientName')
                    row[6] = self.value(target_key, 'StudyDescription')
                    row[7] = self.value(target_key, 'StudyDate')
                    row[8] = new_number
                else:
                    if key in self.dataset:
                        ds = copy.deepcopy(ds)
                        self.dataset[new_key] = ds
                    ds.set_values(
                        attributes + ['SeriesInstanceUID', 'SeriesNumber', 'SOPInstanceUID'], 
                        values + [new_series[s], new_number, dbdataset.new_uid()])
                    if not key in self.dataset:
                        ds.write(self.filepath(new_key), self.status)
                    row = ds.get_values(self.columns)

                # Get new data for the dataframe
                
                copy_data.append(row)
                copy_keys.append(new_key)

        # Update the dataframe in the index

        # If the study is empty and new series have been added
        # then delete the row 
        if self.value(target_key, 'SeriesInstanceUID') is None:
            if copy_keys != []:
                if self.register.at[target_key, 'created']:
                    self.register.drop(index=target_key, inplace=True)
                else:
                    self.register.at[target_key, 'removed'] == True

        self._new_keys += copy_keys
        self._new_data += copy_data
        self.extend()

        if len(new_series) == 1:
            return new_series[0]
        else:
            return new_series


    def copy_to_patient(self, uid, target_key, **kwargs):
        """Copy studies to another patient"""

        attributes, values = self.patient_header(target_key)
        for key in kwargs:
            try:
                ind = attributes.index(key)
            except:
                attributes.append(key)
                values.append(kwargs[key])
            else:
                values[ind] = kwargs[key]

        copy_data = []
        copy_keys = []

        all_studies = self.studies(uid)
        new_studies = dbdataset.new_uid(len(all_studies))
        for s, study in enumerate(all_studies):
            for series in self.series(study):
                new_series_uid = dbdataset.new_uid()
                for key in self.keys(series=series):
                    new_key = self.new_key()
                    instance_uid = self.value(key, 'SOPInstanceUID')
                    ds = self.get_dataset(instance_uid, [key])
                    if ds is None:
                        row = self.value(key, self.columns).tolist()
                        row[0] = self.value(target_key, 'PatientID')
                        row[1] = new_studies[s]
                        row[2] = new_series_uid
                        row[3] = dbdataset.new_uid()
                        row[5] = self.value(target_key, 'PatientName')
                    else:
                        if key in self.dataset:
                            ds = copy.deepcopy(ds)
                            self.dataset[new_key] = ds
                        ds.set_values( 
                            attributes + ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID'], 
                            values + [new_studies[s], new_series_uid, dbdataset.new_uid()])
                        if not key in self.dataset:
                            ds.write(self.filepath(new_key), self.status)
                        row = ds.get_values(self.columns)

                    # Get new data for the dataframe
                    copy_data.append(row)
                    copy_keys.append(new_key)

        # Update the dataframe in the index

        # If the patient is empty and new studies have been added
        # then delete the row 
        if self.value(target_key, 'StudyInstanceUID') is None:
            if copy_keys != []:
                if self.register.at[target_key, 'created']:
                    self.register.drop(index=target_key, inplace=True)
                else:
                    self.register.at[target_key, 'removed'] == True

        self._new_keys += copy_keys
        self._new_data += copy_data
        self.extend()

        if len(new_studies) == 1:
            return new_studies[0]
        else:
            return new_studies

    def copy_to_database(self, uid, **kwargs):
        """Copy patient to the database"""

        copy_data = []
        copy_keys = []

        all_patients = self.patients(uid)
        new_patients = dbdataset.new_uid(len(all_patients))

        for i, patient in enumerate(all_patients):
            keys = self.keys(patient=patient)
            new_patient_uid = new_patients[i]
            new_patient_name = 'Copy of ' + self.value(keys[0], 'PatientName')
            for study in self.studies(patient):
                new_study_uid = dbdataset.new_uid()
                for sery in self.series(study):
                    new_series_uid = dbdataset.new_uid()
                    for key in self.keys(series=sery):
                        new_instance_uid = dbdataset.new_uid()
                        new_key = self.new_key()
                        instance_uid = self.value(key, 'SOPInstanceUID')
                        ds = self.get_dataset(instance_uid, [key])
                        if ds is None:
                            row = self.value(key, self.columns).tolist()
                            row[0] = new_patient_uid
                            row[1] = new_study_uid 
                            row[2] = new_series_uid
                            row[3] = new_instance_uid
                            row[5] = new_patient_name
                        else:
                            if key in self.dataset:
                                ds = copy.deepcopy(ds)
                                self.dataset[new_key] = ds
                            ds.set_values( 
                                list(kwargs.keys())+['PatientID', 'StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID', 'PatientName'], 
                                list(kwargs.values())+[new_patient_uid, new_study_uid, new_series_uid, new_instance_uid, new_patient_name])
                            if not key in self.dataset:
                                ds.write(self.filepath(new_key), self.status)
                            row = ds.get_values(self.columns)

                        # Get new data for the dataframe
                        copy_data.append(row)
                        copy_keys.append(new_key)

        # Update the dataframe in the index

        self._new_keys += copy_keys
        self._new_data += copy_data
        self.extend()

        if len(new_patients) == 1:
            return new_patients[0]
        else:
            return new_patients
            

    # def copy_to(self, source, target, target_type, **kwargs):

    #     #type = self.type(target)
    #     if target_type == 'Database':
    #         return self.copy_to_database(source, target, **kwargs)
    #     if target_type == 'Patient':
    #         return self.copy_to_patient(source, target, **kwargs)
    #     if target_type == 'Study':
    #         return self.copy_to_study(source, target, **kwargs)
    #     if target_type == 'Series':
    #         return self.copy_to_series(source, target, **kwargs)
    #     if target_type == 'Instance':
    #         raise ValueError('Cannot copy to an instance. Please copy to series, study or patient.')
            

    def drop_if_missing(self, key, missing='SOPInstanceUID'):
        # If the series was empty - now it has an instance so the original row can be removed
        if key in self.register.index:
            if self.value(key, missing) is None:
                if self.register.at[key, 'created']:
                    self.register.drop(index=key, inplace=True)
                else:
                    self.register.at[key, 'removed'] == True


    def move_to_series(self, uid, target, **kwargs):
        """Copy datasets to another series"""

        target_keys = self.keys(series=target)
        if target_keys == []:
            msg = 'Moving data to a series that does not exist in the database'
            raise ValueError(msg)

        attributes, values = self.series_header(target_keys[0])
        for key in kwargs:
            try:
                ind = attributes.index(key)
            except:
                attributes.append(key)
                values.append(kwargs[key])
            else:
                values[ind] = kwargs[key]

        n = self.value(target_keys, 'InstanceNumber')
        n = n[n != -1]
        max_number=0 if n.size==0 else np.amax(n)
        
        copy_data = []
        copy_keys = []       

        keys = self.keys(uid)

        for i, key in enumerate(keys):

            # self.status.progress(i+1, len(keys), message='Moving dataset..')

            # If this is the last instance in the series,
            # keep the series as an empty series.
            source_series = self.register.at[key, 'SeriesInstanceUID']
            source_series = (self.register.removed == False) & (self.register.SeriesInstanceUID == source_series)
            source_series_instances = self.register.SOPInstanceUID[source_series]
            source_series_instances_cnt = source_series_instances.shape[0]
            if source_series_instances_cnt == 1:
                row = self.default()
                row[0] = self.register.at[key, 'PatientID']
                row[1] = self.register.at[key, 'StudyInstanceUID']
                row[2] = self.register.at[key, 'SeriesInstanceUID']
                row[5] = self.register.at[key, 'PatientName']
                row[6] = self.register.at[key, 'StudyDescription']
                row[7] = self.register.at[key, 'StudyDate']
                row[8] = self.register.at[key, 'SeriesDescription']
                row[9] = self.register.at[key, 'SeriesNumber']
                copy_keys.append(self.new_key())
                copy_data.append(row)

            instance_uid = self.value(key, 'SOPInstanceUID') 
            ds = self.get_dataset(instance_uid, [key])

            if ds is None:

                row = self.value(key, self.columns).tolist()
                row[0] = self.value(target_keys[0], 'PatientID')
                row[1] = self.value(target_keys[0], 'StudyInstanceUID')
                row[2] = self.value(target_keys[0], 'SeriesInstanceUID')
                row[5] = self.value(target_keys[0], 'PatientName')
                row[6] = self.value(target_keys[0], 'StudyDescription')
                row[7] = self.value(target_keys[0], 'StudyDate')
                row[8] = self.value(target_keys[0], 'SeriesDescription')
                row[9] = self.value(target_keys[0], 'SeriesNumber')
                row[10] = i+1 + max_number
                if self.value(key, 'created'):
                    #self.register.loc[key, self.columns] = row
                    for i, c in enumerate(self.columns):
                        self.register.at[key, c] = row[i]
                    self.drop_if_missing(target_keys[0], 'SOPInstanceUID')
                else:
                    # If the dataset has not yet been edited
                    # find the placeholder row and copy the data
                    #placeholder = (self.register.SOPInstanceUID == instance_uid) & (self.register.placeholder)
                    #placeholder = placeholder[placeholder]
                    placeholder = self.register[self.register.SOPInstanceUID == instance_uid]
                    placeholder = placeholder[placeholder.placeholder]
                    placeholder = placeholder.index[0]
                    #self.register.loc[placeholder, self.columns] = row
                    for i, c in enumerate(self.columns): 
                        self.register.at[placeholder, c] = row[i]
                    # remove the current row and make the placeholder row current
                    self.register.at[key, 'removed'] = True
                    self.register.at[placeholder, 'removed'] = False
                    self.register.at[placeholder, 'placeholder'] = False

                    # self.register.at[key, 'removed'] = True
                    # copy_data.append(row)
                    # copy_keys.append(self.new_key())

            else:

                # If the value has changed before.
                if self.value(key, 'created'): 
                    ds.set_values( 
                        attributes + ['InstanceNumber'], 
                        values + [i+1 + max_number])
                    if not key in self.dataset:
                        ds.write(self.filepath(key), self.status)
                    for i, col in enumerate(attributes):
                        if col in self.columns:
                            self.register.at[key,col] = values[i]
                    self.drop_if_missing(target_keys[0], 'SOPInstanceUID')

                # If this is the first change, then save results in a copy.
                else:  
                    # If the dataset has not yet been edited
                    # find the placeholder row and copy the data
                    # placeholder = (self.register.SOPInstanceUID == instance_uid) & (self.register.placeholder)
                    # placeholder = placeholder[placeholder]
                    placeholder = self.register[self.register.SOPInstanceUID == instance_uid]
                    placeholder = placeholder[placeholder.placeholder]
                    new_key = placeholder.index[0]
                    #new_key = self.new_key()
                    if key in self.dataset:
                        ds = copy.deepcopy(ds)
                        self.dataset[new_key] = ds
                    ds.set_values(
                        attributes + ['InstanceNumber'], 
                        values + [i+1+max_number])
                    if not key in self.dataset:
                        ds.write(self.filepath(new_key), self.status)

                    # Get new data for the dataframe
                    row = ds.get_values(self.columns)
                    #self.register.loc[new_key, self.columns] = row
                    for i, c in enumerate(self.columns): 
                        self.register.at[new_key, c] = row[i]
                    # remove the current row and make the placeholder row current
                    self.register.at[key, 'removed'] = True
                    self.register.at[new_key, 'removed'] = False
                    self.register.at[new_key, 'placeholder'] = False

                    # self.register.at[key,'removed'] = True
                    # copy_data.append(row)
                    # copy_keys.append(new_key)

        # Update the dataframe in the index

        # If the series is empty and new instances have been added
        # then delete the row 
        if copy_keys != []:
            self.drop_if_missing(target_keys[0], 'SOPInstanceUID')

        self._new_keys += copy_keys
        self._new_data += copy_data
        self.extend()

        if len(keys) == 1:
            return self.value(keys, 'SOPInstanceUID')
        else:
            return list(self.value(keys, 'SOPInstanceUID'))


    def move_to_study(self, uid, target, **kwargs):
        """Copy series to another study"""

        target_keys = self.keys(study=target)

        attributes, values = self.study_header(target_keys[0])
        for key in kwargs:
            try:
                ind = attributes.index(key)
            except:
                attributes.append(key)
                values.append(kwargs[key])
            else:
                values[ind] = kwargs[key]

        n = self.value(target_keys, 'SeriesNumber')
        n = n[n != -1]
        max_number=0 if n.size==0 else np.amax(n)
        
        copy_data = []
        copy_keys = []       

        all_series = self.series(uid)

        for s, series in enumerate(all_series):

            self.status.progress(s+1, len(all_series), message='Moving series..')
            new_number = s + 1 + max_number

            keys = self.keys(series=series)

            # If this is the last series in the study
            # Keep the study as an empty study
            source_study = self.register.at[keys[0], 'StudyInstanceUID']
            source_study_series = (self.register.removed == False) & (self.register.StudyInstanceUID == source_study)
            source_study_series = self.register.SeriesInstanceUID[source_study_series]
            source_study_series_cnt = len(source_study_series.unique())
            if source_study_series_cnt == 1:
                row = self.default()
                row[0] = self.register.at[keys[0], 'PatientID']
                row[1] = self.register.at[keys[0], 'StudyInstanceUID']
                row[5] = self.register.at[keys[0], 'PatientName']
                row[6] = self.register.at[keys[0], 'StudyDescription']
                row[7] = self.register.at[keys[0], 'StudyDate']
                copy_keys.append(self.new_key())
                copy_data.append(row)
                
            # Now move the instances one by one
            for key in keys:

                instance_uid = self.value(key, 'SOPInstanceUID')
                ds = self.get_dataset(instance_uid, [key])

                if ds is None:

                    row = self.value(key, self.columns).tolist()
                    row[0] = self.value(target_keys[0], 'PatientID')
                    row[1] = self.value(target_keys[0], 'StudyInstanceUID')
                    row[5] = self.value(target_keys[0], 'PatientName')
                    row[6] = self.value(target_keys[0], 'StudyDescription')
                    row[7] = self.value(target_keys[0], 'StudyDate')
                    row[9] = new_number
                    if self.value(key, 'created'):
                        #self.register.loc[key, self.columns] = row
                        for i, c in enumerate(self.columns):
                            self.register.at[key, c] = data[i]
                        self.drop_if_missing(target_keys[0], 'SeriesInstanceUID')
                    else:
                        # If the dataset has not yet been edited
                        # find the placeholder row and copy the data
                        # placeholder = (self.register.SOPInstanceUID == instance_uid) & (self.register.placeholder)
                        # placeholder = placeholder[placeholder]
                        placeholder = self.register[self.register.SOPInstanceUID == instance_uid]
                        placeholder = placeholder[placeholder.placeholder]
                        placeholder = placeholder.index[0]
                        #self.register.loc[placeholder, self.columns] = row
                        for i, c in enumerate(self.columns): 
                            self.register.at[placeholder, c] = row[i]
                        # remove the current row and make the placeholder row current
                        self.register.at[key, 'removed'] = True
                        self.register.at[placeholder, 'removed'] = False
                        self.register.at[placeholder, 'placeholder'] = False
                        # self.register.at[key,'removed'] = True
                        # copy_data.append(row)
                        # copy_keys.append(self.new_key())

                else:

                    # If the value has changed before.
                    if self.value(key, 'created'): 
                        ds.set_values( 
                            attributes + ['SeriesNumber'], 
                            values + [new_number])
                        if not key in self.dataset:
                            ds.write(self.filepath(key), self.status)
                        for i, col in enumerate(attributes):
                            if col in self.columns:
                                self.register.at[key,col] = values[i]
                        self.drop_if_missing(target_keys[0], 'SeriesInstanceUID')

                    # If this is the first change, then save results in a copy.
                    else:  
                        # If the dataset has not yet been edited
                        # find the placeholder row and copy the data
                        # placeholder = (self.register.SOPInstanceUID == instance_uid) & (self.register.placeholder)
                        # placeholder = placeholder[placeholder]
                        placeholder = self.register[self.register.SOPInstanceUID == instance_uid]
                        placeholder = placeholder[placeholder.placeholder]
                        new_key = placeholder.index[0]
                        #new_key = self.new_key()
                        if key in self.dataset:
                            ds = copy.deepcopy(ds)
                            self.dataset[new_key] = ds
                        ds.set_values(
                            attributes + ['SeriesNumber'], 
                            values + [new_number])
                        if not key in self.dataset:
                            ds.write(self.filepath(new_key), self.status)

                        # Get new data for the dataframe
                        row = ds.get_values(self.columns)
                        #self.register.loc[new_key, self.columns] = row
                        for i, c in enumerate(self.columns): 
                            self.register.at[new_key, c] = row[i]   
                        # remove the current row and make the placeholder row current
                        self.register.at[key, 'removed'] = True
                        self.register.at[new_key, 'removed'] = False
                        self.register.at[new_key, 'placeholder'] = False

                        # self.register.at[key,'removed'] = True
                        # copy_data.append(row)
                        # copy_keys.append(new_key)

        # Update the dataframe in the index
        # If the target study was empty and new series have been added
        # then delete the row 
        if copy_keys != []:
            self.drop_if_missing(target_keys[0], 'SeriesInstanceUID')

        self._new_keys += copy_keys
        self._new_data += copy_data
        self.extend()

        if len(all_series) == 1:
            return all_series[0]
        else:
            return all_series


    def move_to_patient(self, uid, target, **kwargs):
        """Copy series to another study"""

        target_keys = self.keys(patient=target)

        attributes, values = self.patient_header(target_keys[0])
        for key in kwargs:
            try:
                ind = attributes.index(key)
            except:
                attributes.append(key)
                values.append(kwargs[key])
            else:
                values[ind] = kwargs[key]

        copy_data = []
        copy_keys = []  

        all_studies = self.studies(uid)
        for s, study in enumerate(all_studies):
            
            self.status.progress(s+1, len(all_studies), message='Moving study..')

            # If this is the last study in the patient
            # Keep the patient as an empty patient
            keys = self.keys(study=study)
            source_patient = self.register.at[keys[0], 'PatientID']
            source_patient = (self.register.removed == False) & (self.register.PatientID == source_patient)
            source_patient_studies = self.register.StudyInstanceUID[source_patient]
            source_patient_studies_cnt = len(source_patient_studies.unique())
            if source_patient_studies_cnt == 1:
                row = self.default()
                row[0] = self.register.at[keys[0], 'PatientID']
                row[5] = self.register.at[keys[0], 'PatientName']
                copy_keys.append(self.new_key())
                copy_data.append(row)

            #for series in self.series(study):
            for series in self.series(keys=keys):

                for key in self.keys(series=series):

                    instance_uid = self.value(key, 'SOPInstanceUID')
                    ds = self.get_dataset(instance_uid, [key])

                    if ds is None:

                        row = self.value(key, self.columns).tolist()
                        row[0] = self.value(target_keys[0], 'PatientID')
                        row[5] = self.value(target_keys[0], 'PatientName')
                        if self.value(key, 'created'):
                            #self.register.loc[key, self.columns] = row
                            for i, c in enumerate(self.columns):
                                self.register.at[key, c] = row[i]
                            self.drop_if_missing(target_keys[0], 'StudyInstanceUID')
                        else:
                            # If the dataset has not yet been edited
                            # find the placeholder row and copy the data
                            #placeholder = (self.register.SOPInstanceUID == instance_uid) & (self.register.placeholder)
                            #placeholder = placeholder[placeholder]
                            placeholder = self.register[self.register.SOPInstanceUID == instance_uid]
                            placeholder = placeholder[placeholder.placeholder]
                            placeholder = placeholder.index[0]
                            #self.register.loc[placeholder, self.columns] = row
                            for i, c in enumerate(self.columns): 
                                self.register.at[placeholder, c] = row[i]
                            # remove the current row and make the placeholder row current
                            self.register.at[key, 'removed'] = True
                            self.register.at[placeholder, 'removed'] = False
                            self.register.at[placeholder, 'placeholder'] = False
                            # self.register.at[key,'removed'] = True
                            # copy_data.append(row)
                            # copy_keys.append(self.new_key())

                    else:

                        # If the value has changed before.
                        if self.value(key, 'created'): 
                            ds.set_values(attributes, values)
                            if not key in self.dataset:
                                ds.write(self.filepath(key), self.status)
                            for i, col in enumerate(attributes):
                                if col in self.columns:
                                    self.register.at[key,col] = values[i]
                            self.drop_if_missing(target_keys[0], 'StudyInstanceUID')

                        # If this is the first change, then save results in a copy.
                        else:  

                            # If the dataset has not yet been edited
                            # find the placeholder row and copy the data
                            # placeholder = (self.register.SOPInstanceUID == instance_uid) & (self.register.placeholder)
                            # placeholder = placeholder[placeholder]
                            placeholder = self.register[self.register.SOPInstanceUID == instance_uid]
                            placeholder = placeholder[placeholder.placeholder]
                            new_key = placeholder.index[0]
                            #new_key = self.new_key()
                            if key in self.dataset:
                                ds = copy.deepcopy(ds)
                                self.dataset[new_key] = ds
                            ds.set_values(attributes, values)
                            if not key in self.dataset:
                                ds.write(self.filepath(new_key), self.status)

                            # Get new data for the dataframe
                            row = ds.get_values(self.columns)
                            #self.register.loc[new_key, self.columns] = row
                            for i, c in enumerate(self.columns): 
                                self.register.at[new_key, c] = row[i]
                            # remove the current row and make the placeholder row current
                            self.register.at[key, 'removed'] = True
                            self.register.at[new_key, 'removed'] = False
                            self.register.at[new_key, 'placeholder'] = False

                            # self.register.at[key,'removed'] = True
                            # copy_data.append(row)
                            # copy_keys.append(new_key)

            # Update the dataframe in the index

            # If the patient is empty and new studies have been added
            # then delete the row 
            if copy_keys != []:
                self.drop_if_missing(target_keys[0], 'StudyInstanceUID')

            self._new_keys += copy_keys
            self._new_data += copy_data
            self.extend()

        if len(all_studies) == 1:
            return all_studies[0]
        else:
            return all_studies


    def move_to(self, source, target, **kwargs):

        type = self.type(target)
        if type == 'Patient':
            return self.move_to_patient(source, target, **kwargs)
        if type == 'Study':
            return self.move_to_study(source, target, **kwargs)
        if type == 'Series':
            return self.move_to_series(source, target, **kwargs)
        if type == 'Instance':
            raise ValueError('Cannot move to an instance. Please move to series, study or patient.')


    def set_values(self, attributes, values, keys=None, uid=None):
        """Set values in a dataset"""

        uids = ['PatientID', 'StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']
        uids = [i for i in uids if i in attributes]
        if uids != []:
            raise ValueError('UIDs cannot be set using set_value(). Use copy_to() or move_to() instead.')

        #copy_data = []
        #copy_keys = []

        if keys is None:
            keys = self.keys(uid)

        for i, key in enumerate(keys):

            instance_uid = self.value(key, 'SOPInstanceUID')
            ds = self.get_dataset(instance_uid, [key])
            if ds is None:
                ds = new_dataset('MRImage')
                if instance_uid is None: # instance not yet created
                    series_uid = self.value(key, 'SeriesInstanceUID')
                    if series_uid is None:
                        study_uid = self.value(key, 'StudyInstanceUID')
                        if study_uid is None:
                            patient_uid = self.value(key, 'PatientUID')
                            if patient_uid is None:
                                instance_uid, _ = self.new_instance('Database', ds)
                            else:
                                instance_uid, _ = self.new_instance(patient_uid, ds)
                        else:
                            instance_uid, _ = self.new_instance(study_uid, ds)
                    else:
                        instance_uid, _ = self.new_instance(series_uid, ds)
                else:
                    #self.set_dataset(instance_uid, ds)
                    key = self.set_instance_dataset(instance_uid, ds, key)
            # If the value has changed before
            if self.value(key, 'created'): 
                ds.set_values(attributes, values)
                if not key in self.dataset:
                    ds.write(self.filepath(key), self.status)
                for i, col in enumerate(attributes):
                    if col in self.columns:
                        self.register.at[key,col] = values[i]
                new_key = key
            # If this is the first change, then save results in a copy
            else:  
    
                # If the dataset has not yet been edited
                # find the placeholder row and copy the data
                placeholder = self.register[self.register.SOPInstanceUID == instance_uid]
                placeholder = placeholder[placeholder.placeholder]
                new_key = placeholder.index[0]
                #new_key = self.new_key()
        
                if key in self.dataset:
                    ds = copy.deepcopy(ds)
                    self.dataset[new_key] = ds
                ds.set_values(attributes, values)
                if not key in self.dataset:
                    ds.write(self.filepath(new_key), self.status)

                # Get new data for the dataframe
                row = ds.get_values(self.columns)
                # Note this is 10x faster than self.register.loc[new_key, self.columns] = row
                for i, c in enumerate(self.columns): 
                    self.register.at[new_key, c] = row[i]
                # remove the current row and make the placeholder row current
                self.register.at[key, 'removed'] = True
                self.register.at[new_key, 'removed'] = False
                self.register.at[new_key, 'placeholder'] = False

                # self.register.at[key,'removed'] = True
                # copy_data.append(row)
                # copy_keys.append(new_key)

        #self._new_keys += copy_keys
        #self._new_data += copy_data
        #self.extend()

        
        return new_key

 
    def get_values(self, attributes, keys=None, uid=None):

        if keys is None:
            keys = self.keys(uid)
            if keys == []:
                return

        # Single attribute
        if not isinstance(attributes, list):

            if attributes in self.columns:
                value = [self.register.at[key, attributes] for key in keys]
                # trick to get unique elements
                value = [x for i, x in enumerate(value) if i==value.index(x)]
                
            else:
                value = []
                for i, key in enumerate(keys):
                    instance_uid = self.value(key, 'SOPInstanceUID')
                    ds = self.get_dataset(instance_uid, [key])
                    if ds is None:
                        v = None
                    else:
                        v = ds.get_values(attributes)
                    if v not in value:
                        value.append(v)
            if len(value) == 1:
                return value[0]
            return value

        # Multiple attributes
        # Create a np array v with values for each instance and attribute
        if set(attributes) <= set(self.columns):
            v = self.value(keys, attributes)
        else:
            v = np.empty((len(keys), len(attributes)), dtype=object)
            for i, key in enumerate(keys):
                instance_uid = self.value(key, 'SOPInstanceUID')
                ds = self.get_dataset(instance_uid, [key])
                if isinstance(ds, list):
                    instances = self.register.SOPInstanceUID == instance_uid
                    msg = 'Multiple instances with the same SOPInstanceUID \n'
                    msg += instance_uid + '\n'
                    msg += str(self.register.loc[instances].transpose())
                    raise DatabaseCorrupted(msg)
                if ds is None:
                    v[i,:] = [None] * len(attributes)
                else:
                    v[i,:] = ds.get_values(attributes)

        # Return a list with unique values for each attribute
        values = []
        for a in range(v.shape[1]):
            va = v[:,a]
            va = va[va != np.array(None)]
            va = np.unique(va)
            if va.size == 0:
                va = None
            elif va.size == 1:
                va = va[0]
            else:
                va = list(va)
            values.append(va)
        return values


    def import_datasets(self, files):

        # Read manager data
        df = dbdataset.read_dataframe(files, self.columns, self.status)
        df['removed'] = False
        df['created'] = True

        # Do not import SOPInstances that are already in the database
        uids = df.SOPInstanceUID.values.tolist()
        keys = self.keys(instance=uids)
        if keys != []:
            do_not_import = self.value(keys, 'SOPInstanceUID')
            rows = df.SOPInstanceUID.isin(do_not_import)
            df.drop(df[rows].index, inplace=True)
        if df.empty:
            return

        # Add those that are left to the database
        for file in df.index.tolist():
            new_key = self.new_key()
            ds = dbdataset.read(file)
            ds.write(self.filepath(new_key), self.status)
            df.rename(index={file:new_key}, inplace=True)
        self.register = pd.concat([self.register, df])

        # return the UIDs of the new instances
        return df.SOPInstanceUID.values.tolist()


    def export_datasets(self, uids, database):
        
        files = self.filepaths(uids)
        database.import_datasets(files)
