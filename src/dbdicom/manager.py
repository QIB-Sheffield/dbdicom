"""
Maintains an index of all files on disk.
"""

import os
import copy
import pandas as pd
import numpy as np

from dbdicom.message import StatusBar, Dialog
import dbdicom.utils.files as filetools
import dbdicom.utils.dcm4che as dcm4che
import dbdicom.dataset as dbdataset
from dbdicom.dataset_classes.create import read_dataset, SOPClass, new_dataset


class Manager(): 
    """Programming interface for reading and writing a DICOM folder."""

    # The column labels of the dataframe as required by dbdicom
    columns = [    
        'PatientID', 'StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID', 
        'SOPClassUID','NumberOfFrames', 
        'PatientName', 
        'StudyDescription', 'StudyDate', 
        'SeriesDescription', 'SeriesNumber',
        'InstanceNumber', 
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

    def read_dataframe(self, message='Reading database..'):
        """
        Reads all files in the folder and summarises key attributes in a table for faster access.
        """
        if self.path is None:
            raise ValueError('Cant read dataframe - index manages a database in memory')
        files = filetools.all_files(self.path)
        self.register = dbdataset.read_dataframe(files, self.columns, self.status, path=self.path, message=message)
        self.register['removed'] = False
        self.register['created'] = False

    def _pkl(self):
        """ Returns the file path of the .pkl file"""
        if self.path is None:
            raise ValueError('Cant read index file - manager manages a database in memory')
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
        file = self._pkl()
        self.register.to_pickle(file)

    def _read_df(self):
        """Reads the dataFrame from a .pkl file """
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

    def _multiframe_to_singleframe(self):
        """Converts all multiframe files in the folder into single-frame files.
        
        Reads all the multi-frame files in the folder,
        converts them to singleframe files, and delete the original multiframe file.
        """
        if self.path is None:
            # Low priority - we are not create multiframe data from scratch 
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
                    self.register = pd.concat([self.register, df])
                    # delete the original multiframe 
                    os.remove(filepath)
                    self.register.drop(index=relpath, inplace=True)

    def scan(self, unzip=False):
        """
        Reads all files in the folder and summarises key attributes in a table for faster access.
        """
#       Take unzip out until test is developed - less essential feature
#        if unzip:
#            filetools._unzip_files(self.path, self.status)
        self.read_dataframe()
        self._multiframe_to_singleframe()
        return self

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

    def type(self, uid=None):
        """Is the UID a patient, study, series or dataset"""

        if uid is None:
            return None
        if uid == 'Database':
            return uid

        df = self.register
        type = df.columns[df.isin([uid]).any()].values[0]

        if type == 'PatientID':
            return 'Patient'
        if type == 'StudyInstanceUID':
            return 'Study'
        if type == 'SeriesInstanceUID':
            return 'Series'
        if type == 'SOPInstanceUID':
            return 'Instance'

    def keys(self,
        uid = None, 
        patient = None,
        study = None,
        series = None,
        instance = None): 
        """Return a list of indices for all dicom datasets managed by the index.
        
        These indices are strings with unique relative paths 
        that either link to an existing file in the database or can be used for 
        writing a database that is in memory.
        """

        df = self.register
        if df is None:
            raise ValueError('Cant return dicom files - no database open')

        not_deleted = df.removed == False

        # If no arguments are provided
        if (uid is None) & (patient is None) & (study is None) & (series is None) & (instance is None):
            return []

        if uid == 'Database':
            return not_deleted[not_deleted].index.tolist()

        if isinstance(uid, list):
            if 'Database' in uid:
                return not_deleted[not_deleted].index.tolist()

        # If arguments are provided, create a list of unique datasets
        # keys = []
        if uid is not None:
            if not isinstance(uid, list):
                uid = [uid]
            uid = [i for i in uid if i is not None]
            rows = np.isin(df, uid).any(axis=1) & not_deleted
            return df[rows].index.tolist()
            # keys += df[rows].index.tolist()
        if patient is not None:
            if not isinstance(patient, list):
                patient = [patient]
            patient = [i for i in patient if i is not None]
            rows = df.PatientID.isin(patient) & not_deleted
            return df[rows].index.tolist()
            # keys += rows[rows].index.tolist()
        if study is not None:
            if not isinstance(study, list):
                study = [study]
            study = [i for i in study if i is not None]
            rows = df.StudyInstanceUID.isin(study) & not_deleted
            return df[rows].index.tolist()
            # keys += rows[rows].index.tolist()
        if series is not None:
            if not isinstance(series, list):
                series = [series]
            series = [i for i in series if i is not None]
            rows = df.SeriesInstanceUID.isin(series) & not_deleted
            return df[rows].index.tolist()
            # keys += rows[rows].index.tolist()
        if instance is not None: 
            if not isinstance(instance, list):
                instance = [instance]
            instance = [i for i in instance if i is not None]
            rows = df.SOPInstanceUID.isin(instance) & not_deleted
            return df[rows].index.tolist()
            # keys += rows[rows].index.tolist()
        # return list(set(keys))

    def value(self, key, column):
        try:
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

    def filter(self, uids, **kwargs):
        """Filter a list by attributes"""

        if not kwargs:
            return uids

        vals = list(kwargs.values())
        attr = list(kwargs.keys())
        return [id for id in uids if self.get_values(id, attr) == vals]

    def children(self, uid=None, **kwargs):
        """Returns the UIDs of the children"""

        if isinstance(uid, list):
            children = []
            for i in uid:
                children += self.children(i, **kwargs)
            return children

        if uid is None:
            return []

        # Get all children
        keys = self.keys(uid)
        if uid == 'Database':
            children = list(set(self.value(keys, 'PatientID')))
        else:
            row = self.register.loc[keys[0]].values.tolist()
            i = row.index(uid)
            if self.columns[i] == 'SOPInstanceUID':
                return []
            else:
                values = self.register.loc[keys,self.columns[i+1]].values
                values = values[values != np.array(None)]
                children = np.unique(values).tolist()

        return self.filter(children, **kwargs)

    def siblings(self, uid=None, **kwargs):

        if uid is None:
            return None
        if uid == 'Database':
            return None
        parent = self.parent(uid)
        children = self.children(parent)
        children.remove(uid)
        return self.filter(children, **kwargs)

    def instances(self, uid=None, **kwargs):

        keys = self.keys(uid)
        values = list(self.value(keys, 'SOPInstanceUID'))
        values = [v for v in values if v is not None]
        return self.filter(values, **kwargs)

    def series(self, uid=None, **kwargs):

        keys = self.keys(uid)
        values = list(set(self.value(keys, 'SeriesInstanceUID')))
        values = [v for v in values if v is not None]
        return self.filter(values, **kwargs)

    def studies(self, uid=None, **kwargs):

        keys = self.keys(uid)
        values = list(set(self.value(keys, 'StudyInstanceUID')))
        values = [v for v in values if v is not None]
        return self.filter(values, **kwargs)

    def patients(self, uid=None, **kwargs):

        keys = self.keys(uid)
        values = list(set(self.value(keys, 'PatientID')))
        values = [v for v in values if v is not None]
        return self.filter(values, **kwargs)

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
        self.register = pd.concat([self.register, df])
        #sortby = ['PatientID', 'StudyInstanceUID', 'SeriesNumber', 'InstanceNumber']
        #sortby = ['PatientID', 'StudyDescription', 'SeriesDescription', 'InstanceNumber']
        #self.register.sort_values(sortby, inplace=True)

        self._new_data = []
        self._new_keys = []


    def new_patient(self, parent='Database', PatientName='Anonymous'):
        # Allow multiple to be made at the same time

        data = [None] * len(self.columns)
        data[0] = dbdataset.new_uid()
        data[6] = PatientName

        self._new_data.append(data)
        self._new_keys.append(self.new_key())
        self.extend()

        # df = pd.DataFrame([data], index=[self.new_key()], columns=self.columns)
        # df['removed'] = False
        # df['created'] = True
        # self.register = pd.concat([self.register, df])

        return data[0]

    def new_study(self, parent=None, StudyDescription='New Study'):
        # Allow multiple to be made at the same time

        if parent is None:
            parent = self.new_patient()
        if self.type(parent) != 'Patient':
            #parent = self.patients(parent)[0]
            parent = self.new_patient(parent)

        key = self.keys(patient=parent)[0]

        data = [None] * len(self.columns)
        data[0] = self.value(key, 'PatientID')
        data[1] = dbdataset.new_uid()
        data[6] = self.value(key, 'PatientName')
        data[7] = StudyDescription
        
        if self.value(key, 'StudyInstanceUID') is None:
            # New patient without studies - use existing row
            self.register.loc[key, self.columns] = data
        else:
            # Patient with existing study - create new row
            self._new_data.append(data)
            self._new_keys.append(self.new_key())
            self.extend()
            # df = pd.DataFrame([data], index=[self.new_key()], columns=self.columns)
            # df['removed'] = False
            # df['created'] = True
            # self.register = pd.concat([self.register, df])

        return data[1]

    def new_series(self, parent=None, SeriesDescription='New Series'):
        # Allow multiple to be made at the same time

        if parent is None:
            parent = self.new_study()
        if self.type(parent) != 'Study':
            #parent = self.studies(parent)[0]
            parent = self.new_study(parent)

        key = self.keys(study=parent)[0]
        data = self.value(key, self.columns)
        data[2] = dbdataset.new_uid()
        data[3] = None
        data[4] = None
        data[5] = None
        data[9] = SeriesDescription
        data[10] = 1 + len(self.series(parent))
        data[11] = None

        if self.value(key, 'SeriesInstanceUID') is None:
            # New study without series - use existing row
            self.register.loc[key, self.columns] = data
        else:
            # Study with existing series - create new row
            self._new_data.append(data)
            self._new_keys.append(self.new_key())
            self.extend()
            # df = pd.DataFrame([data], index=[self.new_key()], columns=self.columns)
            # df['removed'] = False
            # df['created'] = True
            # self.register = pd.concat([self.register, df])

        return data[2]

    def new_instance(self, parent=None, dataset=None):
        # Allow multiple to be made at the same time

        if parent is None:
            parent = self.new_series()
        if self.type(parent) != 'Series':
            # parent = self.series(parent)[0] 
            parent = self.new_series(parent)

        key = self.keys(series=parent)[0]
        data = self.value(key, self.columns)
        data[3] = dbdataset.new_uid()
        data[4] = None
        data[5] = None
        data[11] = 1 + len(self.instances(parent))

        if self.value(key, 'SOPInstanceUID') is None:
            # New series without instances - use existing row
            self.register.loc[key, self.columns] = data
        else:
            # Series with existing instances - create new row
            self._new_data.append(data)
            self._new_keys.append(self.new_key())
            self.extend()
            # df = pd.DataFrame([data], index=[self.new_key()], columns=self.columns)
            # df['removed'] = False
            # df['created'] = True
            # self.register = pd.concat([self.register, df])

        if dataset is not None:
            self.set_dataset(data[3], dataset)

        return data[3]
    
    def is_empty(self, instance):
        # Needs a unit test
        key = self.keys(instance)[0]
        if key in self.dataset:
            return False
        else:
            file = self.filepath(key)
            if file is None:
                return True
            elif not os.path.exists(file):
                return True
            else:
                return False

    def get_dataset(self, uid, message=None):
        """Gets a list of datasets for a single record
        
        Datasets in memory will be returned.
        If they are not in memory, and the database exists on disk, they will be read from disk.
        If they are not in memory, and the database does not exist on disk, an exception is raised.
        """
        if uid is None:
            return None
        keys = self.keys(uid)
        dataset = []
        for i, key in enumerate(keys):
            if message is not None:
                self.status.progress(i, len(keys), message)
            if key in self.dataset:
                # If in memory, get from memory
                ds = self.dataset[key]
            else:
                # If not in memory, read from disk
                file = self.filepath(key)
                if file is None: # No dataset assigned yet
                    ds = None
                elif not os.path.exists(file):  # New instance, series, study or patient 
                    ds = None 
                else:
                    ds = read_dataset(file, self.dialog)  
            dataset.append(ds)
        if self.type(uid) == 'Instance':
            return dataset[0]
        else:
            return dataset

    def _get_dataset(self, instances):
        """Helper function"""

        for instance in instances:
            ds = self.get_dataset(instance)
            if ds is not None:
                return ds
        return None

    def _get_values(self, instances, attr):
        """Helper function"""

        ds = self._get_dataset(instances)
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
        instances = self.instances(parent)
        if instances != []:
            attr = list(set(dbdataset.module_patient() + dbdataset.module_study() + dbdataset.module_series()))
            vals = self._get_values(instances, attr)
        else:
            parent = self.register.at[key, 'StudyInstanceUID']
            instances = self.instances(parent)
            if instances != []:
                attr = list(set(dbdataset.module_patient() + dbdataset.module_study()))
                vals = self._get_values(instances, attr)
                attr += attr_series
                vals += self.value(key, attr_series).tolist()
            else:
                parent = self.register.at[key, 'PatientID']
                instances = self.instances(parent)
                if instances != []:
                    attr = dbdataset.module_patient()
                    vals = self._get_values(instances, attr)
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
        instances = self.instances(parent)
        if instances != []:
            attr = list(set(dbdataset.module_patient() + dbdataset.module_study()))
            vals = self._get_values(instances, attr)
        else:
            parent = self.register.at[key, 'PatientID']
            instances = self.instances(parent)
            if instances != []:
                attr = dbdataset.module_patient()
                vals = self._get_values(instances, attr)
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
        instances = self.instances(parent)
        if instances != []:
            attr = dbdataset.module_patient()
            vals = self._get_values(instances, attr)
        else:
            attr = attr_patient
            vals = self.value(key, attr).tolist()
        return attr, vals

    def set_instance_dataset(self, instance, ds):

        if isinstance(ds, list):
            if len(ds) > 1:
                raise ValueError('Cannot set multiple datasets to a single instance')
            else:
                ds = ds[0]

        key = self.keys(instance)[0]
        data = self.register.loc[key, self.columns]
        data[4] = ds.SOPClassUID
        if 'NumberOfFrames' in ds:
            data[5] = ds.NumberOfFrames
        ds.set_values(self.columns, data)
        if self.value(key, 'created'):
            self.register.loc[key, self.columns] = data
            self.dataset[key] = ds
        else:
            self.register.at[key,'removed'] = True
            new_key = self.new_key()
            self.dataset[new_key] = ds

            self._new_data.append(data)
            self._new_keys.append(new_key)
            self.extend()
            
            # df = pd.DataFrame([data], index=[new_key], columns=self.columns)
            # df['removed'] = False
            # df['created'] = True
            # self.register = pd.concat([self.register, df])  
                  

    def set_dataset(self, uid, dataset):

        if self.type(uid) == 'Instance':
            self.set_instance_dataset(uid, dataset)
            return

        if not isinstance(dataset, list):
           dataset = [dataset]
         
        parent_keys = self.keys(uid)
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
                if 'NumberOfFrames' in ds:
                    data[5] = ds.NumberOfFrames
                nrs = self.value(parent_keys, 'InstanceNumber')
                nrs = [n for n in nrs if n is not None]
                if nrs == []:
                    data[11] = 1
                else:
                    data[11] = 1 + max(nrs)
                ds.set_values(self.columns, data)

                # Add to database in memory
                new_key = self.new_key()
                self.dataset[new_key] = ds
                new_data.append(data)
                new_keys.append(new_key)

            else: # If the dataset is already in the object

                key = self.keys(instances[ind])[0]
                data = self.value(key, self.columns)
                data[4] = ds.SOPClassUID
                if 'NumberOfFrames' in ds:
                    data[5] = ds.NumberOfFrames
                if self.value(key, 'created'):
                    self.dataset[key] = ds
                else:
                    self.register.at[key,'removed'] = True

                     # Add to database in memory
                    new_key = self.new_key()
                    self.dataset[new_key] = ds
                    new_data.append(data)
                    new_keys.append(new_key)

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

        # if new_keys != []:
        #     df = pd.DataFrame(new_data, index=new_keys, columns=self.columns)
        #     df['removed'] = False
        #     df['created'] = True
        #     self.register = pd.concat([self.register, df])   


    def in_memory(self, uid): # needs a test

        key = self.keys(uid)[0]
        return key in self.dataset

    def new_child(self, uid=None, dataset=None, **kwargs):

        if uid is None:
            return None
        parent_type = self.type(uid)
        if parent_type == 'Database':
            return self.new_patient(uid, **kwargs)
        if parent_type == 'Patient':
            return self.new_study(uid, **kwargs)
        if parent_type == 'Study':
            return self.new_series(uid, **kwargs)
        if parent_type == 'Series':
            return self.new_instance(uid, dataset=dataset, **kwargs)
        if parent_type == 'Instance':
            return None

    def new_sibling(self, uid=None, **kwargs):

        if uid is None:
            return None
        if uid == 'Database':
            return None
        parent = self.parent(uid)
        return self.new_child(parent, **kwargs)

    def new_pibling(self, uid=None):

        if uid is None:
            return None
        if uid == 'Database':
            return None
        parent = self.parent(uid)
        return self.new_sibling(parent)

    def label(self, uid):
        """Return a label to describe a row as Patient, Study, Series or Instance"""

        if self.register is None:
            raise ValueError('Cant provide labels - no database open')

        if uid is None:
            return ''
        if uid == 'Database':
            return 'Database: ' + self.path

        type = self.type(uid)
        key = self.keys(uid)[0]
        row = self.register.loc[key]

        if type == 'Patient':
            name = row.PatientName
            id = row.PatientID
            label = str(name)
            label += ' [' + str(id) + ']'
            return type + " {}".format(label)
        if type == 'Study':
            descr = row.StudyDescription
            date = row.StudyDate
            label = str(descr)
            label += ' [' + str(date) + ']'
            return type + " {}".format(label)
        if type == 'Series':
            descr = row.SeriesDescription
            nr = row.SeriesNumber
            label = str(nr).zfill(3)  
            label += ' [' + str(descr) + ']'
            return type + " {}".format(label)
        if type == 'Instance':
            nr = row.InstanceNumber
            label = str(nr).zfill(6)
            return SOPClass(row.SOPClassUID) + " {}".format(label)

    def print(self):
        """Prints a summary of the project folder to the terminal."""
        
        print('---------- DICOM FOLDER --------------')
        print('DATABASE: ' + self.path)
        for i, patient in enumerate(self.children('Database')):
            print('  PATIENT [' + str(i) + ']: ' + self.label(patient))
            for j, study in enumerate(self.children(patient)):
                print('    STUDY [' + str(j) + ']: ' + self.label(study))
                for k, series in enumerate(self.children(study)):
                    print('      SERIES [' + str(k) + ']: ' + self.label(series))
                    print('        Nr of instances: ' + str(len(self.children(series)))) 

    def read(self, *args, message=None, **kwargs):
        """Read the dataset from disk.
        """
        keys = self.keys(*args, **kwargs)
        for i, key in enumerate(keys):
            if message is not None:
                self.status.progress(i, len(keys), message)
            # do not read if they are already in memory
            # this could overwrite changes made in memory only
            if not key in self.dataset:
                self.dataset[key] = self.get_dataset(self.value(key, 'SOPInstanceUID'))

    def write(self, *args, message=None, **kwargs):
        """Writing data from memory to disk.

        This does nothing if the data are not in memory.
        """
        keys = self.keys(*args, **kwargs)
        for i, key in enumerate(keys):
            if message is not None:
                self.status.progress(i, len(keys), message)
            if key in self.dataset:
                file = self.filepath(key)
                self.dataset[key].write(file, self.dialog)
        self.status.hide()

    def clear(self, *args, **kwargs):
        """Clear all data from memory"""
        # write to disk first so that any changes made in memory are not lost
        self.write(*args, **kwargs)
        # then delete the instances from memory
        for key in self.keys(*args, **kwargs):
            self.dataset.pop(key, None) 

    def close(self):
        """Close an open database.
        """

        if not self.is_open(): 
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
            elif reply == "No":
                self.restore()

        self._write_df()
        self.write()
        self.register = None            
        self.path = None

    def is_saved(self):
        """Check if the folder is saved.
        
        Returns: 
            True if the folder is saved and False otherwise.
        """
        # Needs a formal test for completeness
        if self.register.removed.any(): 
            return False
        if self.register.created.any():
            return False
        return True

    def is_open(self):
        """Check if a database is currently open, either in memory or on disk
        
        Returns: 
            True if a database is open and False otherwise.
        """
        # Needs a formal test for completeness
        return self.register is not None
      
    def delete(self, *args, **kwargs):
        """Deletes some datasets
        
        Deleted datasets are stashed and can be recovered with restore()
        Using save() will delete them permanently
        """
        keys = self.keys(*args, **kwargs)
        self.register.loc[keys,'removed'] = True

    def new_key(self):
        """Generate a new key"""

        return os.path.join('dbdicom', dbdataset.new_uid() + '.dcm') 


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

        n = self.value(target_keys, 'InstanceNumber')
        n = n[n != np.array(None)]
        max_number=0 if n.size==0 else np.amax(n)
   
        copy_data = []
        copy_keys = []

        keys = self.keys(uids)
        new_instances = dbdataset.new_uid(len(keys))

        for i, key in enumerate(keys):

            self.status.progress(i+1, len(keys), message='Copying..')

            new_key = self.new_key()
            ds = self.get_dataset(self.value(key, 'SOPInstanceUID'))
            if ds is None:
                row = self.value(key, self.columns).tolist()
                row[0] = self.value(target_keys[0], 'PatientID')
                row[1] = self.value(target_keys[0], 'StudyInstanceUID')
                row[2] = self.value(target_keys[0], 'SeriesInstanceUID')
                row[3] = new_instances[i]
                row[6] = self.value(target_keys[0], 'PatientName')
                row[7] = self.value(target_keys[0], 'StudyDescription')
                row[8] = self.value(target_keys[0], 'StudyDate')
                row[9] = self.value(target_keys[0], 'SeriesDescription')
                row[10] = self.value(target_keys[0], 'SeriesNumber')
                row[11] = i+1+max_number
            else:
                if key in self.dataset:
                    ds = copy.deepcopy(ds)
                    self.dataset[new_key] = ds
                ds.set_values( 
                    attributes + ['SOPInstanceUID', 'InstanceNumber'], 
                    values + [new_instances[i], i+1+max_number])
                if not key in self.dataset:
                    ds.write(self.filepath(new_key), self.dialog)
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

        # df = pd.DataFrame(copy_data, index=copy_keys, columns=self.columns)
        # df['removed'] = False
        # df['created'] = True
        # self.register = pd.concat([self.register, df])

        if len(new_instances) == 1:
            return new_instances[0]
        else:
            return new_instances



    def copy_to_study(self, uid, target, **kwargs):
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
        n = n[n != np.array(None)]
        max_number=0 if n.size==0 else np.amax(n)

        copy_data = []
        copy_keys = []

        all_series = self.series(uid)
        new_series = dbdataset.new_uid(len(all_series))

        for s, series in enumerate(all_series):

            self.status.progress(s+1, len(all_series), message='Copying..')
            new_number = s + 1 + max_number

            for key in self.keys(series):

                new_key = self.new_key()
                ds = self.get_dataset(self.value(key, 'SOPInstanceUID'))
                if ds is None:
                    row = self.value(key, self.columns).tolist()
                    row[0] = self.value(target_keys[0], 'PatientID')
                    row[1] = self.value(target_keys[0], 'StudyInstanceUID')
                    row[2] = new_series[s]
                    row[3] = dbdataset.new_uid()
                    row[6] = self.value(target_keys[0], 'PatientName')
                    row[7] = self.value(target_keys[0], 'StudyDescription')
                    row[8] = self.value(target_keys[0], 'StudyDate')
                    row[10] = new_number
                else:
                    if key in self.dataset:
                        ds = copy.deepcopy(ds)
                        self.dataset[new_key] = ds
                    ds.set_values(
                        attributes + ['SeriesInstanceUID', 'SeriesNumber', 'SOPInstanceUID'], 
                        values + [new_series[s], new_number, dbdataset.new_uid()])
                    if not key in self.dataset:
                        ds.write(self.filepath(new_key), self.dialog)
                    row = ds.get_values(self.columns)

                # Get new data for the dataframe
                
                copy_data.append(row)
                copy_keys.append(new_key)

        # Update the dataframe in the index

        # If the study is empty and new series have been added
        # then delete the row 
        if self.value(target_keys[0], 'SeriesInstanceUID') is None:
            if copy_keys != []:
                if self.register.at[target_keys[0], 'created']:
                    self.register.drop(index=target_keys[0], inplace=True)
                else:
                    self.register.at[target_keys[0], 'removed'] == True

        self._new_keys += copy_keys
        self._new_data += copy_data
        self.extend()

        # df = pd.DataFrame(copy_data, index=copy_keys, columns=self.columns)
        # df['removed'] = False
        # df['created'] = True
        # self.register = pd.concat([self.register, df])

        if len(new_series) == 1:
            return new_series[0]
        else:
            return new_series

    def copy_to_patient(self, uid, target, **kwargs):
        """Copy studies to another patient"""

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
        new_studies = dbdataset.new_uid(len(all_studies))

        for s, study in enumerate(all_studies):
            
            self.status.progress(s+1, len(all_studies), message='Copying..')

            for series in self.series(study):

                new_series_uid = dbdataset.new_uid()

                for key in self.keys(series):

                    new_key = self.new_key()
                    ds = self.get_dataset(self.value(key, 'SOPInstanceUID'))
                    if ds is None:
                        row = self.value(key, self.columns).tolist()
                        row[0] = self.value(target_keys[0], 'PatientID')
                        row[1] = new_studies[s]
                        row[2] = new_series_uid
                        row[3] = dbdataset.new_uid()
                        row[6] = self.value(target_keys[0], 'PatientName')
                    else:
                        if key in self.dataset:
                            ds = copy.deepcopy(ds)
                            self.dataset[new_key] = ds
                        ds.set_values( 
                            attributes + ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID'], 
                            values + [new_studies[s], new_series_uid, dbdataset.new_uid()])
                        if not key in self.dataset:
                            ds.write(self.filepath(new_key), self.dialog)
                        row = ds.get_values(self.columns)

                    # Get new data for the dataframe
                    copy_data.append(row)
                    copy_keys.append(new_key)

        # Update the dataframe in the index

        # If the patient is empty and new studies have been added
        # then delete the row 
        if self.value(target_keys[0], 'StudyInstanceUID') is None:
            if copy_keys != []:
                if self.register.at[target_keys[0], 'created']:
                    self.register.drop(index=target_keys[0], inplace=True)
                else:
                    self.register.at[target_keys[0], 'removed'] == True

        self._new_keys += copy_keys
        self._new_data += copy_data
        self.extend()

        # df = pd.DataFrame(copy_data, index=copy_keys, columns=self.columns)
        # df['removed'] = False
        # df['created'] = True
        # self.register = pd.concat([self.register, df])

        if len(new_studies) == 1:
            return new_studies[0]
        else:
            return new_studies

    def copy_to(self, source, target, **kwargs):

        type = self.type(target)
        if type == 'Patient':
            return self.copy_to_patient(source, target, **kwargs)
        if type == 'Study':
            return self.copy_to_study(source, target, **kwargs)
        if type == 'Series':
            return self.copy_to_series(source, target, **kwargs)
        if type == 'Instance':
            raise ValueError('Cannot copy to an instance. Please copy to series, study or patient.')


    def move_to_series(self, uid, target, **kwargs):
        """Copy datasets to another series"""

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

        n = self.value(target_keys, 'InstanceNumber')
        n = n[n != np.array(None)]
        max_number=0 if n.size==0 else np.amax(n)
        
        copy_data = []
        copy_keys = []       

        keys = self.keys(uid)
        for i, key in enumerate(keys):

            self.status.progress(i+1, len(keys), message='Moving dataset..')

            ds = self.get_dataset(self.value(key, 'SOPInstanceUID'))

            if ds is None:

                row = self.value(key, self.columns).tolist()
                row[0] = self.value(target_keys[0], 'PatientID')
                row[1] = self.value(target_keys[0], 'StudyInstanceUID')
                row[2] = self.value(target_keys[0], 'SeriesInstanceUID')
                row[6] = self.value(target_keys[0], 'PatientName')
                row[7] = self.value(target_keys[0], 'StudyDescription')
                row[8] = self.value(target_keys[0], 'StudyDate')
                row[9] = self.value(target_keys[0], 'SeriesDescription')
                row[10] = self.value(target_keys[0], 'SeriesNumber')
                row[11] = i+1 + max_number
                if self.value(key, 'created'):
                    self.register.loc[key, self.columns] = row
                else:
                    self.register.at[key,'removed'] = True
                    copy_data.append(row)
                    copy_keys.append(self.new_key())

            else:

                # If the value has changed before.
                if self.value(key, 'created'): 
                    ds.set_values( 
                        attributes + ['InstanceNumber'], 
                        values + [i+1 + max_number])
                    if not key in self.dataset:
                        ds.write(self.filepath(key), self.dialog)
                    for i, col in enumerate(attributes):
                        if col in self.columns:
                            self.register.at[key,col] = values[i]

                # If this is the first change, then save results in a copy.
                else:  
                    new_key = self.new_key()
                    if key in self.dataset:
                        ds = copy.deepcopy(ds)
                        self.dataset[new_key] = ds
                    ds.set_values(
                        attributes + ['InstanceNumber'], 
                        values + [i+1+max_number])
                    if not key in self.dataset:
                        ds.write(self.filepath(new_key), self.dialog)

                    # Get new data for the dataframe
                    self.register.at[key,'removed'] = True
                    row = ds.get_values(self.columns)
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

        # if copy_data != []:
        #     df = pd.DataFrame(copy_data, index=copy_keys, columns=self.columns)
        #     df['removed'] = False
        #     df['created'] = True
        #     self.register = pd.concat([self.register, df])

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
        n = n[n != np.array(None)]
        max_number=0 if n.size==0 else np.amax(n)
        
        copy_data = []
        copy_keys = []       

        all_series = self.series(uid)
        for s, series in enumerate(all_series):

            self.status.progress(s+1, len(all_series), message='Moving series..')
            new_number = s + 1 + max_number

            for key in self.keys(series):

                ds = self.get_dataset(self.value(key, 'SOPInstanceUID'))

                if ds is None:

                    row = self.value(key, self.columns).tolist()
                    row[0] = self.value(target_keys[0], 'PatientID')
                    row[1] = self.value(target_keys[0], 'StudyInstanceUID')
                    row[6] = self.value(target_keys[0], 'PatientName')
                    row[7] = self.value(target_keys[0], 'StudyDescription')
                    row[8] = self.value(target_keys[0], 'StudyDate')
                    row[10] = new_number
                    if self.value(key, 'created'):
                        self.register.loc[key, self.columns] = row
                    else:
                        self.register.at[key,'removed'] = True
                        copy_data.append(row)
                        copy_keys.append(self.new_key())

                else:

                    # If the value has changed before.
                    if self.value(key, 'created'): 
                        ds.set_values( 
                            attributes + ['SeriesNumber'], 
                            values + [new_number])
                        if not key in self.dataset:
                            ds.write(self.filepath(key), self.dialog)
                        for i, col in enumerate(attributes):
                            if col in self.columns:
                                self.register.at[key,col] = values[i]

                    # If this is the first change, then save results in a copy.
                    else:  
                        new_key = self.new_key()
                        if key in self.dataset:
                            ds = copy.deepcopy(ds)
                            self.dataset[new_key] = ds
                        ds.set_values(
                            attributes + ['SeriesNumber'], 
                            values + [new_number])
                        if not key in self.dataset:
                            ds.write(self.filepath(new_key), self.dialog)

                        # Get new data for the dataframe
                        self.register.at[key,'removed'] = True
                        row = ds.get_values(self.columns)
                        copy_data.append(row)
                        copy_keys.append(new_key)

        # Update the dataframe in the index

        # If the study is empty and new series have been added
        # then delete the row 
        if self.value(target_keys[0], 'SeriesInstanceUID') is None:
            if copy_keys != []:
                if self.register.at[target_keys[0], 'created']:
                    self.register.drop(index=target_keys[0], inplace=True)
                else:
                    self.register.at[target_keys[0], 'removed'] == True

        self._new_keys += copy_keys
        self._new_data += copy_data
        self.extend()

        # if copy_data != []:
        #     df = pd.DataFrame(copy_data, index=copy_keys, columns=self.columns)
        #     df['removed'] = False
        #     df['created'] = True
        #     self.register = pd.concat([self.register, df])

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
            for series in self.series(study):

                for key in self.keys(series):

                    ds = self.get_dataset(self.value(key, 'SOPInstanceUID'))

                    if ds is None:

                        row = self.value(key, self.columns).tolist()
                        row[0] = self.value(target_keys[0], 'PatientID')
                        row[6] = self.value(target_keys[0], 'PatientName')
                        if self.value(key, 'created'):
                            self.register.loc[key, self.columns] = row
                        else:
                            self.register.at[key,'removed'] = True
                            copy_data.append(row)
                            copy_keys.append(self.new_key())

                    else:

                        # If the value has changed before.
                        if self.value(key, 'created'): 
                            ds.set_values(attributes, values)
                            if not key in self.dataset:
                                ds.write(self.filepath(key), self.dialog)
                            for i, col in enumerate(attributes):
                                if col in self.columns:
                                    self.register.at[key,col] = values[i]

                        # If this is the first change, then save results in a copy.
                        else:  
                            new_key = self.new_key()
                            if key in self.dataset:
                                ds = copy.deepcopy(ds)
                                self.dataset[new_key] = ds
                            ds.set_values(attributes, values)
                            if not key in self.dataset:
                                ds.write(self.filepath(new_key), self.dialog)

                            # Get new data for the dataframe
                            self.register.at[key,'removed'] = True
                            row = ds.get_values(self.columns)
                            copy_data.append(row)
                            copy_keys.append(new_key)

            # Update the dataframe in the index

            # If the patient is empty and new studies have been added
            # then delete the row 
            if self.value(target_keys[0], 'StudyInstanceUID') is None:
                if copy_keys != []:
                    if self.register.at[target_keys[0], 'created']:
                        self.register.drop(index=target_keys[0], inplace=True)
                    else:
                        self.register.at[target_keys[0], 'removed'] == True

            self._new_keys += copy_keys
            self._new_data += copy_data
            self.extend()

            # if copy_data != []:
            #     df = pd.DataFrame(copy_data, index=copy_keys, columns=self.columns)
            #     df['removed'] = False
            #     df['created'] = True
            #     self.register = pd.concat([self.register, df])

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

    def set_values(self, uid, attributes, values):
        """Set values in a dataset"""

        uids = ['PatientID', 'StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']
        uids = [i for i in uids if i in attributes]
        if uids != []:
            raise ValueError('UIDs cannot be set using set_value(). Use copy_to() or move_to() instead.')

        copy_data = []
        copy_keys = []

        keys = self.keys(uid)
        for i, key in enumerate(keys):

            self.status.progress(i+1, len(keys), message='Setting values..')

            instance_uid = self.value(key, 'SOPInstanceUID')
            ds = self.get_dataset(instance_uid)
            if ds is None:
                ds = new_dataset('MRImage')
                self.set_dataset(instance_uid, ds)
            # If the value has changed before
            if self.value(key, 'created'): 
                ds.set_values(attributes, values)
                if not key in self.dataset:
                    ds.write(self.filepath(key), self.dialog)
                for i, col in enumerate(attributes):
                    if col in self.columns:
                        self.register.at[key,col] = values[i]

            # If this is the first change, then save results in a copy
            else:  
                new_key = self.new_key()
                if key in self.dataset:
                    ds = copy.deepcopy(ds)
                    self.dataset[new_key] = ds
                ds.set_values(attributes, values)
                if not key in self.dataset:
                    ds.write(self.filepath(new_key), self.dialog)

                # Get new data for the dataframe
                self.register.at[key,'removed'] = True
                row = ds.get_values(self.columns)
                copy_data.append(row)
                copy_keys.append(new_key)

        self._new_keys += copy_keys
        self._new_data += copy_data
        self.extend()

        # # Update the dataframe in the index
        # if copy_data != []:
        #     df = pd.DataFrame(copy_data, index=copy_keys, columns=self.columns)
        #     df['removed'] = False
        #     df['created'] = True
        #     self.register = pd.concat([self.register, df])

    def get_values(self, uid, attributes):

        if not isinstance(uid, list):

            keys = self.keys(uid)
            if keys == []:
                return

            if not isinstance(attributes, list):

                if attributes in self.columns:
                    value = self.value(keys, attributes)
                else:
                    value = []
                    for i, key in enumerate(keys):
                        ds = self.get_dataset(self.value(key, 'SOPInstanceUID'))
                        if ds is None:
                            v = None
                        else:
                            v = ds.get_values(attributes)
                        value.append(v)
                value = list(set(value))
                if len(value) == 1:
                    return value[0]
                else:
                    return value

            else:

                # Create a np array v with values for each instance and attribute
                if set(attributes) <= set(self.columns):
                    v = self.value(keys, attributes)
                else:
                    v = np.empty((len(keys), len(attributes)), dtype=object)
                    for i, key in enumerate(keys):
                        ds = self.get_dataset(self.value(key, 'SOPInstanceUID'))
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

        # If a list of UIDs is given, apply the function recursively
        # and generate a list of results - one for each uid.
        else:
            values = []
            for id in uid:
                v = self.get_values(id, attributes)
                values.append(v)
            return values

    def save(self, uid=None): 

        if uid is None:
            return
        df = self.register
        if uid != 'Database':
            df = df[np.isin(df, uid).any(axis=1)]
        created = df.created[df.created]   
        removed = df.removed[df.removed]

        # delete datasets marked for removal
        for key in removed.index.tolist():
            # delete in memory
            if key in self.dataset:
                del self.dataset[key]
            # delete on disk
            file = self.filepath(key) 
            if os.path.exists(file): 
                os.remove(file)

        self.register.loc[created.index, 'created'] = False
        self.register.drop(index=removed.index, inplace=True)

    def restore(self, uid=None):  

        if uid is None:
            return
        df = self.register
        if uid != 'Database':
            # df = df[df[self.type(uid)] == uid]
            df = df[np.isin(df, uid).any(axis=1)]
        created = df.created[df.created]   
        removed = df.removed[df.removed]

        # permanently delete newly created datasets
        for key in created.index.tolist():
            # delete in memory
            if key in self.dataset:
                del self.dataset[key]
            # delete on disk
            file = self.filepath(key) 
            if os.path.exists(file): 
                os.remove(file)

        self.register.loc[removed.index, 'removed'] = False
        self.register.drop(index=created.index, inplace=True)

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
            ds.write(self.filepath(new_key), self.dialog)
            df.rename(index={file:new_key}, inplace=True)
        self.register = pd.concat([self.register, df])

    def export_datasets(self, uids, database):
        
        files = self.filepaths(uids)
        database.import_datasets(files)
