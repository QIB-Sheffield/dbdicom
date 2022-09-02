"""
Maintains an index of all files on disk.
"""

import os
import copy
import pandas as pd
import numpy as np

from dbdicom.message import StatusBar, Dialog
import dbdicom.utils.pydicom as pydcm
import dbdicom.utils.files as filetools
import dbdicom.utils.dcm4che as dcm4che

class DbIndex(): 
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
        # THIS NEEDS A MECHANISM TO PREVENT ANOTHER DbIndex to open the same database.
        self.status = status
        self.dialog = dialog 
        self.path = path
        self.dataframe = dataframe
        self._datasets = {}

    def read_dataframe(self, message='Reading database..'):
        """
        Reads all files in the folder and summarises key attributes in a table for faster access.
        """
        if self.path is None:
            raise ValueError('Cant read dataframe - index manages a database in memory')
        files = [item.path for item in filetools.scan_tree(self.path) if item.is_file()]
        self.dataframe = pydcm.read_dataframe(self.path, files, self.columns, self.status, message=message)
        self.dataframe['removed'] = False
        self.dataframe['created'] = False

    def _pkl(self):
        """ Returns the file path of the .pkl file"""
        if self.path is None:
            raise ValueError('Cant read index file - index manages a database in memory')
        filename = os.path.basename(os.path.normpath(self.path)) + ".pkl"
        return os.path.join(self.path, filename) 

    def _write_df(self):
        """ Writes the dataFrame as a .pkl file"""
        if self.dataframe is None:
            raise ValueError('Cant write index file - no database open')
        file = self._pkl()
        self.dataframe.to_pickle(file)

    def _read_df(self):
        """Reads the dataFrame from a .pkl file """
        if self.dataframe is None:
            raise ValueError('Cant write index file - no dataframe open') 
        file = self._pkl()
        self.dataframe = pd.read_pickle(file)

    def write_csv(self, file):
        """ Writes the dataFrame as a .csv file for visual inspection"""
        if self.dataframe is None:
            raise ValueError('Cant export index file - no database open')
        self.dataframe.to_csv(file)

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
        if self.dataframe is None:
            raise ValueError('Cannot convert multiframe - no database open')
        if self.path is None:
            # Low priority - we are not create multiframe data from scratch 
            # So will always be loaded from disk initially where the solution exists. 
            # Solution: save data in a temporary file, use the filebased conversion, 
            # the upload the solution and delete the temporary file.
            raise ValueError('Multi-frame to single-frame conversion does not yet exist from data in memory')
        singleframe = self.dataframe.NumberOfFrames.isnull() 
        multiframe = singleframe == False
        nr_multiframe = multiframe.sum()
        if nr_multiframe != 0: 
            cnt=0
            for relpath in self.dataframe[multiframe].index.values:
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
                    df = pydcm.read_dataframe(self.path, singleframe_files, self.columns)
                    df['removed'] = False
                    df['created'] = False
                    self.dataframe = pd.concat([self.dataframe, df])
                    # delete the original multiframe 
                    os.remove(filepath)
                    self.dataframe.drop(index=relpath, inplace=True)
                self.status.hide()

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

        df = self.dataframe
        return df.columns[df.isin([uid]).any()].values[0]

    def keys(self,
        uid = None, 
        patient = None,
        study = None,
        series = None,
        dataset = None):
        """Return a list of indices for all dicom datasets managed by the index.
        
        These indices are strings with unique relative paths 
        that either link to an existing file in the database or can be used for 
        writing a database that is in memory.
        """

        df = self.dataframe
        if df is None:
            raise ValueError('Cant return dicom files - no database open')

        not_deleted = df.removed == False

        # If no arguments are provided
        if (uid is None) & (patient is None) & (study is None) & (series is None) & (dataset is None):
            return []

        if uid == 'Database':
            return not_deleted[not_deleted].index.tolist()

        if isinstance(uid, list):
            if 'Database' in uid:
                return not_deleted[not_deleted].index.tolist()

        # If arguments are provided, create a list of unique datasets
        keys = []
        if uid is not None:
            if not isinstance(uid, list):
                uid = [uid]
            uid = [i for i in uid if i is not None]
            rows = np.isin(df, uid).any(axis=1) & not_deleted
            keys += df[rows].index.tolist()
        if patient is not None:
            if not isinstance(patient, list):
                patient = [patient]
            patient = [i for i in patient if i is not None]
            rows = df.PatientID.isin(patient) & not_deleted
            keys += rows[rows].index.tolist()
        if study is not None:
            if not isinstance(study, list):
                study = [study]
            study = [i for i in study if i is not None]
            rows = df.StudyInstanceUID.isin(study) & not_deleted
            keys += rows[rows].index.tolist()
        if series is not None:
            if not isinstance(series, list):
                series = [series]
            series = [i for i in series if i is not None]
            rows = df.SeriesInstanceUID.isin(series) & not_deleted
            keys += rows[rows].index.tolist()
        if dataset is not None: # rephrase dataset -> instance
            if not isinstance(dataset, list):
                dataset = [dataset]
            dataset = [i for i in dataset if i is not None]
            rows = df.SOPInstanceUID.isin(dataset) & not_deleted
            keys += rows[rows].index.tolist()
        return list(set(keys))

    def value(self, key, column):
        try:
            if not isinstance(key, list) and not isinstance(column, list):
                return self.dataframe.at[key, column]
            else:
                return self.dataframe.loc[key, column].values
        except:
            return None

    def parent(self, uid=None):
        """Returns the UID of the parent object"""

        keys = self.keys(uid)
        if keys == []:
            return None
        row = self.dataframe.loc[keys[0]].values.tolist()
        i = row.index(uid)
        if self.columns[i] == 'PatientID':
            return 'Database'
        else:
            return row[i-1]

    def children(self, uid=None):
        """Returns the UIDs of the children"""

        if isinstance(uid, list):
            children = []
            for i in uid:
                children += self.children(i)
            return children

        if uid is None:
            return []

        keys = self.keys(uid)
        if uid == 'Database':
            return list(set(self.value(keys, 'PatientID')))

        row = self.dataframe.loc[keys[0]].values.tolist()
        i = row.index(uid)
        if self.columns[i] == 'SOPInstanceUID':
            return []
        else:
            values = self.dataframe.loc[keys,self.columns[i+1]].values
            values = np.unique(values).tolist()
            return [v for v in values if v is not None]

    def instances(self, uid=None):

        keys = self.keys(uid)
        values = list(self.value(keys, 'SOPInstanceUID'))
        return [v for v in values if v is not None]

    def series(self, uid=None):

        keys = self.keys(uid)
        values = list(set(self.value(keys, 'SeriesInstanceUID')))
        return [v for v in values if v is not None]

    def studies(self, uid=None):

        keys = self.keys(uid)
        values = list(set(self.value(keys, 'StudyInstanceUID')))
        return [v for v in values if v is not None]

    def patients(self, uid=None):

        keys = self.keys(uid)
        values = list(set(self.value(keys, 'PatientID')))
        return [v for v in values if v is not None]
          
    def label(self, uid):
        """Return a label to describe a row as Patient, Study, Series or Instance"""

        if self.dataframe is None:
            raise ValueError('Cant provide labels - no database open')

        if uid is None:
            return ''
        if uid == 'Database':
            return 'Database: ' + self.path

        type = self.type(uid)
        key = self.keys(uid)[0]
        row = self.dataframe.loc[key]

        if type == 'PatientID':
            name = row.PatientName
            id = row.PatientID
            label = str(name)
            label += ' [' + str(id) + ']'
            return 'Patient' + " {}".format(label)
        if type == 'StudyInstanceUID':
            descr = row.StudyDescription
            date = row.StudyDate
            label = str(descr)
            label += ' [' + str(date) + ']'
            return 'Study' + " {}".format(label)
        if type == 'SeriesInstanceUID':
            descr = row.SeriesDescription
            nr = row.SeriesNumber
            label = str(nr).zfill(3)  
            label += ' [' + str(descr) + ']'
            return 'Series' + " {}".format(label)
        if type == 'SOPInstanceUID':
            nr = row.InstanceNumber
            label = str(nr).zfill(6)
            return pydcm.SOPClass(row.SOPClassUID) + " {}".format(label)

    def print(self):
        """Prints a summary of the project folder to the terminal."""
        
        print(' ')
        print('---------- DICOM FOLDER --------------')
        print('DATABASE: ' + self.path)
        for i, patient in enumerate(self.children()):
            print(' ')
            print('    PATIENT [' + str(i) + ']: ' + self.label(patient))
            print(' ')
            for j, study in enumerate(self.children(patient)):
                print('        STUDY [' + str(j) + ']: ' + self.label(study))
                print(' ')
                for k, series in enumerate(self.children(study)):
                    print('            SERIES [' + str(k) + ']: ' + self.label(series))
                    print('                Nr of instances: ' + str(len(self.children(series)))) 

    def read(self, *args, message=None, **kwargs):
        """Read the dataset from disk.
        """
        keys = self.keys(*args, **kwargs)
        for i, key in enumerate(keys):
            if message is not None:
                self.status.progress(i, len(keys), message)
            # do not read if they are already in memory
            # this could overwrite changes made in memory only
            if not key in self._datasets:
                file = self.filepath(key)
                self._datasets[key] = pydcm.read(file, self.dialog)

    def write(self, *args, message=None, **kwargs):
        """Writing data from memory to disk.

        This does nothing if the data are not in memory.
        """
        keys = self.keys(*args, **kwargs)
        for i, key in enumerate(keys):
            if message is not None:
                self.status.progress(i, len(keys), message)
            if key in self._datasets:
                file = self.filepath(key)
                pydcm.write(self._datasets[key], file, self.dialog)
        self.status.hide()

    def clear(self, *args, **kwargs):
        """Clear all data from memory"""
        # write to disk first so that any changes made in memory are not lost
        self.write(*args, **kwargs)
        # then delete the instances from memory
        for key in self.keys(*args, **kwargs):
            self._datasets.pop(key, None) 

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
        self.dataframe = None            
        self.path = None

    def is_saved(self):
        """Check if the folder is saved.
        
        Returns: 
            True if the folder is saved and False otherwise.
        """
        # Needs a formal test for completeness
        if self.dataframe.removed.any(): 
            return False
        if self.dataframe.created.any():
            return False
        return True

    def is_open(self):
        """Check if a database is currently open, either in memory or on disk
        
        Returns: 
            True if a database is open and False otherwise.
        """
        # Needs a formal test for completeness
        return self.dataframe is not None

    def dataset(self, key):
        # Needs a unit test

        if key in self._datasets:
            # If in memory, get from memory
            return self._datasets[key]
        else:
            # If not in memory, read from disk
            # Raise an exception if the database is in memory only.
            try:
                file = self.filepath(key)
            except:
                raise ValueError('Dataset ' + key + ' does not exist.')
            else:
                return pydcm.read(file, self.dialog)       

    def datasets(self, keys, message=None):
        """Gets a list of datasets
        
        Datasets in memory will be returned.
        If they are not in memory, and the database exists on disk, they will be read from disk.
        If they are not in memory, and the database does not exist on disk, an exception is raised.
        """
        datasets = []
        for i, key in enumerate(keys):
            if message is not None:
                self.status.progress(i, len(keys), message)
            ds = self.dataset(key)
            datasets.append(ds)
        self.status.hide()
        return datasets

    def delete(self, *args, **kwargs):
        """Deletes some datasets
        
        Deleted datasets are stashed and can be recovered with restore()
        Using save() will delete them permanently
        """
        keys = self.keys(*args, **kwargs)
        self.dataframe.loc[keys,'removed'] = True

    def new_key(self):
        """Generate a new key"""

        return os.path.join('dbdicom', pydcm.new_uid() + '.dcm') 

    def copy_to_series(self, uid, target):
        """Copy instances to another series"""

        attributes = list(set(
            pydcm.module_patient() + 
            pydcm.module_study() + 
            pydcm.module_series() ))
        target_keys = self.keys(series=target)
        ds = self.dataset(target_keys[0])
        values = pydcm.get_values(ds, attributes)
        max_number = np.amax(self.value(target_keys, 'InstanceNumber'))

        copy_data = []
        copy_keys = []

        keys = self.keys(uid)
        for i, key in enumerate(keys):

            self.status.progress(i+1, len(keys), message='Copying..')

            # Set new UIDs in dataset
            # If the dataset is in memory, the copy is created in memory too
            # Else the copy is created on disk
            new_key = self.new_key()
            ds = self.dataset(key)
            if key in self._datasets:
                ds = copy.deepcopy(ds)
                self._datasets[new_key] = ds
            pydcm.set_values(ds, 
                attributes + ['SOPInstanceUID', 'InstanceNumber'], 
                values + [pydcm.new_uid(), i+1+max_number])
            if not key in self._datasets:
                pydcm.write(ds, self.filepath(new_key), self.dialog)

            # Get new data for the dataframe
            row = pydcm.get_values(ds, self.columns)
            copy_data.append(row)
            copy_keys.append(new_key)

        # Update the dataframe in the index
        df = pd.DataFrame(copy_data, index=copy_keys, columns=self.columns)
        df['removed'] = False
        df['created'] = True
        self.dataframe = pd.concat([self.dataframe, df])

        return copy_keys


    def copy_to_study(self, uid, target):
        """Copy series to another study"""

        attributes = list(set(
            pydcm.module_patient() + 
            pydcm.module_study() ))
        target_keys = self.keys(study=target)
        ds = self.dataset(target_keys[0])
        values = pydcm.get_values(ds, attributes)
        max_number = np.amax(self.value(target_keys, 'SeriesNumber'))

        copy_data = []
        copy_keys = []

        all_series = self.series(uid)
        for s, series in enumerate(all_series):

            self.status.progress(s+1, len(all_series), message='Copying..')
            new_uid = pydcm.new_uid() # should return a list of these instead of keys
            new_number = s + 1 + max_number

            for key in self.keys(series):

                new_key = self.new_key()
                ds = self.dataset(key)
                if key in self._datasets:
                    ds = copy.deepcopy(ds)
                    self._datasets[new_key] = ds
                pydcm.set_values(ds, 
                    attributes + ['SeriesInstanceUID', 'SeriesNumber', 'SOPInstanceUID'], 
                    values + [new_uid, new_number, pydcm.new_uid()])
                if not key in self._datasets:
                    pydcm.write(ds, self.filepath(new_key), self.dialog)
                # Get new data for the dataframe
                row = pydcm.get_values(ds, self.columns)
                copy_data.append(row)
                copy_keys.append(new_key)

        # Update the dataframe in the index
        df = pd.DataFrame(copy_data, index=copy_keys, columns=self.columns)
        df['removed'] = False
        df['created'] = True
        self.dataframe = pd.concat([self.dataframe, df])

        return copy_keys


    def copy_to_patient(self, uid, target):
        """Copy studies to another patient"""

        attributes = pydcm.module_patient()
        target_keys = self.keys(patient=target)
        ds = self.dataset(target_keys[0])
        values = pydcm.get_values(ds, attributes)

        copy_data = []
        copy_keys = []

        all_studies = self.studies(uid)
        for s, study in enumerate(all_studies):
            
            self.status.progress(s+1, len(all_studies), message='Copying..')
            new_study_uid = pydcm.new_uid()

            for series in self.series(study):

                new_series_uid = pydcm.new_uid()
                for key in self.keys(series):

                    new_key = self.new_key()
                    ds = self.dataset(key)
                    if key in self._datasets:
                        ds = copy.deepcopy(ds)
                        self._datasets[new_key] = ds
                    pydcm.set_values(ds, 
                        attributes + ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID'], 
                        values + [new_study_uid, new_series_uid, pydcm.new_uid()])
                    if not key in self._datasets:
                        pydcm.write(ds, self.filepath(new_key), self.dialog)
                    # Get new data for the dataframe
                    row = pydcm.get_values(ds, self.columns)
                    copy_data.append(row)
                    copy_keys.append(new_key)

        # Update the dataframe in the index
        df = pd.DataFrame(copy_data, index=copy_keys, columns=self.columns)
        df['removed'] = False
        df['created'] = True
        self.dataframe = pd.concat([self.dataframe, df])

        return copy_keys

    def copy_to(self, source, target):

        type = self.type(target)
        if type == 'PatientID':
            return self.copy_to_patient(source, target)
        if type == 'StudyInstanceUID':
            return self.copy_to_study(source, target)
        if type == 'SeriesInstanceUID':
            return self.copy_to_series(source, target)
        if type == 'SOPInstanceUID':
            raise ValueError('Cannot copy to an instance. Please copy to series, study or patient.')


    def move_to_series(self, uid, target):
        """Copy datasets to another series"""

        attributes = list(set(
            pydcm.module_patient() + 
            pydcm.module_study() + 
            pydcm.module_series() ))
        target_keys = self.keys(series=target)
        ds = self.dataset(target_keys[0])
        values = pydcm.get_values(ds, attributes)
        max_instance_number = np.amax(self.value(target_keys, 'InstanceNumber'))

        copy_data = []
        copy_keys = []
        curr_keys = []        

        keys = self.keys(uid)
        for i, key in enumerate(keys):

            self.status.progress(i+1, len(keys), message='Moving dataset..')

            ds = self.dataset(key)

            # If the value has changed before.
            if self.value(key, 'created'): 
                pydcm.set_values(ds, 
                    attributes + ['InstanceNumber'], 
                    values + [i+1 + max_instance_number])
                if not key in self._datasets:
                    pydcm.write(ds, self.filepath(key), self.dialog)
                for i, col in enumerate(attributes):
                    if col in self.columns:
                        self.dataframe.at[key,col] = values[i]
                curr_keys.append(key)

            # If this is the first change, then save results in a copy.
            else:  
                new_key = self.new_key()
                if key in self._datasets:
                    ds = copy.deepcopy(ds)
                    self._datasets[new_key] = ds
                pydcm.set_values(ds, 
                    attributes + ['InstanceNumber'], 
                    values + [i+1+max_instance_number])
                if not key in self._datasets:
                    pydcm.write(ds, self.filepath(new_key), self.dialog)

                # Get new data for the dataframe
                self.dataframe.at[key,'removed'] = True
                row = pydcm.get_values(ds, self.columns)
                copy_data.append(row)
                copy_keys.append(new_key)

        # Update the dataframe in the index
        if copy_data != []:
            df = pd.DataFrame(copy_data, index=copy_keys, columns=self.columns)
            df['removed'] = False
            df['created'] = True
            self.dataframe = pd.concat([self.dataframe, df])

        return copy_keys + curr_keys


    def move_to_study(self, uid, target):
        """Copy series to another study"""

        attributes = list(set(
            pydcm.module_patient() + 
            pydcm.module_study() ))
        target_keys = self.keys(study=target)
        ds = self.dataset(target_keys[0])
        values = pydcm.get_values(ds, attributes)
        max_number = np.amax(self.value(target_keys, 'SeriesNumber'))

        copy_data = []
        copy_keys = []
        curr_keys = []        

        all_series = self.series(uid)
        for s, series in enumerate(all_series):

            self.status.progress(s+1, len(all_series), message='Moving series..')
            new_number = s + 1 + max_number

            for key in self.keys(series):

                ds = self.dataset(key)

                # If the value has changed before.
                if self.value(key, 'created'): 
                    pydcm.set_values(ds, 
                        attributes + ['SeriesNumber'], 
                        values + [new_number])
                    if not key in self._datasets:
                        pydcm.write(ds, self.filepath(key), self.dialog)
                    for i, col in enumerate(attributes):
                        if col in self.columns:
                            self.dataframe.at[key,col] = values[i]
                    curr_keys.append(key)

                # If this is the first change, then save results in a copy.
                else:  
                    new_key = self.new_key()
                    if key in self._datasets:
                        ds = copy.deepcopy(ds)
                        self._datasets[new_key] = ds
                    pydcm.set_values(ds, 
                        attributes + ['SeriesNumber'], 
                        values + [new_number])
                    if not key in self._datasets:
                        pydcm.write(ds, self.filepath(new_key), self.dialog)

                    # Get new data for the dataframe
                    self.dataframe.at[key,'removed'] = True
                    row = pydcm.get_values(ds, self.columns)
                    copy_data.append(row)
                    copy_keys.append(new_key)

        # Update the dataframe in the index
        if copy_data != []:
            df = pd.DataFrame(copy_data, index=copy_keys, columns=self.columns)
            df['removed'] = False
            df['created'] = True
            self.dataframe = pd.concat([self.dataframe, df])

        return copy_keys + curr_keys


    def move_to_patient(self, uid, target):
        """Copy series to another study"""

        attributes = pydcm.module_patient()
        target_keys = self.keys(patient=target)
        ds = self.dataset(target_keys[0])
        values = pydcm.get_values(ds, attributes)

        copy_data = []
        copy_keys = []
        curr_keys = []    

        all_studies = self.studies(uid)
        for s, study in enumerate(all_studies):
            
            self.status.progress(s+1, len(all_studies), message='Moving study..')
            for series in self.series(study):

                for key in self.keys(series):

                    ds = self.dataset(key)

                    # If the value has changed before.
                    if self.value(key, 'created'): 
                        pydcm.set_values(ds, 
                            attributes, 
                            values)
                        if not key in self._datasets:
                            pydcm.write(ds, self.filepath(key), self.dialog)
                        for i, col in enumerate(attributes):
                            if col in self.columns:
                                self.dataframe.at[key,col] = values[i]
                        curr_keys.append(key)

                    # If this is the first change, then save results in a copy.
                    else:  
                        new_key = self.new_key()
                        if key in self._datasets:
                            ds = copy.deepcopy(ds)
                            self._datasets[new_key] = ds
                        pydcm.set_values(ds, 
                            attributes, 
                            values)
                        if not key in self._datasets:
                            pydcm.write(ds, self.filepath(new_key), self.dialog)

                        # Get new data for the dataframe
                        self.dataframe.at[key,'removed'] = True
                        row = pydcm.get_values(ds, self.columns)
                        copy_data.append(row)
                        copy_keys.append(new_key)

            # Update the dataframe in the index
            if copy_data != []:
                df = pd.DataFrame(copy_data, index=copy_keys, columns=self.columns)
                df['removed'] = False
                df['created'] = True
                self.dataframe = pd.concat([self.dataframe, df])

            return copy_keys + curr_keys

    def move_to(self, source, target):

        type = self.type(target)
        if type == 'PatientID':
            return self.move_to_patient(source, target)
        if type == 'StudyInstanceUID':
            return self.move_to_study(source, target)
        if type == 'SeriesInstanceUID':
            return self.move_to_series(source, target)
        if type == 'SOPInstanceUID':
            raise ValueError('Cannot move to an instance. Please move to series, study or patient.')


    def set_values(self, uid, attributes, values):
        """Copy datasets to another series"""

        uids = ['PatientID', 'StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']
        uids = [i for i in uids if i in attributes]
        if uids != []:
            raise ValueError('UIDs cannot be set using set_value(). Use copy_to() or move_to() instead.')

        copy_data = []
        copy_keys = []

        keys = self.keys(uid)
        for i, key in enumerate(keys):

            self.status.progress(i+1, len(keys), message='Setting values..')

            ds = self.dataset(key)

            # If the value has changed before
            if self.value(key, 'created'): 
                pydcm.set_values(ds, attributes, values)
                if not key in self._datasets:
                    pydcm.write(ds, self.filepath(key), self.dialog)
                for i, col in enumerate(attributes):
                    if col in self.columns:
                        self.dataframe.at[key,col] = values[i]

            # If this is the first change, then save results in a copy
            else:  
                new_key = self.new_key()
                if key in self._datasets:
                    ds = copy.deepcopy(ds)
                    self._datasets[new_key] = ds
                pydcm.set_values(ds, attributes, values)
                if not key in self._datasets:
                    pydcm.write(ds, self.filepath(new_key), self.dialog)

                # Get new data for the dataframe
                self.dataframe.at[key,'removed'] = True
                row = pydcm.get_values(ds, self.columns)
                copy_data.append(row)
                copy_keys.append(new_key)

        # Update the dataframe in the index
        if copy_data != []:
            df = pd.DataFrame(copy_data, index=copy_keys, columns=self.columns)
            df['removed'] = False
            df['created'] = True
            self.dataframe = pd.concat([self.dataframe, df])

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
                        ds = self.dataset(key)
                        v = pydcm.get_values(ds, attributes)
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
                        ds = self.dataset(key)
                        v[i,:] = pydcm.get_values(ds, attributes)

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
        df = self.dataframe
        if uid != 'Database':
            # df = df[df[self.type(uid)] == uid]
            df = df[np.isin(df, uid).any(axis=1)]
        created = df.created[df.created]   
        removed = df.removed[df.removed]

        # delete datasets marked for removal
        for key in removed.index.tolist():
            # delete in memory
            if key in self._datasets:
                del self._datasets[key]
            # delete on disk
            try:
                file = self.filepath(key) # !!! should return None if in memory only rather than raise exception !!!
                if os.path.exists(file): 
                    os.remove(file)
            except:
                pass

        df.loc[created.index, 'created'] = False
        df.drop(removed.index, inplace=True)

    def restore(self, uid=None):  

        if uid is None:
            return
        df = self.dataframe
        if uid != 'Database':
            # df = df[df[self.type(uid)] == uid]
            df = df[np.isin(df, uid).any(axis=1)]
        created = df.created[df.created]   
        removed = df.removed[df.removed]

        # permanently delete newly created datasets
        for key in created.index.tolist():
            # delete in memory
            if key in self._datasets:
                del self._datasets[key]
            # delete on disk
            try:
                file = self.filepath(key) # !!! should return None if in memory only rather than raise exception !!!
                if os.path.exists(file): 
                    os.remove(file)
            except:
                pass

        df.loc[removed.index, 'removed'] = False
        df.drop(created.index, inplace=True)

    def new_patient(self):

        data = [None] * len(self.columns)
        data[0] = pydcm.new_uid()

        df = pd.DataFrame([data], index=[self.new_key()], columns=self.columns)
        df['removed'] = False
        df['created'] = True
        self.dataframe = pd.concat([self.dataframe, df])

        return data[0]

    def new_study(self, patient=None):

        data = [None] * len(self.columns)
        data[1] = pydcm.new_uid()

        if patient is None:
            data[0] = pydcm.new_uid()
        else:
            key = self.keys(patient=patient)[0]
            data[0] = self.value(key, 'PatientID')
            data[6] = self.value(key, 'PatientName')

        df = pd.DataFrame([data], index=[self.new_key()], columns=self.columns)
        df['removed'] = False
        df['created'] = True
        self.dataframe = pd.concat([self.dataframe, df])

        return data[1]

    def new_series(self, study=None):

        data = [None] * len(self.columns)
        data[2] = pydcm.new_uid()

        if study is None:
            data[0] = pydcm.new_uid()
            data[1] = pydcm.new_uid()
        else:
            key = self.keys(study=study)[0]
            data[0] = self.value(key, 'PatientID')
            data[0] = self.value(key, 'StudyInstanceUID')
            data[6] = self.value(key, 'PatientName')
            data[7] = self.value(key, 'StudyDescription')
            data[8] = self.value(key, 'StudyDate')

        df = pd.DataFrame([data], index=[self.new_key()], columns=self.columns)
        df['removed'] = False
        df['created'] = True
        self.dataframe = pd.concat([self.dataframe, df])

        return data[2]


# Tested until here



    # def row(self, key):
    #     # Needs a formal test
    #     return self.dataframe.loc[key]

    def sortby(self, sortby):
        """Sort dataframe values by given list of column labels"""
        if self.dataframe is None:
            raise ValueError('Cant sort database index - no database open')
        # Needs a formal test for completeness
        self.dataframe.sort_values(sortby, inplace=True)
        return self

    def _new_index(self, instances):
        """Create a new index for a list of instances (DbData or DbRecord)"""

        data = []
        for instance in instances:
            ds = instance.read()._ds
            row = pydcm.get_values(ds, self.columns)
            data.append(row)
        new_files = [self.new_file() for _ in instances]
        df = pd.DataFrame(data, index=new_files, columns=self.columns)
        df['removed'] = False
        df['created'] = True
        return df

    def _update_index(self, instances): # instances is a list of DbDatas or DbRecords

        df = self.dataframe[self.dataframe.removed == False]
        for instance in instances:
            uid = instance.SOPInstanceUID
            filename = df.index[df.SOPInstanceUID == uid].tolist()[0]
            ds = instance.read()._ds
            values = pydcm.get_values(ds, self.columns)
            self.dataframe.loc[filename, self.columns] = values

    def _add(self, instances): # instances is a list of instance DbDatas or DbRecords
        
        # TODO speed up if instances has just one element

        df = self.dataframe[self.dataframe.removed == False]

        # Find existing datasets that are changing for the first time 
        # and existing datasets that have changed before
        uids = [i.SOPInstanceUID for i in instances]
        df_first_change = df.loc[df.SOPInstanceUID.isin(uids) & (df.created == False)]
        df_prevs_change = df.loc[df.SOPInstanceUID.isin(uids) & (df.created == True)]
        uids_first_change = df_first_change.SOPInstanceUID.values
        uids_prevs_change = df_prevs_change.SOPInstanceUID.values
        uids_all = df.SOPInstanceUID.values

        # Create new dataframe for those that are changing for the first time
        first_change = [i for i in instances if i.SOPInstanceUID in uids_first_change]
        df_created = self._new_index(first_change)

        # Update the dataframe values for those that have changed before
        prevs_change = [i for i in instances if i.SOPInstanceUID in uids_prevs_change]
        self._update_index(prevs_change)

        # Find datasets that are new to the database and create a dataframe for them
        new = [i for i in instances if i.SOPInstanceUID not in uids_all]
        df_new = self._new_index(new)
 
        # Extend the dataframe with new rows
        self.dataframe.loc[df_first_change.index, 'removed'] = True
        self.dataframe = pd.concat([self.dataframe, df_created, df_new])

    def update(self, uids, attributes): # instances is a list of instance DbDatas or DbRecords

        df = self.dataframe[self.dataframe.removed == False]

        # Find existing datasets that are changing for the first time 
        # and existing datasets that have changed before
        df_first_change = df.loc[df.SOPInstanceUID.isin(uids) & (df.created == False)]
        df_prevs_change = df.loc[df.SOPInstanceUID.isin(uids) & (df.created == True)]

        # Create new dataframe for those that are changing for the first time
        for key, value in attributes.items():
            for index in df_first_change.index.values():
                df_first_change.at[index, key] = value
        df_first_change['removed'] = False
        df_first_change['created'] = True

        # Update the dataframe values for those that have changed before
        for key, value in attributes.items():
            for index in df_prevs_change.index.values():
                df_first_change.at[index, key] = value
 
        # Extend the dataframe with new rows
        self.dataframe.loc[df_first_change.index, 'removed'] = True
        self.dataframe = pd.concat([self.dataframe, df_first_change])
        
        
    def _append(self, ds):
        """Append a new row to the dataframe from a pydicom dataset.
        
        Args:
            ds: and instance of pydicom dataset
        """

        # Generate a new filename
        file = self.new_file()
#        path = os.path.join(self.path, "Weasel")
#        if not os.path.isdir(path): os.mkdir(path)
#        file = os.path.join(path, pydicom.uid.generate_uid() + '.dcm') 

        row = pd.DataFrame([]*len(self.columns), index=[file], columns=self.columns)
        row['removed'] = False
        row['created'] = True
        self.dataframe = pd.concat([self.dataframe, row]) 
        
        # REPLACED BY CONCAT on 03 june 2022
        # labels = self.columns + ['removed','created']
        #row = pd.Series(data=['']*len(labels), index=labels, name=file)
        #row['removed'] = False
        #row['created'] = True
        #self.__dict__['dataframe'] = self.dataframe.append(row) # REPLACE BY CONCAT

        self._update(file, ds)

    def _update(self, file, ds):
        """Update the values of a dataframe row.
        
        Args:
            file: filepath (or index) of the row to be updated
            ds: instance of a pydicom dataset.
        """
        for tag in self.columns:
            if tag in ds:
                value = pydcm.get_values(ds, tag)
                self.dataframe.loc[file, tag] = value
                #self.dataframe.loc[file, tag] = ds[tag].value

