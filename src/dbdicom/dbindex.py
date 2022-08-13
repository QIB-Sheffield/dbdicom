"""
Maintains an index of all files on disk.
"""

import os
import pandas as pd


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
        # THIS REQUIRES A MECHANISM TO PREVENT ANOTHER DbIndex to open the same database.
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

    def close(self):
        """Close an open record.
        """
        if self.path is None:
            self.dataframe = None
            return
        ### CHECK FOR FILES THAT HAVE BEEN MODIFIED AND/OR DELETED
        ### AND OFFER SAVE/RESTORE OPTIONS AT DATABASE LEVEL.
        self._write_df()
        self.dataframe = None            
        self.path = None

    def keys(self,
        patient = None,
        study = None,
        series = None,
        dataset = None):
        """Return a list of indices for all dicom datasets managed by the index.
        
        These indices are strings with unique relative paths 
        that either link to an existing file in the database or can be used for 
        writing a database that is in memory.
        """
        if self.dataframe is None:
            raise ValueError('Cant return dicom files - no database open')
        if patient is not None:
            column = self.dataframe.PatientID
            return column[column == patient].index.tolist()
        if study is not None:
            column = self.dataframe.StudyInstanceUID
            return column[column == study].index.tolist()
        if series is not None:
            column = self.dataframe.SeriesInstanceUID
            return column[column == series].index.tolist()
        if dataset is not None:
            column = self.dataframe.SOPInstanceUID
            return column[column == dataset].index.tolist()
        return self.dataframe.index.tolist()

    def filepath(self, key):
        """Return the full filepath for a given relative path.
        
        Returns an error for data that live in memory only"""
        # Needs a formal test for completeness
        if self.path is None:
            return ValueError('Cannot get a filepath - this database exists in memory only')
        return os.path.join(self.path, key)

    def filepaths(self, **kwargs):
        """Return a list of full filepaths for all dicom files in the folder"""
        # Needs a formal test for completeness
        return [self.filepath(p) for p in self.keys(**kwargs)]

    def value(self, key, column):
        try:
            return self.dataframe.at[key, column]
        except KeyError:
            raise KeyError('Location (' + key + ', ' + column+') does not exist in the dataframe')

    def label(self, key, type):
        """Return a label to describe a row as Patient, Study, Series or Instance"""

        if self.dataframe is None:
            raise ValueError('Cant provide labels - no database open')

        row = self.dataframe.loc[key]

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
            return pydcm.SOPClass(row.SOPClassUID) + " {}".format(label)

    def read(self, message=None, **kwargs):
        """Read the dataset from disk and return a list of datasets read.
        """
        keys = self.keys(**kwargs)
        for i, key in enumerate(keys):
            if message is not None:
                self.status.progress(i, len(keys), message)
            file = self.filepath(key)
            self._datasets[key] = pydcm.read(file, self.dialog)
        self.status.hide()
        return [self._datasets[key] for key in keys]

    def clear(self, **kwargs):
        """Clear all data from memory"""
        for key in self.keys(**kwargs):
            self._datasets.pop(key, None) 

    def write(self, message=None, **kwargs):
        """Writing data from memory to disk.

        This does nothing if the data are not in memory.
        """
        keys = self.keys(**kwargs)
        for i, key in enumerate(keys):
            if message is not None:
                self.status.progress(i, len(keys), message)
            if key in self._datasets:
                file = self.filepath(key)
                pydcm.write(self._datasets[key], file, self.dialog)
        self.status.hide()

    def datasets(self, message=None, **kwargs):
        """Gets a list of datasets
        
        Datasets in memory will be returned.
        If they are not in memory, and the database exists on disk, they will be read from disk.
        If they are not in memory, and the database does not exist on disk, an exception is raised.
        """
        keys = self.keys(**kwargs)
        datasets = []
        for i, key in enumerate(keys):
            if message is not None:
                self.status.progress(i, len(keys), message)
            if key in self._datasets:
                # If in memory, get from memory
                ds = self._datasets[key]
            else:
                # If not in memory, read from disk
                # Raise an exception if the database is in memory only.
                try:
                    file = self.filepath(key)
                except:
                    raise ValueError('Dataset ' + key + ' does not exist.')
                else:
                    ds = pydcm.read(file, self.dialog)
            datasets.append(ds)
        self.status.hide()
        return datasets




# Tested until here


    def is_saved(self):
        """Check if the folder is saved.
        
        Returns: 
            True if the folder is saved and False otherwise.
        """
        # Needs a formal test for completeness
        if self.dataframe is None:
            return True
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


    def new_file(self):
        """Generate a new filename"""
        # Needs a formal test
        if self.path is None:
            return ValueError('Cannot get a new file - this database exists in memory only')
        path = os.path.join(self.path, "dbdicom")
        if not os.path.isdir(path): 
            os.mkdir(path)
        file = os.path.join("dbdicom", pydcm.new_uid() + '.dcm') 
        return file

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

