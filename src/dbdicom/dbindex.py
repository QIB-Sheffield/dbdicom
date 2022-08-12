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

    def __init__(self, path=None, status=StatusBar(), dialog=Dialog()):
        """Initialise the folder with a path and objects to message to the user.
        
        When used inside a GUI, status and dialog should be instances of the status bar and 
        dialog class defined in `weasel`.
        """  
        # THIS REQUIRES A MECHANISM TO PREVENT ANOTHER DbIndex to open the same database   
        self.status = status
        self.dialog = dialog 
        self.path = path
        self.dataframe = None

    def read_dataframe(self, message='Reading database..'):
        """
        Reads all files in the folder and summarises key attributes in a table for faster access.
        """
        if self.path is None:
            return
        files = [item.path for item in filetools.scan_tree(self.path) if item.is_file()]
        self.dataframe = pydcm.read_dataframe(self.path, files, self.columns, self.status, message=message)
        self.dataframe['removed'] = False
        self.dataframe['created'] = False

    def _pkl(self):
        """ Returns the file path of the .pkl file"""
        filename = os.path.basename(os.path.normpath(self.path)) + ".pkl"
        return os.path.join(self.path, filename) 

    def _write_df(self):
        """ Writes the dataFrame as a .pkl file"""
        if self.path is None:
            return
        file = self._pkl()
        self.dataframe.to_pickle(file)

    def _read_df(self):
        """Reads the dataFrame from a .pkl file """  
        file = self._pkl()
        self.dataframe = pd.read_pickle(file)

    def _write_csv(self):
        """ Writes the dataFrame as a .csv file for visual inspection"""
        if self.path is None:
            return
        filename = os.path.basename(os.path.normpath(self.path)) + ".csv"
        file = os.path.join(self.path, filename)
        self.dataframe.to_csv(file)

    def _multiframe_to_singleframe(self):
        """Converts all multiframe files in the folder into single-frame files.
        
        Reads all the multi-frame files in the folder,
        converts them to singleframe files, and delete the original multiframe file.
        """

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
        as a pkl file when the folder is closed with `folder.close()`. 
        All non-DICOM files in the folder are ignored.
        
        Args:
            path: The full path to the directory that is to be opened.

        Returns:
            The folder instance. This allows using the open() method in a 
            piping notation as in `inst = folder.open().instances()`.
        """
        if path is not None:
            self.path = path
        if self.path is None:
            return
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
        self._write_df()
        self.dataframe = None            
        self.path = None


    def file(self, relpath):
        if self.path is None:
            return ValueError('Cannot access a file - this database exists in memory only')
        return os.path.join(self.path, relpath)

    def files(self):
        relpaths = self.dataframe.index.tolist()
        files = [self.file(p) for p in relpaths]
        return files, relpaths
    
    def is_saved(self):
        """Check if the folder is saved.
        
        Returns: 
            True if the folder is saved and False otherwise.
        """
        if self.dataframe.removed.any(): return False
        if self.dataframe.created.any(): return False
        return True

    def is_open(self):
        """Check if the folder is currently open.
        
        Returns: 
            True if the folder is open and False otherwise.
        """
        return self.path is not None

    def sortby(self, sortby):
        
        self.dataframe.sort_values(sortby, inplace=True)
        return self



    def label(self, row, type):

        if type == 'Patient':
            name = row.PatientName
            id = row.PatientID
            label = str(name)
            label += ' [' + str(id) + ']'
            return type + " - {}".format(label)
        if type == 'Study':
            descr = row.StudyDescription
            date = row.StudyDate
            label = str(descr)
            label += ' [' + str(date) + ']'
            return type + " - {}".format(label)
        if type == 'Series':
            descr = row.SeriesDescription
            nr = row.SeriesNumber
            label = '[' + str(nr).zfill(3) + '] ' 
            label += str(descr)
            return type + " - {}".format(label)
        if type == 'Instance':
            nr = row.InstanceNumber
            label = str(nr).zfill(6)
            return type + " - {}".format(label)

    def new_file(self):

        # Generate a new filename
        path = os.path.join(self.path, "dbdicom")
        if not os.path.isdir(path): os.mkdir(path)
        #file = os.path.join(path, self.new_uid() + '.dcm') 
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

