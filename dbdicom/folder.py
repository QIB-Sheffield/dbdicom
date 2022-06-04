"""Defines the central `Folder` class for reading and writing to DICOM folders.

`Folder()` is the central class of `dbdicom` and the first point of call 
for opening, closing and manipulating a folder with DICOM files. 

    # Example: Get a 3D numpy array from the first series in a folder.

    from dbdicom import Folder

    folder = Folder('C:\\Users\\MyName\\MyData\\DICOMtestData')
    array = folder.open().series(0).array()
"""

__all__ = ['Folder']

import os
import pydicom
import pandas as pd

#from dbdicom import dicm, utilities
from . import dicm, utilities
from .message import StatusBar, Dialog
from .classes.database import Database

class Folder(Database): 
    # This really needs to be Database() with Folder as an attribute. 
    """Programming interface for reading and writing a DICOM folder."""

    # The column labels of the dataframe as required by dbdicom
    required = [    
        'PatientID', 'StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID', 
        'SOPClassUID','NumberOfFrames', 
        'PatientName', 
        'StudyDescription', 'StudyDate', 
        'SeriesDescription', 'SeriesNumber',
        'InstanceNumber', 'AcquisitionTime', 
    ]

    def __init__(self, path=None, status=StatusBar(), dialog=Dialog()):
        """Initialise the folder with a path and objects to message to the user.
        
        When used inside a GUI, status and dialog should be instances of the status bar and 
        dialog class defined in `weasel`.
        """
        # A list of optional dicom attributes that are not used by dbdicom 
        # but can be included for faster access in other applications. 
        # The list can be changed or extended by applications.
        self.__dict__['attributes'] = ['SliceLocation']
        
        self.__dict__['dataframe'] = pd.DataFrame([]*len(self._columns), columns=self._columns)            
        self.__dict__['path'] = path
        self.__dict__['status'] = status
        self.__dict__['dialog'] = dialog
        self.__dict__['dicm'] = dicm

        super().__init__(self)

    @property
    def _columns(self):

        return self.required + self.__dict__['attributes']

    def set_attributes(self, attributes, scan=True):
        """DICOM attributes that are NOT used by dbdicom.
        
        Can be set by applications for fast access to key DICOM attributes such
        as those used for sorting series data.

        Args:
            attributes: list of DICOM attributes.
        """
        # Make sure all columns are unique
        attr = []
        for a in attributes:
            if a not in self.required:
                attr.append(a)
        self.__dict__['attributes'] = attr
        if scan: self.scan()

    def open(self, path=None, message='Opening folder..', unzip=True):
        """Opens a DICOM folder for read and write.
        
        Reads the contents of the folder and summarises all DICOM files
        in a dataframe for faster access next time. The dataframe is saved 
        as a csv file when the folder is closed with `folder.close()`. 
        All non-DICOM files in the folder are ignored.
        
        Args:
            path: The full path to the directory that is to be opened.

        Returns:
            The folder instance. This allows using the open() method in a 
            piping notation as in `inst = folder.open().instances()`.
        """
        if path is not None: 
            self.__dict__['path'] = path
        if self.path is None:
            message = "please set a path before opening."
            message += "\n Use folder.open(path_to_dicom_folder)."
            self.dialog.information(message)
            return self
        if os.path.exists(self._csv):
            self.status.message("Reading register..")
            self._read_csv()
            # If the saved register does not have all required attributes
            # then scan the folder again and create a new register
            labels = self._columns + ['removed','created']
            if labels != list(self.dataframe.columns):
#            for attribute in self.__dict__['attributes']:
#                if attribute not in self.dataframe:
                self.scan(message=message)
                self.status.hide()
                return self
            self.status.hide()
        else:
            self.scan(message=message, unzip=unzip)
        return self

    def scan(self, message='Scanning..', unzip=True):
        """
        Reads all files in the folder and summarises key attributes in a table for faster access.
        """
        if unzip:
            self.status.message('Extracting compressed folders..')
            utilities._unzip_files(self.path, self.status)
        self.status.message('Finding all files..')
        files = [item.path for item in utilities.scan_tree(self.path) if item.is_file()]
        self.__dict__['dataframe'] = utilities.dataframe(self.path, files, self._columns, self.status, message=message)
        self.dataframe['removed'] = [False] * self.dataframe.shape[0]
        self.dataframe['created'] = [False] * self.dataframe.shape[0]
        self._multiframe_to_singleframe()
        self._write_csv()
        return self

    def close(self):
        """Close an open folder.
        
        This method checks if the changes have been saved and prompts the 
        user to save or restore them if not. The user also has the option 
        to cancel closing, in which case the function does nothing. 
        
        `close()` also resets the dataframe and path to default values.

        Returns: 
            True if the user has agreed to close the folder 
            (possible after save or restore) and False if the user 
            has cancelled the closing of the folder.
        """
        if not self.is_open(): 
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

        self._write_csv()
        self.__dict__['dataframe'] = pd.DataFrame([]*len(self._columns), columns=self._columns)            
        self.__dict__['path'] = None

        return True

    def save(self):

        self.status.message("Saving..")
        if self.is_saved():
            self._write_csv()
        else:
            super().save()
        self.status.hide()

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

    def object(self, row, generation=4):
        """Create a new dicom object from a row in the dataframe.
        
        Args:
            row: a row in the dataframe as a series
            generation: determines whether the object returned is 
                at level database (generation=0), patient (generation=1),
                study (generation=2), series (generation=3) 
                or instance (generation=4, default).
        Returns:
            An instance of one of the dicom classes defined in wedicom.classes.
        """
        return dicm.object(self, row, generation)

    def instance(self, file):
        """Create an instance from a filepath"""

        row = self.dataframe.loc[file]
        return self.object(row) 

    def print(self):
        """Prints a summary of the project folder to the terminal."""
        
        print(' ')
        print('---------- DICOM FOLDER --------------')
        print('FOLDER: ' + self.path)
        for i, patient in enumerate(self.children()):
            print(' ')
            print('    PATIENT [' + str(i) + ']: ' + patient.label())
            print(' ')
            for j, study in enumerate(patient.children()):
                print('        STUDY [' + str(j) + ']: ' + study.label())
                print(' ')
                for k, series in enumerate(study.children()):
                    print('            SERIES [' + str(k) + ']: ' + series.label())
                    print('                Nr of instances: ' + str(len(series.children()))) 

    def _write_csv(self):
        """ Writes the dataFrame as a .csv file"""
        file = self._csv
        self.dataframe.to_csv(file)

    def _read_csv(self):
        """Reads the dataFrame from a .csv file """  
        file = self._csv
        self.__dict__['dataframe'] = pd.read_csv(file, index_col=0)

    @property
    def _csv(self):
        """ Returns the file path of the .csv file"""
        filename = os.path.basename(os.path.normpath(self.path)) + ".csv"
        return os.path.join(self.path, filename) 

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

    def new_uid(self, n=1):
        
        if n == 1:
            return pydicom.uid.generate_uid()
        uid = []
        for _ in range(n):
            uid.append(pydicom.uid.generate_uid())
        return uid

    def new_file(self):

        # Generate a new filename
        path = os.path.join(self.path, "dbdicom")
        if not os.path.isdir(path): os.mkdir(path)
        #file = os.path.join(path, self.new_uid() + '.dcm') 
        file = os.path.join("dbdicom", self.new_uid() + '.dcm') 
        return file

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

        row = pd.DataFrame([]*len(self._columns), index=[file], columns=self._columns)
        row['removed'] = False
        row['created'] = True
        self.__dict__['dataframe'] = pd.concat([self.dataframe, row]) 
        
        # REPLACED BY CONCAT on 03 june 2022
        # labels = self._columns + ['removed','created']
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
        for tag in self._columns:
            if tag in ds:
                value = utilities._read_tags(ds, tag)
                self.dataframe.loc[file, tag] = value
                #self.dataframe.loc[file, tag] = ds[tag].value

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
            for filepath, instance in self.dataframe[multiframe].iterrows():
                cnt+=1
                self.status.message("Converting multiframe file " + filepath)
                self.status.progress(cnt, nr_multiframe)
                #
                # Create these in the Weasel folder, not in the original folder.
                #
                singleframe_files = utilities.split_multiframe(filepath, str(instance.SeriesInstanceUID))
                if singleframe_files != []:                    
                    # add the single frame files to the dataframe
                    df = utilities.dataframe(self.path, singleframe_files, self._columns)
                    df['removed'] = [False] * df.shape[0]
                    df['created'] = [False] * df.shape[0]
                    self.__dict__['dataframe'] = pd.concat([self.dataframe, df])
                    # delete the original multiframe 
                    # CHANGE THIS - mark for removal but don't delete
                    # show enhanced DICOM in Tree but set the class to block display etc.
                    os.remove(filepath)
                    self.dataframe.drop(index=filepath, inplace=True)
                self.status.hide()