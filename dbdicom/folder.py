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
    """Programming interface for reading and writing a DICOM folder."""

    
    _columns = [    # The column labels of the dataframe
        'PatientID', 'StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID', 
        'SOPClassUID','NumberOfFrames', 
        'PatientName', 
        'StudyDescription', 'StudyDate', 
        'SeriesDescription', 'SeriesNumber',
        'InstanceNumber'
    ]

    def __init__(self, path=None, status=StatusBar(), dialog=Dialog()):
        """Initialise the folder with a path and objects to message to the user.
        
        When used inside a GUI, status and dialog should be instances of the status bar and 
        dialog class defined in `weasel`.
        """
        self.__dict__['dataframe'] = pd.DataFrame([]*len(self._columns), columns=self._columns)            
        self.__dict__['path'] = path
        self.__dict__['status'] = status
        self.__dict__['dialog'] = dialog
        self.__dict__['dicm'] = dicm

        super().__init__(self)

    def open(self, path=None):
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
            self.status.message("Reading..")
            self._read_csv()
            self.status.hide()
        else:
            self.scan()
        return self

    def scan(self):
        """Reads the folder again.

        Use this function after opening the folder if the files on the folder 
        may have been corrputed or modified by another application.
        """
        files = [item.path for item in utilities.scan_tree(self.path) if item.is_file()] 
        self.__dict__['dataframe'] = utilities.dataframe(files, self._columns, self.status)
        self.dataframe['checked'] = [False] * self.dataframe.shape[0]
        self.dataframe['removed'] = [False] * self.dataframe.shape[0]
        self.dataframe['created'] = [False] * self.dataframe.shape[0]
        self._multiframe_to_singleframe()
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

    def _append(self, ds, checked=False):
        """Append a new row to the dataframe from a pydicom dataset.
        
        Args:
            ds: and instance of pydicom dataset
        """

        # Generate a new filename
        path = os.path.join(self.path, "Weasel")
        if not os.path.isdir(path): os.mkdir(path)
        file = os.path.join(path, pydicom.uid.generate_uid() + '.dcm') 

        # Add a new row in the dataframe
        labels = self._columns + ['checked','removed','created']
        row = pd.Series(data=['']*len(labels), index=labels, name=file)
        row['checked'] = checked
        row['removed'] = False
        row['created'] = True
        self.__dict__['dataframe'] = self.dataframe.append(row) # REPLACE BY CONCAT
        self._update(file, ds)

    def _update(self, file, ds):
        """Update the values of a dataframe row.
        
        Args:
            file: filepath (or index) of the row to be updated
            ds: instance of a pydicom dataset.
        """
        for tag in self._columns:
            if tag in ds:
                self.dataframe.loc[file, tag] = utilities._read_tags(ds, tag)
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
                    df = utilities.dataframe(singleframe_files, self._columns)
                    df['checked'] = [False] * df.shape[0]
                    df['removed'] = [False] * df.shape[0]
                    df['created'] = [False] * df.shape[0]
                    self.__dict__['dataframe'] = pd.concat([self.dataframe, df])
                    # delete the original multiframe 
                    # CHANGE THIS - mark for removal but don't delete
                    # show enhanced DICOM in Tree but set the class to block display etc.
                    os.remove(filepath)
                    self.dataframe.drop(index=filepath, inplace=True)
                self.status.hide()