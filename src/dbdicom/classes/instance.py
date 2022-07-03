import os
import shutil
from copy import deepcopy
from datetime import datetime

import pydicom
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from .record import Record
from .. import utilities

class Instance(Record):

    def __init__(self, folder, UID=[], **attributes):
        super().__init__(folder, UID, generation=4, **attributes)

    def label(self, row=None):

        if row is None:
            data = self.data()
            if data.empty: return "New Instance"
            file = data.index[0]
            nr = data.at[file, 'InstanceNumber']
        else:
            nr = row.InstanceNumber

        return str(nr).zfill(6)

    @property
    def file(self):
        """Returns the filepath to the instance."""
 
        files = self.files
        if len(files) != 0:  
            return files[0]

    def __getitem__(self, tags):
        """Gets the value of the data elements with specified tags.
        
        Arguments
        ---------
        tags : a string, hexadecimal tuple, or a list of strings and hexadecimal tuples

        Returns
        -------
        A value or a list of values
        """
        in_memory = self.in_memory()
        if not in_memory: 
            self.read()
        values = utilities._read_tags(self.ds, tags)   
        if not in_memory: 
            self.clear()
        return values

    def __setitem__(self, tags, values):
        """
        Sets the value of the data element with given tag.
        """
        on_disk = self.on_disk()
        if on_disk: 
            self.read()
        utilities._set_tags(self.ds, tags, values)
        if on_disk:
            self.write()
            self.clear()

    def write(self):
        """Writes the dataset to disk"""

        if self.on_disk(): return
        ds = self.ds

        # Ensure DICOM hierarchy is respected
        if not 'PatientID' in ds:
            ds.PatientID = pydicom.uid.generate_uid()
        if not 'StudyInstanceUID' in ds:
            ds.StudyInstanceUID = pydicom.uid.generate_uid()
        if not 'SeriesInstanceUID' in ds:
            ds.SeriesInstanceUID = pydicom.uid.generate_uid()    
        if not 'SOPInstanceUID' in ds:
            ds.SOPInstanceUID = pydicom.uid.generate_uid()    

        df = self.data()
        if df.empty: # Data exist in memory only.
            self.folder._append(ds)
        else: 
            file = df.index[0] 
            if not df.loc[file,'created']:  # This is the first change 
                self.folder.dataframe.loc[file,'removed'] = True
                self.folder._append(ds)
            else:   # Update values in dataframe row
                self.folder._update(file, ds)

        self._save_ds()

    def _save_ds(self, file=None):

        if file is None: 
            file = self.file
        try:
            self.ds.save_as(file) 
        except:
            message = "Failed to write to " + file
            message += "\n The file is open in another application, or is being synchronised by a cloud service."
            message += "\n Please close the file or pause the synchronisation and try again."
            self.dialog.information(message)

    def clear(self):
        """Clears the dataset in memory"""

        self.__dict__['ds'] = None 

    def read(self):
        """Reads the dataset into memory."""

        file = self.file
        if file is None: 
            return
        try:
            self.ds = pydicom.dcmread(file)
        except:
            message = "Failed to read " + file
            message += "\n Please read the DICOM folder again via File->Read."
            self.dialog.information(message)            
        return self.ds # return self instead so read can be piped: series.read().copy_to(parent)

    def _copy_to_OBSOLETE(self, ancestor): # Replaced by copy_to in record
        """copy instance to a new ancestor.
        
        dicom_object: Root, Patient, Study, or Series
            If the object is not a series, the missing 
            intermediate generations are automatically created.
        """
        # Generate new instance
        copy = self.__class__(self.folder, UID=ancestor.UID)

        if self.in_memory(): # Create the copy in memory
            copy.__dict__['ds'] = deepcopy(self.ds)
            copy._initialize(self.ds)
            if ancestor.in_memory():
                ancestor.ds.append(copy)
        else: # Create copy on disk
            self.read()
            copy.__dict__['ds'] = self.ds
            copy._initialize(self.ds)
            copy.folder._append(copy.ds)
            copy._save_ds()
            self.clear()
            if ancestor.in_memory():
                ancestor.ds.append(copy) 
            else:       
                copy.clear()
            
        return copy

    def save_OBSOLETE(self):
        """
        Saves all changes made in the instance
        """
        self.write() 
        rows = self.folder.dataframe.SOPInstanceUID == self.UID[-1]
        data = self.folder.dataframe[rows] 
        created = data.created[data.created]
        removed = data.removed[data.removed]

        if data.shape[0] == 2: # instance has been modified
            created = created.index
            removed = removed.index
            if created.empty or removed.empty: 
                message = 'DICOM dataset ' + self.file
                message += "\n Source data have been corrupted."
                message += "\n In your DICOM folder, remove the folder Weasel"
                message += "\n and delete the .csv file."
                message += "\n Then read the DICOM folder again."
                self.dialog.error(message)
                return
            try: # save changes in original file
                shutil.copyfile(created[0], removed[0])
            except:
                message = "DICOM files have been removed or are open in another application."
                message += '\n Close the files and try again.'
                self.dialog.error(message)
                return
            os.remove(created[0])
            self.folder.dataframe.drop(created, inplace=True)
            self.folder.dataframe.loc[removed, 'removed'] = False

        elif not removed.empty: # instance has been deleted
            index = removed.index
            os.remove(index[0])
            self.folder.dataframe.drop(index, inplace=True)
            
        elif not created.empty: # instance has been newly created
            self.folder.dataframe.loc[created.index, 'created'] = False

    def restore_OBSOLETE(self):
        """
        Reverses all changes made since the last save.
        """
        in_memory = self.in_memory()
        self.clear() 
        rows = self.folder.dataframe[self.key[-1]] == self.UID[-1]
        data = self.folder.dataframe[rows] 
        created = data.created[data.created]
        removed = data.removed[data.removed]
        if not removed.empty: # restore deleted files
            index = removed.index
            self.folder.dataframe.loc[index, 'removed'] = False
        if not created.empty: # delete new files
            index = created.index
            file = index[0]
            self.folder.dataframe.drop(index, inplace = True)
            if os.path.exists(file): os.remove(file)
        if in_memory: self.read()

    def export(self, path):
        """Export instances to an external folder.

        This will create another copy of the same instance.
        The instance itself will not be removed from the DICOM folder.
        Instead a copy of the file will be copied to the external folder.
        
        Arguments
        ---------
        path : str
            path to an external folder.
        """
        in_memory = self.in_memory()
        if not in_memory: self.read()
        filename = os.path.basename(self.file)
        destination = os.path.join(path, filename)
        self._save_ds(destination)
        if not in_memory: self.clear()

    def _initialize(self, ref_ds=None):
        """Initialize the attributes relevant for the Images"""

        self.ds = utilities._initialize(self.ds, UID=self.UID, ref=ref_ds)
        return

        # overwrite UIDs
        self.ds.PatientID = self.UID[0]
        self.ds.StudyInstanceUID = self.UID[1]
        self.ds.SeriesInstanceUID = self.UID[2]
        self.ds.SOPInstanceUID = self.UID[3]

        # Date and Time of Creation
        dt = datetime.now()
        timeStr = dt.strftime('%H%M%S')  # long format with micro seconds

        self.ds.ContentDate = dt.strftime('%Y%m%d')
        self.ds.ContentTime = timeStr
        self.ds.AcquisitionDate = dt.strftime('%Y%m%d')
        self.ds.AcquisitionTime = timeStr
        self.ds.SeriesDate = dt.strftime('%Y%m%d')
        self.ds.SeriesTime = timeStr
        self.ds.InstanceCreationDate = dt.strftime('%Y%m%d')
        self.ds.InstanceCreationTime = timeStr

        if ref_ds is None: return

        # Series, Instance and Class for Reference
        refd_instance = Dataset()
        refd_instance.ReferencedSOPClassUID = ref_ds.SOPClassUID
        refd_instance.ReferencedSOPInstanceUID = ref_ds.SOPInstanceUID
        refd_instance_sequence = Sequence()
        refd_instance_sequence.append(refd_instance)

        refd_series = Dataset()
        refd_series.ReferencedInstanceSequence = refd_instance_sequence
        refd_series.SeriesInstanceUID = self.ds.SeriesInstanceUID
        refd_series_sequence = Sequence()
        refd_series_sequence.append(refd_series)

        self.ds.ReferencedSeriesSequence = refd_series_sequence