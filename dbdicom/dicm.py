from .classes.database import Database
from .classes.patient import Patient
from .classes.study import Study
from .classes.series import Series
from .classes.instance import Instance
from .classes.MRimage import MRImage
from .classes.enhancedMRimage import EnhancedMRImage
from .classes.secondary_capture_image import SecondaryCaptureImage


def object(folder, row=None, generation=4):
    """Creates an instance of a dicm object from a row in the dataframe"""

    if generation == 0: 
        return Database(folder, UID=[])

    key = folder._columns[0:generation]
    UID = row[key].values.tolist()

    if generation == 1: 
        return Patient(folder, UID=UID)
    if generation == 2: 
        return Study(folder, UID=UID)
    if generation == 3: 
        return Series(folder, UID=UID)
    if generation == 4: 
        if row.SOPClassUID == '1.2.840.10008.5.1.4.1.1.4': 
            return MRImage(folder, UID=UID)
        if row.SOPClassUID == '1.2.840.10008.5.1.4.1.1.4.1': 
            return EnhancedMRImage(folder, UID=UID)
        if row.SOPClassUID == '1.2.840.10008.5.1.4.1.1.7': 
            return SecondaryCaptureImage(folder, UID=UID)
        else: 
            return Instance(folder, UID=UID)

def new_child(obj):
    """Creates a new child object for a DICOM object"""

    if obj.generation == 0:
        return Patient(obj.folder, UID=obj.UID)
    if obj.generation == 1:
        return Study(obj.folder, UID=obj.UID)
    if obj.generation == 2:
        return Series(obj.folder, UID=obj.UID)
    if obj.generation == 3: 
        if obj._SOPClassUID == '1.2.840.10008.5.1.4.1.1.4': 
            return MRImage(obj.folder, UID=obj.UID)
        if obj._SOPClassUID == '1.2.840.10008.5.1.4.1.1.4.1': 
            return EnhancedMRImage(obj.folder, UID=obj.UID)
        if obj._SOPClassUID == '1.2.840.10008.5.1.4.1.1.7': 
            return SecondaryCaptureImage(obj.folder, UID=obj.UID)
        else:
            return Instance(obj.folder, UID=obj.UID)
    if obj.generation == 4: 
        return None

def parent(obj):
    "Returns the parent object"

    if obj.generation == 0: 
        return None
    if obj.generation == 1:
        return Database(obj.folder, UID=obj.UID[:-1])
    if obj.generation == 2:
        return Patient(obj.folder, UID=obj.UID[:-1])
    if obj.generation == 3:
        return Study(obj.folder, UID=obj.UID[:-1])
    if obj.generation == 4:
        return Series(obj.folder, UID=obj.UID[:-1])