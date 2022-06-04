import os
import sys
import zipfile
from datetime import datetime
import math
import pathlib
import subprocess
import pydicom
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
import pandas as pd
import numpy as np

def dataframe(path, files, tags, status=None, message='Reading DICOM folder..'):
    """Reads a list of tags in a list of files.

    Arguments
    ---------
    files : str or list
        A filepath or a list of filepaths
    tags : str or list 
        A DICOM tag or a list of DICOM tags
    status : StatusBar

    Creates
    -------
    dataframe : pandas.DataFrame
        A Pandas dataframe with one row per file
        The index is the file path 
        Each column corresponds to a Tag in the list of Tags
        The returned dataframe is sorted by the given tags.
    """
    if not isinstance(files, list):
        files = [files]
    if not isinstance(tags, list):
        tags = [tags]
    array = []
    dicom_files = []
    if status is not None: status.message(message)
    for i, file in enumerate(files):
        ds = pydicom.dcmread(file, force=True)
        if isinstance(ds, pydicom.dataset.FileDataset):
            if 'TransferSyntaxUID' in ds.file_meta:
                row = _read_tags(ds, tags)
                array.append(row)
                relpath = os.path.relpath(file, path)
                dicom_files.append(relpath) 
        if status is not None: status.progress(i+1, len(files), message)
    if status is not None: status.hide()
    return pd.DataFrame(array, index = dicom_files, columns = tags)

def _initialize(ds, UID=None, ref=None):

    # Date and Time of Creation
    dt = datetime.now()
    timeStr = dt.strftime('%H%M%S')  # long format with micro seconds

    ds.ContentDate = dt.strftime('%Y%m%d')
    ds.ContentTime = timeStr
    ds.AcquisitionDate = dt.strftime('%Y%m%d')
    ds.AcquisitionTime = timeStr
    ds.SeriesDate = dt.strftime('%Y%m%d')
    ds.SeriesTime = timeStr
    ds.InstanceCreationDate = dt.strftime('%Y%m%d')
    ds.InstanceCreationTime = timeStr

    if UID is not None:

        # overwrite UIDs
        ds.PatientID = UID[0]
        ds.StudyInstanceUID = UID[1]
        ds.SeriesInstanceUID = UID[2]
        ds.SOPInstanceUID = UID[3]

    if ref is not None: 

        # Series, Instance and Class for Reference
        refd_instance = Dataset()
        refd_instance.ReferencedSOPClassUID = ref.SOPClassUID
        refd_instance.ReferencedSOPInstanceUID = ref.SOPInstanceUID
        refd_instance_sequence = Sequence()
        refd_instance_sequence.append(refd_instance)

        refd_series = Dataset()
        refd_series.ReferencedInstanceSequence = refd_instance_sequence
        refd_series.SeriesInstanceUID = ds.SeriesInstanceUID
        refd_series_sequence = Sequence()
        refd_series_sequence.append(refd_series)

        ds.ReferencedSeriesSequence = refd_series_sequence

    return ds

def _unzip_files(path, status):
    """
    Unzip any zipped files in a directory.
    
    Checking for zipped files is slow so this only searches the top folder.

    Returns : a list with unzipped files
    """
    files = [entry.path for entry in os.scandir(path) if entry.is_file()]
    zipfiles = []
    for i, file in enumerate(files):
        status.progress(i, len(files), 'Searching for zipped folders..')
        if zipfile.is_zipfile(file):
            zipfiles.append(file)
    if zipfiles == []:
        return
    for i, file in enumerate(zipfiles): # unzip any zip files and delete the original zip
        status.progress(i, len(zipfiles), 'Unzipping file ' + file)
        with zipfile.ZipFile(file, 'r') as zip_folder:
            path = ''.join(file.split('.')[:-1]) # remove file extension
            if not os.path.isdir(path): 
                os.mkdir(path)
            zip_folder.extractall(path)
        os.remove(file)

def _read_tags(ds, tags):
    """Helper function return a list of values"""

    # https://pydicom.github.io/pydicom/stable/guides/element_value_types.html
    if not isinstance(tags, list): 
        if tags not in ds:
            return None
        else:
        #    return ds[tags].value
            return _convert_attribute_type(ds[tags].value)
            
    row = []  
    for tag in tags:
        if tag not in ds:
            value = None
        else:
        #    value = ds[tag].value
            value = _convert_attribute_type(ds[tag].value)
        row.append(value)
    return row

def _convert_attribute_type(value):
    """Convert pyidcom datatypes to the python datatypes used to set the parameter.
    
    While this removes some functionality, this aligns with the principle 
    of `dbdicom` to remove DICOM-native langauge from the API.
    """

    if value.__class__.__name__ == 'PersonName':
        return str(value)
    if value.__class__.__name__ == 'Sequence':
        return [ds for ds in value]
    if value.__class__.__name__ == 'TM': # This can probably do with some formatting
        return str(value) 
    if value.__class__.__name__ == 'UID': 
        return str(value) 
    if value.__class__.__name__ == 'IS': 
        return int(value)
    if value.__class__.__name__ == 'DT': 
        return str(value)
    if value.__class__.__name__ == 'DA': 
        return str(value)
    if value.__class__.__name__ == 'DSfloat': 
        return float(value)
    if value.__class__.__name__ == 'DSdecimal': 
        return int(value)
    else:
        return value

def _set_tags(ds, tags, values):
    """Sets DICOM tags in the dataset in memory"""

    if not isinstance(tags, list): 
        tags = [tags]
        values = [values]
    for i, tag in enumerate(tags):
        if tag in ds:
            ds[tag].value = values[i]
        else:
            if not isinstance(tag, pydicom.tag.BaseTag):
                tag = pydicom.tag.Tag(tag)
            if not tag.is_private: # Add a new data element
                VR = pydicom.datadict.dictionary_VR(tag)
                ds.add_new(tag, VR, values[i])
            else:
                pass # for now
    return ds

def _filter(objects, **kwargs):
    """
    Filters a list of DICOM classes by DICOM tags and values.
    
    Example
    -------
    instances = _filter(instances, PatientName="Harry")
    """

    filtered = []
    for obj in objects:
        select = True
        for tag, value in kwargs.items():
            if getattr(obj, tag) != value:
                select = False
                break
        if select: 
            filtered.append(obj)
    return filtered



def split_multiframe(filepath, description):
    """Splits a multi-frame instance into single frames"""

    multiframeDir = os.path.dirname(filepath)
    fileBase = "SingleFrame_"
    fileBaseFlag = fileBase + "000000_" + description.replace('.', '_')
    command = [program('emf2sf'), "--inst-no", "'%s'", "--not-chseries", "--out-dir", multiframeDir, "--out-file", fileBaseFlag, filepath]
    try:
        fail = subprocess.call(command, stdout=subprocess.PIPE)
    except Exception as e:
        fail = 1
        print('Error in dcm4che: Could not split the detected Multi-frame DICOM file.\n'\
                'The DICOM file ' + filepath + ' was not deleted.')

    # Return a list of newly created files
    multiframe_files_list = []
    if fail == 0:
        for new_file in os.listdir(multiframeDir):
            if new_file.startswith(fileBase):
                new_file_path = os.path.join(multiframeDir, new_file)
                multiframe_files_list.append(new_file_path)     
                # Slice Locations need to be copied from a private field 
                ds = pydicom.dcmread(new_file_path, force=True)
                ds.SliceLocation = ds[0x2001,0x100a].value
                ds.save_as(new_file_path)
    return multiframe_files_list

def program(script):
    """Helper function: Find the program for a script"""

    if os.name =='nt': 
        script += '.bat'
    program = script
    # If running Weasel as executable
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        directory = pathlib.Path(sys._MEIPASS)
    # If running Weasel as normal Python script
    else:
        directory = pathlib.Path().absolute()
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(script):
                sys.path.append(dirpath)
                program = os.path.join(dirpath, filename)
    return program

def scan_tree(directory):
    """Helper function: yield DirEntry objects for the directory."""

    for entry in os.scandir(directory):
        if entry.is_dir(follow_symlinks=False):
            yield from scan_tree(entry.path)
        else:
            yield entry

def _stack_arrays(arrays, align_left=False):
    """Stack a list of arrays of different shapes but same number of dimensions.
    
    This generalises numpy.stack to arrays of different sizes.
    The stack has the size of the largest array.
    If an array is smaller it is zero-padded and centred on the middle.
    """

    # Get the dimensions of the stack
    # For each dimension, look for the largest values across all arrays
    ndim = len(arrays[0].shape)
    dim = [0] * ndim
    for array in arrays:
        for i, d in enumerate(dim):
            dim[i] = max((d, array.shape[i])) # changing the variable we are iterating over!!
    #    for i in range(ndim):
    #        dim[i] = max((dim[i], array.shape[i]))

    # Create the stack
    # Add one dimension corresponding to the size of the stack
    n = len(arrays)
    #stack = np.full([n] + dim, 0, dtype=arrays[0].dtype)
    stack = np.full([n] + dim, None, dtype=arrays[0].dtype)

    for k, array in enumerate(arrays):
        index = [k]
        for i, d in enumerate(dim):
            if align_left:
                i0 = 0
            else: # align center and zero-pad missing values
                i0 = math.floor((d-array.shape[i])/2)
            i1 = i0 + array.shape[i]
            index.append(slice(i0,i1))
        stack[tuple(index)] = array

    return stack
