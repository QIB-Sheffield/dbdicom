import os
import sys
import pathlib
import subprocess
import shutil
import pydicom

import sys

if sys.version_info < (3, 9):
    # importlib.resources either doesn't exist or lacks the files()
    # function, so use the PyPI version:
    import importlib_resources
else:
    # importlib.resources has files(), so use that:
    import importlib.resources as importlib_resources


def findfile(script):
    """Helper function: Find the program for a script"""

    if os.name =='nt': 
        script += '.bat'
    f = importlib_resources.files('dbdicom.external.dcm4che.bin')
    f = str(f.joinpath(script))
    return f


def split_multiframe(filepath):
    """Splits a multi-frame instance into single frames"""
    
    description = os.path.basename(filepath)
    multiframeDir = os.path.dirname(filepath)
    outputDir = os.path.join(multiframeDir, description + '_sf') 
    os.mkdir(outputDir)

    fileBase = 'single_frame_'
    fileBaseFlag = fileBase + "000000"
    command = [findfile('emf2sf'), "--inst-no", "'%s'", "--not-chseries", "--out-dir", outputDir, "--out-file", fileBaseFlag, filepath]
    try:
        subprocess.call(command, stdout=subprocess.PIPE)
    except:
        return []

    # Return a list of newly created files
    # Slice Locations need to be copied from a private field 
    # !!! Keep an eye out for other fields that need adapting !!!

    new_files = [os.path.join(outputDir, f) for f in os.listdir(outputDir) if os.path.isfile(os.path.join(outputDir, f))]
    output_files = [f + '.dcm' for f in new_files]

    for i, file in enumerate(new_files):
        ds = pydicom.dcmread(file, force=True)
        if (0x2001,0x100a) in ds:
            ds.SliceLocation = ds[0x2001,0x100a].value
        ds.save_as(output_files[i])
        os.remove(file)

    if output_files == []:
        os.rmdir(outputDir)

    return output_files





