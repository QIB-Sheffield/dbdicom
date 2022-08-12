import os
import zipfile


def _unzip_files(path, status):
    """
    Unzip any zipped files in a directory.
    
    Checking for zipped files is slow so this only searches the top folder.

    Returns : a list with unzipped files
    """
    files = [entry.path for entry in os.scandir(path) if entry.is_file()]
    zipfiles = []
    for i, file in enumerate(files):
        status.progress(i, len(files), message='Searching for zipped folders..')
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


def scan_tree(directory):
    """Helper function: yield DirEntry objects for the directory."""

    for entry in os.scandir(directory):
        if entry.is_dir(follow_symlinks=False):
            yield from scan_tree(entry.path)
        else:
            yield entry