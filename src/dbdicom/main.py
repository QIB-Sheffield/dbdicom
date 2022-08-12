__all__ = ['open']

from dbdicom.dbrecord import DbRecord
from dbdicom.dbindex import DbIndex
from dbdicom.message import StatusBar, Dialog

def open(path, status=StatusBar(), dialog=Dialog(), attributes=None, message='Opening database..'):

    folder = DbIndex(path, status=status, dialog=dialog, attributes=attributes)
    folder.open(message=message)
    return DbRecord(folder, UID=[], generation=0) 