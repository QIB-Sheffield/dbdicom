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

    def _initialize(self, ref_ds=None): # check if still in use - replace by utilities
        """Initialize the attributes relevant for the Images"""

        if self.__class__.__name__ == 'Record':
            ds = self.read()
        ds_pydicom = utilities._initialize(ds.to_pydicom(), UID=self.UID, ref=ref_ds)
        ds._ds = ds_pydicom