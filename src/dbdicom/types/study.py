import numpy as np
from dbdicom.record import DbRecord

class Study(DbRecord):

    def zeros(*args, **kwargs):
        return zeros(*args, **kwargs)

def zeros(study, shape, dtype='mri'):
    series = study.new_series()
    array = np.zeros(shape, dtype=np.float32)
    if dtype not in ['mri', 'MRImage']:
        message = 'dbdicom can only create images of type MRImage at this stage'
        raise ValueError(message)
    series.set_pixel_array(array)
    return series
