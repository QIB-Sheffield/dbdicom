import os
import shutil
import timeit
import numpy as np
import dbdicom as db
from dbdicom.wrappers.numpy import maximum_intensity_projection, mean_intensity_projection, norm_projection



datapath = os.path.join(os.path.dirname(__file__), 'data')
ct = os.path.join(datapath, '2_skull_ct')
twofiles = os.path.join(datapath, 'TWOFILES')
onefile = os.path.join(datapath, 'ONEFILE')
rider = os.path.join(datapath, 'RIDER')
zipped = os.path.join(datapath, 'ZIP')
multiframe = os.path.join(datapath, 'MULTIFRAME')
iBEAt = os.path.join(datapath, 'Leeds_iBEAt')

# Helper functions

def create_tmp_database(path=None, name='tmp'):
    tmp = os.path.join(os.path.dirname(__file__), name)
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    if path is not None:
        shutil.copytree(path, tmp)
    else:
        os.makedirs(tmp)
    return tmp


def remove_tmp_database(tmp):
    shutil.rmtree(tmp)


def test_maximum_intensity_projection():
    # Test taken from docstring

    # Create a zero-filled array, describing 8 MRI images each measured at 3 flip angles and 2 repetition times:

    coords = {
        'SliceLocation': np.arange(8),
        'FlipAngle': [2, 15, 30],
        'RepetitionTime': [2.5, 5.0],
    }
    series = db.zeros((128,128,8,3,2), coords)

    #Create a maximum intensity projection on the slice locations and check the dimensions:

    mip = maximum_intensity_projection(series)
    array = mip.ndarray(dims=('SliceLocation', 'ImageNumber'))
    assert array.shape == (128, 128, 8, 1)

    # Create a maximum intensity projection along the Slice Location axis:

    mip = maximum_intensity_projection(series, dims=tuple(coords), axis=0)
    array = mip.ndarray(dims=tuple(coords))
    assert array.shape == (128, 128, 1, 3, 2)

    # Create a maximum intensity projection along the Flip Angle axis:

    mip = maximum_intensity_projection(series, dims=tuple(coords), axis=1)
    array = mip.ndarray(dims=tuple(coords))
    assert array.shape == (128, 128, 8, 1, 2)

    # Create a maximum intensity projection along the Repetition Time axis:

    mip = maximum_intensity_projection(series, dims=tuple(coords), axis=2)
    array = mip.ndarray(dims=tuple(coords))
    assert array.shape == (128, 128, 8, 3, 1)


def test_mean_intensity_projection():
    # Test taken from docstring

    # Create a zero-filled array, describing 8 MRI images each measured at 3 flip angles and 2 repetition times:

    coords = {
        'SliceLocation': np.arange(8),
        'FlipAngle': [2, 15, 30],
        'RepetitionTime': [2.5, 5.0],
    }
    series = db.zeros((128,128,8,3,2), coords)

    #Create a mean intensity projection on the slice locations and check the dimensions:

    mip = mean_intensity_projection(series)
    array = mip.ndarray(dims=('SliceLocation', 'ImageNumber'))
    assert array.shape == (128, 128, 8, 1)

    # Create a mean intensity projection along the Slice Location axis:

    mip = mean_intensity_projection(series, dims=tuple(coords), axis=0)
    array = mip.ndarray(dims=tuple(coords))
    assert array.shape == (128, 128, 1, 3, 2)

    # Create a mean intensity projection along the Flip Angle axis:

    mip = mean_intensity_projection(series, dims=tuple(coords), axis=1)
    array = mip.ndarray(dims=tuple(coords))
    assert array.shape == (128, 128, 8, 1, 2)

    # Create a mean intensity projection along the Repetition Time axis:

    mip = mean_intensity_projection(series, dims=tuple(coords), axis=2)
    array = mip.ndarray(dims=tuple(coords))
    assert array.shape == (128, 128, 8, 3, 1)


def test_norm_projection():
    # Test taken from docstring

    # Create a zero-filled array, describing 8 MRI images each measured at 3 flip angles and 2 repetition times:

    coords = {
        'SliceLocation': np.arange(8),
        'FlipAngle': [2, 15, 30],
        'RepetitionTime': [2.5, 5.0],
    }
    series = db.zeros((128,128,8,3,2), coords)

    #Create a norm projection on the slice locations and check the dimensions:

    mip = norm_projection(series)
    array = mip.ndarray(dims=('SliceLocation', 'ImageNumber'))
    assert array.shape == (128, 128, 8, 1)

    # Create a norm projection along the Slice Location axis:

    mip = norm_projection(series, dims=tuple(coords), axis=0)
    array = mip.ndarray(dims=tuple(coords))
    assert array.shape == (128, 128, 1, 3, 2)

    # Create a norm projection along the Flip Angle axis:

    mip = norm_projection(series, dims=tuple(coords), axis=1)
    array = mip.ndarray(dims=tuple(coords))
    assert array.shape == (128, 128, 8, 1, 2)

    # Create a norm projection along the Repetition Time axis:

    mip = norm_projection(series, dims=tuple(coords), axis=2)
    array = mip.ndarray(dims=tuple(coords))
    assert array.shape == (128, 128, 8, 3, 1)



if __name__ == "__main__":

    test_maximum_intensity_projection()
    test_mean_intensity_projection()
    test_norm_projection()

    print('--------------------------------')
    print('wrappers.numpy passed all tests!')
    print('--------------------------------')

