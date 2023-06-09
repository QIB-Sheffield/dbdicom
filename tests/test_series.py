import os
import shutil
import timeit
import numpy as np
import dbdicom as db



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


def test_ndarray():
    # Test taken from docstring
    coords = {
        'SliceLocation': np.arange(8),
        'FlipAngle': [2, 15, 30],
        'RepetitionTime': [2.5, 5.0],
    }
    zeros = db.zeros((128,128,8,3,2), coords)

    dims = ('SliceLocation', 'FlipAngle', 'RepetitionTime')
    array = zeros.ndarray(dims)
    assert array.shape == (128, 128, 8, 3, 2)


def test_set_ndarray():
    # Test taken from docstring

    # Create a zero-filled array, describing 8 MRI slices each measured at 3 flip angles and 2 repetition times:
    coords = {
        'SliceLocation': np.arange(8),
        'FlipAngle': [2, 15, 30],
        'RepetitionTime': [2.5, 5.0],
    }
    shape = (128,128,8,3,2)
    series = db.zeros(shape, coords)

    # Retrieve the array and check that it is populated with zeros:

    array = series.ndarray(tuple(coords))
    assert array.shape == shape
    assert np.mean(array) == 0.0

    # Now overwrite the values with a new array of ones. 
    # Coordinates are not changed so only dimensions need to be specified.

    ones = np.ones(shape)
    series.set_ndarray(ones, dims=tuple(coords))

    #Retrieve the array and check that it is now populated with ones:

    array = series.ndarray(dims=tuple(coords)) 
    assert array.shape == shape
    assert np.mean(array) == 1.0

    # Now set a new array with a just slice location (e.g. T1 map derived from data)

    new_shape = (128,128,8)
    new_coords = {
        'SliceLocation': np.arange(8),
    }
    zeros = np.zeros(new_shape)
    series.set_ndarray(zeros, coords=new_coords)

    # Retrieve the new array and check shape and values
    array = series.ndarray(dims=tuple(new_coords))
    assert array.shape == new_shape
    assert np.mean(array) == 0.0

    # Now set a new array with a completelt different shape

    new_shape = (64,64,3,2)
    new_coords = {
        'SliceLocation': np.arange(3),
        'AcquisitionTime': np.arange(2),
    }
    ones = np.ones(new_shape)
    series.set_ndarray(ones, coords=new_coords)

    # Retrieve the new array and check shape and values
    array = series.ndarray(dims=tuple(new_coords))
    assert array.shape == new_shape
    assert np.mean(array) == 1.0


def test_affine():
    zeros = db.zeros((18,128,10))
    affine = zeros.affine()
    assert np.array_equal(affine[0], np.eye(4))


def test_set_affine():

    zeros = db.zeros((18,128,10))
    affine = zeros.affine()
    assert np.array_equal(affine[0], np.eye(4))

    # Rotate the volume over 90 degrees in the xy-plane:

    affine = np.array([
        [0., -1., 0., 0.],
        [1., 0., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
    ])  
    zeros.set_affine(affine)
    affine_new = zeros.affine()
    assert np.array_equal(affine_new[0], affine) 

    # Apart from the rotation, also change the resolution to (3mm, 3mm, 1.5mm)

    affine = np.array([
        [0., -3., 0., 0.],
        [3., 0., 0., 0.],
        [0., 0., 1.5, 0.],
        [0., 0., 0., 1.],
    ])  
    zeros.set_affine(affine)
    affine_new = zeros.affine()
    assert np.array_equal(affine_new[0], affine)

    # Now rotate, change resolution, and shift the top right hand corner of the lowest slice to position (-30mm, 20mm, 120mm)

    affine = np.array([
        [0., -3., 0., -30.],
        [3., 0., 0., 20.],
        [0., 0., 1.5, 120.],
        [0., 0., 0., 1.0],
    ])  
    zeros.set_affine(affine)
    affine_new = zeros.affine()
    assert np.array_equal(affine_new[0], affine) 
    assert zeros.SliceLocation == [120.0, 121.5, 123.0, 124.5, 126.0, 127.5, 129.0, 130.5, 132.0, 133.5]

def test_subseries():
    coords = {
        'SliceLocation': np.arange(16),
        'FlipAngle': [2, 15, 30],
        'RepetitionTime': [2.5, 7.5],
    }
    zeros = db.zeros((128, 128, 16, 3, 2), coords)
    array = zeros.ndarray(dims=tuple(coords))
    assert array.shape == (128, 128, 16, 3, 2)
    assert zeros.FlipAngle == [2, 15, 30]
    assert zeros.RepetitionTime == [2.5, 7.5]

    volume = zeros.subseries(FlipAngle=2.0, RepetitionTime=7.5)
    array = volume.ndarray(dims=tuple(coords))
    assert array.shape == (128, 128, 16, 1, 1)
    assert (2.0, 7.5) == (volume.FlipAngle, volume.RepetitionTime)
    assert len(volume.study().children()) == 2

def test_split_by():

    coords = {
        'FlipAngle': [2, 15, 30],
        'RepetitionTime': [2.5, 7.5],
    }
    zeros = db.zeros((128, 128, 3, 2), coords)
    zeros.print()

    split = zeros.split_by('FlipAngle')
    zeros.study().print()

    for i in range(3): 
        assert split[i].FlipAngle == coords['FlipAngle'][i]





if __name__ == "__main__":

    test_ndarray()
    test_set_ndarray()
    test_affine()
    test_set_affine()
    test_subseries()
    test_split_by()

    print('------------------------')
    print('series passed all tests!')
    print('------------------------')