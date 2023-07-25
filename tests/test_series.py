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
        'SliceLocation': 10*np.arange(8),
        'FlipAngle': [2, 15, 30],
        'RepetitionTime': [2.5, 5.0],
    }
    zeros = db.zeros((128,128,8,3,2), coords)

    dims = ('SliceLocation', 'FlipAngle', 'RepetitionTime')
    array = zeros.ndarray(dims)
    assert array.shape == (128, 128, 8, 3, 2)

    coords = {
        'SliceLocation': 10*np.arange(8),
        'FlipAngle': [15],
        'RepetitionTime': [2.5, 5.0],
    }
    array = zeros.ndarray(coords=coords)
    assert array.shape == (128, 128, 8, 1, 2)

    inds = {
        'SliceLocation': np.arange(8),
        'FlipAngle': [1],
        'RepetitionTime': np.arange(2),
    }
    array = zeros.ndarray(inds=inds)
    assert array.shape == (128, 128, 8, 1, 2)


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

    ones = np.ones(shape)
    series.set_ndarray(ones, coords)

    #Retrieve the array and check that it is now populated with ones:

    array = series.ndarray(dims=tuple(coords)) 
    assert array.shape == shape
    assert np.mean(array) == 1.0

    # Now set the pixels with flip angle 15 to zero:

    zeros = np.zeros((128,128,8,1,2))
    coords['FlipAngle'] = [15]
    series.set_ndarray(zeros, coords)

    # Extract the complete array again and check the values:

    array = series.ndarray(tuple(coords))
    assert array.shape == shape
    assert np.mean(array[:,:,:,0,:]) == 1.0
    assert np.mean(array[:,:,:,1,:]) == 0.0
    assert np.mean(array[:,:,:,2,:]) == 1.0

    # Set the FA=15 subarray to 1 by index:

    ones = np.ones((128,128,8,1,2))
    inds = {
        'SliceLocation': np.arange(8),
        'FlipAngle': [1],
        'RepetitionTime': np.arange(2),
    }
    series.set_ndarray(ones, inds=inds)
    array = series.ndarray(tuple(inds))
    assert np.mean(array[:,:,:,0,:]) == 1.0
    assert np.mean(array[:,:,:,1,:]) == 1.0
    assert np.mean(array[:,:,:,2,:]) == 1.0    


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


def test_slice_groups():
    shape = (128,128,5,10)
    series = db.ones(shape)
    dims = ('SliceLocation', 'AcquisitionTime')
    sgroups = series.slice_groups(dims)
    assert len(sgroups) == 1
    assert sgroups[0]['ndarray'].shape == shape
    assert np.array_equal(sgroups[0]['affine'], np.eye(4))

def test_slice():

    # Create a zero-filled array, describing 8 MRI images each measured at 3 flip angles and 2 repetition times.
    coords = {
        'SliceLocation': np.arange(8),
        'FlipAngle': [2, 15, 30],
        'RepetitionTime': [2.5, 5.0],
    }
    series = db.zeros((128,128,8,3,2), coords)

    # Slice the series at flip angle 15:
    coords['FlipAngle'] = [15]
    fa15 = series.slice(coords=coords, SeriesDescription='FA15')

    # Retrieve the array and check the dimensions & properties:
    array = fa15.ndarray(dims=tuple(coords))
    assert array.shape == (128, 128, 8, 1, 2)
    assert fa15.SeriesDescription == 'FA15'
    assert fa15.FlipAngle == 15
    assert fa15.RepetitionTime == [2.5, 5.0]

    # Get the same slice but this time use inds to slice:
    inds = {
        'SliceLocation': np.arange(8),
        'FlipAngle': 1 + np.arange(1),
        'RepetitionTime': np.arange(2),
    }
    fa15 = series.slice(inds=inds, SeriesDescription='FA15')

    # Retrieve the array and check the dimensions & properties:
    array = fa15.ndarray(dims=tuple(coords))
    assert array.shape == (128, 128, 8, 1, 2)
    assert fa15.SeriesDescription == 'FA15'
    assert fa15.FlipAngle == 15
    assert fa15.RepetitionTime == [2.5, 5.0]


if __name__ == "__main__":

    test_ndarray()
    test_set_ndarray()
    test_affine()
    test_set_affine()
    test_subseries()
    test_split_by()
    test_slice_groups()
    test_slice()

    print('------------------------')
    print('series passed all tests!')
    print('------------------------')