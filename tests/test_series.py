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


def test_pixel_values():
    # Test taken from docstring
    coords = {
        'SliceLocation': 10*np.arange(8),
        'FlipAngle': [2, 15, 30],
        'RepetitionTime': [2.5, 5.0],
    }
    zeros = db.zeros((128,128,8,3,2), coords)

    dims = ('SliceLocation', 'FlipAngle', 'RepetitionTime')
    array = zeros.pixel_values(dims)
    assert array.shape == (128, 128, 8, 3, 2)

    coords = {
        'SliceLocation': 10*np.arange(8),
        'FlipAngle': [15],
        'RepetitionTime': [2.5, 5.0],
    }
    array = zeros.pixel_values(coords=coords)
    assert array.shape == (128, 128, 8, 1, 2)

    inds = {
        'SliceLocation': np.arange(8),
        'FlipAngle': [1],
        'RepetitionTime': np.arange(2),
    }
    array = zeros.pixel_values(inds=inds)
    assert array.shape == (128, 128, 8, 1, 2)


def test_set_pixel_values():
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

    array = series.pixel_values(tuple(coords))
    assert array.shape == shape
    assert np.mean(array) == 0.0

    # Now overwrite the values with a new array of ones. 

    ones = np.ones(shape)
    series.set_pixel_values(ones, coords)

    #Retrieve the array and check that it is now populated with ones:

    array = series.pixel_values(dims=tuple(coords)) 
    assert array.shape == shape
    assert np.mean(array) == 1.0

    # Now set the pixels with flip angle 15 to zero:

    zeros = np.zeros((128,128,8,1,2))
    coords['FlipAngle'] = [15]
    series.set_pixel_values(zeros, coords)

    # Extract the complete array again and check the values:

    array = series.pixel_values(tuple(coords))
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
    series.set_pixel_values(ones, inds=inds)
    array = series.pixel_values(tuple(inds))
    assert np.mean(array[:,:,:,0,:]) == 1.0
    assert np.mean(array[:,:,:,1,:]) == 1.0
    assert np.mean(array[:,:,:,2,:]) == 1.0    

def test_unique_affines():
    zeros = db.zeros((18,128,10))
    affine = zeros.unique_affines()
    assert np.array_equal(affine[0], np.eye(4))

def test_affine():
    zeros = db.zeros((18,128,10))
    affine = zeros.affine()
    assert np.array_equal(affine, np.eye(4))

def test_set_affine():

    zeros = db.zeros((18,128,10))
    affine = zeros.affine()
    assert np.array_equal(affine, np.eye(4))

    # Rotate the volume over 90 degrees in the xy-plane:

    affine = np.array([
        [0., -1., 0., 0.],
        [1., 0., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
    ])  
    zeros.set_affine(affine)
    affine_new = zeros.affine()
    assert np.array_equal(affine_new, affine) 

    # Apart from the rotation, also change the resolution to (3mm, 3mm, 1.5mm)

    affine = np.array([
        [0., -3., 0., 0.],
        [3., 0., 0., 0.],
        [0., 0., 1.5, 0.],
        [0., 0., 0., 1.],
    ])  
    zeros.set_affine(affine)
    affine_new = zeros.affine()
    assert np.array_equal(affine_new, affine)

    # Now rotate, change resolution, and shift the top right hand corner of the lowest slice to position (-30mm, 20mm, 120mm)

    affine = np.array([
        [0., -3., 0., -30.],
        [3., 0., 0., 20.],
        [0., 0., 1.5, 120.],
        [0., 0., 0., 1.0],
    ])  
    zeros.set_affine(affine)
    affine_new = zeros.affine()
    assert np.array_equal(affine_new, affine) 
    assert zeros.SliceLocation == [120.0, 121.5, 123.0, 124.5, 126.0, 127.5, 129.0, 130.5, 132.0, 133.5]

def test_subseries():
    coords = {
        'SliceLocation': np.arange(16),
        'FlipAngle': [2, 15, 30],
        'RepetitionTime': [2.5, 7.5],
    }
    zeros = db.zeros((128, 128, 16, 3, 2), coords)
    array = zeros.pixel_values(dims=tuple(coords))
    assert array.shape == (128, 128, 16, 3, 2)
    assert zeros.FlipAngle == [2, 15, 30]
    assert zeros.RepetitionTime == [2.5, 7.5]

    volume = zeros.subseries(FlipAngle=2.0, RepetitionTime=7.5)
    array = volume.pixel_values(dims=tuple(coords))
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

    print('Testing slice')

    # Create a zero-filled array, describing 8 MRI images each measured at 3 flip angles and 2 repetition times.
    coords = {
        'SliceLocation': np.arange(8),
        'FlipAngle': [2, 15, 30],
        'RepetitionTime': [2.5, 5.0],
    }
    series = db.zeros((128,128,8,3,2), coords)
    dims = tuple(coords)

    # Slice the series at flip angle 15:
    coords['FlipAngle'] = [15]
    fa15 = series.slice(**coords)
    fa15.SeriesDescription='FA15'

    # Retrieve the array and check the dimensions & properties:
    array = fa15.pixel_values(dims=tuple(coords))
    assert array.shape == (128, 128, 8, 1, 2)
    assert fa15.SeriesDescription == 'FA15'
    assert fa15.FlipAngle == 15
    assert fa15.RepetitionTime == [2.5, 5.0]

    # Slice with a list instead:
    vals = {
        'SliceLocation': np.arange(8),
        'FlipAngle': [15],
        'RepetitionTime': [2.5, 5.0],
    }
    fa15 = series.slice(**vals) 
    assert fa15.pixel_values(dims).shape == (128, 128, 8, 1, 2)

    # Slice with a scalar:
    vals = {
        'SliceLocation': np.arange(8),
        'FlipAngle': 15,
        'RepetitionTime': [2.5, 5.0],
    }
    fa15 = series.slice(**vals) 
    assert fa15.pixel_values(dims).shape == (128, 128, 8, 1, 2)

    # Since the first and second dimensions include all variables, only FA needs to be specified
    vals = {
        'FlipAngle': 15,
    }
    fa15 = series.slice(**vals) 
    assert fa15.pixel_values(dims).shape == (128, 128, 8, 1, 2)

    # Slice using keyword=value notation:
    fa15 = series.slice(FlipAngle=15) 
    assert fa15.pixel_values(dims).shape == (128, 128, 8, 1, 2)

    # Slice using keyword=value notation, providing multiple possible values
    fa15 = series.slice(SliceLocation=[0,5], FlipAngle=15) 
    assert fa15.pixel_values(dims).shape == (128, 128, 2, 1, 2)

    # or specifying all (superfluous but should work):
    fa15 = series.slice(SliceLocation=np.arange(8), FlipAngle=15, RepetitionTime=[2.5, 5.0]) 
    assert fa15.pixel_values(dims).shape == (128, 128, 8, 1, 2)

def test_islice():

    print('Testing islice')

    # Create a zero-filled array, describing 8 MRI images each measured at 3 flip angles and 2 repetition times.
    coords = {
        'SliceLocation': np.arange(8),
        'FlipAngle': [2, 15, 30],
        'RepetitionTime': [2.5, 5.0],
    }
    series = db.zeros((128,128,8,3,2), coords)
    dims = tuple(coords)

    # Slice the series at flip angle 15, using inds to slice:
    inds = {
        'SliceLocation': np.arange(8),
        'FlipAngle': 1 + np.arange(1),
        'RepetitionTime': np.arange(2),
    }
    fa15 = series.islice(**inds)
    fa15.SeriesDescription='FA15'

    # Retrieve the array and check the dimensions & properties:
    array = fa15.pixel_values(dims)
    assert array.shape == (128, 128, 8, 1, 2)
    assert fa15.SeriesDescription == 'FA15'
    assert fa15.FlipAngle == 15
    assert fa15.RepetitionTime == [2.5, 5.0]

    # Slice with a list instead:
    inds = {
        'SliceLocation': np.arange(8),
        'FlipAngle': [1],
        'RepetitionTime': np.arange(2),
    }
    fa15 = series.islice(**inds) 
    assert fa15.pixel_values(dims).shape == (128, 128, 8, 1, 2)

    # Slice with a scalar:
    inds = {
        'SliceLocation': np.arange(8),
        'FlipAngle': 1,
        'RepetitionTime': np.arange(2),
    }
    fa15 = series.islice(**inds) 
    assert fa15.pixel_values(dims).shape == (128, 128, 8, 1, 2)

    # Since the first and second dimensions include all variables, only FA needs to be specified
    inds = {
        'FlipAngle': 1,
    }
    fa15 = series.islice(**inds) 
    assert fa15.pixel_values(dims).shape == (128, 128, 8, 1, 2)

    # Slice using keyword=value notation:
    fa15 = series.islice(FlipAngle=1) 
    assert fa15.pixel_values(dims).shape == (128, 128, 8, 1, 2)

    # or specifying all (superfluous but should work):
    fa15 = series.islice(SliceLocation=np.arange(8), FlipAngle=1, RepetitionTime=np.arange(2)) 
    assert fa15.pixel_values(dims).shape == (128, 128, 8, 1, 2)



def test_coords():

    print('Testing coords')

    # Create a zero-filled array, describing 8 MRI images each measured at 3 flip angles and 2 repetition times.
    coords = {
        'SliceLocation': np.arange(8),
        'FlipAngle': [2, 15, 30],
        'RepetitionTime': [2.5, 5.0],
    }
    series = db.zeros((128,128,8,3,2), coords)
    coords = series.coords(tuple(coords))
    assert coords['SliceLocation'][1,1,1] == 1
    assert coords['FlipAngle'][1,1,1] == 15
    assert coords['RepetitionTime'][1,1,1] == 5


def test_set_coords():

    print('Testing set_coords')

    # Create a zero-filled array:
    gridcoords = {
        'SliceLocation': np.arange(4),
        'FlipAngle': [2, 15, 30],
        'RepetitionTime': [2.5, 5.0],
    }
    series = db.zeros((128,128,4,3,2), gridcoords)
    dims = tuple(gridcoords)

    # Get the coordinates and set them again
    coords = series.coords(dims)
    series.set_coords(coords)
    coords_rec = series.coords(dims)
    for d in dims:
        assert np.array_equal(coords[d], coords_rec[d])

    # Change the flip angle of 15 to 12:
    coords['FlipAngle'][:,1,:] = 12
    series.set_coords(coords)
    new_coords = series.coords(dims)
    fa = new_coords['FlipAngle'][:,1,:]
    assert np.array_equal(np.unique(fa), [12])

    # Set new coordinates
    zloc = np.arange(4)
    tacq = 60*np.arange(6)
    v = np.meshgrid(zloc, tacq, indexing='ij')
    new_coords = {
        'SliceLocation':v[0],
        'AcquisitionTime':v[1],
    }
    series.set_coords(new_coords, dims)
    c = series.coords(tuple(new_coords))
    assert np.array_equal(c['AcquisitionTime'], v[1])


def test_gridcoords():

    print('Testing gridcoords')

    # Create a zero-filled array, describing 8 MRI images each measured at 3 flip angles and 2 repetition times.
    coords = {
        'SliceLocation': np.arange(8),
        'FlipAngle': [2, 15, 30],
        'RepetitionTime': [2.5, 5.0],
    }
    series = db.zeros((128,128,8,3,2), coords)
    coords_recovered = series.gridcoords(tuple(coords))
    for d in coords:
        assert np.array_equal(coords[d], coords_recovered[d])


def test_set_gridcoords():

    print('Testing set_gridcoords')

    # Create a zero-filled array:
    gridcoords = {
        'SliceLocation': np.arange(8),
        'FlipAngle': [2, 15, 30],
        'RepetitionTime': [2.5, 5.0],
    }
    series = db.zeros((128,128,8,3,2), gridcoords)
    dims = tuple(gridcoords)

    # Get the coordinates and set them again
    coords = series.gridcoords(dims)
    series.set_gridcoords(coords)
    coords_rec = series.gridcoords(dims)
    for d in dims:
        assert np.array_equal(coords[d], coords_rec[d])

    # Change the flip angle of 15 to 12:
    coords['FlipAngle'][1] = 12
    series.set_gridcoords(coords)

    # Check coordinates
    new_coords = series.coords(dims)
    fa = new_coords['FlipAngle'][:,1,:]
    assert np.array_equal(np.unique(fa), [12])

    # Check grid coordinates
    new_coords = series.gridcoords(dims)
    fa = new_coords['FlipAngle'][1]
    assert fa == 12

    # Set new coordinates and check
    new_coords = {
        'SliceLocation': np.arange(8),
        'AcquisitionTime': 60*np.arange(6),
    }
    series.set_gridcoords(new_coords, dims)
    c = series.gridcoords(tuple(new_coords))
    assert np.array_equal(c['AcquisitionTime'], 60*np.arange(6))


def test_shape():

    print('Testing shape')

    # Create a zero-filled array with 3 dimensions:
    loc = np.arange(4)
    fa = [2, 15, 30]
    tr = [2.5, 5.0]
    coords = {
        'SliceLocation': loc,
        'FlipAngle': fa,
        'RepetitionTime': tr,
    }
    series = db.zeros((128,128,4,3,2), coords)
    dims = tuple(coords)

    assert series.shape(dims) == (len(loc), len(fa), len(tr))
    assert series.shape(dims[:2]) == (len(loc), len(fa))
    assert series.shape(dims[:1]) == (len(loc),)
    assert series.shape((dims[2], dims[0])) == (len(tr), len(loc))
    assert series.shape(('InstanceNumber',)) == (len(loc)*len(fa)*len(tr),)
    assert series.shape(('Gobbledigook',)) == (1,)


def test_unique():

    print('Testing unique')

    # Create a zero-filled array, describing 8 MRI images each measured at 3 flip angles and 2 repetition times.
    loc = np.arange(4)
    fa = [2, 15, 30]
    tr = [2.5, 5.0]
    coords = {
        'SliceLocation': loc,
        'FlipAngle': fa,
        'RepetitionTime': tr,
    }
    series = db.zeros((128,128,4,3,2), coords)

    # Recover unique values of each coordinate
    assert np.array_equal(series.unique('SliceLocation'), loc)
    assert np.array_equal(series.unique('FlipAngle'), fa)
    assert np.array_equal(series.unique('RepetitionTime'), tr)

    # Get unique Flip Angles for each slice location
    v = series.unique('FlipAngle', dims=('SliceLocation', ))
    assert len(v) == len(loc)
    assert(np.array_equal(v[0], fa))
    assert(np.array_equal(v[-1], fa))

    # Get unique Flip Angles for each slice location and repetition time
    v = series.unique('FlipAngle', dims=('SliceLocation', 'RepetitionTime'))
    assert v.size == len(loc)*len(tr)
    assert(np.array_equal(v[0,0], fa))
    assert(np.array_equal(v[-1,-1], fa))

    # Get unique Flip Angles for each slice location, repetition time and flip angle.
    v = series.unique('FlipAngle', dims=('SliceLocation', 'RepetitionTime', 'FlipAngle'))
    assert v.size == len(loc)*len(tr)*len(fa)
    assert v[0,0,0] == fa[0]
    assert v[0,0,1] == fa[1]
    assert v[-1,-1,1] == fa[1]

    # Get values for a non-existing attribute.
    v = series.unique('Gobbledigook')
    assert v.shape == (0,)
    assert v.size == 0

    # Get values for a non-existing attribute by slice location.
    v = series.unique('Gobbledigook', dims=('SliceLocation',))
    assert v.shape == (4,)
    assert v[-1].size == 0


def test_value():

    print('Testing value')

    # Create a zero-filled array, describing 8 MRI images each measured at 3 flip angles and 2 repetition times.
    loc = np.arange(4)
    fa = [2, 15, 30]
    tr = [2.5, 5.0]
    coords = {
        'SliceLocation': loc,
        'FlipAngle': fa,
        'RepetitionTime': tr,
    }
    series = db.zeros((128,128,4,3,2), coords)

    # Check that an error is raised when multiple values are detected
    try:
        series.value('FlipAngle')
    except:
        pass
    else:
        assert False

    # Check that an error is raised when multiple values are detected
    try: 
        series.value('FlipAngle', dims=('SliceLocation', ))
    except: 
        pass
    else:
        assert False

    # Check that an error is raised when multiple values are detected
    try: 
        series.value('FlipAngle', dims=('SliceLocation', 'RepetitionTime'))
    except:
        pass
    else:
        assert False

    # A value is returned when the flip angle is in the dimensions
    v = series.value('FlipAngle', dims=('SliceLocation', 'RepetitionTime', 'FlipAngle'))
    assert v.shape == (4, 2, 3)
    assert v[0,0,0] == 2.0
    assert v[0,0,1] == 15.0

    v = series.value('FlipAngle', dims=('RepetitionTime', 'FlipAngle'))
    assert v.shape == (2 ,3)
    assert v[0,0] == 2.0
    assert v[1,0] == 2.0
    assert v[0,1] == 15.0

    v = series.value('AcquisitionTime', dims=('SliceLocation', 'RepetitionTime', 'FlipAngle'))
    assert v[0,0,0] == 28609.057496
    assert np.array_equal(np.unique(v), [28609.057496])


def test_set_value():

    print('Testing set_value')

    # Create a zero-filled array, describing 8 MRI images each measured at 3 flip angles and 2 repetition times.
    loc = np.arange(4)
    fa = [2, 15, 30]
    tr = [2.5, 5.0]
    coords = {
        'SliceLocation': loc,
        'FlipAngle': fa,
        'RepetitionTime': tr,
    }
    series = db.zeros((128,128,4,3,2), coords)

    # Get a value and set it again
    v = series.value('SliceLocation', dims=tuple(coords))
    series.set_value('SliceLocation', v, dims=tuple(coords))
    assert np.array_equal(v, series.value('SliceLocation', dims=tuple(coords)))

    # Set the AcquisitionTime to zero for all slices
    assert series.value('AcquisitionTime') == 28609.057496
    series.set_value('AcquisitionTime', 0)
    assert series.value('AcquisitionTime') == 0

    # Set the AcquisitionTime to a different value for each flip angle
    tacq = np.array([0, 60, 120])
    series.set_value('AcquisitionTime', tacq, dims=('FlipAngle',))
    tacq_rec = series.value('AcquisitionTime', dims=('FlipAngle',))
    assert np.array_equal(tacq, tacq_rec)
    tacq_rec = series.value('AcquisitionTime', dims=('RepetitionTime', 'FlipAngle'))
    assert np.array_equal(np.array([tacq, tacq]), tacq_rec)

    # Set the AcquisitionTime to a different value for each flip angle and repetition time
    tacq = np.array([[0, 60], [120, 180], [240, 300]])
    series.set_value('AcquisitionTime', tacq, dims=('FlipAngle','RepetitionTime'))
    tacq_rec = series.value('AcquisitionTime', dims=('FlipAngle','RepetitionTime'))
    assert np.array_equal(tacq, tacq_rec)

    # Check the values in a different shape:
    tacq_rec = series.value('AcquisitionTime', dims=('RepetitionTime', 'FlipAngle'))
    tacq_reshape = np.array([[0, 120, 240], [60, 180, 300]])
    assert np.array_equal(tacq_reshape, tacq_rec)

    # Check that an error is raised if the sizes do not match up:
    try:
        series.set_value('AcquisitionTime', tacq[:,0], dims=('RepetitionTime', 'FlipAngle'))
    except:
        pass
    else:
        assert False

    tacq_rec = series.value('AcquisitionTime', dims=('SliceLocation','FlipAngle','RepetitionTime'))
    assert np.unique(tacq_rec[:,0,0])[0] == 0
    assert np.unique(tacq_rec[:,1,0])[0] == 120
    assert np.unique(tacq_rec[:,2,0])[0] == 240
    assert np.unique(tacq_rec[:,0,1])[0] == 60


def test_spacing():
    series = db.dro.T1_mapping_vFATR()
    assert np.array_equal(series.spacing(), (15, 15, 20))

if __name__ == "__main__":

    # test_pixel_values()
    # test_set_pixel_values()
    # test_unique_affines()
    # test_affine()
    # test_set_affine()
    # test_subseries()
    # test_split_by()
    # test_slice_groups()
    # test_slice()
    # test_islice()
    # test_coords()
    # test_set_coords()
    # test_gridcoords()
    # test_set_gridcoords()
    # test_shape()
    # test_unique()
    # test_value()
    # test_set_value()
    test_spacing()
    

    print('------------------------')
    print('series passed all tests!')
    print('------------------------')