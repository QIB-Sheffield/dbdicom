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


def test_check_if_coords():

    print('Testing check_if_coords')

    # ValueError: These are not proper coordinates. Each coordinate must have the same number of values.
    coords = {
        'SliceLocation': np.array([0,0,0,0]),
        'FlipAngle': np.array([0,0,0]),
        'RepetitionTime': np.array([0,0,0]),
    }
    try:
        db.types.series._check_if_coords(coords) 
    except:
        assert True
    else:
        assert False

    # ValueError: These are not proper coordinates. Coordinate values must be unique.
    coords = {
        'SliceLocation': np.array([0,0,0]),
        'FlipAngle': np.array([0,0,0]),
        'RepetitionTime': np.array([0,0,0]),
    }
    
    # ValueError: These are not proper coordinates. Coordinate values must be unique.
    coords = {
        'SliceLocation': np.array([0,0,0]),
        'FlipAngle': np.array([1,1,1]),
        'RepetitionTime': np.array([2,2,2]),
    }
    try:
        db.types.series._check_if_coords(coords) 
    except:
        assert True
    else:
        assert False 

    # These are proper coordinates - no error
    coords = {
        'SliceLocation': np.array([0,0,0]),
        'FlipAngle': np.array([1,1,1]),
        'RepetitionTime': np.array([1,2,3]),
    }
    try:
        db.types.series._check_if_coords(coords) 
    except:
        assert False
    else:
        assert True

    # Generate proper coordinates from gridcoordinates
    coords = np.meshgrid(
        np.arange(4), 
        np.array([2, 15, 30]), 
        np.array([2.5, 5.0]), 
        indexing='ij')
    coords = {
        'SliceLocation': coords[0],
        'FlipAngle': coords[1],
        'RepetitionTime': coords[2],
    }
    try:
        db.types.series._check_if_coords(coords) 
    except:
        assert False
    else:
        assert True


def test_grid_to_meshcoords():

    print('Testing grid_to_meshcoords')
    
    grid = {
        'SliceLocation': np.arange(4),
        'FlipAngle': np.array([2, 15, 30]),
        'RepetitionTime': np.array([2.5, 5.0]),
    }
    coords = db.types.series._grid_to_meshcoords(grid)
    assert coords['RepetitionTime'][1,2,1] == 5.0
    assert coords['RepetitionTime'][1,2,0] == 2.5

    # ValueError: Grid coordinates have to be unique.
    grid = {
        'SliceLocation': np.arange(4),
        'FlipAngle': np.array([2, 15, 15]),
        'RepetitionTime': np.array([2.5, 5.0]),
    }
    try:
        db.types.series._grid_to_meshcoords(grid)
    except:
        assert True
    else:
        assert False

    #ValueError: Grid coordinates have to be one-dimensionial numpy arrays.
    grid = {
        'SliceLocation': np.arange(4).reshape((2,2)),
        'FlipAngle': np.array([2, 15, 30]),
        'RepetitionTime': np.array([2.5, 5.0]),
    }
    try:
        db.types.series._grid_to_meshcoords(grid)
    except:
        assert True
    else:
        assert False

    # TypeError: Grid coordinates have to be numpy arrays.
    grid = {
        'SliceLocation': np.arange(4),
        'FlipAngle': [2, 15, 30],
        'RepetitionTime': np.array([2.5, 5.0]),
    }
    try:
        db.types.series._grid_to_meshcoords(grid)
    except:
        assert True
    else:
        assert False


def test_as_meshcoords():

    print('Testing as_meshcoords')

    # Proper coordinates that are mesh coordinates
    coords = {
        'SliceLocation': np.array([6,6,6,6,5,5,5,5]),
        'FlipAngle': np.array([3,3,4,4,3,3,4,4]),
        'Gobbledigook': np.array([1,2,4,3,2,1,1,2]),
    }
    coords = db.types.series._as_meshcoords(coords)
    assert coords['SliceLocation'].shape == (2, 2, 2)
    assert np.array_equal(coords['Gobbledigook'][1,1,:], [4,3])

    # ValueError: These are not proper coordinates. Coordinate values must be unique.
    coords = {
        'SliceLocation': np.array([6,6,6,6,5,5,5,5]),
        'FlipAngle': np.array([3,3,4,4,3,3,4,4]),
        'Gobbledigook': np.array([1,2,4,3,2,1,1,1]),
    }
    try:
        db.types.series._as_meshcoords(coords)
    except:
        assert True
    else:
        assert False

    # ValueError: These are not mesh coordinates.
    coords = {
        'SliceLocation': np.array([6,6,6,5]),
        'FlipAngle': np.array([3,4,5,4]),
    }
    try:
        db.types.series._as_meshcoords(coords)
    except:
        assert True
    else:
        assert False


def test_concatenate_coords():

    print('Testing concatenate_coords')

    # Add new slice locations at the same FA and TR
    grid1 = {
        'SliceLocation': np.arange(4),
        'FlipAngle': np.array([2, 15, 30]),
        'RepetitionTime': np.array([2.5, 5.0]),
    }
    coords1 = db.types.series._grid_to_meshcoords(grid1)
    grid2 = {
        'SliceLocation': 4+np.arange(4),
        'FlipAngle': np.array([2, 15, 30]),
        'RepetitionTime': np.array([2.5, 5.0]),
    }
    coords2 = db.types.series._grid_to_meshcoords(grid2)
    coords = db.types.series._concatenate_coords((coords1, coords2), mesh=True)
    assert coords['SliceLocation'].shape == (8,3,2)
    assert np.array_equal(coords['SliceLocation'][:,0,0], np.arange(8))
    assert np.array_equal(coords['SliceLocation'][:,0,1], np.arange(8))

    # ValueError: These are not proper coordinates. Coordinate values must be unique.
    grid2 = {
        'SliceLocation': np.arange(4),
        'FlipAngle': np.array([2, 15, 30]),
        'RepetitionTime': np.array([2.5, 5.0]),
    }
    coords2 = db.types.series._grid_to_meshcoords(grid2)
    try:
        db.types.series._concatenate_coords((coords1, coords2), mesh=True)
    except:
        assert True
    else:
        assert False

    # These are valid coordinates but do not form a mesh, so mesh=True will raise an error.
    # ValueError: These are not mesh coordinates.
    grid2 = {
        'SliceLocation': 4+np.arange(4),
        'FlipAngle': np.array([2, 15, 30, 45]),
        'RepetitionTime': np.array([2.5, 5.0]),
    }
    coords2 = db.types.series._grid_to_meshcoords(grid2)
    try:
        db.types.series._concatenate_coords((coords1, coords2), mesh=True)
    except:
        assert True
    else:
        assert False

    # Setting mesh=False will return a proper set of concatenated coordinates but they no longer fit in a mesh because the second grid has more flip angles for each slice location.
    grid2 = {
        'SliceLocation': 4+np.arange(4),
        'FlipAngle': np.array([2, 15, 30, 45]),
        'RepetitionTime': np.array([2.5, 5.0]),
    }
    coords2 = db.types.series._grid_to_meshcoords(grid2)
    coords = db.types.series._concatenate_coords((coords1, coords2))
    coords_size = coords1['SliceLocation'].size + coords2['SliceLocation'].size
    assert coords['SliceLocation'].size == coords_size


def test_coords():

    print('Testing coords')

    # Create an empty series with 3 coordinates: 
    coords = {
        'SliceLocation': np.array([0,1,2,0,1,2]),
        'FlipAngle': np.array([10,10,10,2,2,2]),
        'RepetitionTime': np.array([1,5,15,1,5,15]),
    }
    series = db.empty_series(coords)
    
    # Extract coordinates and check
    coords = series.coords(tuple(coords))
    assert np.array_equal(coords['FlipAngle'], [2,10,2,10,2,10])
    assert np.array_equal(coords['RepetitionTime'], [1,1,5,5,15,15])

    # Check the default coordinates:
    coords = series.coords()
    assert 'InstanceNumber' in coords
    assert np.array_equal(coords['InstanceNumber'], 1 + np.arange(6))

    # In this case the slice location and flip angle alone are sufficient to identify the frames, so these are valid coordinates:

    coords = series.coords(('SliceLocation', 'FlipAngle'))
    assert np.array_equal(coords['SliceLocation'], [0,0,1,1,2,2])

    # However slice location and acquisition time are not sufficient as coordinates because each combination appears twice. So this throws an error:
    try:
        series.coords(('SliceLocation','RepetitionTime'))
    except:
        assert True
    else:
        assert False

    # Check that an error is thrown if the dimensions are invalid:
    try:
        series.coords(('SliceLocation',))
    except:
        assert True
    else:
        assert False
    try:
        series.coords(('AcquisitionTime', ))
    except:
        assert True
    else:
        assert False

    # Filter the return values:

    coords = series.coords(('SliceLocation', 'FlipAngle'), RepetitionTime=np.array([1,15]))
    assert np.array_equal(coords['FlipAngle'], [2,10,2,10])

    coords = series.coords(('SliceLocation', 'FlipAngle'), AcquisitionTime=0)
    assert np.array_equal(coords['FlipAngle'], [])

    # Filtering with (gr, el) tag can be done using the loc keyword values:

    loc = {(0x0018, 0x0080): np.array([1,15])}
    coords = series.coords(('SliceLocation', 'FlipAngle'), loc=loc)
    assert np.array_equal(coords['FlipAngle'], [2,10,2,10])


    # To test return as mesh, create an empty series with 3 coordinates: 
    coords = {
        'SliceLocation': np.array([0,1,2,0,1,2]),
        'FlipAngle': np.array([10,10,10,2,2,2]),
        'RepetitionTime': np.array([1,5,15,1,5,15]),
    }
    series = db.empty_series(coords)
    
    # Extract as meshcoordinates
    coords = series.coords(tuple(coords), mesh=True)
    assert coords['SliceLocation'].shape == (3,2,1)
    assert coords['SliceLocation'][1,1,0] == 1
    assert coords['FlipAngle'][1,1,0] == 10
    assert coords['RepetitionTime'][1,1,0] == 5

    # Filter the meshed return values:
    coords = series.coords(('SliceLocation', 'FlipAngle'), RepetitionTime=np.array([1,15]), mesh=True)
    assert coords['SliceLocation'].shape == (2,2)
    assert coords['SliceLocation'][1,1] == 2
    assert coords['FlipAngle'][1,1] == 10

    # Use a exclude filter instead:
    coords = series.coords(('SliceLocation', 'FlipAngle'), RepetitionTime=np.array([1,15]), mesh=True, exclude=True)
    assert coords['SliceLocation'].shape == (1,2)
    assert coords['SliceLocation'][0,1] == 1
    assert coords['FlipAngle'][0,0] == 2
    assert coords['FlipAngle'][0,1] == 10

    # If the coordinates are not mesh coordinates, meshcoords() will raise an error:
    # ValueError: These are not mesh coordinates.
    coords = {
        'SliceLocation': np.array([0,1,2,1,1,2]),
        'FlipAngle': np.array([10,10,10,2,2,2]),
        'RepetitionTime': np.array([1,5,15,1,5,15]),
    }
    series = db.empty_series(coords)
    try:
        series.coords(tuple(coords), mesh=True)
    except:
        assert True
    else:
        assert False


def test_values():

    print('Testing values')

    # Create a zero-filled array with 3 slice dimensions.
    coords = {
        'SliceLocation': np.arange(4),
        'FlipAngle': np.array([30, 2, 15]),
        'RepetitionTime': np.array([2.5, 5.0]),
    }
    zeros = db.empty_series(gridcoords=coords)
    dims = tuple(coords)

    # If values() is called without dimensions, a flat array is returned with one value per frame, ordered by instance number:
    in1 = zeros.values('InstanceNumber')
    assert np.array_equal(in1, 1+np.arange(24))
    fa1 = zeros.values('FlipAngle')
    assert set(fa1[:6]) == set([2, 15, 30])

    # Or read both in one go:
    in2, fa2 = zeros.values('InstanceNumber', 'FlipAngle')
    assert np.array_equal(in1, in2)
    assert set(fa1) == set(fa2)

    # Any proper dimensions can be used to sort the values  
    fa = zeros.values('FlipAngle', dims=dims)
    assert fa.shape == (24,)
    assert fa[0] == 2
    assert fa[2] == 15
    assert fa[4] == 30

    # return acquisition time ordered by the original dimensions, check that all values are the same.
    tacq = zeros.values('AcquisitionTime', dims=dims)
    assert tacq.shape == (24,)
    assert tacq[0] == 28609.057496
    assert np.array_equal(np.unique(tacq), [28609.057496])

    # A value of None is returned in locations where the value is missing:
    gbl = zeros.values('Gobbledigook')
    assert np.array_equal(np.full((24,), None), gbl)

    # To test return as mesh, create an empty series with 3 coordinates: 
    coords = {
        'SliceLocation': np.array([0,1,2,0,1,2]),
        'FlipAngle': np.array([10,10,10,2,2,2]),
        'RepetitionTime': np.array([1,5,15,1,5,15]),
    }
    series = db.empty_series(coords)
    
    # Extract as meshcoordinates
    vals = series.values('SliceLocation', dims=tuple(coords), mesh=True)
    assert vals.shape == (3,2,1)
    assert vals[1,1,0] == 1
    vals = series.values('FlipAngle', dims=tuple(coords), mesh=True)
    assert vals[1,1,0] == 10
    vals = series.values('RepetitionTime', dims=tuple(coords), mesh=True)
    assert vals[1,1,0] == 5
    fa, tr, co = series.values('FlipAngle', 'RepetitionTime', dims=tuple(coords), mesh=True, return_coords=True)
    assert fa[1,1,0] == 10
    assert tr[1,1,0] == 5
    assert np.array_equal(co['RepetitionTime'], tr)

    # Add a filter
    fa, tr, co = series.values('FlipAngle', 'RepetitionTime', dims=('SliceLocation', 'FlipAngle'), RepetitionTime=np.array([1,15]), mesh=True, return_coords=True)
    assert co['SliceLocation'].shape == (2,2)
    assert co['SliceLocation'][1,1] == 2
    assert co['FlipAngle'][1,1] == 10
    assert np.array_equal(co['FlipAngle'], fa)


def test_frames():

    print('Testing frames')

    # Create an empty series with 3 slice dimensions 
    grid = {
        'SliceLocation': np.arange(4),
        'FlipAngle': np.array([30,2,15]),
        'RepetitionTime': np.array([2.5, 5.0]),
    }
    coords = db.types.series._grid_to_meshcoords(grid)
    series = db.empty_series(coords)

    frames = series.frames(tuple(coords))
    assert frames[2].FlipAngle == 15
    assert frames[2].RepetitionTime == 2.5

    frames, coords = series.frames(tuple(coords), return_coords=True)
    assert frames[2].FlipAngle == 15
    assert frames[2].RepetitionTime == 2.5
    assert coords['FlipAngle'][2] == 15
    assert coords['RepetitionTime'][2] == 2.5

    # Return values on the fly:
    frames[2].AcquisitionTime = 0
    frames, vals = series.frames(tuple(coords), return_vals=('AcquisitionTime', 'PatientName'))
    assert vals.shape==(2,24)
    assert len(np.unique(vals[0,:])) == 2
    assert len(np.unique(vals[1,:])) == 1
    assert frames[2].AcquisitionTime == 0

    # Now extract as mesh coordinates and test again
    frames = series.frames(tuple(coords), mesh=True)
    assert frames[1,1,1].FlipAngle == 15
    assert frames[1,1,1].RepetitionTime == 5.0

    frames, coords = series.frames(tuple(coords), return_coords=True, mesh=True)
    assert frames[1,1,1].FlipAngle == 15
    assert frames[1,1,1].RepetitionTime == 5.0
    assert coords['FlipAngle'][1,1,1] == 15
    assert coords['RepetitionTime'][1,1,1] == 5.0

    # Add a filter

    frames, coords = series.frames(tuple(coords), return_coords=True, mesh=True, FlipAngle=np.array([15,30]))
    assert frames[1,1,1].FlipAngle == 30
    assert frames[1,1,1].RepetitionTime == 5.0
    assert coords['FlipAngle'][1,1,1] == 30
    assert coords['RepetitionTime'][1,1,1] == 5.0

    # Check empty case

    frames, coords = series.frames(tuple(coords), return_coords=True, mesh=True, FlipAngle=-1)
    assert frames.size == 0
    assert coords['FlipAngle'].size == 0


def test_expand():

    print('Testing expand')

    # Generate grid coordinates
    grid = {
        'SliceLocation': np.arange(4),
        'FlipAngle': np.array([2, 15, 30]),
        'RepetitionTime': np.array([2.5, 5.0]),
    }

    # create an empty series
    series = db.series()

    # Expand the series to the new coordinates
    series.expand(gridcoords=grid)
    coords = series.coords(tuple(grid), mesh=True)
    assert coords['FlipAngle'].shape == (4,3,2)

    # If the series already has data, the new coordinates must be distinct from the exisiting ones:
    grid = {
        'SliceLocation': np.arange(4),
        'FlipAngle': np.array([45, 60]),
        'RepetitionTime': np.array([2.5, 5.0]),
    }
    series.expand(gridcoords=grid)
    coords = series.coords(tuple(grid), mesh=True)
    assert set(np.unique(coords['FlipAngle'])) == set([2, 15, 30, 45, 60])

    # Expanding to the existing coordinates triggers an error
    # ValueError: Cannot expand - the new coordinates overlap with existing coordinates.
    coords = series.coords(tuple(grid), mesh=True)
    try:
        series.expand(coords)
    except:
        assert True
    else:
        assert False

    # Expanding to partially overlapping coordinates triggers an error:
    grid = {
        'SliceLocation': np.arange(4),
        'FlipAngle': np.array([60, 90]),
        'RepetitionTime': np.array([2.5, 5.0]),
    }
    try:
        series.expand(gridcoords=grid)
    except:
        assert True
    else:
        assert False


def test_set_coords():

    print('Testing set_coords')

    # Create an empty series with 3 coordinates: 
    coords = {
        'SliceLocation': np.array([0,1,2,0,1,2]),
        'FlipAngle': np.array([2,2,2,10,10,10]),
        'RepetitionTime': np.array([1,5,15,1,5,15]),
    }
    series = db.empty_series(coords)
    dims = tuple(coords)

    # Get the coordinates and set them again
    coords = series.coords(dims)
    series.set_coords(coords)
    coords_rec = series.coords(dims)
    for d in dims:
        assert np.array_equal(coords[d], coords_rec[d])

    # Change the flip angle of 2 to 5:
    coords = series.coords(dims)
    fa = coords['FlipAngle']
    fa[np.where(fa==2)] = 5
    series.set_coords(coords)
    # Check results
    coords = series.coords(dims)
    assert np.array_equal(coords['FlipAngle'], [5,10,5,10,5,10])

    # Change FA 5 back to FA 2 again, using filtering this time (shorter)
    coords = series.coords(dims, FlipAngle=5)
    coords['FlipAngle'][:] = 2
    series.set_coords(coords, FlipAngle=5)
    # Check results
    coords = series.coords(dims)
    assert np.array_equal(coords['FlipAngle'], [2,10,2,10,2,10])

    # When attempting to set an invalid set of coordinates, an error is raised:
    # ValueError: Cannot set coordinates - this would produce invalid coordinates for the series
    coords = series.coords(dims, FlipAngle=2)
    coords['FlipAngle'][:] = 10
    try:
        series.set_coords(coords, FlipAngle=2)
    except:
        assert True
    else:
        assert False

    # Create a new set of coordinates along different dimensions than the original:
    coords = {
        'SliceLocation': np.array([0,0,1,1,2,2]),
        'AcquisitionTime': np.array([0,60,0,60,0,60]),
    }
    series.set_coords(coords, ('SliceLocation', 'FlipAngle'))

    # Inspect the new coordinates - each slice now has two acquisition times corresponding to the flip angles:
    coords = series.coords(('SliceLocation', 'AcquisitionTime', 'FlipAngle'))
    assert np.array_equal(coords['SliceLocation'], [0,0,1,1,2,2])
    assert np.array_equal(coords['AcquisitionTime'], [0,60,0,60,0,60])
    assert np.array_equal(coords['FlipAngle'], [2,10,2,10,2,10])

    # An error is raised if the new coordinates have different sizes:
    coords = {
        'SliceLocation':np.zeros(24),
        'AcquisitionTime':np.ones(25),
    }
    try:
        series.set_coords(coords, dims)
    except:
        assert True
    else:
        assert False

    # An error is also raised if they have all the same size but the values are not unique:

    coords = {
        'SliceLocation':np.zeros(24),
        'AcquisitionTime':np.ones(25),
    }
    try:
        series.set_coords(coords, dims)
    except:
        assert True
    else:
        assert False

    # .. or when the number does not match up with the size of the series:

    coords = {
        'SliceLocation':np.arange(25),
        'AcquisitionTime':np.arange(25),
    }
    try:
        series.set_coords(coords, dims)
    except:
        assert True
    else:
        assert False


def test_set_values():

    print('Testing set_values')

    # Create a zero-filled array, describing 8 MRI images each measured at 3 flip angles and 2 repetition times.
    coords = {
        'SliceLocation': np.arange(4),
        'FlipAngle': np.array([2, 15, 30]),
        'RepetitionTime': np.array([2.5, 5.0]),
    }
    series = db.empty_series(gridcoords=coords)
    dims = tuple(coords)

    # Get a value and set it again
    v = series.values('SliceLocation', dims=dims)
    series.set_values(v, 'SliceLocation', dims=dims)
    assert np.array_equal(v, series.values('SliceLocation', dims=dims))

    # Get multiple values and set them again
    tags = ('SliceLocation', 'AcquisitionTime')
    v = series.values(*tags, dims=dims)
    series.set_values(v, tags, dims=dims)
    assert np.array_equal(v, series.values(*tags, dims=dims))

    # Set the AcquisitionTime to zero for all slices
    assert np.array_equal(np.unique(series.values('AcquisitionTime')), [28609.057496])
    series.set_values(0,'AcquisitionTime')
    assert np.array_equal(np.unique(series.values('AcquisitionTime')), [0])

    # Set the AcquisitionTime to a different value for each flip angle
    tacq = np.repeat([0, 60, 120], 8)
    series.set_values(tacq, 'AcquisitionTime', dims=('FlipAngle','InstanceNumber'))

    # Check in the original dimensions:
    tacq = series.values('AcquisitionTime', dims=dims, mesh=True)
    assert np.array_equal(np.unique(tacq[:,0,:]), [0])
    assert np.array_equal(np.unique(tacq[:,1,:]), [60])
    assert np.array_equal(np.unique(tacq[:,2,:]), [120])

    # Do the same thing using filters (slower but more transparent):
    series.set_values(30, 'AcquisitionTime', FlipAngle=2)
    series.set_values(90, 'AcquisitionTime', FlipAngle=15)
    series.set_values(150, 'AcquisitionTime', FlipAngle=30)

    # Check in the original dimensions:
    tacq = series.values('AcquisitionTime', dims=dims, mesh=True)
    assert np.array_equal(np.unique(tacq[:,0,:]), [30])
    assert np.array_equal(np.unique(tacq[:,1,:]), [90])
    assert np.array_equal(np.unique(tacq[:,2,:]), [150])

    # Set the acquistion time for each flip angle and TR
    tacq = np.repeat(60*np.arange(6), 4)
    series.set_values(tacq, 'AcquisitionTime', dims=('FlipAngle','RepetitionTime','SliceLocation'))

    # Check in the original dimensions:
    tacq = series.values('AcquisitionTime', dims=dims, mesh=True)
    assert np.array_equal(np.unique(tacq[:,0,0]), [0])
    assert np.array_equal(np.unique(tacq[:,0,1]), [60])
    assert np.array_equal(np.unique(tacq[:,1,0]), [120])
    assert np.array_equal(np.unique(tacq[:,1,1]), [180])

    # Check that an error is raised if the sizes do not match up:
    try:
        series.set_values(np.arange(25), 'AcquisitionTime', dims=tuple(coords))
    except:
        pass
    else:
        assert False


def test_set_gridcoords():

    print('Testing set_gridcoords')

    # Create an empty series and set grid coords:
    gridcoords = {
        'SliceLocation': np.arange(4),
        'FlipAngle': np.array([2, 15, 30]),
        'RepetitionTime': np.array([2.5, 5.0]),
    }
    series = db.empty_series()
    series.set_gridcoords(gridcoords)

    # Get the coordinates as a mesh
    coords = series.coords(tuple(gridcoords), mesh=True)
    assert coords['SliceLocation'].shape == (4,3,2)
    assert coords['FlipAngle'][1,1,1] == 15


def test_gridcoords():

    print('Testing gridcoords')

    # Set grid coordinates and get them again:
    gridcoords = {
        'SliceLocation': np.arange(4),
        'FlipAngle': np.array([2, 15, 30]),
        'RepetitionTime': np.array([2.5, 5.0]),
    }
    series = db.empty_series(gridcoords=gridcoords)

    gridcoords_rec = series.gridcoords(tuple(gridcoords))
    for dim in gridcoords_rec:
        assert np.array_equal(gridcoords_rec[dim], gridcoords[dim])

    # Check that an error is raised if the coordinates are not grid coordinates:

    # Create an empty series with 3 coordinates: 
    coords = {
        'SliceLocation': np.array([0,1,2,0,1,2]),
        'FlipAngle': np.array([10,10,10,2,2,2]),
        'RepetitionTime': np.array([1,5,15,1,5,15]),
    }
    series = db.empty_series(coords)

    # This works:
    try:
        series.coords(tuple(coords), mesh=True)
    except:
        assert False
    else:
        assert True

    # This does not work:
    try:
        series.gridcoords(tuple(coords))
    except:
        assert True
    else:
        assert False


def test_shape():

    print('Testing shape')

    # Create and empty series and test the shape
    assert db.series().shape() == (0,)

    # Create a zero-filled array with 3 dimensions:
    loc = np.arange(4)
    fa = np.array([2, 15, 30])
    tr = np.array([2.5, 5.0])
    coords = {
        'SliceLocation': loc,
        'FlipAngle': fa,
        'RepetitionTime': tr,
    }
    series = db.empty_series(gridcoords=coords)
    dims = tuple(coords)

    assert series.shape() == (len(loc)*len(fa)*len(tr), )
    assert series.shape(dims, mesh=True) == (len(loc), len(fa), len(tr))
    assert series.shape((dims[1], dims[0], dims[2]), mesh=True) == (len(fa), len(loc), len(tr))
    assert series.shape(('FlipAngle', 'InstanceNumber'), mesh=True) == (len(fa), len(loc)*len(tr))
    eshape = series.new_sibling().shape(dims, mesh=True)
    assert eshape == (0,0,0) 
    try:
        series.shape(('FlipAngle', 'Gobbledigook'))
    except:
        assert True
    else:
        assert False
    try:
        series.shape(('FlipAngle', 'AcquisitionTime'))
    except:
        assert True
    else:
        assert False  


def test_unique():

    print('Testing unique')

    # Create a zero-filled array with 3 slice dimensions.
    loc = np.arange(4)
    fa = np.array([2, 15, 30])
    tr = np.array([2.5, 5.0])
    coords = {
        'SliceLocation': loc,
        'FlipAngle': fa,
        'RepetitionTime': tr,
    }
    series = db.empty_series(gridcoords=coords)

    # Recover unique values of each coordinate
    uloc = series.unique('SliceLocation')
    ufa = series.unique('FlipAngle')
    utr = series.unique('RepetitionTime')
    assert np.array_equal(uloc, loc)
    assert np.array_equal(ufa, fa)
    assert np.array_equal(utr, tr)

    # Recover in one call to unique (faster):
    uloc, ufa, utr = series.unique('SliceLocation', 'FlipAngle', 'RepetitionTime')
    assert np.array_equal(uloc, loc)
    assert np.array_equal(ufa, fa)
    assert np.array_equal(utr, tr)

    # Get unique Flip Angles for each slice location
    ufa = series.unique('FlipAngle', sortby=('SliceLocation', ))
    assert len(ufa) == len(loc)
    for zfa in ufa:
        assert np.array_equal(zfa, fa)

    # Get unique repetition times for each slice location
    utr = series.unique('RepetitionTime', sortby=('SliceLocation', ))
    assert len(utr) == len(loc)
    for ztr in utr:
        assert np.array_equal(ztr, tr)

    # Get both in the same call:
    ufa, utr = series.unique('FlipAngle', 'RepetitionTime', sortby=('SliceLocation', ))
    assert len(ufa) == len(loc)
    assert len(utr) == len(loc)
    for zfa in ufa:
        assert np.array_equal(zfa, fa)
    for ztr in utr:
        assert np.array_equal(ztr, tr)

    # Get unique Flip Angles for each slice location and repetition time
    ufa = series.unique('FlipAngle', sortby=('SliceLocation', 'RepetitionTime'))
    assert ufa.size == len(loc)*len(tr)
    assert ufa.shape == (len(loc), len(tr))
    for z in range(len(loc)):
        for t in range(len(tr)):
            assert np.array_equal(ufa[z,t], fa)

    # Extract locations as well
    ufa, co = series.unique('FlipAngle', sortby=('SliceLocation', 'RepetitionTime'), return_locs=True)
    assert ufa.size == len(loc)*len(tr)
    assert ufa.shape == (len(loc), len(tr))
    for z in range(len(loc)):
        for t in range(len(tr)):
            assert np.array_equal(ufa[z,t], fa)
            assert co[0][z,t] == loc[z]
            assert co[1][z,t] == tr[t]

    # Get unique Flip Angles and repetition time for each slice location, repetition time and flip angle.
    ufa, utr = series.unique('FlipAngle', 'RepetitionTime', sortby=('SliceLocation', 'RepetitionTime', 'FlipAngle'))
    for z in range(len(loc)):
        for t in range(len(tr)):
            assert np.array_equal(ufa[z,t,:], fa)
        for a in range(len(fa)):
            assert np.array_equal(utr[z,:,a], tr)

    # Get values for a non-existing attribute.
    gbl = series.unique('Gobbledigook')
    assert gbl.shape == (0,)
    assert gbl.size == 0

    # Get values for a non-existing attribute by slice location.
    gbl = series.unique('Gobbledigook', sortby=('SliceLocation',))
    assert gbl.shape == (len(loc),)
    assert gbl.size == len(loc)
    for zgbl in gbl:
        assert zgbl.shape == (0,)
        assert zgbl.size == 0


def test_pixel_values():
    
    coords = {
        'SliceLocation': 10*np.arange(4),
        'FlipAngle': np.array([2, 15, 30]),
        'RepetitionTime': np.array([2.5, 5.0]),
    }
    zeros = db.zeros((128,64,4,3,2), coords)
    dims = tuple(coords)

    array = zeros.pixel_values(dims)
    assert array.shape == (128, 64, 4, 3, 2)

    array = zeros.pixel_values(dims, FlipAngle=15)
    assert array.shape == (128, 64, 4, 1, 2)

    array = zeros.pixel_values(dims, FlipAngle=0)
    assert array.size == 0

    array = zeros.pixel_values(dims, FlipAngle=np.array([15,30]))
    assert array.shape == (128, 64, 4, 2, 2)

    array = zeros.pixel_values(dims, FlipAngle=np.array([15,30]), SliceLocation=np.array([10,20]))
    assert array.shape == (128, 64, 2, 2, 2)

    array = zeros.pixel_values(dims, FlipAngle=np.array([15,30]), SliceLocation=np.array([1,2]))
    assert array.size == 0

    array = zeros.pixel_values(dims, AcquisitionTime=28609.057496)
    assert array.shape == (128, 64, 4, 3, 2)

    array = zeros.pixel_values(dims, AcquisitionTime=0)
    assert array.size == 0

    array = zeros.pixel_values(dims, select={'FlipAngle': 15})
    assert array.shape == (128, 64, 4, 1, 2)

    array = zeros.pixel_values(dims, select={(0x0018, 0x1314): 15})
    assert array.shape == (128, 64, 4, 1, 2)

    array = zeros.pixel_values(dims, inds={'FlipAngle': 1})
    assert array.shape == (128, 64, 4, 1, 2)

    array = zeros.pixel_values(dims, inds={'FlipAngle':np.arange(2)})
    assert array.shape == (128, 64, 4, 2, 2)

    # ValueError: Indices must be in the dimensions provided.
    try:
        zeros.pixel_values(dims, inds={'AcquisitionTime':np.arange(2)})
    except:
        assert True
    else:
        assert False
    

def test_set_pixel_values():
    # Test taken from docstring

    # Create a zero-filled array with 3 slice dimensions:
    coords = {
        'SliceLocation': 10*np.arange(4),
        'FlipAngle': np.array([2, 15, 30]),
        'RepetitionTime': np.array([2.5, 5.0]),
    }
    series = db.zeros((128,64,4,3,2), coords)
    
    # Retrieve the array and check that it is populated with zeros:
    dims = tuple(coords)
    array = series.pixel_values(dims)
    assert array.shape == (128,64,4,3,2)
    assert np.mean(array) == 0.0

    # Now overwrite the values with a new array of ones. 

    ones = np.ones((128,64,4,3,2))
    series.set_pixel_values(ones, coords)

    #Retrieve the array and check that it is now populated with ones:

    array = series.pixel_values(dims) 
    assert array.shape == (128,64,4,3,2)
    assert np.mean(array) == 1.0

    # Now set only the pixels with flip angle 15 to zero:

    zeros = np.zeros((128,64,8,1,2))
    series.set_pixel_values(zeros, dims, FlipAngle=15)

    # Extract the complete array again and check the values:

    array = series.pixel_values(dims)
    assert array.shape == (128,64,4,3,2)
    assert np.mean(array[:,:,:,0,:]) == 1.0
    assert np.mean(array[:,:,:,1,:]) == 0.0
    assert np.mean(array[:,:,:,2,:]) == 1.0

    # Set the FA=15 subarray back to 1 by index:

    ones = np.ones((128,64,8,1,2))
    series.set_pixel_values(ones, inds={'FlipAngle':1})

    # Extract the complete array again and check the values:

    array = series.pixel_values(dims)
    assert np.mean(array[:,:,:,0,:]) == 1.0
    assert np.mean(array[:,:,:,1,:]) == 1.0
    assert np.mean(array[:,:,:,2,:]) == 1.0   


def test_isnull():

    # Create a zero-filled array with 3 dimensions:
    loc = np.arange(4)
    fa = [2, 15, 30]
    tr = [2.5, 5.0]
    coords = {
        'SliceLocation': loc,
        'FlipAngle': fa,
        'RepetitionTime': tr,
    }
    series = db.zeros((8,8,4,3,2), coords)
    dims = tuple(coords)

    print(series.isnull('SliceLocation', dims))


 

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





def test_spacing():
    series = db.dro.T1_mapping_vFATR()
    assert np.array_equal(series.spacing(), (15, 15, 20))

if __name__ == "__main__":

    
    # Helper functions

    test_check_if_coords()
    test_grid_to_meshcoords()
    test_as_meshcoords()
    test_concatenate_coords()

    # # API
    
    test_coords()
    test_values()
    test_frames()
    test_expand()
    test_set_coords()
    test_set_values()
    test_set_gridcoords()
    test_gridcoords()
    test_shape()
    test_unique()
    # test_pixel_values()
    # test_set_pixel_values()
    
    # test_affine()
    # test_set_affine()

    # test_isnull()
    # test_unique_affines()
    # test_subseries()
    # test_split_by()
    # test_slice_groups()
    # test_slice()
    # test_islice()
    # test_spacing()
    

    print('------------------------')
    print('series passed all tests!')
    print('------------------------')