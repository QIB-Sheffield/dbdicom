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


def test_ellipsoid():
    print('Running test_ellipsoid..')
    ellipsoid = db.dro.ellipsoid(12, 20, 64, spacing=(2,2,4), levelset=True)
    array = ellipsoid.pixel_values()
    affine = np.array(
        [[2., 0., 0., 0.],
         [0., 2., 0., 0.],
         [0., 0., 4., 0.],
         [0., 0., 0., 1.]]
    )
    assert array.shape == (15, 23, 35)
    assert ellipsoid.PixelSpacing == [2.0, 2.0]
    assert ellipsoid.SliceLocation == list(4*np.arange(35))
    assert np.array_equal(ellipsoid.affine(), affine)




if __name__ == "__main__":

    test_ellipsoid()

    print('-----------------------------')
    print('dbdicom.dro passed all tests!')
    print('-----------------------------')

