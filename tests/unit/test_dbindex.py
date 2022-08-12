import os
import shutil
import numpy as np

from dbdicom.dbindex import DbIndex

datapath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fixtures')
twofiles = os.path.join(datapath, 'TWOFILES')
onefile = os.path.join(datapath, 'ONEFILE')
rider = os.path.join(datapath, 'RIDER')
zipped = os.path.join(datapath, 'ZIP')
multiframe = os.path.join(datapath, 'MULTIFRAME')

# Helper functions

def create_tmp_database(path):
    tmp = os.path.join(os.path.dirname(__file__), 'tmp')
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    shutil.copytree(path, tmp)
    return tmp

def remove_tmp_database(tmp):
    shutil.rmtree(tmp)


def test_init():

    dbi = DbIndex()
    assert dbi.dataframe is None

    tmp = create_tmp_database(rider)

    dbi = DbIndex(tmp)
    assert dbi.dataframe is None
    assert dbi.path == tmp

    remove_tmp_database(tmp)


def test_read_dataframe():

    tmp = create_tmp_database(rider)
    dbi = DbIndex(tmp)
    dbi.read_dataframe()
    assert dbi.dataframe.shape == (24, 2+len(dbi.columns))
    remove_tmp_database(tmp)

def test_rw_df():

    tmp = create_tmp_database(twofiles)
    dbi = DbIndex(tmp)
    dbi.read_dataframe()
    dbi._write_df()
    df1 = dbi.dataframe
    dbi._read_df()
    df2 = dbi.dataframe

    remove_tmp_database(tmp)

    assert np.array_equal(df1.to_numpy(), df2.to_numpy())

def test_multiframe_to_singleframe():

    tmp = create_tmp_database(multiframe)
    dbi = DbIndex(tmp)
    dbi.read_dataframe()
    assert dbi.dataframe.shape == (2, 14)
    dbi._multiframe_to_singleframe()
    assert dbi.dataframe.shape == (124, 14)
    remove_tmp_database(tmp)

def test_scan():

    tmp = create_tmp_database(rider)
    dbi = DbIndex(tmp)
    dbi.scan()
    assert dbi.dataframe.shape == (24, 14)
    remove_tmp_database(tmp)

def test_open_close():

    tmp = create_tmp_database(rider)
    dbi = DbIndex(tmp)
    dbi.open()
    assert dbi.dataframe.shape == (24, 14)
    dbi.close()
    assert dbi.dataframe is None
    dbi.open()
    assert dbi.dataframe is None
    dbi.close()
    remove_tmp_database(tmp)

    tmp = create_tmp_database(twofiles)
    dbi.open(tmp)
    assert dbi.dataframe.shape == (2, 14)
    dbi.close()
    assert dbi.dataframe is None
    remove_tmp_database(tmp)
    

if __name__ == "__main__":

    test_init()
    test_read_dataframe()
    test_rw_df()
    test_multiframe_to_singleframe()
    test_scan()
    test_open_close()

    print('-------------------------')
    print('dbindex passed all tests!')
    print('-------------------------')

