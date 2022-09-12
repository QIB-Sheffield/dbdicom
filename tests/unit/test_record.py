import os
import shutil

import dbdicom.record as dbr


datapath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fixtures')
twofiles = os.path.join(datapath, 'TWOFILES')
onefile = os.path.join(datapath, 'ONEFILE')
rider = os.path.join(datapath, 'RIDER')
zipped = os.path.join(datapath, 'ZIP')
multiframe = os.path.join(datapath, 'MULTIFRAME')

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



def test_database():

    tmp = create_tmp_database(rider)

    db = dbr.open(tmp)
    assert 24 == len(db.register.instances('Database'))

    remove_tmp_database(tmp)




if __name__ == "__main__":

    test_database()

    print('------------------------')
    print('record passed all tests!')
    print('------------------------')


