import numpy as np
import dbdicom.register as rg

def get_data():
    columns = ['A', 'B', 'C']
    data = {
        'X': [1,2,3],
        'Y': [4,5,6],
        'Z': [7,8,9],
        'W': [10,11,9]
    }
    return data, columns

def get_data2():
    columns = ['A', 'B', 'C']
    data = {
        'X2': [1,2,3],
        'Y2': [4,5,6],
        'Z2': [7,8,9],
        'W2': [10,11,9]
    }
    return data, columns

def get_data3():
    columns = ['A', 'B', 'C']
    data = {
        'X': [1,1,2],
        'Y': [1,2,3],
        'Z': [5,6,7],
        'T': [3,4,5],
        'U': [1,1,1],
        'V': [5,1,1],
    }
    return data, columns




def test_init():
    data, columns = get_data()
    reg = rg.Register(data, columns)
    assert reg.empty == False
    assert reg.shape == (4,3)

def test_columns():
    data, columns = get_data()
    reg = rg.Register(data, columns)
    assert reg.columns() == columns

def test_index():
    data, columns = get_data()
    reg = rg.Register(data, columns)
    assert reg.index() == list(data.keys())

def test_at():
    data, columns = get_data()
    reg = rg.Register(data, columns)
    assert reg.at('Y','C') == 6

def test_set():
    data, columns = get_data()
    reg = rg.Register(data, columns)
    assert reg.at('Y','C') == 6
    reg.set('Y', 'C', 3)
    assert reg.at('Y','C') == 3

def test_add_column():
    data, columns = get_data()
    reg = rg.Register(data, columns)
    reg.add_column('D', 0)
    assert reg.shape == (4,4)
    assert reg.at('Y','D') == 0

def test_rows_where():
    data, columns = get_data()
    reg = rg.Register(data, columns)
    sub = reg.rows_where('C', [6,9])
    assert sub.shape == (3,3)
    assert sub.at('Y', 'C') == 6
    assert sub.at('Z', 'C') == 9
    assert reg.at('Z', 'C') == 9
    sub = reg.rows_where('C', 9)
    reg.set('Z', 'C', 3)
    assert sub.at('Z', 'C') == 3
    assert reg.at('Z', 'C') == 3
    sub.set('Z', 'C', -3)
    assert sub.at('Z', 'C') == -3
    assert reg.at('Z', 'C') == -3

def test_update():
    data, columns = get_data()
    reg = rg.Register(data, columns)
    data, columns = get_data2()
    reg2 = rg.Register(data, columns)
    reg.update(reg2)
    assert reg.at('Z2', 'C') == 9
    assert reg.shape == (8,3)

def test_drop_row():
    data, columns = get_data()
    reg = rg.Register(data, columns)
    reg.drop_row('Y')
    assert reg.shape == (3,3)
    try:
        reg.at('Y', 'B')
    except:
        assert True
    else:
        assert False

def test_drop_column():
    data, columns = get_data()
    reg = rg.Register(data, columns)
    reg.drop_column('B')
    assert reg.shape == (4,2)
    try:
        reg.at('Y', 'B')
    except:
        assert True
    else:
        assert False

def test_values():
    data, columns = get_data()
    reg = rg.Register(data, columns)
    assert np.array_equal(reg.values(), [[1,2,3],[4,5,6],[7,8,9],[10,11,9]])
    assert np.array_equal(reg.values(index=['X','Y']), [[1,2,3],[4,5,6]])
    assert np.array_equal(reg.values(index='Y'), [4,5,6])
    assert np.array_equal(reg.values(column=['A','B']), [[1,2],[4,5],[7,8],[10,11]])
    assert np.array_equal(reg.values(index=['X','Y'], column=['A','B']), [[1,2],[4,5]])
    assert np.array_equal(reg.values(index='Y', column=['A','B']), [4,5])
    assert np.array_equal(reg.values(column='A'), [1,4,7,10])
    assert np.array_equal(reg.values(index=['X','Y'], column='A'), [1,4])
    assert np.array_equal(reg.values(index='Y', column='A'), 4) 
    assert np.array_equal(reg.values(column='C'), [3,6,9,9])
    assert reg.values(column='R') is None

def test_column():
    data, columns = get_data()
    reg = rg.Register(data, columns)
    assert reg.column(3) == 'C'
    assert reg.column(5) == 'B'

def test_extract_rows():
    data, columns = get_data()
    reg = rg.Register(data, columns)
    sub = reg.extract_rows('X')
    assert sub.shape == (1,3)
    assert sub.at('X', 'C') == 3
    assert reg.at('X', 'C') == 3
    try:
        sub.at('Y', 'C')
    except:
        assert True
    else:
        assert False
    sub = reg.extract_rows(['X','Y'])
    assert sub.shape == (2,3)
    assert sub.at('X', 'C') == 3
    assert reg.at('Y', 'C') == 6
    try:
        sub.at('Y', 'C')
    except:
        assert False
    else:
        assert True
    sub = reg.extract_rows('X', ['A', 'B'])
    assert sub.shape == (1,2)
    assert sub.at('X', 'B') == 2
    assert reg.at('X', 'B') == 2
    try:
        sub.at('X', 'C')
    except:
        assert True
    else:
        assert False
    sub = reg.extract_rows(['X','Y'], ['A', 'B'])
    assert sub.shape == (2,2)
    assert sub.at('Y', 'B') == 5
    assert reg.at('Y', 'B') == 5
    try:
        sub.at('X', 'C')
    except:
        assert True
    else:
        assert False
    sub = reg.extract_rows('X', 'B')
    assert sub.shape == (1,1)
    assert sub.at('X', 'B') == 2
    assert reg.at('X', 'B') == 2
    try:
        sub.at('X', 'C')
    except:
        assert True
    else:
        assert False


def test_rows_where_not():
    data, columns = get_data()
    reg = rg.Register(data, columns)
    sub = reg.rows_where_not('C', 6)
    assert sub.shape == (3,3)


def test_rows_where_any():
    data, columns = get_data()
    reg = rg.Register(data, columns)
    sub = reg.rows_where_any(1)
    assert sub.shape == (1,3)
    assert sub.at('X', 'C') == 3
    sub = reg.rows_where_any([1,5])
    assert sub.shape == (2,3)
    assert set(sub.index()) == {'X', 'Y'}


def test_sort_values():
    data, columns = get_data3()
    reg = rg.Register(data, columns)
    reg.sort_values()
    assert reg.values() == [
        [1, 1, 1], 
        [1, 1, 2], 
        [1, 2, 3], 
        [3, 4, 5], 
        [5, 1, 1], 
        [5, 6, 7],
    ]
    reg.sort_values('C')
    assert reg.values() == [
        [1, 1, 1], 
        [5, 1, 1], 
        [1, 1, 2], 
        [1, 2, 3], 
        [3, 4, 5], 
        [5, 6, 7],
    ]
    reg.sort_values(['B','A'])
    assert reg.values() == [
        [1, 1, 1], 
        [1, 1, 2],
        [5, 1, 1],  
        [1, 2, 3], 
        [3, 4, 5], 
        [5, 6, 7],
    ]

def test_dropna():
    columns = ['A', 'B', 'C']
    data = {
        'X': [1,2,3],
        'Y': [4,float('nan'),6],
        'Z': [None,8,9],
        'W': [10,11,0]
    }
    reg = rg.Register(data, columns)  
    reg.dropna()
    assert reg.values() == [
        [1, 2, 3], 
        [10, 11, 0],
    ] 


if __name__ == "__main__":

    exit()

    test_init()
    test_columns()
    test_index()
    test_at()
    test_set()
    test_add_column()
    test_rows_where()
    test_update()
    test_drop_row()
    test_drop_column()
    test_values()
    test_column()
    test_extract_rows()
    test_rows_where_not()
    test_rows_where_any()
    test_sort_values()
    test_dropna()
    

    print('------------------------------')
    print('The Register passed all tests!')
    print('------------------------------')