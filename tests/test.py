class Dummy:
    def __init__(self, data:dict):
        self._data = data

    @property
    def shape(self):
        return len(self._data)


def extract_rows(register, index):
    """
    Return a view containing only those rows with index given.
    """
    reg = Dummy({})
    for idx in index:
        reg._data[idx] = register._data[idx]
    return reg


def test_extract_2rows():
    data = {'X': 1, 'Y': 2, 'Z': 3}
    reg = Dummy(data)
    extract_rows(reg, ['X','Y'])


def test_extract_1rows():
    reg = Dummy({})
    assert reg.shape == 0


test_extract_2rows()
test_extract_1rows()


