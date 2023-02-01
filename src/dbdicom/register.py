import math

class Register():
    """
    Replaces pandas dataframe with a tree structure.

    Designed to have largely the same interface as a pandas dataframe,
    so that the dataframe can easily be substituted. 

    The Register class was introduced mainly to overcome the critical 
    limitation that extending large pandas dataframes row by row is 
    prohibitively slow. This is problematic when the rows refer 
    to DICOM files that are created one by one, for instance when a 
    Series loops over images in a computation.

    The Register is created pragmatically to replace only the relevant 
    API of dataframes used in dbdicom.
    """

    def __init__(self, data:dict, columns:list):
        self._data = data
        self._cols = {}
        for i, col in enumerate(columns):
            self._cols[col] = i

    @property
    def empty(self):
        return self._data == {}

    @property
    def shape(self):
        return (len(self._data), len(self._cols))

    def columns(self):
        return list(self._cols.keys())

    def index(self):
        return list(self._data.keys())

    def at(self, index, column):
        return self._data[index][self._cols[column]]

    def set(self, index, column, value):
        self._data[index][self._cols[column]] = value

    def add_column(self, column='New column', value=None):
        self._cols[column] = len(self._cols)
        for index in self._data:
            self._data[index].append(value)

    def add_row(self, index, data):
        if len(data) != len(self._cols):
            msg = 'New row does not have the correct number of elements'
            raise ValueError(msg)
        self._data[index] = data

    def rows_where(self, column, value):
        """
        Return a view containing only those rows where column has value
        """
        if not isinstance(value, list):
            value = [value]
        reg = Register({}, self.columns())
        for index in self._data:
            if self.at(index, column) in value:
                reg._data[index] = self._data[index]
        return reg

    def rows_where_any(self, value):
        """
        Return a view containing only those rows where column has value
        """
        if not isinstance(value, list):
            value = [value]
        reg = Register({}, self.columns())
        for index in self._data:
            for column in self.columns():
                if self.at(index, column) in value:
                    reg._data[index] = self._data[index]
                    break
        return reg

    def rows_where_not(self, column, value):
        """
        Return a view containing only those rows where column does not have value
        """
        reg = Register({}, self.columns())
        for index in self._data:
            if self.at(index, column) != value:
                reg._data[index] = self._data[index]
        return reg


    def extract_rows(self, index=None, columns=None):
        """
        Return a view containing only those rows with index given
        """
        if index is None:
            index = self.index()
        if columns is None:
            reg = Register({}, self.columns())
            if not isinstance(index, list):
                reg._data[index] = self._data[index]
            else:
                for idx in index:
                    reg._data[idx] = self._data[idx]
            return reg
        else:
            if not isinstance(columns, list):
                columns = [columns]
            reg = Register({}, columns)
            if not isinstance(index, list):
                reg._data[index] = self.values(index, columns)
            else:
                for idx in index:
                    reg._data[idx] = self.values(idx, columns)
            return reg


    def update(self, register):
        self._data.update(register._data)

    def drop_row(self, index):
        try:
            del self._data[index]
        except:
            raise ValueError('The index ' + str(index) + ' does not exist.')

    def drop_column(self, column):
        icol = self._cols[column]
        for index in self._data:
            del self._data[index][icol]
        del self._cols[column]
        for col in self._cols:
            if self._cols[col] > icol:
                self._cols[col] -= 1


    def values(self, index=None, column=None):
        """Return a list of values in a given index and/or column"""
        try:
            if column is None:
                if index is None:
                    return [self._data[row] for row in self.index()]
                elif isinstance(index, list):
                    return [self._data[row] for row in index]
                else:
                    return self._data[index]
            elif isinstance(column, list):
                if index is None:
                    col = [self._cols[c] for c in column]
                    return [
                        [self._data[row][c] for c in col] 
                        for row in self._data
                    ]
                elif isinstance(index, list):
                    col = [self._cols[c] for c in column]
                    return [
                        [self._data[row][c] for c in col] 
                        for row in index
                    ]
                else: # single row
                    col = [self._cols[c] for c in column]
                    return [self._data[index][c] for c in col]
            else: # single column
                if index is None:
                    col = self._cols[column]
                    return [self._data[row][col] for row in self._data]
                elif isinstance(index, list):
                    col = self._cols[column]
                    return [self._data[row][col] for row in index]
                else:
                    return self.at(index, column)
        # Key error if column or value does not exist
        except: 
            return None
            
            
    def column(self, value):
        """In which column is value?
        
        If multiple -> returns the first one found.
        """
        for data in self._data.values():
            try:
                c = data.index(value)
            except:
                pass
            else:
                for col in self._cols:
                    if self._cols[col] == c:
                        return col

        # slower for long tables - needs to retrieve all values
        # for col in self._cols:
        #     if value in self.values(column=col):
        #         return col


    def sort_values(self, columns=None):
        """Return a view sorted by columns"""
        if columns is None:
            data = sorted(self._data.items(), 
                key = lambda item: item[1],
            )
        else:
            if not isinstance(columns, list):
                columns = [columns]
            cidx = [self._cols[c] for c in columns]
            data = sorted(self._data.items(), 
                key = lambda item: [item[1][i] for i in cidx],
            )
        self._data = dict(data) # This creates a copy
        #self._data.update(data) 


    def dropna(self):
        todelete = []
        for idx in self._data:
            vals = self._data[idx]
            if None in vals:
                todelete.append(idx)
                continue
            for v in vals:
                try:
                    if math.isnan(v):
                        todelete.append(idx)
                        break
                except:
                    pass
        for idx in todelete:
            del self._data[idx]
            


















