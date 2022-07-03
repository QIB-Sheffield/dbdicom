
from .record import Record

class Patient(Record):

    def __init__(self, folder, UID=[], **attributes):
        super().__init__(folder, UID, generation=1, **attributes)


    def label(self, row=None):

        if row is None:
            data = self.data()
            if data.empty: return "New Patient"
            file = data.index[0]
            name = data.at[file, 'PatientName']
            id = data.at[file, 'PatientID']
        else:
            name = row.PatientName
            id = row.PatientID
            
        label = str(name)
        label += ' [' + str(id) + ']'
        return label

