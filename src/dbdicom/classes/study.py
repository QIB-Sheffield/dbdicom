from .record import Record

class Study(Record):

    def __init__(self, folder, UID=[], **attributes):
        
        super().__init__(folder, UID, generation=2, **attributes)

    def label(self, row=None):

        if row is None:
            data = self.data()
            if data.empty: return "New Study"
            file = data.index[0]
            descr = data.at[file, 'StudyDescription']
            date = data.at[file, 'StudyDate']
        else:
            descr = row.StudyDescription
            date = row.StudyDate

        label = str(descr)
        label += ' [' + str(date) + ']'
        return label