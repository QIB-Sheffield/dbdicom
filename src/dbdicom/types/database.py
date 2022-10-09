from dbdicom.record import DbRecord

class Database(DbRecord):

    def open(self, path):
        self.manager.open(path)

    def close(self):
        return self.manager.close()

    def scan(self):
        self.manager.scan()

    def import_dicom(self, files):
        self.manager.import_datasets(files)

    def zeros(*args, **kwargs):
        return zeros(*args, **kwargs)

def zeros(database, shape, dtype='mri'):
    study = database.new_study()
    return study.zeros(shape, dtype=dtype)



