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



