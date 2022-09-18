from dbdicom.record import DbRecord

class Database(DbRecord):

    def open(self, path):
        self.register.open(path)

    def close(self):
        self.register.close()

    def import_datasets(self, files):
        self.register.import_datasets(files)

    def export_datasets(self, records, database):
        uids = [rec.uid for rec in records]
        self.register.export_datasets(uids, database.register)



