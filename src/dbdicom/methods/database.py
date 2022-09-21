from dbdicom.record import DbRecord

class Database(DbRecord):

    def open(self, path):
        self.manager.open(path)

    def close(self):
        self.manager.close()

    def import_datasets(self, files):
        self.manager.import_datasets(files)

    def export_datasets(self, records, database):
        uids = [rec.uid for rec in records]
        self.manager.export_datasets(uids, database.manager)



