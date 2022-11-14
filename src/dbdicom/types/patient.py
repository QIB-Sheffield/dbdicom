from dbdicom.record import DbRecord

class Patient(DbRecord):

    name = 'PatientID'

    def new_study(self, **kwargs):
        attr = {**kwargs, **self.attributes}
        uid, key = self.manager.new_study(parent=self.uid, key=self.key(), **attr)
        return self.record('Study', uid, key, **attr)

    def parent(self):
        return self.record('Database')

    def children(self, **kwargs):
        return self.studies(**kwargs)

    def new_child(self, dataset=None, **kwargs): 
        attr = {**kwargs, **self.attributes}
        return self.new_study(**attr)

    def _copy_from(self, record):
        uids = self.manager.copy_to_patient(record.uid, self.key(), **self.attributes)
        if isinstance(uids, list):
            return [self.record('Study', uid) for uid in uids]
        else:
            return self.record('Study', uids)

