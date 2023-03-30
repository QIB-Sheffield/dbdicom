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
    
    def new_sibling(self, suffix=None, **kwargs):
        if suffix is not None:
            desc = self.instance().PatientName 
            kwargs['PatientName'] = desc + ' [' + suffix + ']'
        return self.parent().new_child(**kwargs)

    def _copy_from(self, record, **kwargs):
        attr = {**kwargs, **self.attributes}
        uids = self.manager.copy_to_patient(record.uid, self.key(), **attr)
        if isinstance(uids, list):
            return [self.record('Study', uid) for uid in uids]
        else:
            return self.record('Study', uids)



