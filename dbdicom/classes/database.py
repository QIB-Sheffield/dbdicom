from .record import Record


class Database(Record):

    def __init__(self, folder, UID=[]):
        super().__init__(folder, UID=[], generation=0) 