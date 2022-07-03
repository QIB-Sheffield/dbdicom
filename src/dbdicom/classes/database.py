from .record import Record


class Database(Record):

    def __init__(self, folder, UID=[], **attributes):
        super().__init__(folder, UID=[], generation=0, **attributes) 