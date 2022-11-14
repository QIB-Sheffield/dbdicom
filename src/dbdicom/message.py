import os

class StatusBar():
    """
    Class with the same interface as StatusBar for use outside weasel.
    """

    def __init__(self):
        self._message = ''
        self.muted = False # Needs adding in wezel status as well.

    def mute(self):
        self.muted = True
    def unmute(self):
        self.muted = False

    def hide(self):
        pass

    def message(self, message):
        self._message = message
        if not self.muted:
            print(message)
        
    def progress(self, value, maximum, message=None):
        if message is not None:
            self._message = message
        if not self.muted:
            perc = str(round(100*value/maximum))
            print(self._message + ' [' + perc + ' %]')
        

class Dialog():
    """Class with the same interface as widgets.Dialog for commandline operation"""

    def information(self, message="Message in the box", title=""):
        """
        Information message. 
        """
        print("INFORMATION")
        if title != "": print(title)
        print(message)

    def warning(self, message="Message in the box", title=""):
        """
        Warning message. 
        """
        print("WARNING!")
        if title != "": print(title)
        print(message)

    def error(self, message="Message in the box", title=""):
        """
        Error message.
        """
        print("ERROR!")
        if title != "": print(title)
        print(message)

    def directory(self, message='Please provide a folder', datafolder=None):
        """
        Select a directory.
        """
        print(message)
        path = input()
        while not os.path.exists(path):
            reply = self.question( 
                title = 'Error!', 
                message = path + ' does not exist. Select another?')
            if reply == "Yes":
                print(message)
                path = input()
            elif reply == 'No':
                return None
        return path

    def question(self, message="Do you wish to proceed?", title="Question", cancel=False):
        """
        Displays a question window in the User Interface.
        
        The user has to click either "OK" or "Cancel" in order to continue using the interface.
        Returns False if reply is "Cancel" and True if reply is "OK".
        """
        if cancel:
            instructions = "y = yes, n = no, c = cancel"
            options = ['y', 'n', 'c']
        else:
            instructions = "y = yes, n = no, c = cancel"
            options = ['y', 'n']
        print(title)
        print(message)
        print(instructions)
        answer = input()
        while answer not in options:
            print("Sorry, I can't interpret that answer")
            print(message)
            print(instructions)
            answer = input()
        if answer == 'y': return "Yes"
        if answer == 'n': return "No"
        if answer == 'c': return "Cancel"

    def file_to_open(self, title='Open file..', initial_folder=None, extension="All files (*.*)", datafolder=None):
        """
        Select a file to read.
        """
        pass

    def file_to_save(self, title='Save as ...', directory=None, filter="All files (*.*)", datafolder=None):
        """
        Select a filename to save.
        """
        pass

    def input(self, *fields, title="User input window"):
        """
        Collect user input of various types.
        """
        pass
