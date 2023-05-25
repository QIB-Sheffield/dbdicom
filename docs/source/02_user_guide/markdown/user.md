# User interactions


`dbdicom` can be used in standalone scripts or interactively. To streamline integration in a GUI, communication with the user is performed via two dedicated attributes `status` and `dialog`. dialog and status attributes are available to any DICOM object. The status attribute is used to send messages to the user, or update on progress of a calculation:

```python
series.message("Starting calculation...")
```

When operating in command line mode this will print the message to the terminal. If `dbdicom` is used in a compatible GUI, this will print the same message to the status bar. Equivalently, the user can be updated on the progress of a calculation via:

```python
for i in range(length):
    series.progress(i, length, 'Calculating..)
```

This will print the message with a percentage progress at each iteration. When used in a GUI, this will update the progress bar of the GUI. 

By default a dbdicom record will always update the user on progress of any calculation. When this beaviour is undersired, the record can be muted as in via `series.mute()`. After this the user will no longer recieve updates. In order to turn messages back on, unmute the record via `series.unmute()`.

Dialogs can be used to send messages to the user or prompt for input. In some cases a dialog may halt the operation of te program until the user has performed the appropriate action, such as hitting enter or entering a value. In command line operator or scripts the user will be prompted for input at the terminal. When using in a GUI, the user will be prompted via a pop-up:

```python
series.dialog.question("Do you wish to proceed?", cancel=True)
```

When used in a script, this will ask the user to enter either "y" (for yes), "n" (for no) or "c" (for cancel) and the program execution will depend on the answer. When the scame script is deployed in a GUI, the question will be asked via a pop-up window and a button push to answer. A number of different dialogs are available via the dialog attribute (see reference guide).