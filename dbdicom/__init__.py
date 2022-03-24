"""

`dbdicom` is a Python interface for reading and writing DICOM databases. 


# Browsing a DICOM folder

### Reading and opening a DICOM folder

Open a DICOM database in a given folder, 
read it and print a summary of the content:

```python
from dbdicom import Folder

folder = Folder('C:\\Users\\MyName\\MyData\\DICOMtestData')
folder.open()
folder.print()
```

The first time the folder is read this will be relatively slow. 
This is because each individual DICOM file in the folder 
is read and summarised in a table (csv file). 
If the folder is reopened again later, 
the table can be read directly and opening will be much faster. 

Use `scan()` to force a rereading of the folder. This may 
be of use when files have become corrupted, 
or have been removed/modified by external applications:

```ruby
folder.scan()
```

After making changes to the DICOM data, the folder should be closed 
properly so any changes can be either saved or rolled back as needed:

```ruby
folder.close()
```

If unsaved changes exist, `close()` will prompt the user to either save or restore to 
the last saved state.

### Retrieving objects from the folder

A DICOM database has a hierarchical structure. 
The files are *instances* of a specific DICOM class and correspond to real-world 
objects such as images or regions-of-interest. Instances are grouped into a *series*, 
and multiple series are grouped into *studies*. Typically a study consist of all the data 
derived in a single examination of a subject. Studies are grouped into *patients*, 
which correspond to the subjects the study is performed upon. 
A *patient* can be an actual patient, but can also be a healthy volunteer, an animal,
a physical reference object, or a digital reference object.

To return a list of all patients, studies, series or instances in the folder: 

```ruby
instances = folder.instances()
series = folder.series()
studies = folder.studies()
patients = folder.patients()
```

The same functions can be used to retrieve the children of 
a certain parent object. For instance, 
to get all studies of a patient:

```ruby
studies = patient.studies()
```

Or all series under the first of those studies:

```ruby
series = studies[0].series()
```

Or all instances of a study:

```ruby
instances = study.instances()
```

And so on for all other levels in the hierarchy. 
Individual objects can also be access directly using 
indices. For instance to retrieve the first instance in the folder:

```ruby
instance = folder.instances(0)
```

These can be chained together for convencience, 
e.g. to get all instances instance of series 5 in study 1 of patient 2:

```ruby
instance = folder.patients(2).studies(1).series(5).instances()
```

These functions also work to find objects higher up in the hierarchy. 
For instance, to find the patient of a given series:

```ruby
patient = series.patients()
```

In this case the function will return a single object rather than a list.

### Finding DICOM objects in the folder

Each DICOM file has a number of attributes describing the properties
of the object. Examples are PatientName, StudyDate, etc. 
A full list of attributes for specific objects can be found here: 
https://dicom.innolitics.com/. 

Each known attribute is identified most easily by a keyword, 
which has a capitalised notation. Objects in the folder 
can be can also be listed by searching on any DICOM tag:

```ruby
instances = folder.instances(PatientName = 'John Dory')
```

This will only return the instances for patient John Dory. 
Objects can also be searched on multiple DICOM tags:

```ruby
series = folder.instances(
    PatientName = 'John Dory', 
    ReferringPhysicianName = 'Dr. No', 
)
```

In this case objects are only returned if both conditions are fullfilled. 
Any arbitrary number of conditions can be entered, and 
higher order objects can be found in the same way:

```ruby
studies = folder.studies(
    PatientName = 'John Dory', 
    ReferringPhysicianName = 'Dr. No', 
)
```

*TO DO* In addition to filtering, the results can also be sorted by attribute:

```ruby
studies = folder.studies(
    sortby = 'StudyDate', 
    PatientName = 'John Dory', 
)
```

In this case the resulting studies will appear in the list in order of Study Date. 
Sorting can also be done based on two or more attributes:

```ruby
studies = folder.studies(
    sortby = ['PatientName', 'StudyDate', 'StudyDescription']
)
```

In this case the result will be a 3-dimensional list. 
For instance to access all studies of patient 3 do:

```ruby
studies[3][:][:]
```

As an alternative to calling explicit object types, 
you can call `children()` and `parent` to move through the hierarchy:

```ruby
studies = patient.children()
patient = studies[0].parent
```

The same convenience functions are available, 
such as using an index or searching by keywords:

```ruby
studies = patient.children(ReferringPhysicianName = 'Dr. No')
study = patient.children(0)
```

### Moving and removing objects

To remove an object from the folder, call `remove()` on the object:

```ruby
study.remove()
instance.remove()
```

remove() can  be called on Patient, Study, Series or Instances.

Moving an object to another parent can be done with `move_to()`
For instance to move a study from one patient to another:

```ruby
study = folder.patients(0).studies(0)
new_parent = folder.patients(1)
study.move_to(new_parent)
```

Objects can also be moved to objects higher up in the hierarchy.
Any missing parents will be automatically created. For instance:

```ruby
instance = folder.instances(0)
study = folder.studies(1)
instance.move_to(study)
```

This will move *instance* from its current parent series to *study*. 
Since no new parent series under *study* has been provided, 
a new series will be created under *study* and used as a parent for *instance*.


### Copying and creating objects

A DICOM object can be copied by calling `copy()`: 

```ruby
study = folder.patients(0).studies(0)
new_study = study.copy()
```

This will create a copy of the object in the same parent object, 
i.e. `study.copy()` in the example above has created a new study in patient 0.
This can be used for instance to copy-paste a study from one patient to another: 

```ruby
study = folder.patients(0).studies(0)
new_parent = folder.patients(1)
study.copy().move_to(new_parent)
```

This is equivalent to using `copy_to()`:

```ruby
study.copy_to(new_parent)   
```

To create a new object, call `new_child()` on the parent:

```ruby
series = study.new_child()
```

*series* will now be a new (empty) series under *study*.


### Export and import

To export an object out of the folder to an external folder, 
call `export()` on any dicom object with the export path as argument:

```ruby
series.export(path)
```

If no path is given then the user will be asked to select one.

*TO DO* Equivalently to import DICOM files from an external folder,
call `import()` with a list of files:

```ruby
folder.import(files)
```



# Creating and modifying DICOM files



### Reading DICOM attributes

An object's DICOM attributes can be read by using the DICOM keyword of the attribute:

```ruby
dimensions = [instance.Rows, instance.Columns]
```

All attributes can also be accessed at series, study, patient or folder level. 
In this case they will return a single value taken from their first instance.

```ruby
rows = folder.patient(0).series(0).Rows
```

To print the Rows for all instances in the series, iterate over them:

```ruby
for instance in series.instances():
    print(instance.Rows)
```

DICOM attributes can also be accessed using the list notation, 
using either the keyword as a string or a (group, element) pair.

```ruby
columns = instance['Columns']
columns = instance[(0x0028, 0x0010)]
```

The tags can also be accessed as a list, for instance:

```ruby
dimensions = ['Rows', (0x0028, 0x0010)]
dimensions = instance[dimensions] 
```

This will return a list with two items. As shown in the example,
the items in the list can be either KeyWord strings or (group, element) pairs. 
This also works on higher-level objects:

```ruby
dimensions = ['Rows', (0x0028, 0x0010)]
dimensions = patient[dimensions] 
```

As for single KeyWord attributes this will return one list
taken from the first instance of the patient.


### Editing attributes


DICOM tags can be modified using the same notations:

```ruby
instance.EchoTime = 23.0
```

or also:

```ruby
instance['EchoTime'] = 23.0
```

or also:

```ruby
instance[(0x0018, 0x0081)] = 23.0
```

Multiple tags can be inserted in the same line:

```ruby
shape = ['Rows', 'Columns']
instance[shape] = [128, 192]
```

When setting values in a series, study or patient, 
all the instances in the object will be modified. 
For instance, to set all the Rows in all instances of a series to 128:

```ruby
series.Rows = 128
```

This is shorthand for:

```ruby
for instance in series.instances():
     instance.Rows = 128
```



### Read and write

By default all changes to a DICOM object are made on disk. 
For instance if a DICOM attribute is changed

```ruby
instance.Rows = 128
```

The data are read from disk, the change is made, the data are
written to disk again and memory is cleared. 
Equally, if a series is copied to another study, all 
its instances will be read, any necessary changes made,
and then written to disk and cleared from memory. 

For many applications reading and writing from disk is too slow. 
For faster access at the cost of some memory usage, the data
can be read into memory before performing any manipulations:

```ruby
series.read()
```

After this all changes are made in memory *only*. 
At any point the changes can be written out again 
by calling `write()`:

```ruby
series.write()
```

This will still retain the data in memory for an further editing.
In order to delete them from memory and free up the space, call `clear()`:

```ruby
series.clear()
```

After calling `clear()`, all subsequent changes are made to disk again.
These operations can be called on patients, studies, series or instances. 


### Save and restore

All changes made in a DICOM folder are reversible until they are saved.
To save all changes, use `save()`:

```ruby
folder.save()
```

This will permanently burn all changes that are made on disk. 
Changes that are only made in memory will *not* be saved in this way. 
In order to save all changes including this that are made in memory, 
make sure to call `write()` first. These commands can also be piped for convenience:

```ruby
folder.write().save()
```

In order to reverse any changes made, use `restore()` to revert back to the last saved state:

```ruby
folder.restore()
```

This will roll back all changes on disk to the last changed state. 
As for `save()`, changes made in memory alone will not be reversed. 
In order to restore all changes in memory as well, read the data again after restoring:

```ruby
folder.restore().read()
```

This will read the entire folder in meomory, which is not usually appropriate. 
However, `save()` and `restore()` can also be called at the level of individual objects:

```ruby
series.restore()
```

will reverse all changes made since the last save, but only for this series. 
Equivalently:

```ruby
series.save()
```

will save all changes made in the series permanently. 


### DICOM Classes

Each DICOM file in a folder holds and instance of a DICOM class, which in turn 
represents an object in the real world such as an MR image, or an image co-registration,
an ECG, etc. The [innolitics DICOM browser](https://dicom.innolitics.com/) shows all possible DICOM Classes
in an easily searchable manner. 

In `dbdicom` such DICOM classes are represented by a separate python class. 
When an instance or list of instances are generated, for instance through: 

```ruby
instances = series.instances()
```

then each instance is automatically returned as an instance of the appropriate class. 
As an example, if the first instance of the series represents an MR Image, 
then `instances[0]` will be an instance of the class "MRImage", which on itself 
inherits functionality from a more general "Image" class. 
This means `instance[0]` automatically has access to functionality relevant for images, 
such as:

```ruby
array = instances[0].array()
```

this will return a 2D numpy array holding the pixel data, 
and will automatically correct for particular MR image issues such 
as the use of private rescale slopes and intercepts for Philips data. 

Other relevant functionality is explained in the reference guide of the individual classes.
At the moment the DICOM classes are very limited in scope, 
but this will be extended over time as needs arise in ongoing projects.


### Creating DICOM files from scratch


*TO DO* DICOM data can be created from scratch by instantiating one of the 
DICOM classes: 

```ruby
new_image = MRImage()
```

This will create an MRI image with empty pixel data. Since no parent series/study/patient are 
provide, defaults will be automatically created. At this point 
the image will only exist in memory but can be edited in the usual way. 
For instance to assign pixel data based on an empty numpy array:

```ruby
array = numpy.zeros(128, 128)
new_image.set_array(array)
```

In order to save the image to disk an instance of the folder class needs to be provided. 
This can point to an empty folder, or to an existing DICOM database where the new data will be added:

```ruby
new_image.folder = Folder('C:\\Users\\MyName\\MyData\\New Folder')
```

After setting a folder, the image can be written to disk:

```ruby
new_image.write()
```

An instance can also be read from a single file:

```ruby
image = MRImage('C:\\Users\\steve\\Data\\my_dicom_file.ima')
```

Changes to the file can then be made as usual:

```ruby
image.PatientName = 'John Dory'
image.array = numpy.zeros((128,128)
```

and then saved as `image.write()`. When used in this way the class is just a simple 
wrapped for a `pydicom` dataset.


# User interactions


`dbdicom` can be used in standalone scripts or at command line, to streamline
integration in a GUI, communication with the user should be performed 
via two dedicated attributes `status` and `dialog`. 
dialog and status attributes are available to the folder class, and to any DICOM object.

The status attribute is used to send messages to the user, or update on progress of a calculation:

```ruby
series.status.message("Starting calculation...")
```

When operating in command line mode this will simply print the message to the terminal.
If `dbdicom` is used in a GUI, this will print the same message to the status bar.
Equivalently, the user can be updated on the progress of a calculation via:

```ruby
series.status.message("Calculating..")
for i in range(length):
    series.status.progress(i, length)
```

This will print the message with a percentage progress at each iteraion. 
When used in a GUI, this will update the porgress bar of the GUI. For use in a GUI,
it is required to reset the progress bar after exiting the loop:

```ruby
series.status.hide()
```

When operating in command line, this statement does nothing, but it makes the 
pipeline ready to be deloyed in a GUI without modification.

In addition, dialogs can be used to send messages to the user or prompt for input.
In some cases a dialog may halt the operation of te program until the user 
has performed the appropriate action, such as hitting enter or entering a value. 
In command line operator or scripts the user will be prompted for input at the terminal. 
When using in a GUI, the user will be prompted via a pop-up. Example:

```ruby
series.dialog.question("Do you wish to proceed?", cancel=True)
```

When used in a script, this will ask the user to enter either 
"y" (for yes), "n" (for no) or "c" (for cancel) and the program execution will
depend on the answer. When the scame script is deployed in a GUI, the question
will be asked via a pop-up window and a button push to answer. 
A number of different dialogs are available via the dialog attribute (see reference guide). 


# About ***dbdicom***

## Why ***dbdicom***?

DICOM is scary. *And* it has been the universally accepted standard for medical images for decades. 
Why is that? It is *because* it is scary. DICOM is extremely detailed and rigorous in the description of its 
terminology and structure. It has to be, because DICOM deals with the most complex and sensitive data possible: your 
medical history. All of it. Every single one of your DICOM images in a clinical archive contains 
the key to access all of your medical details. This allows doctors to link your images to your blood tests, 
family history, previous diagnosis treatments, other imaging, and so on. And this is important to make 
the best possible informed decisions when it comes to your health. 

In medical imaging research this additional information is often seen as a nuisance and discarded prior to 
processing of the images. Typically a data array of some sort is extracted, perhaps also some key geometrical descriptors such as 
pixel sizes or a transformation matrix, and all the other information is ignored. 

Conversion into a *lossy* data format is often sufficient for method development or basic scientific research, 
but when it comes to deploying these methods in clinical studies, 
all this additional (discarded) information is just as important as in clinical practice. 
It ensures that all derived data are properly traceable to the source, 
and can be compared between subjects and within a subject over time. 
It allows to test for instance whether a new (expensive) imaging method provides an *additive* benefit 
over and above (cheap) data from medical history, clinical exams or blood tests. 
And so, if we accept that new image analysis methods ultimately will need to be tested clinically 
(and ideally sooner rather than later), then we simply can't avoid the need to convert results back to DICOM. 
In practice this step often requires a major rewrite of image processing pipelines set up for basic research, 
creating a major barrier to clinical translation of new methods. 

Quantitative imaging is another area where the information discarded by conversion to lossy formats is important. 
Quantification involves the application of complex signal models to multi-dimensional imaging 
data that are acquired by varying contrast parameters such as (in MRI) echo times, b-values, 
gradient directions, inversion times, flip angle etc. Often many of these are varied at the same 
time, and not necessarily in some clean incremental order -  as in MR fingerprinting. The models that interpret these data 
need access to this information, and often also need to encode it alongside the images that 
are produced. When images are first converted from DICOM into some lossy data format, 
this is often not possible because the application was not foreseen when the lossy format was first defined. 
This then requires ad-hoc solutions retaining part of the 
original DICOM information in unstructured free text fields or separate newly defined header files. 

All these problems can be solved, for current and any imaginable or unimaginable future applications, 
by dropping conversions into lossy image formats and simply reading from DICOM and writing to DICOM. 

If only DICOM wasn't so scary!!

## What is ***dbdicom***?

`dbdicom` is a programming interface that makes reading and writing DICOM data intuitive 
for the practicing medical imaging scientist working in Python. 
We promise you won't even know it's DICOM. In fact the documentation 
hardly even mentions DICOM at all. It will certainly not mention things like composite information 
object definitions, application entities, service-object pairs, unique identifiers, etc etc. 
This is the language of DICOM, and it's confusing in part because the concepts date back to 
the 1970's and 1980's when the standard was developed. But then again, that is exactly what you 
would expect from a successful standard. It doesn't change. It shouldn't change. 

`dbdicom` wraps around DICOM using a language and code structure that is native to the 2020's. 
It allows you to develop your medical imaging methods using DICOM files only, which 
means your prototypes of new analysis methods can be deployed in clinical trials just like that. 
It also means that any result you generate can easily be integrated in open access DICOM databases 
and can be visualised along with any other images of the same subject 
by anyone with a DICOM viewer (i.e. literally anyone).

Since `dbdicom` is primarily a development tool, it can be used from command line or to write stand-alone scripts. 
However, since `dbdicom` is all about facilitating translation into clinical trials and ultimately 
clinical practice, all scripts written in `dbdicom` are set up for deployment in a graphical user interface. 
Convenience classes are provided for user interaction that print to a terminal when used 
in a script, but will automatically generate pop-up windows or progress bars when the same script is 
deployed inside a `dbdicom` compatible graphical user interface. 

## Acknowledgements

`dbdicom` relies heavily on `pydicom` for basic read/write of DICOM files, 
with some additional features provided by `nibabel` and `dcm4che`. 
Graphical user interface compatibility is provided by `PyQt5`. 
Documentation is generated by `pdoc3`. Basic image manipulation is provided by `numpy` and `scipi`,
 and sorting and tabulating of data by `pandas`. Export to other formats is provided by `matplotlib`.

"""

# do not show in documentation
__pdoc__ = {}
__pdoc__["utilities"] = False 
__pdoc__["external"] = False 
__pdoc__["dicm"] = False 

from .folder import *