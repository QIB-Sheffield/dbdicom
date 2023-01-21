## Installation
Run `pip install dbdicom`.


# Summary

The DICOM format is the universally recognised standard for medical imaging, but reading and writing DICOM data remains a challenging task for most data scientists. 

The excellent python package `pydicom` is very commonly used and well-supported, but it is limited to reading and writing individual files, and still requires a fairly high level of understanding of DICOM to ensure compliance with the standard. 

`dbdicom` wraps around `pydicom` to provide an intuitive programming interface for reading and writing data from entire DICOM databases, replacing unfamiliar DICOM-native concepts by language and notations that will be more familiar to data scientists. 

The sections below list some basic uses of `dbdicom`. The package is currently deployed in several larger scale multicentre clinical studies led by the authors, such as the [iBEAt study](https://bmcnephrol.biomedcentral.com/articles/10.1186/s12882-020-01901-x) and the [AFiRM study](https://www.uhdb.nhs.uk/afirm-study/). The package will continue to be shaped through use in these studies and we expect it will attain a more final form when these analysis pipelines are fully operational.


# Browsing a DICOM folder

### Reading and opening a DICOM database

Open a DICOM database in a given folder, and print a summary of the content:

```python
import dbdicom as db

database = db.database('C:\\Users\\MyName\\MyData\\DICOMtestData')
database.print()
```

The first time the database is opened this will be relatively slow because all files need to be read and summarized. If the folder is reopened again later, the table can be read directly and opening will be much faster. 

Use `scan()` to force a rereading of the database. This may be of use when files have become corrupted, or have been removed/modified by external applications:

```python
database.scan()
```

After making changes to the DICOM data, the folder should be closed 
properly so any changes can be either saved or rolled back as needed:

```python
database.close()
```

If unsaved changes exist, `close()` will prompt the user to either save or restore to the last saved state.

### Retrieving objects from the folder

A DICOM database has a hierarchical structure. 

```
database/
|
|---- Patient 1/
|    |
|    |---- Study 1/
|    |     |
|    |     |---- Series 1/
|    |     |    |----Instance 1
|    |     |    |----Instance 2
|    |     |    |----Instance 3
|    |     |    
|    |     |----Series 2/
|    |    
|    |---- Study 2/
|
|---- Patient 2/  
| 
```

A *patient* can be an actual patient but can also be a healthy volunteer, an animal, a physical reference object, or a digital reference object. Typically a *study* consist of all the data derived in a single examination of a subject. A *series* usually represents and individual examination in a study, such an MR sequence. The files contain the data and are *instances* of real-world objects such as images or regions-of-interest. 

To return a list of all patients, studies, series or instances in the folder: 

```python
instances = database.instances()
series = database.series()
studies = database.studies()
patients = database.patients()
```

The same functions can be used to retrieve the children of a certain parent object. For instance, 
to get all studies of a patient:

```python
studies = patient.studies()
```

Or all series under the first of those studies:

```python
series = studies()[0].series()
```

Or all instances of a study:

```python
instances = study.instances()
```

And so on for all other levels in the hierarchy. These functions also work to find objects higher up in the hierarchy. For instance, to find the patient of a given series:

```python
patient = series.patients()
```

In this case the function will return a single item.

### Finding DICOM objects in the folder

Each DICOM file has a number of attributes describing the properties of the object. Examples are PatientName, StudyDate, etc. A convenient list of attributes for specific objects can be found [here](https://dicom.innolitics.com/):

Each known attribute is identified most easily by a keyword, which has a capitalised notation. Objects in the folder can be can also be listed by searching on any DICOM tag:

```python
instances = database.instances(PatientName = 'John Dory')
```

This will only return the instances for patient John Dory. This also works with multiple DICOM tags:

```python
series = database.instances(
    PatientName = 'John Dory', 
    ReferringPhysicianName = 'Dr. No', 
)
```

In this case objects are only returned if both conditions are fullfilled. Any arbitrary number of conditions can be entered, and higher order objects can be found in the same way:

```python
studies = database.studies(
    PatientName = 'John Dory', 
    ReferringPhysicianName = 'Dr. No', 
)
```

As an alternative to calling explicit object types, you can call `children()` and `parent` to move through the hierarchy:

```python
studies = patient.children()
patient = studies[0].parent
```

The same convenience functions are available, such as searching by keywords:

```python
studies = patient.children(ReferringPhysicianName = 'Dr. No')
```

### Moving and removing objects

To remove an object from the folder, call `remove()` on the object:

```python
study.remove()
instance.remove()
```

remove() can  be called on Patient, Study, Series or Instances.

Moving an object to another parent can be done with `move_to()`. For instance to move a study from one patient to another:

```python
study = folder.patients()[0].studies()[0]
new_parent = folder.patients()[1]
study.move_to(new_parent)
```


### Copying and creating objects

Any object can be copied by calling `copy()`: 

```python
study = folder.patients()[0].studies()[0]
new_study = study.copy()
```

This will create a copy of the object in the same parent object, i.e. `study.copy()` in the example above has created a new study in patient 0. This can be used for instance to copy-paste a study from one patient to another: 

```python
study = folder.patients()[0].studies()[0]
new_parent = folder.patients()[1]
study.copy().move_to(new_parent)
```

This is equivalent to using `copy_to()`:

```python
study.copy_to(new_parent)   
```

Instead of copying, and object can also be moved:

```python
study.move_to(new_parent)   
```

To create a new object, call `new_child()` on the parent:

```python
series = study.new_child()
```

*series* will now be a new (empty) series under *study*. This can also be written more explicitly for clarity:

```python
series = study.new_series()
```

And equivalently for `new_patient`, `new_study` and `new_instance`. New sibling objects under the same parent can be created by:

```python
new_series = series.new_sibling()
```

here `new_series` will be a series under the same study as `series`. Objects higher up the hierarchy can be created using `new_pibling` (i.e. sibling of the parent):

```python
new_study = series.new_pibling()
```

This is shorthand for:

```python
new_study = series.parent().new_sibling()
```

When new objects are created, they can be assigned properties up front, for instance:

```python
new_study = series.new_pibling(
    StudyDescription='Parametric maps',
    StudyDate = '12.12.22')
```

This will ensure that all data that appear under the new study will have these attributes. 


### Export and import

To import DICOM files from an external folder, call `import_dicom()` on a database with a list of files:

```python
database.import_dicom(files)
```

To export dicom datasets out of the folder to an external folder, call `export_as_dicom()` on any dicom object with the export path as argument:

```python
series.export_as_dicom(path)
```

Exporting in other formats is similar:

```python
study.export_as_csv(path)
study.export_as_nifti(path)
study.export_as_png(path)
```

The pixel data from a series can also be exported in numpy format:

```python
series.export_as_npy(path)
```

This exports the array in dimensions `(n,x,y)` where `n` enumerates the images and `x,y` are the pixels. To export in different dimensions use the `sortby` keyword with one or more DICOM tags:

```python
series.export_as_npy(path, sortby=['SliceLocation','AcquisitionTime'])
```

This exports an array with dimensions `(z,t,n,x,y)` sorted by slice location and acquisition time.


# Creating and modifying DICOM files

### Reading DICOM attributes

An object's DICOM attributes can be read by using the DICOM keyword of the attribute:

```python
nr_of_rows = instance.Rows
```

All attributes can also be accessed at series, study, patient or folder level. In this case they will return a list of unique values. For instance to return a list with all distinct series descriptions in a study:

```python
desc = study.SeriesDescription
```

DICOM attributes can also be accessed using the list notation, using either the keyword as a string or a (group, element) pair:

```python
columns = instance['Columns']
columns = instance[(0x0028, 0x0010)]
```

The tags can also be accessed as a list, for instance:

```python
dimensions = ['Rows', (0x0028, 0x0010)]
dimensions = instance[dimensions] 
```

This will return a list with two items. As shown in the example, the items in the list can be either KeyWord strings or (group, element) pairs. This also works on higher-level objects:

```python
dimensions = ['Rows', (0x0028, 0x0010)]
dimensions = patient[dimensions] 
```


### Editing attributes


DICOM tags can be modified using the same notations:

```python
instance.EchoTime = 23.0
```

or also:

```python
instance['EchoTime'] = 23.0
```

or also:

```python
instance[(0x0018, 0x0081)] = 23.0
```

Multiple tags can be inserted in the same line:

```python
shape = ['Rows', 'Columns']
instance[shape] = [128, 192]
```

When setting values in a series, study or patient, all the instances in the object will be modified. For instance, to set all the Rows in all instances of a series to 128:

```python
series.Rows = 128
```

### Custom attributes

Apart from the predefined public and private DICOM keywords, `dbdicom` also provides a number of custom attributes for more convenient access to higher level properties. In order to distinguish these from existing DICOM attributes which are defined in `CamelCase`, the custom attributes follow the `lower_case` notation. 

For instance, to set one of the standard [matplotlib color maps](https://matplotlib.org/stable/tutorials/colors/colormaps.html), you can do:

```python
image.colormap = 'YlGnBu'
series.colormap = 'Oranges'
```

and so on.. The colormaps can be retrieved the same way:

```python
cm_image = image.colormap
cm_series = series.colormap
```

As for standard DICOM attributes this returns a list if unique values for the series. 

Custom attributes can easily be added to any DICOM dataset type and the number of available attributes is set to grow as the need arises.


### Read and write

By default all changes to a database are made on disk. For instance if a DICOM attribute is changed

```python
instance.Rows = 128
```

The data are read from disk, the change is made, the data are written to disk again and memory is cleared. Equally, if a series is copied to another study, all its instances will be read, any necessary changes made, and then written to disk and cleared from memory. 

For many applications reading and writing from disk is too slow. For faster access at the cost of some memory usage, the data can be read into memory before performing any manipulations:

```python
series.read()
```

After this all changes are made in memory. To clear the data from memory and continue working from disk, use `clear()`:


```python
series.clear()
```

These operations can be called on the entire database, on patients, studies, series or instances. 


### Save and restore

All changes made in a DICOM folder are reversible until they are saved.
To save all changes, use `save()`:

```python
database.save()
```

This will permanently burn all changes that are made on disk. In order to reverse any changes made, use `restore()` to revert back to the last saved state:

```python
database.restore()
```

This will roll back all changes on disk to the last changed state. `save()` and `restore()` can also be called at the level of individual objects:

```python
series.restore()
```

will reverse all changes made since the last save, but only for this series. Equivalently:

```python
series.save()
```

will save all changes made in the series (but not other objects in the database) permanently. 


### Working with series

A DICOM series typically represents images that are acquired together, such as 3D volumes or time series. Some dedicated functionality exists for series that is not relevant for objects elsewhere in the hierarchy. 

To extract the images in a series as a numpy array, use `array()`:

```python
array, _ = series.array()
```

This will return an array with dimensions `(n,x,y)` where `n` enumerates the images in the series. The array can also be returned with other dimensions:

```python
array, _ = series.array(['SliceLocation', 'FlipAngle'])
```

This returns an array with dimensions `(z,t,n,x,y)` where `z` corresponds to slice locations and `t` to flip angles. The 3d dimension `n` enumerates images at the same slice location and flip angle. Any number of dimensions can be added in this way. If an application requires the pixels to be listed first, use the `pixels_first` keyword:

```python
array, _ = series.array(['SliceLocation', 'FlipAngle'], pixels_first=True)
```

In this case the array has dimensions `(x,y,z,t,n)`. Replacing the images of a series with a given numpy array works the same way:

```python
series.array(array)
```

The function `array()` also returns the header information for each slice in a second return value:

```python
array, header = series.array(['SliceLocation', 'FlipAngle'])
```

The header is a numpy array of instances with the same dimensions as the array - except for the pixel coordinates: in this case `(z,t,n)`. This can be used to access any additional data in a transparent way. For instance, to list the flip angles of the first slice `z=0, n=0`:

```python
FA = [hdr.FlipAngle for hdr in header[0,:,0]]
```

The header array is also useful when a calculation is performed on the array and the results need to be saved in the DICOM database again. In this case `header` can be used to carry over the metadata. 

As an example, let's calculate a maximum intensity projection (MIP) of a 4D time series and write the result out in the same series:

```python
array, header = series.array(['SliceLocation', 'AcquisitionTime'])
mip = np.amax(array, axis=0)
series.set_array(mip, header[0,:,:])
```

In this case the header information of the MIP is taken from the first image of the time series. Provding header information is not required - if the header argument is not specified then a template header is used.

Another useful tool on series level is extracting a subseries. Let's say we have an MRI series with phase and magnitude data mixed, and we want to split it up into separate series:


```python
phase = series.subseries(image_type='PHASE')
magn = series.subseries(image_type='MAGNITUDE')
```

This will create two new series in the same study. The `image_type` keyword is defined in dbdicom for MR images to simplify access to phase or magnitude data, but the method also works for any standard DICOM keyword, or combinations thereof. For instance, to extract a subseries of all images with a flip angle of 20 and a TR of 5:

```python
sub = series.subseries(FlipAngle=20, RepetitionTime=5)
```

Another useful feature at series level is to overlay one series on another. 

```python
overlay = series.map_to(target)
# replaced by:
from dbdicom.wrappers import scipy
overlay = scipy.map_to(series, target)
```

If series is a binary mask (or can be interpreted as one), a similar function can be used to overlay the mask on another series:

```python
overlay = series.map_mask_to(target)
```


### Creating DICOM data from scratch

To create a DICOM series from a numpy array, use `dbdicom.series()`:

```python
import numpy as np
import dbdicom as db

array = np.random.normal(size=(10, 128, 192))
series = db.series(array)
```

After this you can save it to a folder in DICOM, or set some header elements before saving:

```python
series.PatientName = 'Random noise'
series.StudyDate = '19112022'
series.AcquisitionTime = '120000'
series.save(path)
```

You can build an entire database explicitly as well. For instance, the following code builds a database with two patients (James Bond and Scarface) who each underwent and MRI and an XRay study:

```python
database = db.database()

james_bond = database.new_patient(PatientName='James Bond')
james_bond_mri = james_bond.new_study(StudyDescription='MRI')
james_bond_mri_localizer = james_bond_mri.new_series(SeriesDescription='Localizer')
james_bond_mri_T2w = james_bond_mri.new_series(SeriesDescription='T2w')
james_bond_xray = james_bond.new_study(StudyDescription='Xray')
james_bond_xray_chest = james_bond_xray.new_series(SeriesDescription='Chest')
james_bond_xray_head = james_bond_xray.new_series(SeriesDescription='Head')

scarface = database.new_patient(PatientName='Scarface')
scarface_mri = scarface.new_study(StudyDescription='MRI')
scarface_mri_localizer = scarface_mri.new_series(SeriesDescription='Localizer')
scarface_mri_T2w = scarface_mri.new_series(SeriesDescription='T2w')
scarface_xray = scarface.new_study(StudyDescription='Xray')
scarface_xray_chest = scarface_xray.new_series(SeriesDescription='Chest')
scarface_xray_head = scarface_xray.new_series(SeriesDescription='Head')
```

### Work in progress: a numpy-like interface

We are currently building a `numpy`-type interface for creating new DICOM objects. For instance to create a new series with given dimensions in a study you can do:

```python
img = study.zeros((10, 128, 192), dtype='mri')
```

This will create a DICOM series of type 'MRImage' (shorthand 'mri') with 10 slices of 128 columns and 192 rows each. This can also be done from scratch:

```python
import dbdicom as db

series = db.series((10, 128, 192))
```

Currently, writing in data types other than 'MRImage' is not supported, so the data type argument is not necessary.

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


# About ***dbdicom***

## Why DICOM?

``*[...] after 2 hours of reading, I still cannot figure out how to determine the 3D orientation of a multi-slice (Supplement 49) DICOM file. I'm sure it is in there somewhere, but if this minor factoid can't be deciphered in 2 hours, then the format and its documentation is too intricate.*''. Robert W. Cox, PhD, Director, Scientific and Statistical Computing Core, National Institute of Mental Health [link](https://afni.nimh.nih.gov/pub/dist/doc/nifti/nifti_revised.html).

This echoes a common frustration for anyone who has ever had a closer to look at DICOM. DICOM seems to make simple things very difficult, and the language often feels outdated to modern data scientists. 

But there are good reasons for that. DICOM not only retains imaging data, but also all other relevant data about the subject and context in which the data are taken. Detailing provenance of the data and linkage to other data is critical in radiology, but the nature of these meta data is very broad, complex and constantly changing. Storing them in some consistent and standardised way that is future proof therefore requires a systematic approach and some necessary level of abstraction. 

DICOM does this well and has for that reason grown to be the single accepted standard in medical imaging. This also explains the outdated look and feel. DICOM standardises not only the format, but also the language of medical imaging. And successful standards, by definition, don't change.

## Why ***dbdicom***?

Reading and especially writing DICOM data remains a challenging enterprise for the practicing data scientist. A typical image processing pipeline might use the excellent python package `pydicom` for extracting image arrays and any required header information from DICOM data, but will then write out the results in more manageable format such as nifty. In the process the majority of header information will have to be discarded, including detailed imaging parameters and linkage between original and derived images, follow-up studies, etc.

The practice of converting outputs in a lossy image format may be sufficient in the early stages of method development, but forms a major barrier to research or deployment of these processing methods in a real-world context. This requires results in DICOM format so they can be linked to other data of the same patients, integrated in the radiological workflow, and reviewed and edited through integrated radiological viewers. Integration of datasets ensures that all derived data are properly traceable to the source, and can be compared between subjects and within a subject over time. It also allows to test for instance whether a new (expensive) imaging method provides an *additive* benefit over and above (cheap) data from medical history, clinical exams or blood tests. 

DICOM integration of processing outputs is typically performed by DICOM specialists in the private sector, for new products that have proven clinical utility. However, this requires a major separate investment, delays the point of real-world validation until after commercialisation and massively increases the risk of costly late-stage failures. 


## What is ***dbdicom***?

`dbdicom` is a programming interface that makes reading and writing DICOM data intuitive for the practicing medical imaging scientist working in Python. DICOM-native language and terminology is hidden and replaced by concepts that are more natural for those developing in Python. The documentation therefore does not reference confusing DICOM concepts such as composite information object definitions, application entities, service-object pairs, unique identifiers, etc.

`dbdicom` wraps around DICOM using a language and code structure that is native to the 2020's. This should allow DICOM integration from the very beginning of development of new image processing methods, which means they can be deployed in clinical workflows from the very beginning. It also means that any result you generate can easily be integrated in open access DICOM databases and can be visualised along with any other images of the same subject with a standard DICOM viewer such as [OHIF](https://ohif.org/).

`dbdicom` is developed by through the [UKRIN-MAPS](https://www.nottingham.ac.uk/research/groups/spmic/research/uk-renal-imaging-network/ukrin-maps.aspx) project of the UK renal imaging network, which aims to provide clinical translation of quantitative renal MRI on a multi-vendor platform. UKRIN-MAPS is funded by the UK's [Medical Research Council](https://gtr.ukri.org/projects?ref=MR%2FR02264X%2F1).

## Acknowledgements

`dbdicom` relies heavily on `pydicom` for read/write of individual DICOM files, with some additional features provided by `nibabel` and `dcm4che`. Basic array manipulation is provided by `numpy`, and sorting and tabulating of data by `pandas`. Export to other formats is provided by `matplotlib`.