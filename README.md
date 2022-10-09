# dbdicom

`dbdicom` is a Python interface for reading and writing DICOM databases. 

***CAUTION: dbdicom is work in progress!!!***

## Installation
Run `pip install dbdicom`.


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

And so on for all other levels in the hierarchy. These can be chained together for convencience, 
e.g. to get all instances instance of series 5 in study 1 of patient 2:

```python
instance = database.patients()[2].studies()[1].series()[5].instances()
```

These functions also work to find objects higher up in the hierarchy. 
For instance, to find the patient of a given series:

```python
patient = series.patients()
```

In this case the function will return a single item.

### Finding DICOM objects in the folder

Each DICOM file has a number of attributes describing the properties of the object. Examples are PatientName, StudyDate, etc. A convenient list of attributes for specific objects can be found [here]: (https://dicom.innolitics.com/). 

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

To extract the images in a series as a numpy array, use `get_pixel_array`:

```python
array, _ = series.get_pixel_array()
```

This will return an array with dimensions `(n,x,y)` where `n` enumerates the images in the series. The array can also be returned with other dimensions:

```python
array, _ = series.get_pixel_array(['SliceLocation', 'FlipAngle'])
```

This returns an array with dimensions `(z,t,n,x,y)` where `z` corresponds to slice locations and `t` to flip angles. The 3d dimension `n` enumerates images at the same slice location and flip angle. Any number of dimensions can be added in this way. If an application requires the pixels to be listed first, use the `pixels_first` keyword:

```python
array, _ = series.get_pixel_array(['SliceLocation', 'FlipAngle'], pixels_first=True)
```

In this case the array has dimensions `(x,y,z,t,n)`. Replacing the images of a series with a given numpy array works the same way:

```python
series.set_pixel_array(array)
```

The `get_pixel_array()` also returns the header information for each slice in a second return value:

```python
array, header = series.get_pixel_array(['SliceLocation', 'FlipAngle'])
```

The header is a numpy array of instances with the same dimensions as the array - except for the pixel coordinates: in this case `(z,t,n)`. This can be used to access any additional data in a transparent way. For instance, to list the flip angles of the first slice `z=0, n=0`:

```python
FA = [hdr.FlipAngle for hdr in header[0,:,0]]
```

The header array is also useful when a calculation is performed on the array and the results need to be saved in the DICOM database again. In this case `header` can be used to carry over the metadata. 

As an example, let's calculate a maximum intensity projection (MIP) of a 4D time series and write the result out in the same series:

```python
array, header = series.get_pixel_array(['SliceLocation', 'AcquisitionTime'])
mip = np.amax(array, axis=0)
series.set_pixel_array(mip, header[0,:,:])
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


`dbdicom` can be used in standalone scripts or at command line. To streamline integration in a GUI, communication with the user is performed via two dedicated attributes `status` and `dialog`. dialog and status attributes are available to any DICOM object. The status attribute is used to send messages to the user, or update on progress of a calculation:

```python
series.status.message("Starting calculation...")
```

When operating in command line mode this will print the message to the terminal. If `dbdicom` is used in a compatible GUI, this will print the same message to the status bar. Equivalently, the user can be updated on the progress of a calculation via:

```python
series.status.message("Calculating..")
for i in range(length):
    series.status.progress(i, length)
```

This will print the message with a percentage progress at each iteration. When used in a GUI, this can be used to update the progress bar of the GUI. If messaging is not desired, and/or when it slows down the execution too much, it can be muted and unmuted:

```python
series.mute()
compute(series)
series.unmute()
```

In this case, even if the function `compute` directs messaging to the terminal, these will not be printed until the series is unmuted again.

Dialogs can be used to send messages to the user or prompt for input. In some cases a dialog may halt the operation of te program until the user has performed the appropriate action, such as hitting enter or entering a value. In command line operator or scripts the user will be prompted for input at the terminal. When using in a GUI, these prompts can be redirected to a pop-up window:

```python
series.dialog.question("Do you wish to proceed?", cancel=True)
```

When used in a script, this will ask the user to enter either "y" (for yes), "n" (for no) or "c" (for cancel) and the program execution will depend on the answer. When the scame script is deployed in a GUI, the question will be asked via a pop-up window and a button push to answer. A number of different dialogs are available via the dialog attribute (see reference guide). 


# About ***dbdicom***

## Why ***dbdicom***?

This statement echoes a common frustration for anyone who has ever had a closer to look at DICOM: 

``*[...] after 2 hours of reading, I still cannot figure out how to determine the 3D orientation of a multi-slice (Supplement 49) DICOM file. I'm sure it is in there somewhere, but if this minor factoid can't be deciphered in 2 hours, then the format and its documentation is too intricate.*''. Robert W. Cox, PhD, Director, Scientific and Statistical Computing Core, National Institute of Mental Health [link](https://afni.nimh.nih.gov/pub/dist/doc/nifti/nifti_revised.html).

DICOM is scary. But it has also been the universally accepted standard for medical images for decades. Why is that? DICOM is extremely detailed and rigorous in the description of its terminology and structure. It has to be, because DICOM deals with the most complex and sensitive data possible: your body. All of it. Every single one of your DICOM images in a clinical archive contains the key to access all of your medical details. This allows doctors to link your images to your blood tests, family history, previous diagnosis treatments, other imaging, and so on. And this is important to make the best possible informed decisions when it comes to your health. 

In medical imaging research this additional information is often seen as a nuisance and discarded prior to processing of the images. Typically a data array of some sort is extracted, perhaps also some key geometrical descriptors such as pixel sizes or a transformation matrix, and all the other information is ignored. Conversion into such a *lossy* data format may be sufficient for method development or basic scientific research, but when it comes to deploying these methods in clinical studies, all this additional information is just as important as in clinical practice. It ensures that all derived data are properly traceable to the source, and can be compared between subjects and within a subject over time. It allows to test for instance whether a new (expensive) imaging method provides an *additive* benefit over and above (cheap) data from medical history, clinical exams or blood tests. 

And so, if we accept that new image analysis methods ultimately will need to be tested clinically (and ideally sooner rather than later), then we simply can't avoid the need to convert results back to DICOM. In practice this step often requires a major rewrite of image processing pipelines set up for basic research, creating a significant barrier to deployment of new methods in clinical trials. 

Quantitative imaging is another area where the information discarded by conversion to lossy formats is important. Quantification involves the application of complex signal models to multi-dimensional imaging data. These are acquired by varying contrast parameters such as (in MRI) echo times, b-values, gradient directions, inversion times, flip angle etc. Often many of these are varied at the same time, and not necessarily in some clean incremental order -  as in MR fingerprinting. The models that interpret these data need access to this information. When DICOM data have been converted to some lossy data format, this then requires ad-hoc solutions retaining part of the original DICOM information in unstructured free text fields or separate newly defined header files. 

All these problems can be solved, for current and any imaginable or unimaginable future applications, by dropping conversions into lossy image formats and simply reading from DICOM and writing to DICOM. 

If only DICOM wasn't so scary!!

## What is ***dbdicom***?

`dbdicom` is a programming interface that makes reading and writing DICOM data intuitive for the practicing medical imaging scientist working in Python. We promise you won't even know it's DICOM. In fact the documentation hardly even mentions DICOM at all. It will certainly not mention things like composite information object definitions, application entities, service-object pairs, unique identifiers, etc etc. This is the language of DICOM, and it's confusing in part because the concepts date back to the 1970's and 1980's when the standard was developed. But then again, that is exactly what you would expect from a successful standard. It doesn't change. It shouldn't change. But we *can* wrap it up real nice.

`dbdicom` wraps around DICOM using a language and code structure that is native to the 2020's. It allows you to develop your medical imaging methods using DICOM files only, which means your prototypes of new analysis methods can be deployed in clinical trials just like that. It also means that any result you generate can easily be integrated in open access DICOM databases and can be visualised along with any other images of the same subject by anyone with a DICOM viewer (i.e. literally anyone).

Since `dbdicom` is primarily a development tool, it can be used from command line or to write stand-alone scripts. However, since `dbdicom` is all about facilitating translation into clinical trials and ultimately clinical practice, all scripts written in `dbdicom` are set up for deployment in a graphical user interface. Convenience classes are provided for user interaction that print to a terminal when used 
in a script, but will automatically generate pop-up windows or progress bars when the same script is deployed inside a `dbdicom` compatible graphical user interface. 

## Acknowledgements

`dbdicom` relies heavily on `pydicom` for read/write of individual DICOM files, with some additional features provided by `nibabel` and `dcm4che`. Basic array manipulation is provided by `numpy`, and sorting and tabulating of data by `pandas`. Export to other formats is provided by `matplotlib`.
