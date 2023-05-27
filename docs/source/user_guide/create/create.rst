

Reading DICOM attributes
^^^^^^^^^^^^^^^^^^^^^^^^

An object's DICOM attributes can be read by using the DICOM keyword of the attribute:

.. code-block:: python

    nr_of_rows = instance.Rows


All attributes can also be accessed at series, study, patient or folder level. In this case they will return a list of unique values. For instance to return a list with all distinct series descriptions in a study:

.. code-block:: python

    desc = study.SeriesDescription

DICOM attributes can also be accessed using the list notation, using either the keyword as a string or a (group, element) pair:

.. code-block:: python

    columns = instance['Columns']
    columns = instance[(0x0028, 0x0010)]

The tags can also be accessed as a list, for instance:

.. code-block:: python

    dimensions = ['Rows', (0x0028, 0x0010)]
    dimensions = instance[dimensions] 


This will return a list with two items. As shown in the example, the items in the list can be either KeyWord strings or (group, element) pairs. This also works on higher-level objects:

.. code-block:: python

    dimensions = ['Rows', (0x0028, 0x0010)]
    dimensions = patient[dimensions] 


Editing attributes
^^^^^^^^^^^^^^^^^^


DICOM tags can be modified using the same notations:

.. code-block:: python

    instance.EchoTime = 23.0

or also:

.. code-block:: python

    instance['EchoTime'] = 23.0

or also:

.. code-block:: python

    instance[(0x0018, 0x0081)] = 23.0

Multiple tags can be inserted in the same line:

.. code-block:: python

    shape = ['Rows', 'Columns']
    instance[shape] = [128, 192]

When setting values in a series, study or patient, all the instances in the object will be modified. For instance, to set all the Rows in all instances of a series to 128:

.. code-block:: python

    series.Rows = 128


Custom attributes
^^^^^^^^^^^^^^^^^

Apart from the predefined public and private DICOM keywords, ``dbdicom`` also provides a number of custom attributes for more convenient access to higher level properties. In order to distinguish these from existing DICOM attributes which are defined in ``CamelCase``, the custom attributes follow the ``lower_case`` notation. 

For instance, to set one of the standard `matplotlib color maps <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_, you can do:

.. code-block:: python

    image.colormap = 'YlGnBu'
    series.colormap = 'Oranges'


and so on.. The colormaps can be retrieved the same way:

.. code-block:: python

    cm_image = image.colormap
    cm_series = series.colormap


As for standard DICOM attributes this returns a list if unique values for the series. 

Custom attributes can easily be added to any DICOM dataset type and the number of available attributes is set to grow as the need arises.


Read and write
^^^^^^^^^^^^^^

By default all changes to a database are made on disk. For instance if a DICOM attribute is changed

.. code-block:: python

    instance.Rows = 128

The data are read from disk, the change is made, the data are written to disk again and memory is cleared. Equally, if a series is copied to another study, all its instances will be read, any necessary changes made, and then written to disk and cleared from memory. 

For many applications reading and writing from disk is too slow. For faster access at the cost of some memory usage, the data can be read into memory before performing any manipulations:

.. code-block:: python

    series.read()

After this all changes are made in memory. To clear the data from memory and continue working from disk, use `clear()`:


.. code-block:: python

    series.clear()


These operations can be called on the entire database, on patients, studies, series or instances. 


Save and restore
^^^^^^^^^^^^^^^^

All changes made in a DICOM folder are reversible until they are saved.
To save all changes, use ``save()``:

.. code-block:: python

    database.save()

This will permanently burn all changes that are made on disk. In order to reverse any changes made, use ``restore()`` to revert back to the last saved state:

.. code-block:: python

    database.restore()


This will roll back all changes on disk to the last changed state. ``save()`` and ``restore()`` can also be called at the level of individual objects:

.. code-block:: python

    series.restore()

will reverse all changes made since the last save, but only for this series. Equivalently:

.. code-block:: python

    series.save()


will save all changes made in the series (but not other objects in the database) permanently. 


Working with series
^^^^^^^^^^^^^^^^^^^

A DICOM series typically represents images that are acquired together, such as 3D volumes or time series. Some dedicated functionality exists for series that is not relevant for objects elsewhere in the hierarchy. 

To extract the images in a series as a numpy array, use ``array()``:

.. code-block:: python

    array, _ = series.array()


This will return an array with dimensions ``(n,x,y)`` where ``n`` enumerates the images in the series. The array can also be returned with other dimensions:

.. code-block:: python

    array, _ = series.array(['SliceLocation', 'FlipAngle'])


This returns an array with dimensions ``(z,t,n,x,y)`` where ``z`` corresponds to slice locations and ``t`` to flip angles. The 3d dimension ``n`` enumerates images at the same slice location and flip angle. Any number of dimensions can be added in this way. If an application requires the pixels to be listed first, use the ``pixels_first`` keyword:

.. code-block:: python

    array, _ = series.array(['SliceLocation', 'FlipAngle'], pixels_first=True)


In this case the array has dimensions ``(x,y,z,t,n)``. Replacing the images of a series with a given numpy array works the same way:

.. code-block:: python

    series.array(array)

The function ``array()`` also returns the header information for each slice in a second return value:

.. code-block:: python

    array, header = series.array(['SliceLocation', 'FlipAngle'])


The header is a numpy array of instances with the same dimensions as the array - except for the pixel coordinates: in this case ``(z,t,n)``. This can be used to access any additional data in a transparent way. For instance, to list the flip angles of the first slice ``z=0, n=0``:

.. code-block:: python

    FA = [hdr.FlipAngle for hdr in header[0,:,0]]

The header array is also useful when a calculation is performed on the array and the results need to be saved in the DICOM database again. In this case ``header`` can be used to carry over the metadata. 

As an example, let's calculate a maximum intensity projection (MIP) of a 4D time series and write the result out in the same series:

.. code-block:: python

    array, header = series.array(['SliceLocation', 'AcquisitionTime'])
    mip = np.amax(array, axis=0)
    series.set_array(mip, header[0,:,:])

In this case the header information of the MIP is taken from the first image of the time series. Provding header information is not required - if the header argument is not specified then a template header is used.

Another useful tool on series level is extracting a subseries. Let's say we have an MRI series with phase and magnitude data mixed, and we want to split it up into separate series:


.. code-block:: python

    phase = series.subseries(image_type='PHASE')
    magn = series.subseries(image_type='MAGNITUDE')

This will create two new series in the same study. The ``image_type`` keyword is defined in dbdicom for MR images to simplify access to phase or magnitude data, but the method also works for any standard DICOM keyword, or combinations thereof. For instance, to extract a subseries of all images with a flip angle of 20 and a TR of 5:

.. code-block:: python

    sub = series.subseries(FlipAngle=20, RepetitionTime=5)

Another useful feature at series level is to overlay one series on another. 

.. code-block:: python

    from dbdicom.wrappers import scipy
    overlay = scipy.map_to(series, target)

If series is a binary mask (or can be interpreted as one), a similar function can be used to overlay the mask on another series:

.. code-block:: python

    overlay = scipy.map_mask_to(series, target)


Creating DICOM data from scratch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To create a DICOM series from a numpy array, use ``dbdicom.series()``:

.. code-block:: python

    import numpy as np
    import dbdicom as db

    array = np.random.normal(size=(10, 128, 192))
    series = db.series(array)


After this you can save it to a folder in DICOM, or set some header elements before saving:

.. code-block:: python

    series.PatientName = 'Random noise'
    series.StudyDate = '19112022'
    series.AcquisitionTime = '120000'
    series.save(path)

You can build an entire database explicitly as well. For instance, the following code builds a database with two patients (James Bond and Scarface) who each underwent and MRI and an XRay study:

.. code-block:: python

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


Creating objects
^^^^^^^^^^^^^^^^

Some routines are available for creating DICOM objects from scratch, modelled on ``numpy`` creation routines. For instance, to create a new series with given dimensions in a study you can do:

.. code-block:: python

    img = study.zeros((10, 128, 192), dtype='mri')

This will create a DICOM series of type 'MRImage' (shorthand 'mri') with 10 slices of 128 columns and 192 rows each. This can also be done from scratch:

.. code-block:: python

    import dbdicom as db
    series = db.series((10, 128, 192))

Currently, writing in data types other than 'MRImage' is not supported, so the data type argument is not necessary.