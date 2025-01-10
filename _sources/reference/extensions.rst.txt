**********
Extensions
**********

Extensions are functions of ``dbdicom`` records that require additional packages to be installed. Often these are DICOM to DICOM functions that take well-known functions which operate on numpy arrays, and lift them to functions that operate on ``dbdicom`` series. Where appropriate these will produce ``dbdicom`` objects in the same database as output, propagating both data and header information correctly. 

The extensions listed in this section are derived from use cases in ongoing studies. The list will grow over time, providing a library of components that can be assembled to build more complex DICOM to DICOM pipelines. The extensions are generally structured along the libraries that they depend on. For instance, the ``skimage`` section lists a number of routines that wrap around common ``skimage`` functions. The idea is to allow an intuitive translation of ``numpy``-type pipelines to ``dbdicom`` type pipelines. 


.. warning::

   Documentation for extensions is work in progress. At this stage, these pages show *all* functions that are currently in use, whether they are documented or not.

.. toctree::
   :maxdepth: 2

   extensions.numpy
   extensions.matplotlib
   extensions.scipy
   extensions.skimage
   extensions.sklearn
   extensions.dipy
   extensions.elastix
   extensions.vreg
   