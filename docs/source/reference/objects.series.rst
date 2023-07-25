.. currentmodule:: dbdicom


Series
======


The Series level is the natural level of interaction for most use cases in imaging, since individual scans are typically stored in separate series. 

Nearly all of the pipelines provided so far operate on series and return other series, for relatively general use cases including reslicing, rotations, coregistration, segmentation and general image processing. 

The documentation of these features is currently catching up with these use cases. Please come back later for a more extensively documented interface for the Series class.


Working with arrays
-------------------

.. autosummary::
   :toctree: generated/

   Series.ndarray
   Series.set_ndarray
   Series.affine
   Series.set_affine
   Series.slice_groups


Series editing
--------------

.. autosummary::
   :toctree: generated/

   Series.slice
   Series.split_by
   Series.subseries
   