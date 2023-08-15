.. currentmodule:: dbdicom


Series
======

The Series level is the natural level of interaction for most use cases in imaging, since individual scans are typically stored in separate series. 

Reading values
--------------

.. autosummary::
   :toctree: ../generated/api/
   :template: autosummary.rst
   :nosignatures:

   Series.coords
   Series.values
   Series.frames
   Series.shape
   Series.spacing
   Series.unique
   Series.unique_affines
   Series.gridcoords
   Series.pixel_values
   Series.affine


Setting values
--------------

.. autosummary::
   :toctree: ../generated/api/
   :template: autosummary.rst
   :nosignatures:

   Series.set_coords
   Series.set_gridcoords
   Series.set_values
   Series.set_pixel_values
   Series.set_affine


Slicing
-------

.. autosummary::
   :toctree: ../generated/api/
   :template: autosummary.rst
   :nosignatures:

   Series.slice
   Series.islice
   Series.split_by
   