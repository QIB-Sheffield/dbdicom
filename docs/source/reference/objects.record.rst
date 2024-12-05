.. currentmodule:: dbdicom


Record
======

`Record` is a base class grouping properties that are common to all types of DICOM records  (`Database`, `Patient`, `Study`, `Series`). 

`Record` is an *abstract* base class, meaning that it is inherited by all key classes but is not intended to be instantiated directly. It exists only to avoid duplicating properties in the individual derived classes.


Properties
----------

.. autosummary::
   :toctree: ../generated/api/
   :template: autosummary.rst
   :nosignatures:

   Record.path
   Record.empty
   Record.files
   Record.label


Navigating the database
------------------------

.. autosummary::
   :toctree: ../generated/api/
   :template: autosummary.rst
   :nosignatures:

   Record.parent
   Record.children
   Record.siblings
   Record.series
   Record.studies
   Record.patients
   Record.database


Creating new records
--------------------

.. autosummary::
   :toctree: ../generated/api/
   :template: autosummary.rst
   :nosignatures:

   Record.new_patient
   Record.new_study
   Record.new_series
   Record.new_sibling
   Record.new_pibling
   Record.new_child


Copy and move
-------------

.. autosummary::
   :toctree: ../generated/api/
   :template: autosummary.rst
   :nosignatures:

   Record.remove
   Record.move_to
   Record.copy_to
   Record.copy
   
   
Save and restore
----------------

.. autosummary::
   :toctree: ../generated/api/
   :template: autosummary.rst
   :nosignatures:

   Record.save
   Record.restore


Export to other formats
-----------------------

.. autosummary::
   :toctree: ../generated/api/
   :template: autosummary.rst
   :nosignatures:

   Record.export_as_dicom
   Record.export_as_png
   Record.export_as_csv
   Record.export_as_nifti
   Record.export_as_npy


Messaging
---------

.. autosummary::
   :toctree: ../generated/api/
   :template: autosummary.rst
   :nosignatures:

   Record.print
   Record.set_log
   Record.log
   Record.message
   Record.progress
   Record.mute
   Record.unmute



