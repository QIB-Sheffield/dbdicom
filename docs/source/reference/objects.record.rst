.. currentmodule:: dbdicom


Record
======

``Record`` is a base class grouping properties that are common to all types of DICOM records  (``Database``, ``Patient``, ``Study``, ``Series``). 

``Record`` is an `abstract` base class, meaning that it is inherited by all key classes but is not intended to be instantiated directly. It exists only to avoid duplicating properties in the individual derived classes.


Properties
----------

.. autosummary::
   :toctree: generated/

   Record.print
   Record.path
   Record.empty
   Record.files
   Record.label


Up and down the database
------------------------

.. autosummary::
   :toctree: generated/

   Record.parent
   Record.children
   Record.siblings
   Record.series
   Record.studies
   Record.patients
   Record.database


Editing a record
----------------

.. autosummary::
   :toctree: generated/

   Record.new_patient
   Record.new_study
   Record.new_series
   Record.new_sibling
   Record.new_pibling
   Record.new_child
   Record.remove
   Record.move_to
   Record.copy_to
   Record.copy
   
   
Loading and saving
------------------

.. autosummary::
   :toctree: generated/

   Record.save
   Record.restore
   Record.load
   Record.clear



User interactions
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   Record.message
   Record.progress
   Record.mute
   Record.unmute



