.. dbdicom documentation master file, created by
   sphinx-quickstart on Mon Oct 10 07:46:23 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

#####################
dbdicom documentation
#####################

A Python interface for reading and writing DICOM databases.

.. warning::

   ``dbdicom`` is developed in public and currently being trialled in ongoing multi-centre clinical studies `iBEAt <https://bmcnephrol.biomedcentral.com/articles/10.1186/s12882-020-01901-x>`_ and `AFiRM <https://www.uhdb.nhs.uk/afirm-study/>`_. However, ``dbdicom`` is work in progress and **not yet sufficiently stable for wider use**. Current dissemination activities, such as on the `ISMRM (Toronto 2023) <https://www.ismrm.org/23m/>`_, are limited in scope and intended only to get early feedback from the community. 


Ambition
^^^^^^^^

The DICOM format is the universally recognised standard for medical imaging, but working with DICOM data remains a challenging task for data scientists. 

``dbdicom`` aims to provide an *intuitive* programming interface for reading and writing DICOM databases - replacing unfamiliar DICOM-native concepts by more pythonic language and syntax. 


.. toctree::
   :maxdepth: 2
   
   user_guide/index
   reference/index
   tutorials/index
   developers_guide/index
   about/index