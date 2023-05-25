.. dbdicom documentation master file, created by
   sphinx-quickstart on Mon Oct 10 07:46:23 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the dbdicom documentation!
=====================================

``dbdicom`` is a new Python interface for reading and writing DICOM databases.

.. warning::

   ``dbdicom`` is developed in public, however it is a work in progress. 
   Therefore, please bear in mind some features are still in development and **backwards compatibility is not likely to happen**.

Summary
============

The DICOM format is the universally recognised standard for medical imaging,
but reading and writing DICOM data remains a challenging task for most data
scientists.

The excellent python package ``pydicom`` is very commonly used and
well-supported, but it is limited to reading and writing individual
files, and still requires a fairly high level of understanding of
DICOM to ensure compliance with the standard.

``dbdicom`` wraps around ``pydicom`` to provide an intuitive programming
interface for reading and writing data from entire DICOM databases,
replacing unfamiliar DICOM-native concepts by language and notations
that will be more familiar to data scientists. 

The sections below list some basic uses of ``dbdicom``. The package is
currently deployed in several larger scale multicentre clinical studies
led by the authors, such as the
`iBEAt study <https://bmcnephrol.biomedcentral.com/articles/10.1186/s12882-020-01901-x>`_
and the `AFiRM study <https://www.uhdb.nhs.uk/afirm-study/>`_. The
package will continue to be shaped through use in these studies and we
expect it will attain a more final form when these analysis pipelines
are fully operational.

.. grid-item-card:: Getting started
   :shadow: md

      .. button-ref:: 01_getting_started/getting_started
         :ref-type: ref
         :click-parent:
         :color: secondary
         :expand:

         To the getting started guides

.. grid-item-card:: User guide
   :shadow: md
   
      .. button-ref:: 02_user_guide/user
         :ref-type: ref
         :click-parent:
         :color: secondary
         :expand:

         To the user guides

.. toctree::
   :maxdepth: 4
   :caption: Contents:
   
   01_getting_started/getting_started
   02_user_guide/user
   03_developers_guide/installation
   04_API/modules
   05_example_notebooks/examples
   06_about/about