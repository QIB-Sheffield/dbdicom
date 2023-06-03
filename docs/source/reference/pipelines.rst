Pipelines
=========

Pipelines are functions that take ``dbdicom`` records as input and, where appropriate, produce ``dbdicom`` records as output - propagating both data and header information correctly.

The pipelines listed in this section are derived from use cases in ongoing studies. The list will grow over time, providing a library of components that can be assembled to build more complex pipelines. 

Most of the pipelines wrap around array-to-array functions from well-established packages that are not included in the dbdicom distribution. These pipelines will need a separate installation of these packages to run.

The list of pipelines is structured along the libraries that they wrap around. For instance, the ``skimage`` section lists a number of routines that wrap around common ``skimage`` functions - making these available to operate directly on ``dbdicom`` series. 


.. warning::

   Documentation for pipelines is work in progress. At this stage, these pages show *all* functions that are currently in use, whether they are documented or not.

.. toctree::
   :maxdepth: 2

   pipelines.numpy
   pipelines.scipy
   pipelines.skimage
   pipelines.sklearn
   pipelines.dipy
   pipelines.elastix
   pipelines.vreg