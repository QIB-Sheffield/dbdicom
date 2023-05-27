Developer guide
===============

.. warning::

    This page is currently under construction. Please check back soon for updates.


Environment and dependencies
----------------------------

A working Python environment is required to run dbdicom. All required Python
dependencies are specified within the ``requirements.txt`` and  ``pyproject.toml``
files located within the root directory of the dbdicom source code.

When integrating ``dbdicom`` within a custom project, ``conda`` virtual
environments can be useful for managing project dependencies in isolation.
Anaconda may be installed within the user's directory without causing
conflicts with a system's Python installation, therefore it is recommended
to set up a working environment by downloading the ``conda`` package manager
from `Anaconda's Python distribution <https://www.anaconda.com/download/>`_.

.. warning::

    The following steps assume that Anaconda has already been installed and that commands are run from a Windows OS. If replicating from a different OS, please adapt commands to the appropriate related invocation (`Some examples here <https://kinsta.com/blog/python-commands/>`_).


Project setup and installation
------------------------------

#. From the project root directory, run the following command to create a separate virtual environment:

.. code-block:: console

    conda create --name <environment_name> python=<version_number>

#. Activate the virtual environment:

.. code-block:: console
    
    conda activate <environment_name>

#. Install required ``dbdicom`` dependencies:

.. code-block:: console

    pip install -e .

#. Install ``dbdicom`` with optional dependencies specified in pyproject.toml file (e.g., wrappers):

.. code-block:: console

    pip install -e .[wrappers]

