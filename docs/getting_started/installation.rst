Installation
============

Python version support
----------------------

BioMol supports Python 3.10 and above.


Install with pip
----------------

BioMol can be installed from PyPI:

.. code-block:: bash

   pip install biomol


Install from source
-------------------

For development, BioMol uses `uv <https://docs.astral.sh/uv/>`_ for fast dependency management:

.. code-block:: bash

   git clone https://github.com/CSSB-SNU/BioMol
   cd BioMol
   uv sync

For development with all tools (linting, testing, etc.):

.. code-block:: bash

   uv sync --extra dev
