Overview
========

BioMol is a molecular data engine that brings PyMOL-like selections and NumPy-style operations into Python.
It enables seamless navigation between atoms, residues, and chains, while providing efficient data structures for machine learning and large-scale analysis.


Key features
------------

- **Single Object Representation**: Unified representation of molecular structures
- **Flexible View**: Navigate between atoms, residues, and chains easily
- **Performance**: Fast and memory-efficient data structures with NumPy-style operations


Core Philosophy
---------------

The core philosophy of BioMol is to provide a single object that can represent complex molecular structures with multiple levels of detail. 

.. code-block:: python

   from biomol import BioMol

   mol = BioMol(...)
   mol.atoms  # Access atom-level data
   mol.residues  # Access residue-level data
   mol.chains  # Access chain-level 
   

This unified approach allows users to handle molecular data more intuitively like PyMOL.

.. code-block:: python

   chain_a_atoms = mol.chains.select(id="A").atoms
   chain_a_carbons = chain_a_atoms.select(element="C")
   chain_a_non_carbons = chain_a_atoms - chain_a_carbons


Also, BioMol supports NumPy-style operations for efficient data manipulation.

.. code-block:: python

   chain_a_ca = chain_a_carbons[chain_a_carbons.name == "CA"]
   first_five_ca = chain_a_ca[:5]


.. seealso::

   :doc:`../concepts/biomol/index`
      Dive deeper into the core concepts of BioMol, including Views and Features.