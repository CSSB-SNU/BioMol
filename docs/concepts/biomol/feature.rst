.. _concepts_feature:

.. currentmodule:: biomol

Feature
=======

A :class:`Feature <core.Feature>` provides a NumPy array–like interface to the underlying data of a :class:`View <core.View>`, such as atom coordinates or residue names.  
The primary purpose of the :class:`Feature <core.Feature>` is to enable efficient, vectorized computations while preserving the structural context of the molecule.

Key characteristics include:

- **NumPy-like Interface**: Access and manipulate data using familiar NumPy syntax and functions.  
- **Topology Preservation**: Operations return a Feature of the same shape, ensuring consistency with the parent view.
- **Type-Specific**: Features are categorized into :class:`NodeFeature <core.NodeFeature>` for per-element data (e.g., atoms) and :class:`EdgeFeature <core.EdgeFeature>` for connectivity data (e.g., bonds).  


Node and Edge Features
----------------------

A :class:`Feature <core.Feature>` wraps an underlying NumPy array and adds metadata for structural context.  
The raw array can be accessed via the :attr:`value <core.Feature.value>` attribute or by converting the feature to a NumPy array using :func:`numpy.asarray()`.

Features are divided into two categories:

- :class:`NodeFeature <core.NodeFeature>`: Represents data associated with individual nodes in a view.
- :class:`EdgeFeature <core.EdgeFeature>`: Represents data associated with connections between nodes.

An :class:`EdgeFeature <core.EdgeFeature>` additionally stores connectivity information:

- :attr:`src <core.EdgeFeature.src>`: source node indices  
- :attr:`dst <core.EdgeFeature.dst>`: destination node indices  

These arrays have the same length as the feature itself, ensuring that each edge corresponds to a specific pair of nodes.

.. tip::

   To access the unique node indices referenced by an edge feature, use :attr:`nodes <core.EdgeFeature.nodes>`.


Basic Usage
-----------

Features can be accessed either as attributes of a :class:`View <core.View>` or by name using the :meth:`get_feature <core.View.get_feature>` method.

.. code-block:: python

    # Attribute access is convenient for common features
    positions = mol.atoms.positions
    res_names = mol.residues.name

    # Method access is useful for dynamic access
    chain_ids = mol.chains.get_feature("id")

A key behavior is that features are automatically reindexed from the :class:`View <core.View>` they are accessed from. This ensures that feature values always correspond to the current selection in the view.

.. code-block:: python

    chain_a_atoms = mol.atoms[mol.chains.id == "A"]
    chain_a_positions = chain_a_atoms.positions

.. note::

   :class:`Feature <core.Feature>` objects are reindexed on-the-fly when accessed from a :class:`View <core.View>`.  
   Features are not reindexed immediately when the view is filtered; instead, they are reindexed lazily upon access.


To use feature data as a raw NumPy array, you can convert a :class:`Feature <core.Feature>` using :func:`numpy.asarray()` or its :attr:`value <core.Feature.value>` attribute.

.. code-block:: python

   positions = mol.atoms.positions
   positions_arr = positions.value

   # or equivalently
   import numpy as np
   positions_arr = np.asarray(positions)


Operations
----------


Features support a wide range of NumPy-style operations, with one key distinction:

- **Topology-preserving operations** (same shape, same type) return a new :class:`Feature <core.Feature>`.

  .. code-block:: python
   
   mol.atoms.positions + [1.0, 0.0, 0.0]  # Feature
      

- **Shape- or type-changing operations** return a standard NumPy array. This includes aggregations, comparisons, and passing the feature to most NumPy functions.

  .. code-block:: python

   mol.atoms.positions.mean(axis=0)  # np.ndarray
   mol.atoms.name == "CA"  # np.ndarray
   
   import numpy as np
   np.linalg.norm(mol.atoms.positions, axis=1) # np.ndarray

.. warning::

   :class:`Feature <core.Feature>` objects are immutable; their underlying data cannot be modified in place.  
   Always create a new feature when modifying data.
