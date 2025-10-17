.. _concepts_view:

.. currentmodule:: biomol

View
====

A :class:`View <core.View>` represents a *selection* at a specific structural level—**atoms, residues, or chains**.
Views are lightweight, index-based projections of the parent :class:`BioMol`.

Key characteristics of Views include:

- **Powerful Selections**: Select and filter data using indexing and set-like operations.
- **Flexible Transitions**: Transition between different structural levels with ease.
- **Lightweight**: Views only store indices, ensuring memory efficiency.

Basic Usage
-----------

To enter a view, first create a BioMol object:

.. code-block:: python

   mol = BioMol(...)


You can then access views via the :meth:`atoms <core.View.atoms>`, :meth:`residues <core.View.residues>`, and :meth:`chains <core.View.chains>` attributes:

.. code-block:: python

    mol.atoms # AtomView
    mol.residues # ResidueView
    mol.chains # ChainView


Access features with attribute syntax or the :meth:`get_feature <core.View.get_feature>` method:

.. code-block:: python

    mol.atoms.name
    mol.atoms.get_feature("name")


From any view, you can return to the parent :class:`BioMol` via the :attr:`mol <core.View.mol>` attribute:

.. code-block:: python
    
    mol.atoms.mol


Indexing and Selection
----------------------

Views support two primary ways to filter data: NumPy-style indexing and the PyMOL-like :meth:`select <core.View.select>` method.

**NumPy-style Indexing**

Views support standard NumPy indexing operations, which always return a new :class:`View <core.View>` object.

- **Slicing:**

  .. code-block:: python

      atoms = mol.atoms[:5]

- **Integer indexing:**

  .. code-block:: python

      atoms = mol.atoms[[0, 2, 4]]
      last_residue = mol.residues[-1]

- **Boolean indexing:**

  .. code-block:: python

      ca_atoms = mol.atoms[mol.atoms.name == "CA"]
      gly_residues = mol.residues[mol.residues.name == "GLY"]
      protein_chains = mol.chains[mol.chains.entity == "PROTEIN"]

- **Combined conditions:**

  .. code-block:: python

      polar_residues = mol.residues[
          (mol.residues.name == "SER") | (mol.residues.name == "THR")
      ]


Indexing operations can be chained—an indexed view can itself be indexed again:

.. code-block:: python

    atoms = mol.atoms[:10]
    ca_atoms = atoms[atoms.name == "CA"]


.. note::

    View objects maintain their own index space. Always index using the view’s own indices:

    .. code-block:: python

        atoms = mol.atoms[:10]
        atoms[mol.atoms.name == "CA"] # ❌ Incorrect
        atoms[atoms.name == "CA"]     # ✅ Correct


**PyMOL-like Selection**

The :meth:`select <core.View.select>` method provides a **PyMOL-like but Pythonic** way of filtering.

- **Single condition:**

  .. code-block:: python

      ca_atoms = mol.atoms.select(name="CA")
      gly_residues = mol.residues.select(name="GLY")
      protein_chains = mol.chains.select(entity="PROTEIN")

- **Multiple values for one feature:**

  .. code-block:: python

      backbone_atoms = mol.atoms.select(name=["N", "CA", "C", "O"])
      positive_residues = mol.residues.select(name=["LYS", "ARG", "HIS"])
      chain_a_and_b = mol.chains.select(name=["A", "B"])

- **Multiple features:**

  .. code-block:: python

      disulfide_bond_atoms = mol.atoms.select(element="S", bond="disulfide")


.. tip::
    
    Each keyword argument is combined with **logical AND**.
    In this example, ``disulfide_bond_atoms`` returns sulfur atoms that also participate in disulfide bonds.


Transitioning Between Levels
----------------------------
  
Views can freely transition between structural levels—**atom**, **residue**, and **chain**. 
This is not strictly hierarchical: any view can navigate to another level. 

**From lower to higher level:**

.. code-block:: python
    
    ca_atoms = mol.atoms.select(name="CA")
    ca_residues = ca_atoms.residues
    ca_chains = ca_residues.chains

**From higher to lower level:**

.. code-block:: python

    chain_a = mol.chains.select(name="A")
    gly_residues = chain_a.select(name="GLY")
    gly_atoms = gly_residues.atoms


.. note::
    
    Transitions using properties (:attr:`atoms <core.View.atoms>`, :attr:`residues <core.View.residues>`, :attr:`chains <core.View.chains>`) always return views with **unique indices**. 
    If you want to preserve duplicates (element-wise), use the explicit methods (:meth:`to_atoms <core.View.to_atoms>`, :meth:`to_residues <core.View.to_residues>`, :meth:`to_chains <core.View.to_chains>`).

    .. code-block:: python
        
        gly_residues = gly_atoms.residues # unique indices
        gly_residues_dup = gly_atoms.to_residues() # preserve duplicates


.. tip::
    
    Transitions always consider **only the indices contained in the current view**.
    In the example above, ``gly_atoms`` is restricted to the atoms within ``gly_residues``, not all atoms of `mol`.


Operations
----------

:class:`View <core.View>` objects support concatenation and set-wise operations.

.. note::
    Operations are only supported between views of the same structural level within the same parent :class:`BioMol`.

**Concatenation:**

.. code-block:: python

    protein_chains = mol.chains.select(entity="PROTEIN")
    rna_chains = mol.chains.select(entity="RNA")
    protein_and_rna = protein_chains + rna_chains

**Set-like operations:**

.. code-block:: python

    five_chains = mol.chains[:5]
    protein_chains = mol.chains.select(entity="PROTEIN")

    union = five_chains | protein_chains
    intersection = five_chains & protein_chains
    difference = five_chains - protein_chains
    sym_difference = five_chains ^ protein_chains
    invert = ~protein_chains # all chains except protein chains

.. tip::
    
    Concatenation (``+``) preserves duplicates, while set-like operations (``|``, ``&``, ``-``, ``^``, ``~``) return unique indices.


Additional Methods
------------------

**Index Manipulation**

- :attr:`.indices <core.View.indices>`: Accesses the raw NumPy array of indices.
- :meth:`unique <core.View.unique>`: Returns a new view with duplicate indices removed.
- :meth:`sort <core.View.sort>`: Returns a new view with indices sorted in ascending order.
- :meth:`new <core.View.new>`: Creates a new view from a raw array of indices.

**State Checking**

- :meth:`is_empty <core.View.is_empty>`: Checks if a view contains no elements.
- :meth:`is_subset <core.View.is_subset>`: Checks if one view is fully contained within another.
