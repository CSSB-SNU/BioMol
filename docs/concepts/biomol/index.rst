.. _concepts:

.. currentmodule:: biomol

BioMol
========

.. figure:: /_static/overview.svg
   :align: center
   :alt: Overview of BioMol Concepts
   :width: 600px


The main :class:`BioMol` object serves as the primary entry point.
It manages a separate :class:`FeatureContainer <core.FeatureContainer>` at each structural level (atoms, residues, and chains), which store the relevant features.

You interact with this data through two powerful abstractions:

- :class:`View <core.ViewProtocol>`: Accesses the data's *structure*, allowing you to select and navigate between atoms, residues, and chains.
- :class:`Feature <core.Feature>`: Accesses the *data itself* as NumPy-like arrays for efficient computation.


This design provides an intuitive yet powerful interface for complex molecular analysis.
To see how these components enable powerful workflows, dive deeper into their specific roles:

.. grid:: 1 1 2 2
    :gutter: 2

    .. grid-item-card:: View
        :shadow: md
        :link: view
        :link-type: doc

        Learn how to select, filter, and navigate between atoms, residues, and chains.

    .. grid-item-card:: Feature
        :shadow: md
        :link: feature
        :link-type: doc

        Discover how to perform efficient NumPy-style operations on molecular data.

.. toctree::
   :maxdepth: 1
   :hidden:

   view
   feature
