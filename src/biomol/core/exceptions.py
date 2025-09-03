class BioMolError(Exception):
    """Base exception class for the biomol package."""


class FeatureShapeError(BioMolError, ValueError):
    """Custom exception for errors related to feature shapes."""


class FeatureIndicesError(BioMolError, IndexError):
    """Custom exception for errors related to feature indices."""


class FeatureKeyError(BioMolError, KeyError):
    """Custom exception for errors related to feature keys."""


class ViewProtocolError(BioMolError, TypeError):
    """Custom exception for errors related to view protocols."""


class StructureLevelError(BioMolError, ValueError):
    """Custom exception for errors related to structure levels."""
