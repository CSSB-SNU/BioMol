class BioMolError(Exception):
    """Base exception class for the biomol package."""


class FeatureKeyError(BioMolError, KeyError):
    """Custom exception for errors related to feature keys."""


class FeatureOperationError(BioMolError, ValueError):
    """Custom exception for errors related to feature operations."""


class ViewOperationError(BioMolError, ValueError):
    """Custom exception for errors related to view operations."""


class StructureLevelError(BioMolError, ValueError):
    """Custom exception for errors related to structure levels."""


class IndexInvalidError(BioMolError, ValueError):
    """Custom exception for errors related to invalid indices."""


class IndexMismatchError(BioMolError, ValueError):
    """Custom exception for errors related to index mismatches."""


class IndexOutOfBoundsError(BioMolError, IndexError):
    """Custom exception for errors related to index out of bounds."""
