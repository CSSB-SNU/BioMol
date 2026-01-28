import enum


@enum.unique
class StructureLevel(enum.Enum):
    """Levels of structural hierarchy in biomolecules."""

    ATOM = "atom"
    RESIDUE = "residue"
    CHAIN = "chain"
