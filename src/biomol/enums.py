import enum


@enum.unique
class StructureLevel(str, enum.Enum):
    """Levels of structural hierarchy in biomolecules."""

    ATOM = "atom"
    RESIDUE = "residue"
    CHAIN = "chain"
