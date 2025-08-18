from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float

from biomol.utils.typecheck import typecheck


@typecheck
@dataclass
class BioMol:
    atom_positions: Float[np.ndarray, "N 3"]

    def from_path(self, path: str | Path) -> "BioMol":
        """
        Load BioMol data from a file. Supports `.pdb`, `.cif` formats.

        Parameters
        ----------
        path : str or Path
            The path to the file containing BioMol data.

        Returns
        -------
        BioMol
            BioMol instance with loaded data.
        """
        path = Path(path)
        if not path.exists():
            msg = f"File not found: {path}"
            raise FileNotFoundError(msg)
        if path.suffix not in [".pdb", ".cif"]:
            msg = f"Unsupported file format: {path.suffix}."
            raise ValueError(msg)

        raise NotImplementedError
