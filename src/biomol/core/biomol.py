from pathlib import Path

from biotite.structure import io as strucio


class BioMol:
    @classmethod
    def from_path(cls, path: str | Path) -> "BioMol":
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

        suffix = path.suffix.lower()
        match suffix:
            case ".pdb" | ".cif":
                return cls(atom_array=strucio.load_structure(path))
            case _:
                msg = f"Unsupported file format: {suffix}."
                raise ValueError(msg)
