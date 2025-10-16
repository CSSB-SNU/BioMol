from pathlib import Path

from biomol.io.cache import ParsingCache
from biomol.io.cooker import Cooker


def read_data(a3m_path: Path) -> dict:
    """Parse a a3m file and return its data as a dictionary."""
    with open(a3m_path) as f:
        lines = f.readlines()
    if len(lines) < 2 or not lines[0].startswith(">"):  # noqa: PLR2004
        msg = f"Unsupported file format: {a3m_path}"
        raise ValueError(msg)
    headers = []
    raw_sequences = []
    for ii, _line in enumerate(lines):
        line = _line.strip()
        if ii % 2 == 0:
            if not line.startswith(">"):
                msg = f"Unsupported file format: {a3m_path}"
                raise ValueError(msg)
            headers.append(line)
        else:
            raw_sequences.append(line)
    if len(headers) != len(raw_sequences):
        msg = f"Mismatch between headers and sequences in: {a3m_path}"
        raise ValueError(msg)
    return {"headers": headers, "raw_sequences": raw_sequences, "a3m_type": "protein"}


def parse(
    recipe_path: Path,
    a3m_path: Path,
    targets: list[str] | None = None,
) -> dict:
    """Parse a a3m file using a predefined recipe."""
    a3m_data = read_data(a3m_path)
    parse_cache = ParsingCache()
    cooker = Cooker(parse_cache=parse_cache, recipebook=str(recipe_path))
    cooker.prep(a3m_data, fields=list(a3m_data.keys()))
    cooker.cook()
    return cooker.serve(targets=targets)
