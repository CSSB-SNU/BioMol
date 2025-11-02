import glob
import importlib
import os
from pathlib import Path

import click
import lmdb

from biomol.db.lmdb_handler import build_lmdb


# ------------------------------
# Helper: Load a3m file list
# ------------------------------
def load_a3m_list(a3m_dir: Path, pattern: str = "*.a3m*") -> list[Path]:
    """Load a list of a3m file paths from a directory."""
    return list(a3m_dir.rglob(pattern))


def load_a3m(a3m_path: Path) -> dict[str, str]:
    """Load a3m file into a dictionary."""
    a3m_dict = {}
    with a3m_path.open("r") as f:
        lines = f.readlines()
    current_header = ""
    for _line in lines:
        line = _line.strip()
        if line.startswith(">"):
            current_header = line[1:]
            a3m_dict[current_header] = ""
        else:
            a3m_dict[current_header] += line
    return a3m_dict


# ------------------------------
# Helper: Build LMDB config
# ------------------------------
def load_lmdb_config(
    env_path: Path,
    map_size: int = int(1e8),
    shard_idx: int | None = None,
    **kwargs: str,
) -> dict:
    """Build the LMDB configuration dictionary."""
    config = {"env_path": env_path, "map_size": map_size}
    config["n_jobs"] = int(os.environ.get("SLURM_CPUS_PER_TASK", -1))
    if shard_idx is not None:
        config["env_path"] = Path(config["env_path"]).with_name(
            f"{Path(config['env_path']).stem}_shard{shard_idx}{Path(config['env_path']).suffix}",
        )
    config.update(kwargs)
    return config


# ==============================================================
# Command Group
# ==============================================================
@click.group()
def cli() -> None:
    """Build and merge LMDB databases from a3m files."""


# ==============================================================
# 1. Build Command
# ==============================================================
@cli.command("build")
@click.argument("a3m_dir", type=click.Path(exists=True, path_type=Path))
@click.argument("env_path", type=click.Path(path_type=Path))
@click.argument("parser", type=str)
@click.argument("recipe_path", type=click.Path(path_type=Path))
@click.option("--map-size", "-m", type=float, default=1e8, show_default=True)
@click.option(
    "--shard-idx",
    "-i",
    type=int,
    default=None,
    help="Index of the shard to process (0-based).",
)
@click.option("--n-shards", "-n", type=int, default=1, show_default=True)
def build(
    a3m_dir: Path,
    env_path: Path,
    parser: str,
    recipe_path: Path,
    map_size: float,
    shard_idx: int | None,
    n_shards: int,
) -> None:
    """
    Build an LMDB database from a3m_DIR into ENV_PATH.

    Example:
        python a3m_lmdb.py build ./a3m ./a3m.lmdb biomol.io.parsers.a3m_parser:parse\\
              ./plans/recipe.py \\
            --map-size 1e12 --shard-idx 0 --n-shards 4
    """
    map_size = int(map_size)
    a3m_list = load_a3m_list(a3m_dir)

    if shard_idx is not None:
        if shard_idx < 0 or shard_idx >= n_shards:
            msg = f"Invalid shard index {shard_idx} for {n_shards} shards."
            raise click.BadParameter(msg)
        # Split a3m list into n_shards and take only the shard_idx part
        a3m_list = [a3m for i, a3m in enumerate(a3m_list) if i % n_shards == shard_idx]
        click.echo(
            f"Processing shard {shard_idx}/{n_shards} with {len(a3m_list)} files.",
        )
    else:
        click.echo(f"Processing all {len(a3m_list)} files as a single shard.")

    config = load_lmdb_config(
        env_path,
        map_size=map_size,
        shard_idx=shard_idx,
    )

    # Dynamically import parser function
    module_name, func_name = parser.split(":", 1)
    module = importlib.import_module(module_name)
    parser_func = getattr(module, func_name)

    build_lmdb(*a3m_list, **config, parser=parser_func, recipe=recipe_path)


# ==============================================================
# 2. Merge Command (auto-detect *.lmdb by stem pattern)
# ==============================================================
@cli.command("merge")
@click.argument("shard_pattern", type=str)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output merged LMDB path.",
)
@click.option("--map-size", "-m", type=float, default=1e12, show_default=True)
@click.option("--overwrite", is_flag=True, help="Overwrite existing LMDB if it exists.")
def merge(shard_pattern: str, output: Path, map_size: float, overwrite: bool) -> None:
    """
    Merge multiple LMDB shard databases into a single LMDB file.

    Example:
        python a3m_lmdb.py merge "/data/BioMolDBv2_2024Oct21/a3m_shard*.lmdb" -o /data/BioMolDBv2_2024Oct21/a3m_merged.lmdb
    """
    map_size = int(map_size)

    # Expand wildcard pattern
    shard_paths = sorted(Path(p) for p in glob.glob(shard_pattern))
    if not shard_paths:
        msg = f"No LMDB files found for pattern: {shard_pattern}"
        raise click.ClickException(msg)

    if output.exists() and not overwrite:
        msg = f"{output} already exists. Use --overwrite to replace it."
        raise click.ClickException(
            msg,
        )

    click.echo(f"Found {len(shard_paths)} shards:")
    for s in shard_paths:
        click.echo(f"  - {s}")

    merged_env = lmdb.open(str(output), map_size=map_size)
    total_keys = 0

    for shard_path in shard_paths:
        click.echo(f"Merging {shard_path}")
        shard_env = lmdb.open(str(shard_path), readonly=True, lock=False)
        with shard_env.begin() as shard_txn, merged_env.begin(write=True) as merged_txn:
            cursor = shard_txn.cursor()
            for key, value in cursor:
                merged_txn.put(key, value)
                total_keys += 1
        shard_env.close()

    merged_env.sync()
    merged_env.close()

    click.echo(f"[Done] Merged {len(shard_paths)} shards into {output}")
    click.echo(f"Total keys merged: {total_keys}")


# ==============================================================
# Entrypoint
# ==============================================================
if __name__ == "__main__":
    cli()
