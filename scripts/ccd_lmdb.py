import os
import importlib
from collections.abc import Callable
from pathlib import Path

import click

from biomol.db.lmdb_handler import build_lmdb


def load_cif_list(cif_dir: Path, pattern: str = "*.cif*") -> list[Path]:
    """Load a list of CIF file paths from a directory."""
    return list(cif_dir.rglob(pattern))


def load_lmdb_config(
    env_path: Path,
    map_size: int = int(1e8),
    shard_idx: int | None = None,
) -> dict:
    """Build the LMDB configuration dictionary."""
    config = {"env_path": env_path, "map_size": map_size}
    config["n_jobs"] = os.cpu_count()
    if shard_idx is not None:
        config["env_path"] = Path(config["env_path"]).with_name(
            f"{Path(config['env_path']).stem}_shard{shard_idx}{Path(config['env_path']).suffix}",
        )
    return config


@click.command()
@click.argument("cif_dir", type=click.Path(exists=True, path_type=Path))
@click.argument("env_path", type=click.Path(path_type=Path))
@click.argument("parser", type=click.Path(path_type=Path))
@click.argument("recipe_path", type=click.Path(path_type=Path))
@click.option(
    "--map-size",
    "-m",
    type=float,
    default=1e8,
    show_default=True,
    help="Maximum size of the LMDB database in bytes.",
)
@click.option(
    "--shard-idx",
    "-i",
    type=int,
    default=None,
    help="Index of the shard to process (0-based).",
)
@click.option(
    "--n-shards",
    "-n",
    type=int,
    default=1,
    show_default=True,
    help="Total number of shards to split the dataset.",
)
def main(
    cif_dir: Path,
    env_path: Path,
    parser: str,
    recipe_path: Path,
    map_size: float,
    shard_idx: int | None,
    n_shards: int,
) -> None:
    """
    Build an LMDB database from CIF_DIR into ENV_PATH.

    Example:
        python ccd_lmdb.py ./cif ./ccd.lmdb --map-size 1e9 --shard-idx 0 --n-shards 10
    """
    map_size = int(map_size)
    cif_list = load_cif_list(cif_dir)

    if shard_idx is not None:
        if shard_idx < 0 or shard_idx >= n_shards:
            msg = f"Invalid shard index {shard_idx} for {n_shards} shards."
            raise click.BadParameter(msg)
        # Split the cif_list into n_shards parts and select the shard_idx-th part
        cif_list = [cif for i, cif in enumerate(cif_list) if i % n_shards == shard_idx]
        click.echo(
            f"Processing shard {shard_idx}/{n_shards} with {len(cif_list)} files.",
        )
    else:
        shard_idx = None

    config = load_lmdb_config(env_path, map_size=map_size, shard_idx=shard_idx)

    # load parser function
    module_name, func_name = str(parser).split(":", 1)
    module = importlib.import_module(module_name)
    parser_func = getattr(module, func_name)

    build_lmdb(*cif_list, **config, parser=parser_func, recipe=recipe_path)


if __name__ == "__main__":
    main()
