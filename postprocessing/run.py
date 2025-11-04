from pathlib import Path

import click
import lmdb
from joblib import Parallel, delayed
import os

from biomol.cif import CIFMol
from biomol.core.utils import from_bytes
from biomol.io.cache import ParsingCache
from biomol.io.cooker import Cooker
import networkx as nx


def extract_key_list(env_path: Path) -> list[str]:
    """Extract all keys from the LMDB database."""
    env = lmdb.open(str(env_path), readonly=True, lock=False)
    with env.begin() as txn:
        key_list = [
            key.decode() for key in txn.cursor().iternext(keys=True, values=False)
        ]
    env.close()
    return key_list


def load_cif(key: str, env_path: Path) -> dict[str, CIFMol]:
    """
    Read a value from the LMDB database by key.

    Args:
        env_path: Path to the LMDB environment.
        key: Key of the data to retrieve.

    Returns
    -------
        dict
            The data dictionary retrieved from the LMDB database.
    """
    cache = getattr(load_cif, "_env_cache", None)
    if cache is None:
        cache = {}
        setattr(load_cif, "_env_cache", cache)

    env_key = str(env_path)
    env = cache.get(env_key)
    if env is None:
        env = lmdb.open(
            env_key,
            readonly=True,
            lock=False,
            max_readers=4096,
            readahead=True,
        )
        cache[env_key] = env

    with env.begin(buffers=True) as txn:
        value = txn.get(key.encode())

    if value is None:
        msg = f"Key '{key}' not found in LMDB database at '{env_path}'."
        raise KeyError(msg)

    value = from_bytes(bytes(value))
    value, metadata = value["assembly_dict"], value["metadata_dict"]

    cifmol_dict: dict[str, dict[str, CIFMol]] = {}
    for cif_key, item in value.items():
        assembly_id, model_id, alt_id = cif_key.split("_")

        md = dict(metadata)
        md["assembly_id"] = assembly_id
        md["model_id"] = model_id
        md["alt_id"] = alt_id

        item = dict(item)
        item["metadata"] = md

        cifmol_dict[cif_key] = {"cifmol": CIFMol.from_dict(item)}

    return cifmol_dict


def load_fasta(fasta_path: Path) -> dict[str, str]:
    """Load fasta file into a dictionary."""
    fasta_dict = {}
    with fasta_path.open("r") as f:
        lines = f.readlines()
    current_header = ""
    for _line in lines:
        line = _line.strip()
        if line.startswith(">"):
            current_header = line[1:]
            fasta_dict[current_header] = ""
        else:
            fasta_dict[current_header] += line
    return fasta_dict


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


def load_graph(key: str, env_path: Path) -> nx.Graph:
    """
    Read a value from the LMDB database by key.

    Args:
        env_path: Path to the LMDB environment.
        key: Key of the data to retrieve.
    Returns
    -------
        dict
            The data dictionary retrieved from the LMDB database.
    """
    cache = getattr(load_graph, "_env_cache", None)
    if cache is None:
        cache = {}
        setattr(load_graph, "_env_cache", cache)

    env_key = str(env_path)
    env = cache.get(env_key)
    if env is None:
        env = lmdb.open(
            env_key,
            readonly=True,
            lock=False,
            max_readers=4096,
            readahead=True,
        )
        cache[env_key] = env

    with env.begin(buffers=True) as txn:
        value = txn.get(key.encode())

    if value is None:
        msg = f"Key '{key}' not found in LMDB database at '{env_path}'."
        raise KeyError(msg)

    value = from_bytes(bytes(value))["cluster_graph"]
    node_list = value["nodes"]["seq_clusters"]["value"]
    edge_list = value["edges"]["contact_edges"]
    src, dst = edge_list["src_indices"], edge_list["dst_indices"]
    graph = nx.Graph()
    for ii, seq_cluster in enumerate(node_list):
        graph.add_node(ii, label=seq_cluster)
    for s, d in zip(src, dst, strict=True):
        graph.add_edge(s, d)
    return graph


def base_process(
    data_dict: dict,
    recipe_path: Path,
    targets: list[str] | None = None,
) -> dict:
    """Parse a CIFMol object using a predefined recipe."""
    parse_cache = ParsingCache()
    cooker = Cooker(parse_cache=parse_cache, recipebook=str(recipe_path))
    cooker.prep(data_dict, fields=list(data_dict.keys()))
    cooker.cook()
    return cooker.serve(targets=targets)


def cifmol_process(
    cif_id: str,
    recipe_path: Path,
    cif_db_path: Path,
    targets: list[str] | None = None,
) -> dict:
    """Parse a CIF file using a predefined recipe."""
    cifmol_dict = load_cif(cif_id, env_path=cif_db_path)
    output = {}
    for key in cifmol_dict:
        cifmol = cifmol_dict[key]
        output[key] = base_process(
            cifmol,
            recipe_path=recipe_path,
            targets=targets,
        )
    return output


def cifmol_meta_process(
    cif_ids: list[str],
    recipe_path: Path,
    cif_db_path: Path,
    targets: list[str] | None = None,
) -> dict:
    """Parse a CIF file using a predefined recipe."""
    cifmols = {}
    for cif_id in cif_ids:
        _cifmol_dict = load_cif(cif_id, env_path=cif_db_path)
        for key in _cifmol_dict:
            full_key = f"{cif_id}_{key}"
            cifmols[full_key] = _cifmol_dict[key]
    return base_process(
        {"cifmol_dict": cifmols},
        recipe_path=recipe_path,
        targets=targets,
    )


@click.group()
def cli() -> None:
    """Build and merge LMDB databases from CIF files."""


@cli.command("extract_fasta")
@click.argument("cif_db_path", type=click.Path(path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option("--njobs", "-j", type=int, default=-1, show_default=True)
def extract_fasta(
    cif_db_path: Path,
    output_path: Path,
    njobs: int = -1,
) -> None:
    """Extract fasta from CIFMol objects stored in LMDB database."""
    # python -m postprocessing.run extract_fasta /public_data/BioMolDBv2_2024Oct21/cif.lmdb/ /public_data/BioMolDBv2_2024Oct21/fasta/
    key_list = extract_key_list(cif_db_path)
    recipe_path = "postprocessing/recipes/extract_fasta.py"

    results = Parallel(n_jobs=njobs, verbose=10)(
        delayed(cifmol_process)(
            cif_id=key,
            recipe_path=recipe_path,
            cif_db_path=cif_db_path,
        )
        for key in key_list
    )
    results = dict(zip(key_list, results, strict=False))

    # remove redundancy
    path_to_lines = {}
    all_fasta = []

    for cif_key, outer_data in results.items():
        for inner_data in outer_data.values():
            for chain_id in inner_data["fasta"]:
                path = output_path / cif_key[1:3] / f"{cif_key}_{chain_id}.fasta"
                path_to_lines[path] = inner_data["fasta"][chain_id]
                all_fasta.append(inner_data["fasta"][chain_id])

    def _write_fasta(
        output_path: Path,
        lines: str,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            f.writelines(lines)

    click.echo(f"Writing {len(path_to_lines)} fasta files to {output_path}...")
    Parallel(n_jobs=njobs, verbose=10)(
        delayed(_write_fasta)(
            output_path=path,
            lines=lines,
        )
        for path, lines in path_to_lines.items()
    )

    # write merged_fasta file
    merged_fasta_path = output_path / "merged.fasta"
    click.echo(f"Writing merged fasta file to {merged_fasta_path}...")
    _write_fasta(
        output_path=merged_fasta_path,
        lines="".join(all_fasta),
    )


@cli.command("build_seq_hash_map")
@click.argument("merged_fasta_path", type=click.Path(path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
def build_seq_hash_map(
    merged_fasta_path: Path,
    output_path: Path,
) -> None:
    """Build a sequence hash map from the merged fasta file."""
    # python -m postprocessing.run build_seq_hash_map /public_data/BioMolDBv2_2024Oct21/fasta/merged.fasta /public_data/BioMolDBv2_2024Oct21/metadata/seq_hash_map.tsv
    fasta_dict = load_fasta(merged_fasta_path)
    recipe_path = "postprocessing/recipes/build_seq_hash_map.py"

    results = base_process(
        data_dict={"fasta_dict": fasta_dict},
        recipe_path=recipe_path,
        targets=["seq_hash_map"],
    )
    seq_hash_map = results["seq_hash_map"]

    # sort by 1. identifier, 2. sequence length
    seq_hash_map = dict(
        sorted(
            seq_hash_map.items(),
            key=lambda item: (item[0][0], len(item[1])),
        ),
    )

    # write to output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for seq_hash, sequence in seq_hash_map.items():
            f.write(f"{seq_hash}\t{sequence}\n")


@cli.command("seq_cluster")
@click.argument("seq_hash_map", type=click.Path(path_type=Path))
@click.argument("sabdab_summary_path", type=click.Path(path_type=Path))
@click.argument("tmp_dir", type=click.Path(path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option("--mmseqs2_seq_id", type=float, default=0.3, show_default=True)
@click.option("--mmseqs2_cov", type=float, default=0.8, show_default=True)
@click.option("--mmseqs2_covmode", type=int, default=0, show_default=True)
@click.option("--mmseqs2_clustermode", type=int, default=1, show_default=True)
def seq_cluster(
    seq_hash_map: Path,
    sabdab_summary_path: Path,
    tmp_dir: Path,
    output_path: Path,
    mmseqs2_seq_id: float,
    mmseqs2_cov: float,
    mmseqs2_covmode: int,
    mmseqs2_clustermode: int,
) -> None:
    """
    Cluster sequences...

    - Antibody: SabDab, CD-HIT (H3 if exists, else L3) cluster ID = A1234567
    - Peptides <10 residues, 100%
    - Polypeptide(D) (따로 나눠서)
    - Other Proteins: MMseqs2 easy-cluster (seq_id=0.3,cov=0.8,covmode=0,clustmode=1) cluster ID = P1234567
    - RNA, DNA, Ligands: 100% sequence identity cluster ID = N1234567 (same as sequence hash)
        /public_data/BioMolDBv2_2024Oct21/metadata/seq_hash_map.tsv ./postprocessing/tmp/ /public_data/BioMolDBv2_2024Oct21/cluster/seq_clusters.tsv

    Example:
    python -m postprocessing.run seq_cluster \
        /public_data/BioMolDBv2_2024Oct21/metadata/seq_hash_map.tsv \
        ./postprocessing/external_source/SabDab/sabdab_summary_all.tsv \
        /public_data/BioMolDBv2_2024Oct21/seq_cluster/tmp/ \
        /public_data/BioMolDBv2_2024Oct21/seq_cluster/seq_clusters.tsv \
        --mmseqs2_seq_id 0.3 \
        --mmseqs2_cov 0.8 \
        --mmseqs2_covmode 0 \
        --mmseqs2_clustermode 1
    """
    # tmp dir
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # recipe path
    recipe_path = "postprocessing/recipes/seq_cluster.py"
    cluster_dict = base_process(
        data_dict={
            "tmp_dir": tmp_dir,
            "seq_hash_map": seq_hash_map,
            "sabdab_summary_path": sabdab_summary_path,
            "mmseqs2_params": {
                "mmseqs2_seq_id": mmseqs2_seq_id,
                "mmseqs2_cov": mmseqs2_cov,
                "mmseqs2_covmode": str(mmseqs2_covmode),
                "mmseqs2_clustermode": str(mmseqs2_clustermode),
            },
        },
        recipe_path=recipe_path,
    )

    # write down the cluster_dict
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cluster_num = len(cluster_dict["cluster_dict"])
    total_members = 0
    with output_path.open("w") as f:
        for rep_seq_hash, member_list in cluster_dict["cluster_dict"].items():
            members = ",".join(member_list)
            f.write(f"c{rep_seq_hash}\t{members}\n")
            total_members += len(member_list)
    click.echo(f"Saved {cluster_num} clusters with {total_members} total members.")


@cli.command("graph_lmdb")
@click.argument("cif_db_path", type=click.Path(path_type=Path))
@click.argument("seq_hash_map", type=click.Path(path_type=Path))
@click.argument("seq_cluster_map", type=click.Path(path_type=Path))
@click.argument("graph_lmdb_path", type=click.Path(path_type=Path))
def graph_lmdb(
    cif_db_path: Path,
    seq_hash_map: Path,
    seq_cluster_map: Path,
    graph_lmdb_path: Path,
) -> None:
    """
    Extract graphs of CIFMol objects stored in LMDB database.

    Example:
    python -m postprocessing.run graph_lmdb \
        /public_data/BioMolDBv2_2024Oct21/cif.lmdb \
        /public_data/BioMolDBv2_2024Oct21/metadata/seq_hash_map.tsv \
        /public_data/BioMolDBv2_2024Oct21/seq_cluster/seq_clusters.tsv \
        /public_data/BioMolDBv2_2024Oct21/graph.lmdb
    """
    key_list = extract_key_list(cif_db_path)
    graph_lmdb_recipe_path = "postprocessing/recipes/graph_lmdb.py"

    seq_hash_to_seq: dict[str, int] = {}
    with seq_hash_map.open("r") as f:
        for line in f:
            seq_hash, seq = line.strip().split("\t")
            seq_hash_to_seq[seq_hash] = seq

    seq_hash_to_cluster: dict[int, str] = {}
    with seq_cluster_map.open("r") as f:
        for line in f:
            rep, members = line.strip().split("\t")
            for m in members.split(","):
                seq_hash_to_cluster[m] = rep

    seq_to_seq_hash_list = {}
    for seq_hash, seq in seq_hash_to_seq.items():
        if seq in seq_to_seq_hash_list:
            seq_to_seq_hash_list[seq].append(seq_hash)
        else:
            seq_to_seq_hash_list[seq] = [seq_hash]

    # ---------------------------------------
    # Worker on CHUNK basis
    # ---------------------------------------
    def _process_chunk(keys: list[str]) -> dict[str, bytes]:
        out: dict[str, bytes] = {}
        for cif_id in keys:
            cifmol_dict = load_cif(cif_id, env_path=cif_db_path)
            for inner_key, obj in cifmol_dict.items():
                cifmol = obj["cifmol"]
                result = base_process(
                    {
                        "cifmol": cifmol,
                        "seq_to_seq_hash_list": seq_to_seq_hash_list,
                        "seq_hash_to_cluster": seq_hash_to_cluster,
                    },
                    recipe_path=graph_lmdb_recipe_path,
                )
                graph_bytes = result["graph_bytes"]
                full_key = f"{cif_id}_{inner_key}"
                out[full_key] = graph_bytes
        return out

    # make chunks
    CHUNK = 200  # tuneable
    key_chunks = [key_list[i : i + CHUNK] for i in range(0, len(key_list), CHUNK)]
    click.echo(f"Processing {len(key_chunks)} chunks of size {CHUNK}...")

    # parallel batch
    chunk_results = Parallel(n_jobs=-1, verbose=10)(
        delayed(_process_chunk)(chunk) for chunk in key_chunks
    )

    # merge
    graph_dict: dict[str, bytes] = {}
    for d in chunk_results:
        graph_dict.update(d)

    # write to lmdb (original behaviour)
    all_keys = list(graph_dict.keys())
    click.echo(f"Writing {len(graph_dict)} graphs to LMDB at {graph_lmdb_path}...")
    env = lmdb.open(str(graph_lmdb_path), map_size=int(1e12))

    WRITE_CHUNK = 10_000
    for i in range(0, len(all_keys), WRITE_CHUNK):
        click.echo(
            f"Processing files {i} to {min(i + WRITE_CHUNK, len(all_keys))} / {len(all_keys)}"
        )
        data_chunk = all_keys[i : i + WRITE_CHUNK]
        with env.begin(write=True) as txn:
            for key in data_chunk:
                zcompressed = graph_dict[key]
                txn.put(key.encode(), zcompressed)


@cli.command("graph_cluster")
@click.argument("graph_lmdb_path", type=click.Path(path_type=Path))
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.argument("unique_graph_lmdb_path", type=click.Path(path_type=Path))
def graph_cluster(
    graph_lmdb_path: Path,
    output_dir: Path,
    unique_graph_lmdb_path: Path,
) -> None:
    """
    Clustering graphs of CIFMol objects stored in LMDB database.

    Example:
    python -m postprocessing.run graph_cluster \
        /public_data/BioMolDBv2_2024Oct21/graph.lmdb \
        /public_data/BioMolDBv2_2024Oct21/cluster/graph_cluster/ \
        /public_data/BioMolDBv2_2024Oct21/unique_graph.lmdb
    """
    key_list = extract_key_list(graph_lmdb_path)
    graph_cluster_recipe_path = "postprocessing/recipes/graph_cluster.py"

    # ---------------------------------------
    # Worker on CHUNK basis
    # ---------------------------------------
    def _process_chunk(keys: list[str]) -> dict[str, nx.Graph]:
        out: dict[str, nx.Graph] = {}
        for cif_id in keys:
            graph = load_graph(cif_id, env_path=graph_lmdb_path)
            out[cif_id] = graph
        return out

    # make chunks
    CHUNK = 200  # tuneable
    key_chunks = [key_list[i : i + CHUNK] for i in range(0, len(key_list), CHUNK)]
    click.echo(f"Processing {len(key_chunks)} chunks of size {CHUNK}...")

    # parallel batch
    graph_list = Parallel(n_jobs=-1, verbose=10)(
        delayed(_process_chunk)(chunk) for chunk in key_chunks
    )
    graph_map = {}
    for d in graph_list:
        graph_map.update(d)

    click.echo(f"Loaded {len(graph_map)} graphs from {graph_lmdb_path}.")

    results = base_process(
        {
            "graph_map": graph_map,
        },
        recipe_path=graph_cluster_recipe_path,
    )
    graph_clusters = results["graph_clusters"]  # list of set of graph IDs
    graph_hash_map = results["graph_hash_map"]
    graph_dict = results["graph_dict"]
    graph_cluster_save_path = output_dir / "graph_clusters.tsv"
    graph_hash_map_save_path = output_dir / "graph_hash_map.tsv"

    with graph_cluster_save_path.open("w") as f:
        for _cluster_id, _graph_hash_set in enumerate(graph_clusters):
            # 123 -> 000123
            graph_hash_set = [f"g{int(gid):06d}" for gid in _graph_hash_set]
            cluster_id = f"gc{_cluster_id:06d}"  # gc = graph cluster
            f.write(f"{cluster_id}\t{','.join(graph_hash_set)}\n")
    with graph_hash_map_save_path.open("w") as f:
        for cif_id in graph_hash_map:
            graph_hash = f"g{graph_hash_map[cif_id]:06d}"
            f.write(f"{cif_id}\t{graph_hash}\n")

    key_list = list(graph_dict.keys())  # graph_hash -> graph bytes
    click.echo(
        f"Writing {len(graph_dict)} graphs to LMDB at {unique_graph_lmdb_path}..."
    )
    env = lmdb.open(str(unique_graph_lmdb_path), map_size=int(1e12))  # ~1TB
    chunk = 10_000  # write in chunks of 10000
    for i in range(0, len(graph_dict), chunk):
        click.echo(
            f"Processing files {i} to {min(i + chunk, len(graph_dict))} / {len(graph_dict)}",
        )
        data_chunk = key_list[i : i + chunk]

        # --- Write results to LMDB ---
        with env.begin(write=True) as txn:
            for key in data_chunk:
                zcompressed_data = graph_dict[key]
                key = f"g{int(key):06d}"
                txn.put(key.encode(), zcompressed_data)


if __name__ == "__main__":
    cli()
