from __future__ import annotations

import gzip
import io
import json
import pickle
from pathlib import Path
import os

import click
import lmdb
from joblib import Parallel, delayed


@click.group()
def cli() -> None:
    """Build and merge LMDB databases from CIF files."""


@cli.command("compare_fasta")
@click.argument("v1_path", type=click.Path(path_type=Path))
@click.argument("v2_path", type=click.Path(path_type=Path))
@click.option("--njobs", "-j", type=int, default=-1, show_default=True)
def compare_fasta(
    v1_path: Path,
    v2_path: Path,
    njobs: int = -1,
) -> None:
    """Compare fasta files from v1 and v2 paths."""
    # python -m postprocessing.compare compare_fasta /public_data/BioMolDB_2024Oct21/entity/merged_all_molecules.fasta /public_data/BioMolDBv2_2024Oct21/fasta/merged.fasta

    def _load_seq_dict_from_fasta(fasta: Path) -> dict:
        with fasta.open("r") as f:
            lines = f.readlines()
        type_dict = {}
        seq_dict = {}
        for line in lines:
            if line.startswith(">"):
                header = line.strip()[1:]
                cif_ID, entity_type = header.split("|")[0:2]
                cif_ID = cif_ID.strip().upper()
                entity_type = entity_type.strip()
                seq_dict[cif_ID] = ""
                type_dict[cif_ID] = entity_type
            else:
                seq_dict[cif_ID] += line.strip()
        return seq_dict, type_dict

    v1_seq_dict, v1_type_dict = _load_seq_dict_from_fasta(v1_path)
    v2_seq_dict, v2_type_dict = _load_seq_dict_from_fasta(v2_path)

    cif1_list = list(v1_seq_dict.keys())
    cif2_list = list(v2_seq_dict.keys())
    common = set(cif1_list) & set(cif2_list)
    diff1 = set(cif1_list) - set(cif2_list)
    diff2 = set(cif2_list) - set(cif1_list)

    # for diff1, it should be unknown sequences
    # for diff2, it should be H2O

    to_check_1 = []
    to_check_2 = []
    print(f"Number of common fasta files: {len(common)}")
    print(f"Number of fasta files only in v1: {len(diff1)}")
    print(f"Number of fasta files only in v2: {len(diff2)}")
    for cif_id in diff1:
        seq = v1_seq_dict[cif_id]
        if all(residue == "X" for residue in seq) or seq in ("(UNK)", "(UNL)"):
            continue
        to_check_1.append(cif_id)
    for cif_id in diff2:
        seq = v2_seq_dict[cif_id]
        if seq in ("(HOH)", "(DOD)"):
            continue
        to_check_2.append(cif_id)

    def _compare_each_fasta(seq1: str, seq2: str) -> tuple[bool, int, int]:
        len1 = len(seq1)
        len2 = len(seq2)
        is_match = seq1 == seq2
        return is_match, len1, len2

    # test
    results = Parallel(n_jobs=njobs, verbose=10)(
        delayed(_compare_each_fasta)(v1_seq_dict[cif_id], v2_seq_dict[cif_id])
        for cif_id in common
    )
    n_total = len(results)
    n_match = sum(1 for res in results if res[0])
    n_mismatch = n_total - n_match
    click.echo(f"Total fasta files: {n_total}")
    click.echo(f"Matching fasta files: {n_match}")
    click.echo(f"Non-matching fasta files: {n_mismatch}")
    if n_mismatch > 0:
        results = {cif_id: res for cif_id, res in zip(common, results)}
        click.echo("Mismatched files:")
        wrong_cif_ids = [cif_id for cif_id, res in results.items() if not res[0]]
        wrong_cif_id_types = [
            (v1_type_dict[cif_id], v2_type_dict[cif_id]) for cif_id in wrong_cif_ids
        ]
        # -> all branched.


def _ensure_env(lmdb_path: Path) -> lmdb.Environment:
    global _ENV, _LMDB_PATH
    if _ENV is None or _LMDB_PATH != lmdb_path:
        # readonly + lock=False + readahead=False : 다수 리더 병렬 읽기에 유리
        _ENV = lmdb.open(
            str(lmdb_path),
            readonly=True,
            lock=False,
            readahead=False,
            max_readers=4096,
        )
        _LMDB_PATH = lmdb_path
    return _ENV


_ENV = None  # worker-local (프로세스 전역)
_LMDB_PATH = None


def load_data_from_lmdb_and_write(
    lmdb_path: Path,
    key: str,
    to_path: Path,
    skip_if_exists: bool = True,
) -> None:
    if skip_if_exists and to_path.exists():
        return 0
    new_hash = to_path.stem

    env = _ensure_env(lmdb_path)

    with env.begin(buffers=True) as txn:
        buf = txn.get(key.encode("utf-8"))
    if buf is None:
        return 0 # 혹은 raise

    gz_bytes = pickle.loads(bytes(buf))
    decompressed = gzip.decompress(gz_bytes)
    with io.BytesIO(decompressed) as bio:
        lines = bio.read().decode("utf-8")
    lines = lines.splitlines()
    # change first header
    text = f">{new_hash}\n" + "\n".join(lines[1:])

    to_path.parent.mkdir(parents=True, exist_ok=True)
    with to_path.open("w") as f:
        f.write(text)
    return 1


@cli.command("remap_seq_hash")
@click.argument("v1_path", type=click.Path(path_type=Path))
@click.argument("v2_path", type=click.Path(path_type=Path))
@click.option("--njobs", "-j", type=int, default=-1, show_default=True)
def remap_seq_hash(v1_path: Path, v2_path: Path, njobs: int = -1) -> None:
    """Remap protein sequence hash from v1 to v2."""
    # python -m postprocessing.compare remap_seq_hash /public_data/BioMolDB_2024Oct21/metadata/metadata_psk_new.csv /public_data/BioMolDBv2_2024Oct21/metadata/seq_hash_map.tsv
    # load v2 seq_hash_map
    v2_seq_hash_map = {}
    with v2_path.open("r") as f:
        lines = f.readlines()
    for line in lines:
        seq_hash, sequence = line.strip().split("\t")
        if seq_hash.startswith("P"):
            v2_seq_hash_map[sequence] = seq_hash

    # load v1 metadata and build seq_hash map (v1)
    v1_seq_hash_map = {}
    with v1_path.open("r") as f:
        lines = f.readlines()
    for ii, line in enumerate(lines):
        if ii == 0:
            continue
        _split = line.strip().split(",")
        seq_hash, sequence = _split[-3], _split[-1]
        if sequence.startswith("[PROTEIN]:"):
            sequence = sequence[len("[PROTEIN]:") :]
            v1_seq_hash_map[sequence] = seq_hash
        else:
            continue
    v1_seq_hash_to_seq = {v1_seq_hash_map[seq]: seq for seq in v1_seq_hash_map}
    v1_to_v2_seq_hash = {}
    for sequence in v2_seq_hash_map:
        if sequence in v1_seq_hash_map:
            v1_seq_hash = v1_seq_hash_map[sequence]
            v2_seq_hash = v2_seq_hash_map[sequence]
            v1_to_v2_seq_hash[v1_seq_hash] = v2_seq_hash

    # remap signalp files
    v1_signalp_path = v1_path.parent.parent / "signalp"
    signalp_lines = {}
    for file in os.listdir(v1_signalp_path):
        if not file.endswith(".gff3"):
            continue
        with (v1_signalp_path / file).open("r") as f:
            lines = f.readlines()
        v1_seq_hash = file.split(".")[0]
        sequence = v1_seq_hash_to_seq[v1_seq_hash]
        if sequence not in v2_seq_hash_map:
            continue
        v2_seq_hash = v2_seq_hash_map[sequence]
        signalp_lines[v2_seq_hash] = lines
    # write to v2 signalp path
    v2_signalp_path = v2_path.parent.parent / "signalp"
    for v2_seq_hash in signalp_lines:
        with (v2_signalp_path / f"{v2_seq_hash}.gff3").open("w") as f:
            f.writelines(signalp_lines[v2_seq_hash])

    v1_seq_hash_list = list(v1_seq_hash_to_seq.keys())
    # remap a3m.lmdb
    v1_a3m_lmdb_path = v1_path.parent.parent / "a3m.lmdb"

    v2_a3m_path = v2_path.parent.parent / "a3m"
    print(f"Total a3m to remap: {len(v1_seq_hash_list)}")
    total_rewritten = 0
    for start in range(0, len(v1_seq_hash_list), 1000):
        end = min(start + 1000, len(v1_seq_hash_list))
        batch = v1_seq_hash_list[start:end]
        print(f"Processing batch {start}–{end - 1}...")

        results = Parallel(n_jobs=njobs, verbose=10)(
            delayed(load_data_from_lmdb_and_write)(
                lmdb_path=v1_a3m_lmdb_path,
                key=v1_seq_hash,
                to_path=v2_a3m_path
                / f"{v1_to_v2_seq_hash[v1_seq_hash][:4]}"
                / f"{v1_to_v2_seq_hash[v1_seq_hash]}.a3m",
            )
            for v1_seq_hash in batch
            if v1_seq_hash in v1_to_v2_seq_hash
        )
        total_rewritten += sum(results)
    print(f"Total a3m files remapped: {total_rewritten}")


if __name__ == "__main__":
    cli()
