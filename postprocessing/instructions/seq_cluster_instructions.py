from pathlib import Path
from collections import deque
from collections.abc import Callable, Iterable
from typing import TypeVar
from anarci import run_anarci

import os

from numpy.typing import NDArray

from biomol.cif.mol import CIFMol

InputType = TypeVar("InputType", str, int, float)
FeatureType = TypeVar("FeatureType")
NumericType = TypeVar("NumericType", int, float)


def run_mmseqs2(
    fasta_path: Path,
    tmp_dir: Path,
    mmseqs2_seq_id: float = 0.3,
    mmseqs2_cov: float = 0.8,
    mmseqs2_covmode: str = "0",
    mmseqs2_clustermode: str = "1",
):
    os.system(
        f"mmseqs easy-cluster {fasta_path} {tmp_dir} {tmp_dir}/tmp/ --min-seq-id {mmseqs2_seq_id} -c {mmseqs2_cov} --cov-mode {mmseqs2_covmode} --cluster-mode {mmseqs2_clustermode}"
    )


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


def separate_sequences() -> Callable[..., type[InputType]]:
    """Separate sequences into different types and write to different fasta files."""

    def _worker(
        tmp_dir: Path,
        seq_hash_map: Path,
    ) -> type[InputType]:
        # 1. load seq_hash_map and fasta
        seq_hash_dict = {}
        seq_to_hash = {}
        with seq_hash_map.open("r") as f:
            for line in f:
                line = line.strip()
                seq_hash, sequence = line.split("\t")
                seq_hash_dict[seq_hash] = sequence
                seq_to_hash[sequence] = seq_hash

        # # 2. Parse SabDab summary to get antibody sequences
        # fasta_dict = load_fasta(merged_fasta_path)
        # chain_ID_to_seq = {}
        # for header in fasta_dict:
        #     sequence = fasta_dict[header]
        #     chain_ID = header.split("|")[0].strip()
        #     chain_ID_to_seq[chain_ID] = sequence

        # ab_H_chain_list, ab_L_chain_list = [], []
        # with sabdab_summary_path.open("r") as f:
        #     lines = f.readlines()
        # for line in lines[1:]:
        #     line = line.strip()
        #     items = line.split("\t")
        #     pdb_ID, ab_H_chain, ab_L_chain = items[0], items[1], items[2]
        #     ab_H_chain = f"{pdb_ID}_{ab_H_chain}".upper() if ab_H_chain != "NA" else None
        #     ab_L_chain = f"{pdb_ID}_{ab_L_chain}".upper() if ab_L_chain != "NA" else None
        #     if ab_H_chain in chain_ID_to_seq:
        #         seq_hash = seq_to_hash[chain_ID_to_seq[ab_H_chain]]
        #         ab_H_chain_list.append(seq_hash)
        #     if ab_L_chain in chain_ID_to_seq:
        #         seq_hash = seq_to_hash[chain_ID_to_seq[ab_L_chain]]
        #         ab_L_chain_list.append(seq_hash)

        # 20251102, for now cif lmdb doesn't have auth_asym_id so I use already_parsed files
        to_be_replaced = Path("/public_data/BioMolDB_2024Oct21/AbAg/Ab.fasta")
        ab_fasta_dict = load_fasta(to_be_replaced)
        ab_H_hash_list, ab_L_hash_list = [], []

        for header in ab_fasta_dict:
            sequence = ab_fasta_dict[header]
            chain_ID, hl = header.split("|")
            chain_ID, hl = chain_ID.strip(), hl.strip()
            seq_hash = seq_to_hash[sequence]
            if hl == "Heavy":
                ab_H_hash_list.append(seq_hash)
            elif hl == "Light":
                ab_L_hash_list.append(seq_hash)

        # 3. Divide sequences into protein, peptide, nucleotide etc.
        entity_types = [
            "antibody_H",
            "antibody_L",
            "protein",
            "peptide",
            "protein_D",
            "nucleotide",
            "ligand",
        ]
        entity_dict = {etype: [] for etype in entity_types}
        for seq_hash, sequence in seq_hash_dict.items():
            if seq_hash in ab_H_hash_list:
                etype = "antibody_H"
            elif seq_hash in ab_L_hash_list:
                etype = "antibody_L"
            else:
                entity_identifier = seq_hash[0]
                match entity_identifier:
                    case "P":
                        etype = "protein"
                    case "Q":
                        etype = "protein_D"
                    case "N":
                        etype = "nucleotide"
                    case "R":
                        etype = "nucleotide"
                    case "D":
                        etype = "nucleotide"
                    case "L":
                        etype = "ligand"
                    case "B":
                        etype = "ligand"
                    case "X":
                        etype = "ligand"
                    case _:
                        etype = "ligand"
                if etype in ("protein", "protein_D") and len(sequence) < 10:
                    etype = "peptide"
            entity_dict[etype].append(seq_hash)
        fasta_tmp_dir = tmp_dir / "fasta"
        fasta_tmp_dir.mkdir(parents=True, exist_ok=True)
        fasta_path_dict = {}
        total_count = sum(len(v) for v in entity_dict.values())
        for etype, seq_hash_list in entity_dict.items():
            fasta_path = fasta_tmp_dir / f"{etype}.fasta"
            fasta_path_dict[etype] = fasta_path
            with fasta_path.open("w") as f:
                for seq_hash in seq_hash_list:
                    sequence = seq_hash_dict[seq_hash]
                    f.write(f">{seq_hash}\n{sequence}\n")
        return fasta_path_dict

    return _worker


def parse_mmseqs2_cluster(
    mmseqs2_cluster_tsv: Path,
) -> dict[str, list[str]]:
    """Parse MMSeqs2 cluster TSV file into a dictionary mapping representative to members."""
    cluster_dict = {}
    with mmseqs2_cluster_tsv.open("r") as f:
        for _line in f:
            line = _line.strip()
            representative, member = line.split("\t")
            if representative not in cluster_dict:
                cluster_dict[representative] = []
            cluster_dict[representative].append(member)
    return cluster_dict


def protein_cluster() -> Callable[..., type[InputType]]:
    """Cluster protein sequences using MMSeqs2."""

    def _worker(
        tmp_dir: Path,
        fasta_path_dict: dict[str, Path],
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        protein_path = fasta_path_dict["protein"]
        protein_d_path = fasta_path_dict["protein_D"]
        mmseqs2_protein_tmp_dir = tmp_dir / "mmseqs2_protein"
        mmseqs2_protein_d_tmp_dir = tmp_dir / "mmseqs2_protein_D"
        mmseqs2_protein_tmp_dir.mkdir(parents=True, exist_ok=True)
        mmseqs2_protein_d_tmp_dir.mkdir(parents=True, exist_ok=True)
        protein_clustered_path = tmp_dir / "mmseqs2_protein_cluster.tsv"
        protein_d_clustered_path = tmp_dir / "mmseqs2_protein_D_cluster.tsv"
        run_mmseqs2(protein_path, mmseqs2_protein_tmp_dir)
        run_mmseqs2(protein_d_path, mmseqs2_protein_d_tmp_dir)
        protein_cluster_dict = parse_mmseqs2_cluster(protein_clustered_path)
        protein_d_cluster_dict = parse_mmseqs2_cluster(protein_d_clustered_path)
        return protein_cluster_dict, protein_d_cluster_dict

    return _worker


chotia_map = {
    "L1": list(range(24, 34 + 1)),
    "L2": list(range(50, 56 + 1)),
    "L3": list(range(89, 97 + 1)),
    "H1": list(range(26, 32 + 1)),
    "H2": list(range(50, 65 + 1)),
    "H3": list(range(95, 102 + 1)),
}


def _extract_H3L3_sequence(cdr_type: str, sequence: str) -> str:
    """Return H3 or L3 sequence from full antibody sequence using ANARCI."""
    chotia_idx = chotia_map[cdr_type]
    result = run_anarci(sequence, scheme="chothia", ncpu=16)
    result = result[1][0][0][0]
    output = ""
    for res in result:
        idx = res[0][0]
        seq = res[1]
        if idx in chotia_idx:
            output += seq
    return output


def extract_H3L3_sequence(full_fasta: Path, output_fasta: Path, cdr_type: str) -> None:
    """Extract H3 or L3 sequences from full antibody sequences in a fasta file."""
    # if light chain only, get L3
    Ab_chain = {}
    with full_fasta.open("r") as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith(">"):
            seq_hash = line.strip()[1:]
        else:
            sequence = line.strip()
            Ab_chain[seq_hash] = sequence

    H3L3_chain = {}
    for seq_hash, sequence in Ab_chain.items():
        wo_unknown_sequence = sequence.replace("X", "")
        H3L3_sequence = _extract_H3L3_sequence(cdr_type, wo_unknown_sequence)
        H3L3_chain[seq_hash] = H3L3_sequence

    with output_fasta.open("w") as f:
        for seq_hash, sequence in H3L3_chain.items():
            f.write(f">{seq_hash}_{cdr_type}\n")
            f.write(f"{sequence}\n")


def run_CDHIT(
    input_fasta: Path,
    output_path: Path,
    seq_id: float = 0.9,
) -> None:
    """
    Run CD-HIT to cluster sequences.

    Other parameters are set to recommended values for antibody clustering.
    """
    os.system(
        f"cd-hit -i {input_fasta} -o {output_path} -c {seq_id} -n 2 -M 256000 -d 0 -T 0 -l 1 -s 0.8 -aL 0.8 -aS 0.8 -g 1"
    )


def parse_cdhit_cluster(
    cdhit_cluster_dat: Path,
) -> dict[str, list[str]]:
    """Parse CD-HIT cluster .dat file into a dictionary mapping representative to members."""
    cluster_dict = {}
    current_rep = ""
    with cdhit_cluster_dat.open("r") as f:
        for _line in f:
            line = _line.strip()
            if line.startswith(">Cluster"):
                continue
            else:
                parts = line.split()
                seq_info = parts[2]
                seq_hash = seq_info.split("...")[0]
                seq_hash = (
                    seq_hash.replace(">", "").replace("_H3", "").replace("_L3", "")
                )
                if line.startswith("0"):
                    current_rep = seq_hash.replace("P", "A")  # representative ID P->A
                    cluster_dict[current_rep] = [seq_hash]
                else:
                    cluster_dict[current_rep].append(seq_hash)
    return cluster_dict


def antibody_cluster() -> Callable[..., type[InputType]]:
    """Cluster antibody sequences using CD-HIT."""

    def _worker(
        tmp_dir: Path,
        fasta_path_dict: dict[str, Path],
    ) -> None:
        ab_heavy_fasta_path = fasta_path_dict["antibody_H"]
        ab_light_fasta_path = fasta_path_dict["antibody_L"]
        ab_H3_fasta_path = tmp_dir / "fasta" / "antibody_H3.fasta"
        ab_L3_fasta_path = tmp_dir / "fasta" / "antibody_L3.fasta"

        ab_seq_hash_list = []
        with ab_heavy_fasta_path.open("r") as f:
            for line in f:
                if line.startswith(">"):
                    seq_hash = line.strip()[1:]
                    ab_seq_hash_list.append(seq_hash)
        with ab_light_fasta_path.open("r") as f:
            for line in f:
                if line.startswith(">"):
                    seq_hash = line.strip()[1:]
                    ab_seq_hash_list.append(seq_hash)

        if not ab_H3_fasta_path.exists():
            extract_H3L3_sequence(ab_heavy_fasta_path, ab_H3_fasta_path, cdr_type="H3")
        if not ab_L3_fasta_path.exists():
            extract_H3L3_sequence(ab_light_fasta_path, ab_L3_fasta_path, cdr_type="L3")
        ab_H_clustered_path = tmp_dir / "cdhit_ab_H_clustered.dat"
        ab_L_clustered_path = tmp_dir / "cdhit_ab_L_clustered.dat"
        if not ab_H_clustered_path.exists():
            run_CDHIT(ab_H3_fasta_path, ab_H_clustered_path)
        if not ab_L_clustered_path.exists():
            run_CDHIT(ab_L3_fasta_path, ab_L_clustered_path)
        ab_H_cluster_output = tmp_dir / "cdhit_ab_H_clustered.dat.clstr"
        ab_L_cluster_output = tmp_dir / "cdhit_ab_L_clustered.dat.clstr"
        ab_H_cluster_dict = parse_cdhit_cluster(ab_H_cluster_output)
        ab_L_cluster_dict = parse_cdhit_cluster(ab_L_cluster_output)
        ab_cluster_dict = {**ab_H_cluster_dict, **ab_L_cluster_dict}
        # TODO(psk) 20251103 : 이유는 모르겠지만, 씹히는 애들이 있음 (CDHIT에서)
        output_ab = sum(ab_cluster_dict.values(), [])
        for ab_hash in ab_seq_hash_list:
            if ab_hash not in output_ab:
                ab_cluster_dict[ab_hash.replace("P", "A")] = [ab_hash]
        return ab_cluster_dict

    return _worker


def write_cluster() -> Callable[..., type[InputType]]:
    """Write cluster dictionaries to pickle files."""

    def _worker(
        fasta_path_dict: dict[str, Path],
        protein_cluster_dict: dict[str, list[str]],
        protein_d_cluster_dict: dict[str, list[str]],
        antibody_cluster_dict: dict[str, list[str]],
    ) -> None:
        cluster_dict = {
            **protein_cluster_dict,
            **protein_d_cluster_dict,
            **antibody_cluster_dict,
        }

        for etype, fasta_path in fasta_path_dict.items():
            if etype in ("protein", "protein_D", "antibody_H", "antibody_L"):
                continue
            with fasta_path.open("r") as f:
                lines = f.readlines()
            for line in lines:
                if line.startswith(">"):
                    seq_hash = line.strip()[1:]
                    cluster_dict[seq_hash] = [seq_hash]
        return cluster_dict

    return _worker
