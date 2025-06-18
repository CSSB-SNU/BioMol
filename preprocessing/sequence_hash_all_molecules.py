import os
import hashlib
import pickle
from BioMol import DB_PATH, SIGNALP_PATH

pickle_file = f"{DB_PATH}/entity/sequence_hashes.pkl"

with open(pickle_file, "rb") as pf:
    protein_only_seq_hashes = pickle.load(pf)
protein_only_seq_hashes = {k :  str(v).zfill(6) for k, v in protein_only_seq_hashes.items()}

def filter_protein(fasta_txt):
    first_line = fasta_txt.split("\n")[0]
    is_protein = first_line.split("| ")[-1] == "PolymerType.PROTEIN"
    return is_protein


def read_fasta(fasta_path):
    with open(fasta_path, "r") as f:
        fasta_txt = f.read()
    return fasta_txt


def merge_all_fasta(entity_dir, save_path):
    fasta = []
    for inner_dir in os.listdir(entity_dir):
        if not os.path.isdir(os.path.join(entity_dir, inner_dir)):
            continue
        for file in os.listdir(os.path.join(entity_dir, inner_dir)):
            if file.endswith(".fasta"):
                fasta_txt = read_fasta(os.path.join(entity_dir, inner_dir, file))
                fasta.append(fasta_txt)

    with open(save_path, "w") as f:
        for fasta_txt in fasta:
            f.write(fasta_txt)


def parse_signalp(signalp_path: str) -> list[str]:
    """
    Parse the signalp output file and extract the sequence IDs.
    """
    if not os.path.exists(signalp_path):
        return None
    with open(signalp_path) as f:
        lines = f.readlines()

    result = lines[1].split("\t")
    return int(result[3]) - 1, int(result[4]) - 1


def get_unique_hash(sequence, digits, used_candidates):
    """
    Compute a unique hash for a given sequence:
    - Start with an MD5 hash reduced modulo 10**digits.
    - If that candidate is already used by a different sequence, use linear probing to find the next available number.
    """
    mod = 10**digits
    candidate = int(hashlib.md5(sequence.encode("utf-8")).hexdigest(), 16) % mod
    original_candidate = candidate
    while candidate in used_candidates and used_candidates[candidate] != sequence:
        candidate = (candidate + 1) % mod
        if candidate == original_candidate:
            raise ValueError("Exhausted all hash values; too many collisions.")
    used_candidates[candidate] = sequence
    return candidate


# def parse_fasta_unique_hash(file_path, digits):
#     """
#     Parse the FASTA file and compute a unique hash (6 or 8 digits) for each unique sequence.
#     Uses a dictionary to store the mapping and a helper dict for collision resolution.
#     """
#     unique_seq_hash = {}
#     used_candidates = {}
#     current_seq = ""
#     with open(file_path, "r") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             if line.startswith(">"):
#                 # When encountering a header, process the previous sequence (if any)
#                 if current_seq:
#                     if current_seq not in unique_seq_hash:
#                         candidate = get_unique_hash(current_seq, digits, used_candidates)
#                         unique_seq_hash[current_seq] = candidate
#                     current_seq = ""
#             else:
#                 current_seq += line
#         # Process the final sequence in the file
#         if current_seq and current_seq not in unique_seq_hash:
#             candidate = get_unique_hash(current_seq, digits, used_candidates)
#             unique_seq_hash[current_seq] = candidate
#     return unique_seq_hash

def parse_fasta_unique_hash(file_path: str, digits: int) -> dict[str, str]:
    mod = 10**digits
    width = digits

    unique_seqs = set()
    seq = ""
    molecule_type_map = {
        'PolymerType.PROTEIN': '[PROTEIN]:',
        'PolymerType.DNA': '[DNA]:',
        'PolymerType.RNA': '[RNA]:',
        'PolymerType.NA_HYBRID': '[NA_HYBRID]:',
        'NONPOLYMER': '[NONPOLYMER]:',
        'BRANCHED': '[BRANCHED]:',
    }
    molecule_type_tag = ''

    with open(file_path, "r") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line[0] == ">":
                if seq:
                    if '  | ' in seq:
                        seq = seq.replace('  | ', '|')
                    unique_seqs.add(f"{molecule_type_tag}{seq}")
                molecule_type_tag = molecule_type_map[line.split("| ")[-1]]
                seq = ""
            else:
                seq += line
        if seq:
            unique_seqs.add(seq)
    updated_hashes = protein_only_seq_hashes.copy()
    used_hashes = set(protein_only_seq_hashes.values())
    unique_seqs = list(unique_seqs)

    for s in unique_seqs:
        if s.replace("[PROTEIN]:", "") in protein_only_seq_hashes:
            updated_hashes[s] = protein_only_seq_hashes[s.replace("[PROTEIN]:", "")]
            updated_hashes.pop(s.replace("[PROTEIN]:", ""), None)
            continue
        base = hashlib.md5(s.encode()).hexdigest()
        h_int = int(base, 16) % mod
        h_str = str(h_int).zfill(width)
        while h_str in used_hashes:
            base = hashlib.md5((s + h_str).encode()).hexdigest()
            h_int = int(base, 16) % mod
            h_str = str(h_int).zfill(width)
        updated_hashes[s] = h_str
        used_hashes.add(h_str)
    return updated_hashes


if __name__ == "__main__":
    # merge all fasta files into one
    # merge_all_fasta(
    #     f"{DB_PATH}/entity",
    #     f"{DB_PATH}/entity/merged_all_molecules.fasta",
    # )


    fasta_file = f"{DB_PATH}/entity/merged_all_molecules.fasta"
    pickle_file = f"{DB_PATH}/entity/sequence_hashes_all_molecules.pkl"

    # Parse the FASTA file and get the hash for each unique sequence
    seq_hashes = parse_fasta_unique_hash(fasta_file, digits=6,)

    # Save the dictionary to a pickle file using the highest protocol for fast serialization
    with open(pickle_file, "wb") as pf:
        pickle.dump(seq_hashes, pf, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved {len(seq_hashes)} unique sequences to {pickle_file}")
