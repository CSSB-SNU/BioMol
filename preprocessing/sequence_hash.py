import sys
import hashlib
import pickle


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


def parse_fasta_unique_hash(file_path, digits):
    """
    Parse the FASTA file and compute a unique hash (6 or 8 digits) for each unique sequence.
    Uses a dictionary to store the mapping and a helper dict for collision resolution.
    """
    unique_seq_hash = {}
    used_candidates = {}
    current_seq = ""
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # When encountering a header, process the previous sequence (if any)
                if current_seq:
                    if current_seq not in unique_seq_hash:
                        candidate = get_unique_hash(current_seq, digits, used_candidates)
                        unique_seq_hash[current_seq] = candidate
                    current_seq = ""
            else:
                current_seq += line
        # Process the final sequence in the file
        if current_seq and current_seq not in unique_seq_hash:
            candidate = get_unique_hash(current_seq, digits, used_candidates)
            unique_seq_hash[current_seq] = candidate
    return unique_seq_hash


if __name__ == "__main__":
    fasta_file = "/public_data/psk6950/PDB_2024Mar18/entity/merged_protein.fasta"
    pickle_file = "/public_data/psk6950/PDB_2024Mar18/entity/sequence_hashes.pkl"

    # Parse the FASTA file and get the hash for each unique sequence
    seq_hashes = parse_fasta_unique_hash(fasta_file, digits=6)

    # Save the dictionary to a pickle file using the highest protocol for fast serialization
    with open(pickle_file, "wb") as pf:
        pickle.dump(seq_hashes, pf, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved {len(seq_hashes)} unique sequences to {pickle_file}")
