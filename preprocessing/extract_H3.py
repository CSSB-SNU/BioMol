from anarci import run_anarci

chotia_map = {
    'L1' : [ii for ii in range(24,34+1)],
    'L2' : [ii for ii in range(50,56+1)],
    'L3' : [ii for ii in range(89,97+1)],
    'H1' : [ii for ii in range(26,32+1)],
    'H2' : [ii for ii in range(50,65+1)],
    'H3' : [ii for ii in range(95,102+1)]
}

def remove_unknown(sequence : str):
    return sequence.replace('X','')

def _extract_H3L3_sequence(cdr_type:str, sequence: str) -> str:
    chotia_idx = chotia_map[cdr_type]
    result = run_anarci(sequence, scheme='chothia',ncpu=16)
    result = result[1][0][0][0]
    output = ''
    for res in result:
        idx = res[0][0]
        seq = res[1]
        if idx in chotia_idx:
            output += seq
    return output

def extract_H3L3_sequence(full_fasta_file: str, output_fasta_file: str):
    # if light chain only, get L3
    Ab_chain = {}
    with open(full_fasta_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith('>'):
            chain_id = line.split('|')[0][1:].strip()
            chain_type = line.split('|')[1].strip()
        else:
            sequence = line.strip()
            Ab_chain[chain_id] = {chain_type: sequence}

    heavy_chain = {}

    for chain_id in Ab_chain:
        if 'Heavy' in Ab_chain[chain_id].keys():
            heavy_chain[chain_id] = ('H3', Ab_chain[chain_id]['Heavy'])
        else :
            print(f"Light chain only : {chain_id}")
            heavy_chain[chain_id] = ('L3', Ab_chain[chain_id]['Light'])

    H3L3_chain = {}
    error_items = []
    for chain_id, (cdr_type, sequence) in heavy_chain.items():
        wo_unknown_sequence = remove_unknown(sequence)
        try:
            H3L3_sequence = _extract_H3L3_sequence(cdr_type, wo_unknown_sequence)
        except:
            print(f"Error in {chain_id}")
            error_items.append((chain_id, cdr_type))
        H3L3_chain[chain_id] = H3L3_sequence
    print(f"Error items : {error_items}")

    with open(output_fasta_file, 'w') as f:
        for chain_id, sequence in H3L3_chain.items():
            f.write(f">{chain_id}\n")
            f.write(f"{sequence}\n")

if __name__ == "__main__":
    full_fasta_file = '/data/psk6950/PDB_2024Mar18/AbAg/Ab.fasta'
    output_fasta_file = '/data/psk6950/PDB_2024Mar18/AbAg/H3L3.fasta'
    extract_H3L3_sequence(full_fasta_file, output_fasta_file)